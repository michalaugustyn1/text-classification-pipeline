#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/apptainer/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/apptainer/.models_dir
_OLLAMA_SERVER_PID=""

# Return the physical /dev/nvidia minor number for the SLURM-allocated GPU.
# SLURM's CUDA_VISIBLE_DEVICES is a remapped logical index (always 0,1,...),
# not the physical GPU number. We reverse-engineer the physical index so
# NVIDIA_VISIBLE_DEVICES can tell Apptainer to bind only that one device.
_physical_gpu_minor() {
    # Get the PCI bus ID of the allocated GPU via the SLURM-visible nvidia-smi.
    local BUS; BUS=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader 2>/dev/null | head -1)
    [[ -z "$BUS" ]] && return

    # Method 1: /proc/driver/nvidia — most reliable, gives kernel minor directly.
    local PCI="0000:$(echo "${BUS#00000000:}" | tr '[:upper:]' '[:lower:]')"
    local MINOR; MINOR=$(grep -i "DeviceMinor" "/proc/driver/nvidia/gpus/${PCI}/information" 2>/dev/null | awk '{print $NF}')
    if [[ -n "$MINOR" ]]; then echo "$MINOR"; return; fi

    # Method 2: enumerate all GPUs with nvidia-smi, match by bus ID.
    local IDX; IDX=$(CUDA_VISIBLE_DEVICES="" nvidia-smi --query-gpu=gpu_bus_id \
                     --format=csv,noheader 2>/dev/null \
                     | grep -in "$BUS" | cut -d: -f1 | head -1)
    [[ -n "$IDX" ]] && echo "$((IDX - 1))"
}

# Build Apptainer GPU passthrough flags.
# Tries --nvccli first (requires nvidia-container-toolkit on the host).
# Falls back to --nv.  The caller is responsible for setting
# NVIDIA_VISIBLE_DEVICES before invoking apptainer so --nv binds only
# the correct physical GPU device (see ollama_start).
_build_nv_flags() {
    local SIF="$1"

    if ! nvidia-smi &>/dev/null 2>&1; then
        echo ""; return
    fi

    if apptainer exec --nvccli "$SIF" true &>/dev/null 2>&1; then
        echo "--nvccli"; return
    fi

    local FLAGS="--nv"
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
               /dev/nvidia-modeset /dev/nvidia-caps; do
        [[ -e "$dev" ]] && FLAGS="$FLAGS --bind $dev:$dev"
    done
    echo "$FLAGS"
}

ollama_start() {
    [[ -f "$_OLLAMA_SIF" ]]         || { echo "ERROR: $_OLLAMA_SIF not found. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }
    [[ -f "$_OLLAMA_MODELS_FILE" ]] || { echo "ERROR: $_OLLAMA_MODELS_FILE not found. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }
    local MODELS_DIR; MODELS_DIR="$(cat "$_OLLAMA_MODELS_FILE")"
    local PORT=$(( 11500 + (SLURM_JOB_ID % 500) ))
    export OLLAMA_HOST="http://localhost:$PORT"
    export OLLAMA_PORT="$PORT"

    local NV_FLAGS; NV_FLAGS="$(_build_nv_flags "$_OLLAMA_SIF")"
    local NV_ENV=()

    if [[ -n "$NV_FLAGS" ]]; then
        # On multi-GPU nodes SLURM remaps the allocated GPU to logical index 0
        # via CUDA_VISIBLE_DEVICES. Apptainer --nv binds ALL /dev/nvidia* files,
        # so cuInit tries physical GPU 0 which may not be the allocated one —
        # SLURM's cgroup blocks it and returns CUDA_ERROR_NO_DEVICE (100).
        #
        # Fix: set NVIDIA_VISIBLE_DEVICES to the PHYSICAL minor number of the
        # allocated GPU. Apptainer --nv will then bind only that device, so
        # CUDA inside the container enumerates exactly one accessible GPU.
        local PHY_MINOR; PHY_MINOR=$(_physical_gpu_minor)
        if [[ -n "$PHY_MINOR" ]]; then
            NV_ENV+=(NVIDIA_VISIBLE_DEVICES="$PHY_MINOR")
            echo "GPU: physical /dev/nvidia${PHY_MINOR} → NVIDIA_VISIBLE_DEVICES=${PHY_MINOR}" >&2
        fi

        # Bind the driver library to a path Ollama's CUDA detection scans.
        local CUDA_LIB; CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        [[ -n "$CUDA_LIB" ]] && NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        echo "GPU passthrough flags: $NV_FLAGS" >&2
    else
        echo "No GPU detected — running CPU-only" >&2
    fi

    # Use exec so CUDA_VISIBLE_DEVICES is unset INSIDE the container before
    # Ollama starts. SLURM's inherited CUDA_VISIBLE_DEVICES=0 can cause
    # cuInit to return CUDA_ERROR_NO_DEVICE (100) inside the container namespace
    # even when the device files are correct. Ollama's own warning message
    # says: "if GPUs are not correctly discovered, unset and try again."
    env "${NV_ENV[@]}" apptainer exec $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "$_OLLAMA_SIF" \
        bash -c 'unset CUDA_VISIBLE_DEVICES; exec ollama serve' &
    _OLLAMA_SERVER_PID=$!
    trap ollama_stop EXIT
    for i in $(seq 1 60); do
        curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1 && { echo "Ollama ready (${i}s)"; return 0; }
        kill -0 "$_OLLAMA_SERVER_PID" 2>/dev/null || { echo "ERROR: Ollama died during startup." >&2; exit 1; }
        sleep 1
    done
    echo "ERROR: Ollama not ready after 60s." >&2; exit 1
}

ollama_stop() {
    if [[ -n "$_OLLAMA_SERVER_PID" ]]; then
        kill "$_OLLAMA_SERVER_PID" 2>/dev/null || true
        wait "$_OLLAMA_SERVER_PID" 2>/dev/null || true
        _OLLAMA_SERVER_PID=""
    fi
}
