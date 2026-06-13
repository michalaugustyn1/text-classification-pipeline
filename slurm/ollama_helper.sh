#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/apptainer/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/apptainer/.models_dir
_OLLAMA_SERVER_PID=""

_physical_gpu_minor() {
    local BUS; BUS=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader 2>/dev/null | head -1)
    [[ -z "$BUS" ]] && return
    local PCI="0000:$(echo "${BUS#00000000:}" | tr '[:upper:]' '[:lower:]')"
    local MINOR; MINOR=$(grep -i "DeviceMinor" "/proc/driver/nvidia/gpus/${PCI}/information" 2>/dev/null | awk '{print $NF}')
    if [[ -n "$MINOR" ]]; then echo "$MINOR"; return; fi
    local IDX; IDX=$(CUDA_VISIBLE_DEVICES="" nvidia-smi --query-gpu=gpu_bus_id \
                     --format=csv,noheader 2>/dev/null \
                     | grep -in "$BUS" | cut -d: -f1 | head -1)
    [[ -n "$IDX" ]] && echo "$((IDX - 1))"
}

# --nvccli requires nvidia-container-toolkit on the host; falls back to --nv.
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

    local CONTAINER_ENV=()
    if [[ -n "$NV_FLAGS" ]]; then
        local PHY_MINOR; PHY_MINOR=$(_physical_gpu_minor)
        if [[ -n "$PHY_MINOR" ]]; then
            echo "GPU: physical /dev/nvidia${PHY_MINOR}" >&2
            # Pass physical minor as CUDA_VISIBLE_DEVICES inside the container.
            # SLURM's value is a logical remapping (always 0); we need the
            # physical index so CUDA opens the device the cgroup actually permits.
            # Also pass NVIDIA_VISIBLE_DEVICES for Apptainer --nv device selection.
            NV_ENV+=(NVIDIA_VISIBLE_DEVICES="$PHY_MINOR")
            CONTAINER_ENV+=(--env "CUDA_VISIBLE_DEVICES=$PHY_MINOR")
        fi

        # Trigger NVIDIA UVM driver initialization on the HOST before entering
        # the container. Inside the container the setuid nvidia-modprobe binary
        # cannot escalate privileges, so cuInit fails if UVM isn't already up.
        if command -v nvidia-modprobe &>/dev/null; then
            nvidia-modprobe -u 2>/dev/null || true
            echo "nvidia-modprobe -u: done" >&2
        fi

        # Bind the driver library to a path Ollama's CUDA detection scans.
        local CUDA_LIB; CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        [[ -n "$CUDA_LIB" ]] && NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        echo "GPU passthrough flags: $NV_FLAGS" >&2
    else
        echo "No GPU detected — running CPU-only" >&2
        # Silence SLURM's ROCm/Intel visible-device overrides (no GPU, so no warns).
        CONTAINER_ENV+=(--env "ROCR_VISIBLE_DEVICES=" --env "GPU_DEVICE_ORDINAL=")
    fi

    env "${NV_ENV[@]}" apptainer exec $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "${CONTAINER_ENV[@]}" \
        "$_OLLAMA_SIF" ollama serve &
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
