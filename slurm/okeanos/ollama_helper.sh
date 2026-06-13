#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/singularity/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/singularity/.models_dir
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

# Return the path to a native ollama binary installed outside the container.
_native_ollama() {
    for c in "$HOME/bin/ollama" "$HOME/.local/bin/ollama"; do
        [[ -x "$c" ]] && echo "$c" && return
    done
}

# Used by test_gpu.sh to probe flags; --nvccli requires nvidia-container-toolkit.
_build_nv_flags() {
    local SIF="$1"
    if ! nvidia-smi &>/dev/null 2>&1; then echo ""; return; fi
    if singularity exec --nvccli "$SIF" true &>/dev/null 2>&1; then echo "--nvccli"; return; fi
    local FLAGS="--nv"
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
               /dev/nvidia-modeset /dev/nvidia-caps; do
        [[ -e "$dev" ]] && FLAGS="$FLAGS --bind $dev:$dev"
    done
    echo "$FLAGS"
}

ollama_start() {
    [[ -f "$_OLLAMA_SIF" ]]         || { echo "ERROR: $_OLLAMA_SIF not found." >&2; exit 1; }
    [[ -f "$_OLLAMA_MODELS_FILE" ]] || { echo "ERROR: $_OLLAMA_MODELS_FILE not found." >&2; exit 1; }
    local MODELS_DIR; MODELS_DIR="$(cat "$_OLLAMA_MODELS_FILE")"
    local PORT=$(( 11500 + (SLURM_JOB_ID % 500) ))
    export OLLAMA_HOST="http://localhost:$PORT"
    export OLLAMA_PORT="$PORT"

    if ! nvidia-smi &>/dev/null 2>&1; then
        # ── No GPU: container CPU mode ────────────────────────────────────────
        echo "No GPU detected — container CPU mode" >&2
        singularity exec \
            --bind "$MODELS_DIR:$MODELS_DIR" \
            --env "OLLAMA_HOST=0.0.0.0:$PORT" \
            --env "OLLAMA_MODELS=$MODELS_DIR" \
            "$_OLLAMA_SIF" ollama serve &

    elif singularity exec --nvccli "$_OLLAMA_SIF" true &>/dev/null 2>&1; then
        # ── Container + --nvccli (nvidia-container-toolkit installed) ─────────
        echo "GPU mode: container --nvccli" >&2
        singularity exec --nvccli \
            --bind "$MODELS_DIR:$MODELS_DIR" \
            --env "OLLAMA_HOST=0.0.0.0:$PORT" \
            --env "OLLAMA_MODELS=$MODELS_DIR" \
            "$_OLLAMA_SIF" ollama serve &

    else
        local NATIVE; NATIVE=$(_native_ollama)
        if [[ -n "$NATIVE" ]]; then
            # ── Native binary: inherits SLURM cgroup GPU access directly ──────
            # SLURM's CUDA_VISIBLE_DEVICES is correct for native processes;
            # no remapping needed. nvidia-modprobe -u ensures UVM is initialised.
            # Unset ROCm/Intel overrides — they cause spurious Ollama warnings.
            echo "GPU mode: native binary ($NATIVE)" >&2
            command -v nvidia-modprobe &>/dev/null && { nvidia-modprobe -u 2>/dev/null || true; }
            env ROCR_VISIBLE_DEVICES= GPU_DEVICE_ORDINAL= HIP_VISIBLE_DEVICES= \
                OLLAMA_HOST="0.0.0.0:$PORT" OLLAMA_MODELS="$MODELS_DIR" \
                "$NATIVE" serve &

        else
            # ── Container + --nv fallback ─────────────────────────────────────
            # cuInit will likely fail (no cgroup device setup without --nvccli).
            # To enable GPU: install ~/bin/ollama (native) or ask HPC admin for
            # nvidia-container-toolkit so --nvccli works.
            echo "GPU mode: container --nv (likely CPU — install ~/bin/ollama for GPU)" >&2
            local PHY_MINOR; PHY_MINOR=$(_physical_gpu_minor)
            command -v nvidia-modprobe &>/dev/null && { nvidia-modprobe -u 2>/dev/null || true; }
            local NV_FLAGS="--nv"
            for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
                       /dev/nvidia-modeset /dev/nvidia-caps; do
                [[ -e "$dev" ]] && NV_FLAGS="$NV_FLAGS --bind $dev:$dev"
            done
            local CUDA_LIB; CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
            [[ -n "$CUDA_LIB" ]] && NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
            local NV_ENV=(); [[ -n "$PHY_MINOR" ]] && NV_ENV=(NVIDIA_VISIBLE_DEVICES="$PHY_MINOR")
            local CV_ENV=(); [[ -n "$PHY_MINOR" ]] && CV_ENV=(--env "CUDA_VISIBLE_DEVICES=$PHY_MINOR")
            env "${NV_ENV[@]}" singularity exec $NV_FLAGS \
                --bind "$MODELS_DIR:$MODELS_DIR" \
                --env "OLLAMA_HOST=0.0.0.0:$PORT" \
                --env "OLLAMA_MODELS=$MODELS_DIR" \
                "${CV_ENV[@]}" \
                --env "ROCR_VISIBLE_DEVICES=" \
                --env "GPU_DEVICE_ORDINAL=" \
                --env "HIP_VISIBLE_DEVICES=" \
                "$_OLLAMA_SIF" ollama serve &
        fi
    fi

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
