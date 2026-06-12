#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/apptainer/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/apptainer/.models_dir
_OLLAMA_SERVER_PID=""

# Build Apptainer GPU passthrough flags.
# Tries --nvccli first; falls back to --nv + explicit /dev/nvidia* binds
# + explicit libcuda.so.1 bind (needed when Apptainer's --nv auto-discovery
# can't find the CUDA driver library on the host).
_build_nv_flags() {
    local SIF="$1"
    local FLAGS=""

    if ! nvidia-smi &>/dev/null 2>&1; then
        echo "" ; return
    fi

    if apptainer exec --nvccli "$SIF" true &>/dev/null 2>&1; then
        echo "--nvccli"
        return
    fi

    FLAGS="--nv"

    # Explicit GPU device nodes
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
        [[ -e "$dev" ]] && FLAGS="$FLAGS --bind $dev:$dev"
    done

    # Explicit CUDA driver library — Apptainer's --nv sometimes fails to
    # locate libcuda.so.1 when it lives in a non-standard path on the host.
    local CUDA_LIB
    CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
    if [[ -z "$CUDA_LIB" ]]; then
        CUDA_LIB=$(find /usr/lib /usr/lib64 /usr/local/lib /opt \
                        -name "libcuda.so.1" -maxdepth 6 2>/dev/null | head -1)
    fi
    if [[ -n "$CUDA_LIB" ]]; then
        local CUDA_DIR; CUDA_DIR="$(dirname "$CUDA_LIB")"
        FLAGS="$FLAGS --bind $CUDA_DIR:$CUDA_DIR"
    fi

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
    [[ -n "$NV_FLAGS" ]] && echo "GPU passthrough flags: $NV_FLAGS" >&2 \
                         || echo "No GPU detected — running CPU-only" >&2
    apptainer run $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "$_OLLAMA_SIF" &
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
