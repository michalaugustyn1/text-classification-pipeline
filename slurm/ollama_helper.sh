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

    # Explicit GPU device nodes (belt-and-suspenders alongside --nv).
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
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
    local EXTRA_ENV=()
    if [[ -n "$NV_FLAGS" ]]; then
        # --nv binds libnvidia-ml.so but Ollama's Go-level CUDA detection often
        # fails in Apptainer. Bind libcuda.so.1 to a path Ollama searches and
        # set OLLAMA_LLM_LIBRARY=cuda_v12 to skip Ollama's Go detection entirely,
        # directly launching the CUDA 12 llama-server binary which does its own
        # cuInit() against the bound driver library and device nodes.
        local CUDA_LIB; CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        if [[ -n "$CUDA_LIB" ]]; then
            NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        fi
        EXTRA_ENV=(--env "OLLAMA_LLM_LIBRARY=cuda_v12")
        echo "GPU passthrough flags: $NV_FLAGS" >&2
    else
        echo "No GPU detected — running CPU-only" >&2
    fi
    apptainer run $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "${EXTRA_ENV[@]}" \
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
