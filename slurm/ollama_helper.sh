#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/apptainer/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/apptainer/.models_dir
_OLLAMA_SERVER_PID=""

ollama_start() {
    [[ -f "$_OLLAMA_SIF" ]]         || { echo "ERROR: $_OLLAMA_SIF not found. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }
    [[ -f "$_OLLAMA_MODELS_FILE" ]] || { echo "ERROR: $_OLLAMA_MODELS_FILE not found. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }
    local MODELS_DIR; MODELS_DIR="$(cat "$_OLLAMA_MODELS_FILE")"
    local PORT=$(( 11500 + (SLURM_JOB_ID % 500) ))
    export OLLAMA_HOST="http://localhost:$PORT"
    export OLLAMA_PORT="$PORT"
    local NV_FLAG=""
    nvidia-smi &>/dev/null 2>&1 && NV_FLAG="--nv"
    apptainer run $NV_FLAG \
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
