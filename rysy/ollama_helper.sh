#!/bin/bash
_OLLAMA_SIF=~/HPAI/text_classification/apptainer/ollama.sif
_OLLAMA_MODELS_FILE=~/HPAI/text_classification/apptainer/.models_dir
_OLLAMA_SERVER_PID=""

_native_ollama() {
    for c in "$HOME/bin/ollama" "$HOME/.local/bin/ollama"; do
        [[ -x "$c" ]] && echo "$c" && return
    done
}

ollama_start() {
    [[ -f "$_OLLAMA_SIF" ]]         || { echo "ERROR: $_OLLAMA_SIF not found." >&2; exit 1; }
    [[ -f "$_OLLAMA_MODELS_FILE" ]] || { echo "ERROR: $_OLLAMA_MODELS_FILE not found." >&2; exit 1; }
    local MODELS_DIR; MODELS_DIR="$(cat "$_OLLAMA_MODELS_FILE")"
    local PORT=$(( 11500 + (SLURM_JOB_ID % 500) ))
    export OLLAMA_HOST="http://localhost:$PORT"
    export OLLAMA_PORT="$PORT"

    if nvidia-smi &>/dev/null 2>&1; then
        local NATIVE; NATIVE=$(_native_ollama)
        if [[ -n "$NATIVE" ]]; then
            echo "GPU mode: native binary ($NATIVE)" >&2
            OLLAMA_HOST="0.0.0.0:$PORT" OLLAMA_MODELS="$MODELS_DIR" \
                ROCR_VISIBLE_DEVICES= GPU_DEVICE_ORDINAL= HIP_VISIBLE_DEVICES= \
                "$NATIVE" serve &
        elif singularity exec --nvccli "$_OLLAMA_SIF" true &>/dev/null 2>&1; then
            echo "GPU mode: container --nvccli" >&2
            singularity exec --nvccli \
                --bind "$MODELS_DIR:$MODELS_DIR" \
                "$_OLLAMA_SIF" \
                bash -c "export OLLAMA_HOST='0.0.0.0:${PORT}' OLLAMA_MODELS='${MODELS_DIR}'; exec ollama serve" &
        else
            echo "GPU mode: container --nv" >&2
            singularity exec --nv \
                --bind "$MODELS_DIR:$MODELS_DIR" \
                "$_OLLAMA_SIF" \
                bash -c "export OLLAMA_HOST='0.0.0.0:${PORT}' OLLAMA_MODELS='${MODELS_DIR}'; exec ollama serve" &
        fi
    else
        echo "No GPU — CPU mode" >&2
        singularity exec \
            --bind "$MODELS_DIR:$MODELS_DIR" \
            "$_OLLAMA_SIF" \
            bash -c "export OLLAMA_HOST='0.0.0.0:${PORT}' OLLAMA_MODELS='${MODELS_DIR}'; exec ollama serve" &
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
