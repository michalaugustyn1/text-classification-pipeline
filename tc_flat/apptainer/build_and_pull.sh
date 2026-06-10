#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SIF_PATH="$PROJECT_DIR/apptainer/ollama.sif"
DEF_PATH="$PROJECT_DIR/apptainer/ollama.def"

MODELS_BASE="${SCRATCH:-$HOME}"
MODELS_DIR="$MODELS_BASE/.ollama_models"

OLLAMA_PORT=11435
OLLAMA_URL="http://localhost:$OLLAMA_PORT"

EXTRA_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --extra) EXTRA_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NV_FLAG=""
if command -v nvidia-smi &>/dev/null 2>&1; then
    NV_FLAG="--nv"
    echo "GPU detected — enabling CUDA passthrough"
fi

echo "=== Step 1: Building Apptainer image ==="
if [[ -f "$SIF_PATH" ]]; then
    echo "  $SIF_PATH already exists — skipping."
    echo "  Delete it and rerun to rebuild."
else
    echo "  Building from $DEF_PATH ..."
    apptainer build "$SIF_PATH" "$DEF_PATH"
    echo "  Built: $SIF_PATH  ($(du -sh "$SIF_PATH" | cut -f1))"
fi

echo ""
echo "=== Step 2: Preparing model store ==="
mkdir -p "$MODELS_DIR"
echo "  Model store: $MODELS_DIR"
echo "$MODELS_DIR" > "$PROJECT_DIR/apptainer/.models_dir"
echo "  Path saved → $PROJECT_DIR/apptainer/.models_dir"

echo ""
echo "=== Step 3: Starting Ollama server on port $OLLAMA_PORT ==="

OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" \
OLLAMA_MODELS="$MODELS_DIR" \
apptainer run $NV_FLAG \
    --bind "$MODELS_DIR:$MODELS_DIR" \
    --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
    --env "OLLAMA_MODELS=$MODELS_DIR" \
    "$SIF_PATH" &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

cleanup() {
    echo ""
    echo "Stopping Ollama server (PID $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "  Waiting for server ..."
for i in $(seq 1 60); do
    if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        echo "  Server ready after ${i}s."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Ollama server process died unexpectedly." >&2
        echo "       Try running manually to see the error:" >&2
        echo "       apptainer run $NV_FLAG --bind $MODELS_DIR:$MODELS_DIR \\" >&2
        echo "         --env OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT \\" >&2
        echo "         --env OLLAMA_MODELS=$MODELS_DIR $SIF_PATH" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "ERROR: Server did not become ready within 60 s." >&2
    echo ""
    echo "Diagnostics — try running the server manually:" >&2
    echo "  apptainer run $NV_FLAG \\" >&2
    echo "    --bind $MODELS_DIR:$MODELS_DIR \\" >&2
    echo "    --env OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT \\" >&2
    echo "    --env OLLAMA_MODELS=$MODELS_DIR \\" >&2
    echo "    $SIF_PATH" >&2
    echo "" >&2
    echo "Then in another terminal:" >&2
    echo "  curl http://localhost:$OLLAMA_PORT/api/tags" >&2
    exit 1
fi

echo ""
echo "=== Step 4: Pulling models ==="

pull_model() {
    local model="$1"
    echo "  Pulling $model ..."
    OLLAMA_HOST="localhost:$OLLAMA_PORT" \
    apptainer exec $NV_FLAG \
        --env "OLLAMA_HOST=localhost:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        "$SIF_PATH" \
        ollama pull "$model"
    echo "  Done: $model"
}

pull_model "llama3"
pull_model "mistral"

[[ -n "$EXTRA_MODEL" ]] && pull_model "$EXTRA_MODEL"

echo ""
echo "=== Setup complete ==="
echo "  Image:      $SIF_PATH"
echo "  Models:     $MODELS_DIR"
echo "  Disk usage: $(du -sh "$MODELS_DIR" | cut -f1)"
echo ""
echo "  Submit the LLM job with:  sbatch slurm/run_llm.sh"

