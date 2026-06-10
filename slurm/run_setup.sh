#!/bin/bash
#SBATCH --job-name=tc_setup
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# One-time setup: build ollama.sif (if missing) and pull llama3 + mistral.
# Run before any LLM job:  sbatch slurm/run_setup.sh
# If the cluster blocks outbound HTTP on compute nodes, run interactively:
#   bash apptainer/build_and_pull.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

SIF_PATH="$PROJECT_DIR/apptainer/ollama.sif"
DEF_PATH="$PROJECT_DIR/apptainer/ollama.def"
MODELS_DIR="${SCRATCH:-$HOME}/.ollama_models"
OLLAMA_PORT=11435
OLLAMA_URL="http://localhost:$OLLAMA_PORT"

module purge
module load gcc/11.2

echo "=== JOB INFO ==="
echo "Node:    $(hostname)"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo ""

# ── Step 1: build image ───────────────────────────────────────────────────────
echo "=== Step 1: Apptainer image ==="
if [[ -f "$SIF_PATH" ]]; then
    echo "  $SIF_PATH already exists — skipping build."
    echo "  Delete it and resubmit to force a rebuild."
else
    echo "  Building from $DEF_PATH  (pulls ollama/ollama Docker layer ~10 min) ..."
    apptainer build "$SIF_PATH" "$DEF_PATH"
    echo "  Built: $SIF_PATH  ($(du -sh "$SIF_PATH" | cut -f1))"
fi
echo ""

# ── Step 2: model store ───────────────────────────────────────────────────────
echo "=== Step 2: Model store ==="
mkdir -p "$MODELS_DIR"
echo "$MODELS_DIR" > "$PROJECT_DIR/apptainer/.models_dir"
echo "  Path: $MODELS_DIR"
echo "  Saved → $PROJECT_DIR/apptainer/.models_dir"
echo ""

# ── Step 3: start server ──────────────────────────────────────────────────────
echo "=== Step 3: Starting Ollama server on port $OLLAMA_PORT ==="

NV_FLAG=""
if nvidia-smi &>/dev/null 2>&1; then
    NV_FLAG="--nv"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

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
for i in $(seq 1 90); do
    if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        echo "  Server ready after ${i}s."; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Ollama server process died unexpectedly." >&2
        echo "  Try: apptainer run $NV_FLAG --bind $MODELS_DIR:$MODELS_DIR \\" >&2
        echo "         --env OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT \\" >&2
        echo "         --env OLLAMA_MODELS=$MODELS_DIR $SIF_PATH" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "ERROR: Server did not become ready within 90 s." >&2; exit 1
fi
echo ""

# ── Step 4: pull models ───────────────────────────────────────────────────────
echo "=== Step 4: Pulling models ==="

pull_model() {
    local model="$1"
    echo "  Pulling $model ..."
    OLLAMA_HOST="localhost:$OLLAMA_PORT" \
    apptainer exec $NV_FLAG \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=localhost:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "$SIF_PATH" \
        ollama pull "$model"
    echo "  Done: $model"
    echo ""
}

pull_model "llama3"
pull_model "mistral"

[[ -n "${EXTRA_MODEL:-}" ]] && pull_model "$EXTRA_MODEL"

# ── summary ───────────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo "  Image:    $SIF_PATH  ($(du -sh "$SIF_PATH" | cut -f1))"
echo "  Models:   $MODELS_DIR  ($(du -sh "$MODELS_DIR" | cut -f1))"
echo "  Finished: $(date)"
echo ""
echo "  Next: sbatch slurm/run_llm.sh"
