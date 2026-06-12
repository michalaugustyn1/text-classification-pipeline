#!/bin/bash
#SBATCH --job-name=tc_setup
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

mkdir -p logs

SIF_PATH=~/HPAI/text_classification/apptainer/ollama.sif
DEF_PATH=~/HPAI/text_classification/apptainer/ollama.def
MODELS_DIR="${SCRATCH:-$HOME}/.ollama_models"
OLLAMA_PORT=11435

module purge
module load gcc/11.2 2>/dev/null || true

if [[ ! -f "$SIF_PATH" ]]; then
    apptainer build "$SIF_PATH" "$DEF_PATH"
fi

mkdir -p "$MODELS_DIR"
echo "$MODELS_DIR" > ~/HPAI/text_classification/apptainer/.models_dir

NV_FLAGS="--nv"
if apptainer exec --nvccli "$SIF_PATH" true &>/dev/null 2>&1; then
    NV_FLAGS="--nvccli"
else
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
        [[ -e "$dev" ]] && NV_FLAGS="$NV_FLAGS --bind $dev:$dev"
    done
fi
echo "GPU passthrough flags: $NV_FLAGS" >&2

apptainer run $NV_FLAGS \
    --bind "$MODELS_DIR:$MODELS_DIR" \
    --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
    --env "OLLAMA_MODELS=$MODELS_DIR" \
    "$SIF_PATH" &

SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

for i in $(seq 1 90); do
    curl -sf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: Ollama server died." >&2; exit 1; }
    sleep 1
done

curl -sf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 \
    || { echo "ERROR: Server not ready after 90 s." >&2; exit 1; }

pull_model() {
    apptainer exec $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=localhost:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "$SIF_PATH" ollama pull "$1"
}

pull_model "llama3"
pull_model "mistral"
[[ -n "${EXTRA_MODEL:-}" ]] && pull_model "$EXTRA_MODEL"

echo "Finished: $(date)"
