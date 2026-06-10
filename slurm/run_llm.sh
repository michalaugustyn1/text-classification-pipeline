#!/bin/bash
#SBATCH --job-name=tc_llm
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

PARTITION="${SLURM_PARTITION:-short}"

set -euo pipefail

if ! sinfo -h -p "$PARTITION" &>/dev/null 2>&1; then
    echo "WARNING: partition '$PARTITION' not found."
    echo "Available partitions:"; sinfo -h -o "%P" | sort -u
fi
mkdir -p logs results

module purge
module load python/3.10
module load gcc/11.2

source $HOME/venvs/tc_env/bin/activate
export PYTHONPATH=$PWD

SIF_PATH="$PWD/apptainer/ollama.sif"
MODELS_DIR_FILE="$PWD/apptainer/.models_dir"

[[ -f "$SIF_PATH" ]] || { echo "ERROR: $SIF_PATH missing. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }
[[ -f "$MODELS_DIR_FILE" ]] || { echo "ERROR: .models_dir missing. Run: bash apptainer/build_and_pull.sh" >&2; exit 1; }

MODELS_DIR="$(cat "$MODELS_DIR_FILE")"

OLLAMA_PORT=$(( 11500 + (SLURM_JOB_ID % 500) ))
OLLAMA_URL="http://localhost:$OLLAMA_PORT"

echo "Node:        $(hostname)"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Image:       $SIF_PATH"
echo "Model store: $MODELS_DIR"
echo "Port:        $OLLAMA_PORT"
echo "Started:     $(date)"

NV_FLAG=""
if nvidia-smi &>/dev/null 2>&1; then
    NV_FLAG="--nv"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" \
OLLAMA_MODELS="$MODELS_DIR" \
apptainer run $NV_FLAG \
    --bind "$MODELS_DIR:$MODELS_DIR" \
    --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
    --env "OLLAMA_MODELS=$MODELS_DIR" \
    "$SIF_PATH" &

SERVER_PID=$!
echo "Ollama server PID: $SERVER_PID"

cleanup() {
    echo "Stopping Ollama server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama server ..."
for i in $(seq 1 60); do
    if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        echo "Server ready after ${i}s."; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Ollama server process died." >&2; exit 1
    fi
    sleep 1
done

curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 \
    || { echo "ERROR: Server not ready after 60 s." >&2; exit 1; }

OLLAMA_HOST="http://localhost:$OLLAMA_PORT" python - <<'PYEOF'
import sys, os, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

from configs.config import RAW_CSV
from utils.data_utils import prepare_data
from utils.metrics import evaluate, save_results, timed
from models.llm_models import LlamaClassifier, MistralClassifier

X_train, y_train, _, _, X_test, y_test, le = prepare_data(RAW_CSV)
all_res = []

for ModelCls in [LlamaClassifier, MistralClassifier]:
    model = ModelCls()
    res = {"model": model.name, "feature": "raw_text"}
    with timed("train", res):
        model.fit(X_train, y_train, label_encoder=le)
    with timed("inference_test", res):
        y_pred = model.predict(X_test)
    n = min(len(X_test), model.sample_size)
    m = evaluate(y_test[:n], y_pred[:n], le.classes_)
    res.update({"test_accuracy":    m["accuracy"],
                "test_f1_macro":    m["f1_macro"],
                "test_f1_weighted": m["f1_weighted"]})
    all_res.append(res)
    print(f"{model.name}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

save_results({"runs": all_res}, "results/results_llm.json")
PYEOF

echo "Done: $(date)"
