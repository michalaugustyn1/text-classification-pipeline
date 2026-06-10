#!/bin/bash
#SBATCH --job-name=tc_llm
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

mkdir -p logs results

module purge
module load python/3.10
module load gcc/11.2
module load cuda/11.8
module load rapids/24.06

source ~/HPAI/text_classification/venv/bin/activate
export PYTHONPATH=~/HPAI/text_classification

source ~/HPAI/text_classification/slurm/ollama_helper.sh
ollama_start

time python - <<'PYEOF'
import sys
sys.path.insert(0, ".")

from config import RAW_CSV
from data_utils import prepare_data
from metrics import evaluate, save_results, timed
from models import LlamaClassifier, MistralClassifier

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
    res.update({"test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                "test_f1_weighted": m["f1_weighted"]})
    all_res.append(res)
    print(f"{model.name}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

save_results({"runs": all_res}, "results/results_llm.json")
PYEOF

echo "Finished: $(date)"
