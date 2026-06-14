#!/bin/bash
#SBATCH --job-name=tc_setup
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

mkdir -p logs

module purge
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv

export HF_HOME=~/HPAI/text_classification/rysy/hf_cache
mkdir -p "$HF_HOME"

[[ -z "${HF_TOKEN:-}" ]] && echo "WARNING: HF_TOKEN not set — Llama3 download will fail" >&2

python - <<'EOF'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

token = os.environ.get("HF_TOKEN")

for model_id in [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]:
    print(f"Downloading {model_id} ...", flush=True)
    try:
        AutoTokenizer.from_pretrained(model_id, token=token)
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=token)
        print(f"  OK: {model_id}", flush=True)
    except Exception as e:
        print(f"  FAILED: {model_id}: {e}", flush=True)
EOF

echo "Finished: $(date)"
