#!/bin/bash
#SBATCH --job-name=tc_llm_test
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/llm_test_%j.out
#SBATCH --error=logs/llm_test_%j.err

mkdir -p logs

module purge
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv
export PYTHONPATH=~/HPAI/text_classification
export HF_HOME=~/HPAI/text_classification/rysy/hf_cache

python - <<'EOF'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
token = os.environ.get("HF_TOKEN")

print(f"\nLoading {model_id} ...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto", token=token)
model.eval()
print(f"Loaded on {next(model.parameters()).device}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]
out = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt")
input_ids = (out.input_ids if hasattr(out, 'input_ids') else out).to(model.device)

print("\nRunning inference ...")
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(f"Response: {response!r}")
print("OK")
EOF

echo "Finished: $(date)"
