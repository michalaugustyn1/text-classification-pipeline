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

# Classification smoke test — single and batched
print("\n--- Classification test (single) ---")
classes = ["business", "entertainment", "politics", "sport", "tech"]
test_texts = [
    "Manchester United beat Arsenal 2-1 in last night's Premier League clash.",
    "The Federal Reserve raised interest rates by 25 basis points amid inflation concerns.",
]
for text in test_texts:
    msgs = [{"role": "user", "content":
        f"Classify the following document into exactly one of these categories: {', '.join(classes)}.\n\n"
        f"Document:\n{text}\n\n"
        f"Rules:\n- Reply with ONLY the category name.\n"
        f"- Do not add any explanation, punctuation, or extra words.\nCategory:"}]
    inp = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    inp_ids = (inp.input_ids if hasattr(inp, 'input_ids') else inp).to(model.device)
    with torch.no_grad():
        out2 = model.generate(inp_ids, max_new_tokens=10, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out2[0][inp_ids.shape[-1]:], skip_special_tokens=True)
    print(f"  Text : {text[:60]}...")
    print(f"  Label: {resp!r}")

print("\n--- Classification test (batched, left-pad with eos) ---")
def make_input_ids(text):
    msgs = [{"role": "user", "content":
        f"Classify the following document into exactly one of these categories: {', '.join(classes)}.\n\n"
        f"Document:\n{text}\n\n"
        f"Rules:\n- Reply with ONLY the category name.\n"
        f"- Do not add any explanation, punctuation, or extra words.\nCategory:"}]
    out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    t = out.input_ids if hasattr(out, 'input_ids') else out
    return t.squeeze(0)

id_list = [make_input_ids(t) for t in test_texts]
max_len = max(x.shape[0] for x in id_list)
pad_id = tokenizer.pad_token_id
input_ids = torch.full((len(id_list), max_len), pad_id, dtype=torch.long)
attn_mask = torch.zeros_like(input_ids)
for i, ids in enumerate(id_list):
    offset = max_len - ids.shape[0]
    input_ids[i, offset:] = ids
    attn_mask[i, offset:] = 1
input_ids = input_ids.to(model.device)
attn_mask = attn_mask.to(model.device)
with torch.no_grad():
    out_batch = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                               max_new_tokens=20, do_sample=False,
                               pad_token_id=pad_id)
for i, text in enumerate(test_texts):
    raw = tokenizer.decode(out_batch[i][max_len:], skip_special_tokens=True)
    print(f"  Text : {text[:60]}...")
    print(f"  Label: {raw!r}")
EOF

echo "Finished: $(date)"
