#!/bin/bash
#SBATCH --job-name=tc_test_gpu
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err

mkdir -p logs

SIF_PATH=~/HPAI/text_classification/apptainer/ollama.sif
MODELS_DIR="$(cat ~/HPAI/text_classification/apptainer/.models_dir 2>/dev/null || echo "${SCRATCH:-$HOME}/.ollama_models")"
OLLAMA_PORT=11499

PASS=0; FAIL=0
ok()   { echo "[PASS] $*"; (( PASS++ )); }
fail() { echo "[FAIL] $*"; (( FAIL++ )); }

echo "========================================"
echo " GPU / Apptainer / Ollama diagnostics"
echo " $(date)"
echo "========================================"
echo ""

# --- 1. Host GPU ---
echo "--- 1. Host nvidia-smi ---"
if nvidia-smi; then
    ok "nvidia-smi"
else
    fail "nvidia-smi not available on host"
fi
echo ""

# --- 2. SLURM GPU allocation ---
echo "--- 2. SLURM GPU env vars ---"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  ROCR_VISIBLE_DEVICES = ${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "  GPU_DEVICE_ORDINAL   = ${GPU_DEVICE_ORDINAL:-<unset>}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && ok "CUDA_VISIBLE_DEVICES is set" \
                                     || fail "CUDA_VISIBLE_DEVICES not set — GPU may not be allocated"
echo ""

# --- 3. NVIDIA devices ---
echo "--- 3. /dev/nvidia* devices ---"
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
    if [[ -e "$dev" ]]; then
        ok "$dev exists"
    else
        echo "      $dev missing (may be optional)"
    fi
done
echo ""

# --- 4. Apptainer .sif ---
echo "--- 4. Apptainer image ---"
if [[ -f "$SIF_PATH" ]]; then
    ok "SIF found: $SIF_PATH ($(du -sh "$SIF_PATH" | cut -f1))"
else
    fail "SIF not found at $SIF_PATH — run: sbatch slurm/run_setup.sh"
fi
echo ""

# --- 5. nvidia-smi inside container with --nv ---
echo "--- 5. nvidia-smi inside container (--nv) ---"
if apptainer exec --nv "$SIF_PATH" nvidia-smi 2>&1; then
    ok "--nv passthrough works"
    NV_FLAGS="--nv"
else
    fail "--nv passthrough failed"
    NV_FLAGS=""
fi
echo ""

# --- 6. nvidia-smi inside container with --nvccli ---
echo "--- 6. nvidia-smi inside container (--nvccli) ---"
if apptainer exec --nvccli "$SIF_PATH" nvidia-smi 2>&1; then
    ok "--nvccli passthrough works"
    NV_FLAGS="--nvccli"
else
    echo "      --nvccli not available or failed (not fatal)"
fi
echo ""

# --- 7. Ollama server starts with GPU ---
echo "--- 7. Ollama server startup ---"
if [[ -z "$NV_FLAGS" ]]; then
    echo "      Skipping — no GPU passthrough method worked"
    fail "Ollama GPU startup (skipped)"
else
    # Build device bind flags (fallback if --nv needs explicit binds)
    DEV_BINDS=""
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
        [[ -e "$dev" ]] && DEV_BINDS="$DEV_BINDS --bind $dev:$dev"
    done

    apptainer run $NV_FLAGS $DEV_BINDS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "$SIF_PATH" &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

    echo "  Waiting for Ollama server..."
    READY=0
    for i in $(seq 1 30); do
        curl -sf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && { READY=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || { fail "Ollama server died during startup"; break; }
        sleep 1
    done

    if [[ $READY -eq 1 ]]; then
        ok "Ollama server is up"

        # --- 8. Check GPU vs CPU in Ollama logs ---
        echo ""
        echo "--- 8. Ollama compute backend ---"
        # Give it a moment to finish logging startup
        sleep 2
        COMPUTE=$(curl -sf "http://localhost:$OLLAMA_PORT/api/ps" 2>/dev/null || echo "")
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            # Check running process output for GPU markers
            PROC_OUT=$(ls /proc/$SERVER_PID/fd 2>/dev/null | wc -l)
            echo "  Server PID $SERVER_PID is running"
            # Infer from env: if CUDA_VISIBLE_DEVICES is set and --nv worked, likely GPU
            if [[ "$NV_FLAGS" != "" ]]; then
                ok "GPU passthrough flags were applied ($NV_FLAGS) — check logs for 'library=cuda' to confirm"
            fi
        fi

        # --- 9. Quick inference test ---
        echo ""
        echo "--- 9. Quick inference (llama3) ---"
        if curl -sf "http://localhost:$OLLAMA_PORT/api/tags" | grep -q "llama3"; then
            RESPONSE=$(curl -sf -X POST "http://localhost:$OLLAMA_PORT/api/chat" \
                -H "Content-Type: application/json" \
                -d '{"model":"llama3","messages":[{"role":"user","content":"Reply with just the word: ok"}],"options":{"num_predict":5},"stream":false}' \
                2>&1)
            if echo "$RESPONSE" | grep -qi "ok\|message"; then
                ok "llama3 inference responded"
                echo "  Response: $(echo "$RESPONSE" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["message"]["content"])' 2>/dev/null || echo "$RESPONSE" | head -c 200)"
            else
                fail "llama3 inference gave unexpected response: $RESPONSE"
            fi
        else
            echo "      llama3 not pulled yet — skipping inference test (run: sbatch slurm/run_setup.sh)"
        fi
    else
        fail "Ollama server did not become ready within 30s"
    fi
fi

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
