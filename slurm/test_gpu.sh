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

source ~/HPAI/text_classification/slurm/ollama_helper.sh

SIF_PATH=~/HPAI/text_classification/apptainer/ollama.sif
MODELS_DIR="$(cat ~/HPAI/text_classification/apptainer/.models_dir 2>/dev/null \
              || echo "${SCRATCH:-$HOME}/.ollama_models")"
OLLAMA_PORT=11499

PASS=0; FAIL=0
ok()   { echo "[PASS] $*"; (( PASS++ )); }
fail() { echo "[FAIL] $*"; (( FAIL++ )); }

echo "========================================"
echo " GPU / Apptainer / Ollama diagnostics"
echo " $(date)"
echo "========================================"
echo ""

# --- 1. Host nvidia-smi ---
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
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
    && ok "CUDA_VISIBLE_DEVICES is set" \
    || fail "CUDA_VISIBLE_DEVICES not set — GPU may not be allocated"
echo ""

# --- 3. /dev/nvidia* devices ---
echo "--- 3. /dev/nvidia* devices ---"
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
           /dev/nvidia-modeset /dev/nvidia-caps; do
    if [[ -e "$dev" ]]; then
        ok "$dev exists"
    elif [[ "$dev" == /dev/nvidia0 || "$dev" == /dev/nvidiactl || "$dev" == /dev/nvidia-uvm ]]; then
        fail "$dev missing (required)"
    else
        echo "      $dev missing (optional)"
    fi
done
echo ""

# --- 4. libcuda.so.1 on host ---
echo "--- 4. libcuda.so.1 (CUDA driver library) ---"
CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
if [[ -n "$CUDA_LIB" ]]; then
    ok "libcuda.so.1 found at $CUDA_LIB (will be bound into container)"
else
    echo "      libcuda.so.1 not in ldconfig — --nv may still find it via other means"
fi
echo ""

# --- 5. Apptainer .sif ---
echo "--- 5. Apptainer image ---"
if [[ -f "$SIF_PATH" ]]; then
    ok "SIF found: $SIF_PATH ($(du -sh "$SIF_PATH" | cut -f1))"
else
    fail "SIF not found at $SIF_PATH — run: sbatch slurm/run_setup.sh"
fi
echo ""

# --- 6. nvidia-smi inside container (--nv) ---
echo "--- 6. nvidia-smi inside container (--nv) ---"
if apptainer exec --nv "$SIF_PATH" nvidia-smi 2>&1; then
    ok "--nv passthrough works"
else
    fail "--nv passthrough failed"
fi
echo ""

# --- 7. nvidia-smi inside container (--nvccli) ---
echo "--- 7. nvidia-smi inside container (--nvccli) ---"
if apptainer exec --nvccli "$SIF_PATH" nvidia-smi 2>&1; then
    ok "--nvccli passthrough works"
else
    echo "      --nvccli not available (not fatal)"
fi
echo ""

# --- 7.5. --nv library passthrough + cuInit test ---
echo "--- 7.5. --nv passthrough contents + cuInit test ---"
DIAG_CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
DIAG_BIND=${DIAG_CUDA_LIB:+--bind $DIAG_CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1}
apptainer exec --nv $DIAG_BIND "$SIF_PATH" bash -c '
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  /.singularity.d/libs/ (cuda entries):"
ls /.singularity.d/libs/ 2>/dev/null | grep -i cuda || echo "    (none)"
echo "  libcuda search:"
for d in /.singularity.d/libs /usr/lib/x86_64-linux-gnu /usr/lib64 /lib64 /usr/local/cuda/lib64; do
    [[ -e "$d/libcuda.so.1" ]] && echo "    FOUND: $d/libcuda.so.1 ($(stat -c%s $d/libcuda.so.1) bytes)"
done
echo "  /dev/nvidia* devices inside container:"
ls /dev/nvidia* 2>/dev/null | tr "\n" " " && echo
echo "  cuInit test (python3):"
python3 -c "
import ctypes, sys
for path in [\"libcuda.so.1\", \"/usr/lib/x86_64-linux-gnu/libcuda.so.1\", \"/.singularity.d/libs/libcuda.so.1\"]:
    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        ret = lib.cuInit(ctypes.c_uint(0))
        print(f\"  cuInit({path}) = {ret}  (0=SUCCESS)\")
        break
    except Exception as e:
        print(f\"  {path}: {e}\")
" 2>/dev/null || echo "    (python3 not available)"
echo "  /usr/lib/ollama/cuda_v12/ contents:"
ls /usr/lib/ollama/cuda_v12/ 2>/dev/null | grep -E "(cuda|ggml)" | head -10 | sed "s/^/    /"
' 2>/dev/null
echo ""

# --- 8. Ollama server with _build_nv_flags ---
echo "--- 8. Ollama server startup ---"
if [[ ! -f "$SIF_PATH" ]]; then
    fail "Skipping — SIF not found"
else
    NV_FLAGS="$(_build_nv_flags "$SIF_PATH")"
    EXTRA_ENV=()
    if [[ -n "$NV_FLAGS" ]]; then
        CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        if [[ -n "$CUDA_LIB" ]]; then
            NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
            echo "  libcuda.so.1 → bound from $CUDA_LIB"
        fi
        EXTRA_ENV=()
        echo "  NOTE: --nvccli unavailable; cuInit will likely fail (cgroup device perms)"
    fi
    echo "  Using flags: ${NV_FLAGS:-none (CPU-only)}"

    apptainer run $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
        "${EXTRA_ENV[@]}" \
        "$SIF_PATH" &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

    echo "  Waiting for server..."
    READY=0
    for i in $(seq 1 30); do
        curl -sf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null 2>&1 && { READY=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || { fail "Ollama server died during startup"; break; }
        sleep 1
    done

    if [[ $READY -eq 1 ]]; then
        ok "Ollama server is up"

        # --- 9. GPU vs CPU backend ---
        echo ""
        echo "--- 9. Ollama compute backend ---"
        sleep 2
        # Trigger a model load to force backend selection to appear in logs
        curl -sf -X POST "http://localhost:$OLLAMA_PORT/api/chat" \
            -H "Content-Type: application/json" \
            -d '{"model":"llama3","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":1},"stream":false}' \
            >/dev/null 2>&1 || true
        sleep 1
        # The compute backend is logged to stderr by Ollama — check via /api/ps
        PS_OUT=$(curl -sf "http://localhost:$OLLAMA_PORT/api/ps" 2>/dev/null || echo "")
        if echo "$PS_OUT" | grep -q '"library":"cuda"'; then
            ok "Ollama is using CUDA (GPU)"
        elif echo "$PS_OUT" | grep -q '"library":"cpu"'; then
            fail "Ollama fell back to CPU (cuInit=100: CUDA_ERROR_NO_DEVICE)"
            echo "  Root cause: Apptainer --nv binds device files but not cgroup device"
            echo "  permissions. cuInit() needs write-access set up by nvidia-container-toolkit."
            echo "  Fix: ask HPC admin to enable --nvccli (nvidia-container-toolkit) for Apptainer."
            echo "  CPU inference works correctly in the meantime."
        else
            echo "  /api/ps response: $PS_OUT"
        fi

        # --- 10. Quick inference ---
        echo ""
        echo "--- 10. Quick inference (llama3) ---"
        if curl -sf "http://localhost:$OLLAMA_PORT/api/tags" | grep -q "llama3"; then
            RESPONSE=$(curl -sf -X POST "http://localhost:$OLLAMA_PORT/api/chat" \
                -H "Content-Type: application/json" \
                -d '{"model":"llama3","messages":[{"role":"user","content":"Reply with just the word: ok"}],"options":{"num_predict":5},"stream":false}' \
                2>&1)
            CONTENT=$(echo "$RESPONSE" | python3 -c \
                'import sys,json; print(json.load(sys.stdin)["message"]["content"])' 2>/dev/null \
                || echo "$RESPONSE")
            if echo "$CONTENT" | grep -qi "ok"; then
                ok "llama3 inference responded: $CONTENT"
            else
                fail "Unexpected response: $CONTENT"
            fi
        else
            echo "  llama3 not pulled yet — run: sbatch slurm/run_setup.sh"
        fi
    else
        fail "Ollama server did not become ready within 30s"
    fi
fi

echo ""
echo "========================================"
printf " Results: %d passed, %d failed\n" "$PASS" "$FAIL"
echo "========================================"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
