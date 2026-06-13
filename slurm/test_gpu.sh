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

# --- 7.5. Physical GPU identification + cuInit test ---
echo "--- 7.5. Physical GPU index + cuInit test ---"
DIAG_ALLOC_BUS=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader 2>/dev/null | head -1)
DIAG_PCI="0000:$(echo "${DIAG_ALLOC_BUS#00000000:}" | tr '[:upper:]' '[:lower:]')"
DIAG_PHY_MINOR=$(grep -i "DeviceMinor" "/proc/driver/nvidia/gpus/${DIAG_PCI}/information" 2>/dev/null | awk '{print $NF}')
if [[ -z "$DIAG_PHY_MINOR" ]]; then
    DIAG_IDX=$(CUDA_VISIBLE_DEVICES="" nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader 2>/dev/null \
               | grep -in "$DIAG_ALLOC_BUS" | cut -d: -f1 | head -1)
    DIAG_PHY_MINOR=$((DIAG_IDX - 1))
fi
echo "  Allocated GPU bus ID: ${DIAG_ALLOC_BUS:-not found}"
echo "  Physical /dev/nvidia minor: ${DIAG_PHY_MINOR:-unknown}"
[[ -n "$DIAG_PHY_MINOR" ]] && ok "Physical GPU is /dev/nvidia${DIAG_PHY_MINOR}" \
                             || echo "      Could not determine physical GPU device"

DIAG_CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
DIAG_BIND=${DIAG_CUDA_LIB:+--bind $DIAG_CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1}
DIAG_GPU_ENV=${DIAG_PHY_MINOR:+NVIDIA_VISIBLE_DEVICES=$DIAG_PHY_MINOR}
echo "  Testing cuInit with NVIDIA_VISIBLE_DEVICES=${DIAG_PHY_MINOR:-<not set>}:"
env $DIAG_GPU_ENV apptainer exec --nv $DIAG_BIND "$SIF_PATH" bash -c '
echo "  /dev/nvidia* inside container:"
ls /dev/nvidia[0-9]* 2>/dev/null | tr "\n" " " && echo
python3 -c "
import ctypes
try:
    lib = ctypes.CDLL(\"libcuda.so.1\", mode=ctypes.RTLD_GLOBAL)
    ret = lib.cuInit(ctypes.c_uint(0))
    print(f\"  cuInit = {ret}  (0=SUCCESS, 100=NO_DEVICE)\")
except Exception as e:
    print(f\"  cuInit test failed: {e}\")
" 2>/dev/null || echo "    (python3 not available)"
' 2>/dev/null
echo ""

# --- 8. Ollama server with _build_nv_flags ---
echo "--- 8. Ollama server startup ---"
if [[ ! -f "$SIF_PATH" ]]; then
    fail "Skipping — SIF not found"
else
    NV_FLAGS="$(_build_nv_flags "$SIF_PATH")"
    NV_ENV=()
    if [[ -n "$NV_FLAGS" ]]; then
        PHY_MINOR="$(_physical_gpu_minor)"
        if [[ -n "$PHY_MINOR" ]]; then
            NV_ENV+=(NVIDIA_VISIBLE_DEVICES="$PHY_MINOR")
            echo "  Physical GPU: /dev/nvidia${PHY_MINOR} → NVIDIA_VISIBLE_DEVICES=${PHY_MINOR}"
        fi
        CUDA_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        [[ -n "$CUDA_LIB" ]] && NV_FLAGS="$NV_FLAGS --bind $CUDA_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    fi
    echo "  Using flags: ${NV_FLAGS:-none (CPU-only)}"

    env "${NV_ENV[@]}" apptainer run $NV_FLAGS \
        --bind "$MODELS_DIR:$MODELS_DIR" \
        --env "OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT" \
        --env "OLLAMA_MODELS=$MODELS_DIR" \
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
