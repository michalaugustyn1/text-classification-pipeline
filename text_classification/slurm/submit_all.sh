#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PARTITION="${PARTITION:-short}"

if [[ ! -f "apptainer/ollama.sif" ]]; then
    echo "ERROR: apptainer/ollama.sif not found." >&2
    echo "       Run first:  bash apptainer/build_and_pull.sh" >&2
    exit 1
fi

if [[ ! -f "apptainer/.models_dir" ]]; then
    echo "ERROR: apptainer/.models_dir not found." >&2
    echo "       Run first:  bash apptainer/build_and_pull.sh" >&2
    exit 1
fi

mkdir -p logs results

echo "=== Submitting pipeline jobs (partition: $PARTITION) ==="
echo ""

submit() {
    local script="$1"
    local dep="$2"
    local extra="${3:-}"

    local sbatch_args=(
        --partition="$PARTITION"
        --parsable
    )

    if [[ -n "$dep" ]]; then
        sbatch_args+=(--dependency="afterok:$dep")
    fi

    local job_id
    job_id=$(sbatch "${sbatch_args[@]}" $extra "$SCRIPT_DIR/$script")
    echo "$job_id"
}

JOB_SERIAL=$(submit run_serial.sh "")
echo "  [1] serial       → job $JOB_SERIAL"

JOB_OPT=$(submit run_optimized.sh "$JOB_SERIAL")
echo "  [2] optimized    → job $JOB_OPT  (after $JOB_SERIAL)"

JOB_PAR=$(submit run_parallel.sh "$JOB_OPT")
echo "  [3] parallel     → job $JOB_PAR  (after $JOB_OPT)"

JOB_MPI=$(submit run_mpi.sh "$JOB_PAR")
echo "  [4] mpi          → job $JOB_MPI  (after $JOB_PAR)"

JOB_SCALE=$(submit run_scaling.sh "$JOB_PAR")
echo "  [5] scaling      → job $JOB_SCALE  (after $JOB_PAR)"

JOB_ANALYSE=$(submit run_analyse.sh "${JOB_MPI}:${JOB_SCALE}")
echo "  [6] analyse      → job $JOB_ANALYSE  (after $JOB_MPI and $JOB_SCALE)"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Monitor with:"
echo "  watch -n 10 squeue -u \$USER"
echo ""
echo "Job chain:"
echo "  $JOB_SERIAL → $JOB_OPT → $JOB_PAR → $JOB_MPI ─┐"
echo "                                       └→ $JOB_SCALE ─┘→ $JOB_ANALYSE"
