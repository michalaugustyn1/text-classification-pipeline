#!/bin/bash
set -euo pipefail

cd ~/HPAI/text_classification/rysy
mkdir -p logs results

submit() {
    local script="$1" dep="$2"
    local args=(--parsable)
    [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
    sbatch "${args[@]}" "$script"
}

echo "=== Submitting pipeline jobs ==="

JOB_SETUP=$(submit run_setup.sh "")
echo "  [0] setup       → job $JOB_SETUP"

JOB_SERIAL=$(submit run_serial.sh "$JOB_SETUP")
echo "  [1] serial      → job $JOB_SERIAL  (after $JOB_SETUP)"

JOB_OPT=$(submit run_optimized.sh "$JOB_SERIAL")
echo "  [2] optimized   → job $JOB_OPT  (after $JOB_SERIAL)"

JOB_PAR=$(submit run_parallel.sh "$JOB_OPT")
echo "  [3] parallel    → job $JOB_PAR  (after $JOB_OPT)"

JOB_MPI=$(submit run_mpi.sh "$JOB_PAR")
echo "  [4] mpi         → job $JOB_MPI  (after $JOB_PAR)"

JOB_SCALE=$(submit run_scaling.sh "$JOB_PAR")
echo "  [5] scaling     → job $JOB_SCALE  (after $JOB_PAR)"

JOB_MPI_SCALE=$(submit run_mpi_scaling.sh "$JOB_PAR")
echo "  [6] mpi_scaling → job $JOB_MPI_SCALE  (after $JOB_PAR)"

JOB_ANALYSE=$(submit run_analyse.sh "${JOB_MPI}:${JOB_SCALE}:${JOB_MPI_SCALE}")
echo "  [7] analyse     → job $JOB_ANALYSE  (after $JOB_MPI, $JOB_SCALE, $JOB_MPI_SCALE)"

echo ""
echo "Monitor: watch -n 10 squeue -u \$USER"
