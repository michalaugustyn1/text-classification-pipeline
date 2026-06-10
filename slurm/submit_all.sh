#!/bin/bash
set -euo pipefail

cd ~/HPAI/text_classification
mkdir -p logs results

submit() {
    local script="$1" dep="$2"
    local args=(--parsable)
    [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
    sbatch "${args[@]}" "$script"
}

echo "=== Submitting pipeline jobs ==="

JOB_SETUP=$(submit slurm/run_setup.sh "")
echo "  [0] setup     → job $JOB_SETUP"

JOB_SERIAL=$(submit slurm/run_serial.sh "$JOB_SETUP")
echo "  [1] serial    → job $JOB_SERIAL  (after $JOB_SETUP)"

JOB_OPT=$(submit slurm/run_optimized.sh "$JOB_SERIAL")
echo "  [2] optimized → job $JOB_OPT  (after $JOB_SERIAL)"

JOB_PAR=$(submit slurm/run_parallel.sh "$JOB_OPT")
echo "  [3] parallel  → job $JOB_PAR  (after $JOB_OPT)"

JOB_MPI=$(submit slurm/run_mpi.sh "$JOB_PAR")
echo "  [4] mpi       → job $JOB_MPI  (after $JOB_PAR)"

JOB_SCALE=$(submit slurm/run_scaling.sh "$JOB_PAR")
echo "  [5] scaling   → job $JOB_SCALE  (after $JOB_PAR)"

JOB_ANALYSE=$(submit slurm/run_analyse.sh "${JOB_MPI}:${JOB_SCALE}")
echo "  [6] analyse   → job $JOB_ANALYSE  (after $JOB_MPI, $JOB_SCALE)"

echo ""
echo "Monitor: watch -n 10 squeue -u \$USER"
