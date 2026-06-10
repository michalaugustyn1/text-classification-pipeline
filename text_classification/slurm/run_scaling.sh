#!/bin/bash
#SBATCH --job-name=tc_scale
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs results/scaling

module purge
module load python/3.10
module load gcc/11.2
module load cuda/11.8
module load rapids/24.06

source $HOME/venvs/tc_env/bin/activate
export PYTHONPATH=$PWD

N_JOBS_ARR=(1 2 4 8 16 32)
N_JOBS=${N_JOBS_ARR[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$N_JOBS

echo "Array task $SLURM_ARRAY_TASK_ID → n_jobs=$N_JOBS | $(date)"

source "$SCRIPT_DIR/ollama_helper.sh"
ollama_start

time python pipeline_parallel.py \
    --data       data/train_data.txt \
    --out        results/scaling/njobs_${N_JOBS}/ \
    --feat-jobs  $N_JOBS \
    --model-jobs $N_JOBS \
    --llm

echo "Done n_jobs=$N_JOBS: $(date)"
