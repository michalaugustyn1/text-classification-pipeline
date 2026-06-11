#!/bin/bash
#SBATCH --job-name=tc_scale
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err

mkdir -p logs results/scaling

module purge
module load python/3.10 2>/dev/null || true
module load gcc/11.2 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true
module load rapids/24.06 2>/dev/null || true

source ~/HPAI/text_classification/venv/bin/activate
export PYTHONPATH=~/HPAI/text_classification

N_JOBS_ARR=(1 2 4 8 16 32)
N_JOBS=${N_JOBS_ARR[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$N_JOBS

source ~/HPAI/text_classification/slurm/ollama_helper.sh
ollama_start

time python ~/HPAI/text_classification/pipeline_parallel.py \
    --data       ~/HPAI/text_classification/data/train_data.txt \
    --out        ~/HPAI/text_classification/results/scaling/njobs_${N_JOBS}/ \
    --feat-jobs  $N_JOBS \
    --model-jobs $N_JOBS \
    --llm

echo "Finished n_jobs=$N_JOBS: $(date)"
