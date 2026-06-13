#!/bin/bash
#SBATCH --job-name=tc_parallel
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/parallel_%j.out
#SBATCH --error=logs/parallel_%j.err

mkdir -p logs results

module purge
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv
export PYTHONPATH=~/HPAI/text_classification
export OMP_NUM_THREADS=2

source ~/HPAI/text_classification/rysy/ollama_helper.sh
ollama_start

time python ~/HPAI/text_classification/pipeline_parallel.py \
    --data       ~/HPAI/text_classification/data/train_data.txt \
    --out        ~/HPAI/text_classification/rysy/results/ \
    --feat-jobs  5 \
    --model-jobs 7 \
    --llm

echo "Finished: $(date)"
