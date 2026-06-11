#!/bin/bash
#SBATCH --job-name=tc_serial
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/serial_%j.out
#SBATCH --error=logs/serial_%j.err

mkdir -p logs results

module purge
module load python/3.10 2>/dev/null || true
module load gcc/11.2 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true
module load rapids/24.06 2>/dev/null || true

source ~/miniconda3/bin/activate
export PYTHONPATH=~/HPAI/text_classification

source ~/HPAI/text_classification/slurm/ollama_helper.sh
ollama_start

time python ~/HPAI/text_classification/pipeline_serial.py \
    --data ~/HPAI/text_classification/data/train_data.txt \
    --out  ~/HPAI/text_classification/results/ \
    --llm

echo "Finished: $(date)"
