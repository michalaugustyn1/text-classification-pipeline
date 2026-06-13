#!/bin/bash
#SBATCH --job-name=tc_serial
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/serial_%j.out
#SBATCH --error=logs/serial_%j.err

mkdir -p logs results

module purge
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv
export PYTHONPATH=~/HPAI/text_classification

source ~/HPAI/text_classification/rysy/ollama_helper.sh
ollama_start

time python ~/HPAI/text_classification/pipeline_serial.py \
    --data ~/HPAI/text_classification/data/train_data.txt \
    --out  ~/HPAI/text_classification/rysy/results/ \
    --llm

echo "Finished: $(date)"
