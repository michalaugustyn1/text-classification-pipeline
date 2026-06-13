#!/bin/bash
#SBATCH --job-name=tc_serial
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
export HF_HOME=~/HPAI/text_classification/rysy/hf_cache

time python ~/HPAI/text_classification/pipeline_serial.py \
    --data ~/HPAI/text_classification/data/train_data.txt \
    --out  ~/HPAI/text_classification/rysy/results/ \
    --llm

echo "Finished: $(date)"
