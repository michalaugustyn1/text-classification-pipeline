#!/bin/bash
#SBATCH --job-name=tc_optimized
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/optimized_%j.out
#SBATCH --error=logs/optimized_%j.err

mkdir -p logs results

module purge
module load python/3.10
module load gcc/11.2
module load cuda/11.8
module load rapids/24.06

source ~/HPAI/text_classification/venv/bin/activate
export PYTHONPATH=~/HPAI/text_classification

source ~/HPAI/text_classification/slurm/ollama_helper.sh
ollama_start

time python ~/HPAI/text_classification/pipeline_optimized.py \
    --data ~/HPAI/text_classification/data/train_data.txt \
    --out  ~/HPAI/text_classification/results/ \
    --llm

echo "Finished: $(date)"
