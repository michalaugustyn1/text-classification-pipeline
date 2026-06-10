#!/bin/bash
#SBATCH --job-name=tc_analyse
#SBATCH --account=g103-2499
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyse_%j.out
#SBATCH --error=logs/analyse_%j.err

mkdir -p logs

module purge
module load python/3.10
module load gcc/11.2

source ~/HPAI/text_classification/venv/bin/activate
export PYTHONPATH=~/HPAI/text_classification

time python ~/HPAI/text_classification/analyse.py \
    --results-dir ~/HPAI/text_classification/results/

echo "Finished: $(date)"
