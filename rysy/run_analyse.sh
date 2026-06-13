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
module load python 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv
export PYTHONPATH=~/HPAI/text_classification

time python ~/HPAI/text_classification/analyse.py \
    --results-dir ~/HPAI/text_classification/rysy/results/

echo "Finished: $(date)"
