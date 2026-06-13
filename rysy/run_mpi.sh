#!/bin/bash
#SBATCH --job-name=tc_mpi
#SBATCH --account=g103-2499
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mpi_%j.out
#SBATCH --error=logs/mpi_%j.err

mkdir -p logs results

module purge
module load cuda 2>/dev/null || true
module load python 2>/dev/null || true
module load openmpi 2>/dev/null || true

source ~/miniconda3/bin/activate
conda activate myenv
export PYTHONPATH=~/HPAI/text_classification

source ~/HPAI/text_classification/rysy/ollama_helper.sh
ollama_start

time srun python ~/HPAI/text_classification/pipeline_mpi.py \
    --data ~/HPAI/text_classification/data/train_data.txt \
    --out  ~/HPAI/text_classification/rysy/results/ \
    --llm

echo "Finished: $(date)"
