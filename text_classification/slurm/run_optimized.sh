#!/bin/bash
#SBATCH --job-name=tc_optimized
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/optimized_%j.out
#SBATCH --error=logs/optimized_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs results

module purge
module load python/3.10
module load gcc/11.2
module load cuda/11.8
module load rapids/24.06

source $HOME/venvs/tc_env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | $(date)"

source "$SCRIPT_DIR/ollama_helper.sh"
ollama_start

time python pipeline_optimized.py \
    --data data/train_data.txt \
    --out  results/ \
    --llm

echo "Done: $(date)"
