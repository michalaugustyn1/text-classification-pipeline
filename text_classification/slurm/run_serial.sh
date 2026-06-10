#!/bin/bash
#SBATCH --job-name=tc_serial
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/serial_%j.out
#SBATCH --error=logs/serial_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

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

echo "=== JOB INFO ==="
echo "Node:    $(hostname)"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"

source "$SCRIPT_DIR/ollama_helper.sh"
ollama_start

time python pipeline_serial.py \
    --data data/train_data.txt \
    --out  results/ \
    --llm

echo "Finished: $(date)"
