#!/bin/bash
#SBATCH --job-name=tc_mpi
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/mpi_%j.out
#SBATCH --error=logs/mpi_%j.err

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
module load openmpi/4.1.4
module load mpi4py/3.1.4

source $HOME/venvs/tc_env/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD

echo "Nodes: $SLURM_JOB_NUM_NODES | Tasks: $SLURM_NTASKS | $(date)"

source "$SCRIPT_DIR/ollama_helper.sh"
ollama_start

time srun python parallel/pipeline_mpi.py \
    --data data/train_data.txt \
    --out  results/ \
    --llm

echo "Done: $(date)"
