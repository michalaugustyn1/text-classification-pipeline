#!/bin/bash
#SBATCH --job-name=tc_analyse
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyse_%j.out
#SBATCH --error=logs/analyse_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs results

module purge
module load python/3.10
module load gcc/11.2

source $HOME/venvs/tc_env/bin/activate
export PYTHONPATH=$PWD

echo "Node: $(hostname) | $(date)"

python analysis/analyse_results.py --results-dir results/

echo "Done: $(date)"
