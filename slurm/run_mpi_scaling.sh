#!/bin/bash
#SBATCH --job-name=tc_mpi_scale
#SBATCH --account=g103-2499
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mpi_scale_%j.out
#SBATCH --error=logs/mpi_scale_%j.err

mkdir -p logs

module purge
module load python/3.10
module load gcc/11.2
module load cuda/11.8
module load rapids/24.06
module load openmpi/4.1.4
module load mpi4py/3.1.4

source ~/HPAI/text_classification/venv/bin/activate
export PYTHONPATH=~/HPAI/text_classification

N_NODES_ARR=(1 2 4 8)
NTASKS_ARR=(9 18 36 72)

for i in "${!N_NODES_ARR[@]}"; do
    n_nodes=${N_NODES_ARR[$i]}
    n_tasks=${NTASKS_ARR[$i]}
    out_dir=~/HPAI/text_classification/results/mpi_scaling/nodes_${n_nodes}
    mkdir -p "$out_dir"
    echo "--- nodes=$n_nodes  tasks=$n_tasks ---"
    srun --ntasks=$n_tasks --nodes=$n_nodes \
        python ~/HPAI/text_classification/pipeline_mpi.py \
        --data ~/HPAI/text_classification/data/train_data.txt \
        --out  "$out_dir"
done

echo "Finished: $(date)"
