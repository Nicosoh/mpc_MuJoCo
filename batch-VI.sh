#!/bin/sh

#SBATCH --job-name=VI_train
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=0
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1024MB
#SBATCH --account=education-me-msc-ro

module load 2025
module load openmpi
module load python
module load miniconda3/4.12.0
module load gcc/12.4.0

# Activate conda, run job, deactivate conda
conda activate /home/nsoh/.conda/envs/mpc_mujoco
cd ~/mpc_MuJoCo
srun python -m value_iteration.value_iteration
conda deactivate