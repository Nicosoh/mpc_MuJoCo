#!/bin/sh

#SBATCH --job-name=VI_train
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=0
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1024MB
#SBATCH --account=research-me-cor

module load 2025
module load python
module load gcc/12.4.0
module load cuda/12.5

source ~/.bashrc
cd /scratch/nsoh/mpc_MuJoCo
srun pixi run VI