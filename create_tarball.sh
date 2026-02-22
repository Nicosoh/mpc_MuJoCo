#!/bin/bash

#SBATCH --job-name=create_tarball
#SBATCH --partition=compute
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=0
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256MB
#SBATCH --account=education-me-msc-ro

# Set the specific directory path
DIR_PATH="/scratch/nsoh/mpc_MuJoCo/value_iteration/output/2026-02-21_22-11-41_Pendulum_VI"

# Check if the provided path exists and is a directory
if [ ! -d "$DIR_PATH" ]; then
  echo "Error: $DIR_PATH is not a valid directory."
  exit 1
fi

# Extract the directory name (basename) and the parent directory (dirname)
DIR_NAME=$(basename "$DIR_PATH")
PARENT_DIR=$(dirname "$DIR_PATH")

# Print start message
echo "Creating Tarball: ${PARENT_DIR}/${TARBALL_NAME}"

# Create the tarball in the same directory using tar and pipe it to pigz for compression
TARBALL_NAME="${DIR_NAME}.tar.gz"
srun tar -cf - -C "$PARENT_DIR" "$DIR_NAME" | pigz -9 -p 32 > "$PARENT_DIR/$DIR_NAME.tar.gz"

# Print success message
echo "Tarball created: ${PARENT_DIR}/${TARBALL_NAME}"