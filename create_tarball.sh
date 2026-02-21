#!/bin/bash

# Check if a directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

# Get the provided directory path
DIR_PATH="$1"

# Check if the provided path exists and is a directory
if [ ! -d "$DIR_PATH" ]; then
  echo "Error: $DIR_PATH is not a valid directory."
  exit 1
fi

# Extract the directory name (basename) and the parent directory (dirname)
DIR_NAME=$(basename "$DIR_PATH")
PARENT_DIR=$(dirname "$DIR_PATH")

# Create the tarball in the same directory using pigz for compression
TARBALL_NAME="${DIR_NAME}.tar.gz"
tar -caf "${PARENT_DIR}/${TARBALL_NAME}" -C "$PARENT_DIR" "$DIR_NAME" --use-compress-program=pigz

# Print success message
echo "Tarball created: ${PARENT_DIR}/${TARBALL_NAME}"