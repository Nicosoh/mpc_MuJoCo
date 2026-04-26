#!/usr/bin/env bash

REMOTE_USER="nsoh"
REMOTE_HOST="login.delftblue.tudelft.nl"

# Full remote path
# REMOTE_PATH="/scratch/nsoh/mpc_MuJoCo/value_iteration/output/2026-04-18_18-50-34_TwoDofArm_VI"
# REMOTE_PATH="/scratch/nsoh/mpc_MuJoCo_2/value_iteration/output/2026-04-18_18-50-36_TwoDofArm_VI"
REMOTE_PATH="/scratch/nsoh/mpc_MuJoCo_3/value_iteration/output/2026-04-25_04-50-09_iiwa14_VI"

# Extract only the final folder name
FOLDER_NAME=$(basename "$REMOTE_PATH")

# Local destination = ~/Downloads/<folder_name>
LOCAL_DEST="$PWD/value_iteration/output/${FOLDER_NAME}"

rsync -av \
  --include="loop_*/" \
  --include="loop_1/data_collection/" \
  --include="loop_1/data_collection/data_config.yaml" \
  --include="loop_*/metrics.json" \
  --include="loop_*/training/***" \
  --include="/*.log" \
  --include="/*.png" \
  --include="/*.yaml" \
  --exclude="*" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_DEST}"