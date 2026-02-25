#!/usr/bin/env bash

REMOTE_USER="nsoh"
REMOTE_HOST="login.delftblue.tudelft.nl"

# Full remote path
REMOTE_PATH="/scratch/nsoh/mpc_MuJoCo/value_iteration/output/2026-02-22_15-19-00_TwoDofArm_VI"

# Extract only the final folder name
FOLDER_NAME=$(basename "$REMOTE_PATH")

# Local destination = ~/Downloads/<folder_name>
LOCAL_DEST="$HOME/Downloads/${FOLDER_NAME}"

rsync -av \
  --include="loop_*/" \
  --include="loop_*/metrics.json" \
  --include="loop_*/training/***" \
  --include="/*.log" \
  --include="/*.png" \
  --exclude="*" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_DEST}"