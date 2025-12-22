import yaml
import argparse
import os
import re

from utils import *
from simulator import MujocoReplay
from controller import CONTROLLER_REGISTRY
from data_collection.data_utils import load_npz

def main(config_path):
    # Load model configuration
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Load replay configuration
    with open("configs/replay_config.yaml", "r") as f:
        replay_config = yaml.safe_load(f)

    # Extract playback speed from replay_config
    playback_speed = replay_config["playback_speed"]

    # Base output directory
    run_dir = os.path.dirname(config_path)
    dir_name = os.path.basename(run_dir)

    # extract timestamp from directory name
    match = re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", dir_name)
    if not match:
        raise RuntimeError(f"No timestamp found in directory name: {dir_name}")

    timestamp = match.group(0)

    # Load logs from directory
    npz_path = os.path.join(run_dir, f"{timestamp}_logs.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing log file: {npz_path}")

    logs_dict = load_npz(npz_path)['default']

    # Initialize visualizer
    replay = MujocoReplay(model_config, replay_config, logs_dict)
    replay.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a simulation with given config")
    parser.add_argument("config_path", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()

    main(args.config_path)