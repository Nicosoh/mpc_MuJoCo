# Data Collector
# Runs simulation loop and collects data at each step untill termination condition is met
# Save full trajectory and control at each step and cost
# Format should be something like traj = (x_0, x_1, x_2, ..., x_N), u = (u_0, u_1, ..., u_{N-1}), cost
# ie. max steps or reached goal
# Saves collected data to .npz file

import numpy as np
from data_collection import save_npz
import argparse
from main import main
from utils import save_summary
import yaml

def run_data_collector(model_name):
    # Load data collection config
    with open("data_collection/data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)["data_collector"]

    runs = data_config["runs"]

    total_elapsed = 0.0
    all_logs = {}

    # Run main simulation loop and collect data
    for step in range(runs):
        # Simulate one OCP, get state, control, cost
        logs, elapsed, config = main(model_name, data_collection=True)
        total_elapsed += elapsed
        run_key = f"run_{step:03d}"
        all_logs[run_key] = logs
        print(f"Completed data collection run {step+1}/{runs}")
    
    save_summary(config, config_path="config.yaml", output_dir="data")
    save_summary(data_config, elapsed=total_elapsed, config_path="data_collection/data_config.yaml", output_dir="data", sub_name="data_collection")
    
    # Save collected data
    save_npz("logs.npz", data=all_logs, output_dir="data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model by name")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    model_name = args.model

    print(f"\n Starting run for model: {model_name}")
    run_data_collector(model_name)

    print("\n Done.")