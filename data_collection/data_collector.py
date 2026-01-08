# Data Collector
# Runs simulation loop and collects data at each step untill termination condition is met
# Save full trajectory and control at each MPC step and cost
# Format should be something like traj = (x_0, x_1, x_2, ..., x_N), u = (u_0, u_1, ..., u_{N-1}), cost
# ie. max steps or termination condition
# Saves collected data to .npz file

import os
import argparse
import yaml
import time
from datetime import datetime
from main import main
from utils import save_yaml
from data_collection import save_npz

def run_data_collector(model_name, data_config_path="data_collection/data_config.yaml"):
    # Load data collection config
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)["data_collector"]

    runs = data_config["runs"]
    # total_elapsed = 0.0
    all_logs = {}
    base_dir = "data"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output and log paths
    output_dir = os.path.join(base_dir, f"{timestamp}_{model_name}_data_collection")
    os.makedirs(output_dir, exist_ok=True)
    output_log_path = os.path.join(output_dir, "output.log")

    # Yaml save path
    yaml_save_path = os.path.join(output_dir, "data_config.yaml")
    save_yaml(data_config, yaml_save_path)

    # Initialize / clear the log file
    with open(output_log_path, "w") as f:
        f.write(f"=== Data Collection Log for model '{model_name}' ===\n")
        f.write(f"Started at {timestamp}\n")

    # Record start time
    start_time = time.time()

    success_count = 0
    attempt_count = 0
    
    # Run main simulation loop and collect data
    while success_count < runs:
        attempt_count += 1
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"\n--- Starting data collection run {success_count+1}/{runs} ---")

        try:
            # Simulate one OCP, get state, control, cost
            logs = main(model_name, data_collection=True, output_dir=output_dir, timestamp=run_timestamp, data_config=data_config)
            all_logs[run_timestamp] = logs
            success_count += 1

            print(f"Completed run: {run_timestamp}, {success_count}/{runs}")
            with open(output_log_path, "a") as f:
                f.write(f"\n--- Run {success_count}/{runs}, {run_timestamp}, Completed ---\n")

        except Exception as e:
            # Catch and log the error
            import traceback
            tb = traceback.format_exc()

            print(f"Attempt {attempt_count} failed: {e}")
            with open(output_log_path, "a") as f:
                f.write(f"\n--- Attempt {attempt_count}, FAILED, {run_timestamp} ---\n")
                f.write(f"Error: {str(e)}\n")
                f.write(tb)
                f.write("\n-----------------------------\n")

            # Continue with the next run instead of crashing
            continue
    
    # Record end time and print elapsed time
    end_time = time.time()
    elapsed = end_time - start_time

    # Save data
    save_npz(f"{timestamp}_{model_name}_logs.npz", data=all_logs, output_dir=output_dir)
    
    # Initialize / clear the log file
    with open(output_log_path, "a") as f:
        f.write(f"Ended at {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")

    print("\n=== Data collection finished ===")
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print(f"Log file: {output_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model by name")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    print(f"\nStarting data collection for model: {args.model}")
    run_data_collector(args.model)
    print("\nDone.")