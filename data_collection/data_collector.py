# Data Collector
# Runs simulation loop and collects data at each step untill termination condition is met
# Save full trajectory and control at each step and cost
# Format should be something like traj = (x_0, x_1, x_2, ..., x_N), u = (u_0, u_1, ..., u_{N-1}), cost
# ie. max steps or reached goal
# Saves collected data to .npz file

import numpy as np
from data_collection.data_utils import *

import argparse
from main import main

def run_data_collector(model_name):
    # Load data collection config
    import yaml
    with open("data_collection/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)["data_collector"]

    max_steps = config["max_steps"]
    goal_tolerance = config["goal_tolerance"]

    # Run main simulation loop and collect data
    # This is a placeholder for actual data collection logic
    # Replace with your actual data collection code
    traj = []
    controls = []
    costs = []

    for step in range(max_steps):
        # Simulate one step, get state, control, cost
        state = ...  # Get current state from simulation
        control = ...  # Get control input applied
        cost = ...  # Get cost at this step

        traj.append(state)
        controls.append(control)
        costs.append(cost)

        # Check termination condition
        if np.linalg.norm(state - desired_state) < goal_tolerance:
            print(f"Goal reached at step {step}")
            break

    # Save collected data
    save_npz(
        filename=f"data_collection/{model_name}_data.npz",
        traj=np.array(traj),
        controls=np.array(controls),
        costs=np.array(costs),
        compressed=True
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model by name")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    model_name = args.model

    print(f"\n Starting run for model: {model_name}")
    run_data_collector(model_name)

    print("\n Done.")