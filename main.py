from simulator import load_model, run_simulation, apply_model_config
from Utils import save_video, plot_signals, save_summary
from controller import AcadosMPCController
import numpy as np
import time
import os
import yaml
import argparse

def main(model_name):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]
        mujoco_config = config["mujoco"]
        mpc_config = config["mpc"]
        plots_config = config["plots"]

    # Simulation parameters

    sim_framerate = mujoco_config["sim_framerate"]
    path = mujoco_config["model_path"]

    # Record start time
    start_time = time.time()

    # Load MuJoCo model
    model, data = load_model(path)

    # Update model parameters from config for MuJoCo
    apply_model_config(config, model)

    # Create MPC controller
    mpc = AcadosMPCController(mpc_config)

    # Run MuJoCo simulation with MPC in the loop
    logs, frames = run_simulation(
        mpc_config,
        mujoco_config,
        model,
        data,
        controller=mpc,
    )

    # Record end time and print elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    # Save run summary
    save_summary(config=config, elapsed=elapsed, config_path="config.yaml")

    # Save video if frames were recorded
    if frames:
        save_video(frames, fps=sim_framerate)

    # Plot logged signals
    plot_signals(
        time=logs["time"],
        logs=logs,
        model=model,
        plots_config=plots_config,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)
