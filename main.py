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

    # MPC parameters
    Fmax = mpc_config["Fmax"]
    N_horizon = mpc_config["N_horizon"]
    use_RTI = mpc_config["use_RTI"]
    mpc_timestep = mpc_config["mpc_timestep"]
    Tf = N_horizon * mpc_timestep  # Time horizon
    x0 = np.array(mpc_config["x0"])  # Initial state

    # Simulation parameters
    sim_duration = mujoco_config["sim_duration"]
    sim_framerate = mujoco_config["sim_framerate"]
    verbose = mujoco_config["verbose"]
    render = mujoco_config["render"]
    path = mujoco_config["model_path"]

    # Record start time
    start_time = time.time()

    # Load MuJoCo model
    model, data = load_model(path)

    # Update model parameters from config for MuJoCo
    apply_model_config(config, model)

    # Create MPC controller
    mpc = AcadosMPCController(x0=x0, Fmax=Fmax, N_horizon=N_horizon, Tf=Tf, use_RTI=use_RTI)

    # Run MuJoCo simulation with MPC in the loop
    logs, frames = run_simulation(
        x0,
        model,
        data,
        sim_duration=sim_duration,
        mpc_timestep=mpc_timestep,
        sim_framerate=sim_framerate,
        render=render,
        controller=mpc,
        verbose=verbose,
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
