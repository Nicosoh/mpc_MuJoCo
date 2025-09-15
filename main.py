from simulator import load_model, run_simulation
from visualization import save_video, plot_signals
from controller import AcadosMPCController
import numpy as np
import time
import os
import yaml
import argparse

def main(model):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model]
        mujoco_config = config["mujoco"]
        mpc_config = config["mpc"]

    # MPC parameters
    Fmax = mpc_config["Fmax"]
    N_horizon = mpc_config["N_horizon"]
    use_RTI = mpc_config["use_RTI"]
    mpc_timestep = mpc_config["mpc_timestep"]
    Tf = N_horizon * mpc_timestep  # Time horizon

    # Simulation parameters
    sim_duration = mujoco_config["sim_duration"]
    sim_framerate = mujoco_config["sim_framerate"]
    verbose = mujoco_config["verbose"]
    render = mujoco_config["render"]
    path = mujoco_config["model_path"]

    # Initial condition
    x0 = np.array(mpc_config["x0"])

    # Record start time
    start_time = time.time()

    # Load MuJoCo model
    model, data = load_model(path)

    # Create MPC controller
    mpc = AcadosMPCController(x0, Fmax=Fmax, N_horizon=N_horizon, Tf=Tf, use_RTI=use_RTI)

    # Run MuJoCo simulation with MPC in the loop
    results, frames = run_simulation(
        x0,
        model,
        data,
        duration=sim_duration,
        mpc_timestep=mpc_timestep,
        framerate=sim_framerate,
        render=render,
        controller=mpc,
        verbose=verbose,
    )

    # Record end time and print elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    # Save run summary
    save_summary(
        x0,
        Fmax,
        N_horizon,
        Tf,
        use_RTI,
        sim_duration,
        sim_framerate,
        verbose,
        render,
        path,
        elapsed,
        mpc_timestep,
    )

    # Save video if frames were recorded
    if frames:
        save_video(frames, fps=sim_framerate)

    # Plot results
    plot_signals(
        results["time"],
        {
            "Cart Position": results["cart_pos"],
            "Pendulum Angle": results["pend_angle"],
            "Cart Velocity": results["cart_vel"],
            "Pendulum Angular Velocity": results["pend_angvel"],
            "Control Input": results["u_applied"],
        },
        ylabel_units={
            "Cart Position": "m",
            "Cart Velocity": "m/s",
            "Pendulum Angle": "rad",
            "Pendulum Angular Velocity": "rad/s",
            "Control Input": "N",
        },
    )

def get_next_filename(base_name="simulation_summary", ext="txt", folder="outputs"):
    """Find the next available numbered filename in the given folder."""
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
    i = 1
    while True:
        filename = os.path.join(folder, f"{base_name}_{i}.{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

def save_summary(
    x0,
    Fmax,
    N_horizon,
    Tf,
    use_RTI,
    sim_duration,
    sim_framerate,
    verbose,
    render,
    path,
    elapsed,
    mpc_frequency,
):
    """Save simulation configuration and runtime details into a text file."""
    summary_file = get_next_filename()
    with open(summary_file, "w") as f:
        f.write("Simulation Summary\n")
        f.write("=================\n\n")

        f.write("General:\n")
        f.write(f"  Model file: {path}\n")
        f.write(f"  Initial condition (x0): {x0.tolist()}\n\n")

        f.write("MPC Parameters:\n")
        f.write(f"  Max Force (Fmax): {Fmax}\n")
        f.write(f"  Horizon (N_horizon): {N_horizon}\n")
        f.write(f"  Time Horizon (Tf): {Tf}\n")
        f.write(f"  Real-Time Iteration (use_RTI): {use_RTI}\n\n")
        f.write(f"  MPC Frequency: {mpc_frequency} Hz\n\n")

        f.write("Simulation Parameters:\n")
        f.write(f"  Duration: {sim_duration} s\n")
        f.write(f"  Framerate: {sim_framerate} fps\n")
        f.write(f"  Verbose: {verbose}\n")
        f.write(f"  Render: {render}\n\n")

        f.write(f"Total execution time: {elapsed:.2f} seconds\n")

    print(f"Simulation details saved to {os.path.abspath(summary_file)}")

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)
