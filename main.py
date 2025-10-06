from simulator import load_model, run_simulation, apply_model_config, load_model_from_robot_descriptions
from Utils import save_video, plot_signals, save_summary
from controller import AcadosMPCController
import time
import yaml
import argparse

def main(model_name):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]

    # Simulation parameters
    sim_framerate = config["mujoco"]["sim_framerate"]
    # path = config["mujoco"]["model_path"]
    urdf_available = config["mujoco"]["urdf_available"]
    # menagerie_name =  config["mujoco"]["menagerie_name"]
    # import pdb; pdb.set_trace()
    # Record start time
    start_time = time.time()

    # Load MuJoCo model from path or URDF if available
    if urdf_available:
        menagerie_name =  config["mujoco"]["menagerie_name"]
        model, data = load_model_from_robot_descriptions(menagerie_name)
    else:
        path = config["mujoco"]["model_path"]
        model, data = load_model(path)
        # Update model parameters from config for MuJoCo
        apply_model_config(config, model)

    # Create MPC controller
    mpc = AcadosMPCController(config)

    # Run MuJoCo simulation with MPC in the loop
    logs, frames = run_simulation(
        config,
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
        plots_config=config["plots"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)
