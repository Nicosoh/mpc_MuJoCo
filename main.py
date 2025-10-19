from simulator import load_model, run_simulation, apply_model_config, load_model_from_robot_descriptions
from utils import save_video, plot_signals, save_summary, load_yref, randomise_x0
from controller import AcadosMPCController
import time
import yaml
import argparse

def main(model_name, data_collection=False):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]

    # Simulation parameters
    sim_framerate = config["mujoco"]["sim_framerate"]
    urdf_available = config["mujoco"]["urdf_available"]

    # Randomise inital state if specified
    if config["mpc"]["x0_random"]:
        x0 = randomise_x0(config)
        config["mpc"]["x0"] = x0
        print(f"Randomised initial state: {x0}")

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

    # Load reference trajectory/endpoint
    yref = load_yref(model_name)

    # Create MPC controller
    mpc = AcadosMPCController(config, yref)

    # Run MuJoCo simulation with MPC in the loop
    logs, frames = run_simulation(
        config=config,
        model=model,
        data=data,
        yref=yref,
        data_collection=data_collection,
        controller=mpc
    )

    # Record end time and print elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    if data_collection: #quit here if data collection
        return logs, elapsed, config
    
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
        yref = yref,
        plots_config=config["plots"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)
