import time
import yaml
import argparse
import os
from datetime import datetime
from utils import save_video, plot_signals, save_summary, ocp_plot
from simulator import MuJoCoSimulator

def main(model_name, data_collection=False, output_dir=None):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]
    
    # Base output directory
    output_dir = output_dir or "data"
    os.makedirs(output_dir, exist_ok=True)  

    # Create a subfolder with date, time and model name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{model_name}"
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving current run data to: {run_dir}")

    # Record start time
    start_time = time.time()

    # Create simulator object
    simulator = MuJoCoSimulator(config)

    # Run simulation
    simulator.run()

    # Record end time and print elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    # Save run summary
    save_summary(config=simulator.config, elapsed=elapsed, output_dir=run_dir)
    
    if data_collection: #quit here if data collection
        return simulator.logs
    elif config["mpc"]["solve_ocp"]:
        ocp_plot(simulator, run_dir)
    
    # Save video if frames were recorded
    if simulator.frames:
        save_video(simulator.frames, output_dir=run_dir, fps=simulator.config["mujoco"]["sim_framerate"])

    # Plot logged signals
    plot_signals(
        time=simulator.logs["time"],
        logs=simulator.logs,
        model=simulator.model,
        plots_config=simulator.config["plots"],
        yref=simulator.yref,
        output_dir=run_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)
