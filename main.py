from utils import save_video, plot_signals, save_summary,
from simulator import MuJoCoSimulator
import time
import yaml
import argparse

def main(model_name, data_collection=False):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]

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

    if data_collection: #quit here if data collection
        return simulator.logs, elapsed, config
    
    # Save run summary
    save_summary(config=config, elapsed=elapsed, config_path="config.yaml")

    # Save video if frames were recorded
    if simulator.frames:
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
