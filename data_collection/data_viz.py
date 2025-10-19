import argparse
import yaml
from data_collection import load_npz
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl

def main(model_name, log_file):
    # Load config for the model
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f).get(model_name)

    # Load data
    all_logs = load_npz(f"{log_file}.npz", input_dir="data")
    plot_traj(all_logs, samples=1, config=config)

def plot_traj(all_logs, samples=None, seed=42, config=None):
    run_keys = sorted(all_logs.keys())

    if samples is not None and samples < len(run_keys):
        random.seed(seed)
        run_keys = random.sample(run_keys, samples)

    # Extract plot configuration
    qpos_plots = {
        plot_name: index
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qpos"
    }

    base_cmap = plt.get_cmap("plasma")

    # Find maximum number of timesteps (for color normalization)
    max_t = max(all_logs[run_key]["x_traj"].shape[0] for run_key in run_keys)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_t - 1)

    # Loop through each qpos plot and create individual figures
    for state_name, idx in qpos_plots.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for run_key in run_keys:
            x_traj = all_logs[run_key]["x_traj"]  # shape: (timesteps, horizon, state_dim)
            qpos = all_logs[run_key]["qpos"]      # (timesteps, 2)
            num_timesteps, horizon = x_traj.shape[:2]

            for t in range(num_timesteps):
                traj = x_traj[t, :, :]  # shape: (horizon, state_dim)
                color = base_cmap(norm(t))
                time_axis = np.arange(t, t + horizon)
                state_values = traj[:, idx]
                ax.plot(time_axis, state_values, color=color, alpha=0.05)
            
            # Plot ground truth (solid black line)
            true_state = qpos[:, idx]  # (timesteps,)
            ax.plot(np.arange(len(true_state)), true_state, color="black", linewidth=2, label="Ground Truth")

        ax.set_ylabel(state_name)
        ax.set_title(f"{state_name} over timestep (colored by prediction time t)")
        ax.set_xlabel("Timestep (not seconds)")
        ax.grid(True)

        fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data for a model")
    parser.add_argument("model", type=str, help="Model name to load from config.yaml")
    parser.add_argument("log_file", type=str, help="Base name of the logs npz file (without extension)")
    args = parser.parse_args()

    main(args.model, args.log_file)
