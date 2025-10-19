import argparse
import yaml
from data_collection import load_npz
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl

def main(model_name, log_file, run, samples):
    # Load config for the model
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f).get(model_name)

    # Load data
    all_logs = load_npz(f"{log_file}.npz", input_dir="data")
    plot_traj(all_logs, samples=samples, config=config, run_filter=run)

def plot_traj(all_logs, samples=None, seed=44, config=None, run_filter=None):
    run_keys = sorted(all_logs.keys())

    # Filter if a specific run is provided
    if run_filter:
        if run_filter in run_keys:
            run_keys = [run_filter]
        else:
            print(f"Run '{run_filter}' not found in logs.")
            return

    # Else, sample randomly
    elif samples is not None and samples < len(run_keys):
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
            
            # Plot ground truth line
            time_series = np.arange(len(qpos))
            true_state = qpos[:, idx]
            ax.plot(time_series, true_state, color="black", linewidth=2, label=run_key)

            # Annotate the run_key at the start point
            ax.annotate(
                run_key,
                xy=(time_series[0], true_state[0]),
                xytext=(-5, 5),
                textcoords="offset points",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
                ha='right',
                va='bottom',
            )

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
    parser.add_argument("--run", type=str, default=None, help="Optional: specific run key (e.g., run_001)")
    parser.add_argument("--samples", type=int, default=5, help="Optional: number of samples to plot")
    args = parser.parse_args()

    main(args.model, args.log_file, args.run, args.samples)
