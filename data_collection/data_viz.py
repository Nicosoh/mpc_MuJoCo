import argparse
import yaml
from data_collection import load_npz
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
import os
import mplcursors
import seaborn as sns

def main(model_name, log_dir, run, samples):
        # Load config for the model
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f).get(model_name)

    log_file= log_dir[:-16]
    log_file = os.path.join("data", log_dir, f"{log_file}_logs.npz")
    plots_dir = os.path.join("data", log_dir, "plots")

    # Load data
    all_logs = load_npz(log_file)

    # Run plotting functions
    # plot_traj(all_logs, save_dir=plots_dir, samples=samples, config=config, run_filter=run)
    plot_dist(all_logs, save_dir=plots_dir, samples=samples, config=config, run_filter=run)

def plot_traj(
    all_logs,
    save_dir,
    samples=None,
    seed=44,
    config=None,
    run_filter=None,
    tstep=1,    # <- Plot predicted trajectories every tstep steps
    hstep=10,     # <- Subsample points within each horizon prediction
    ):

    os.makedirs(save_dir, exist_ok=True) # Create output directory if it doesn't exist

    run_keys = sorted(all_logs.keys())

    # Filter specific run
    if run_filter:
        if run_filter in run_keys:
            run_keys = [run_filter]
        else:
            print(f"Run '{run_filter}' not found in logs.")
            return
    elif samples is not None and samples < len(run_keys):
        random.seed(seed)
        run_keys = random.sample(run_keys, samples)
    # If 'samples' is None, plot all runs (explicit)
    else:
        print(f"No sampling requested — plotting all {len(run_keys)} runs.")

    # Get all qpos plots
    qpos_plots = {
        plot_name: index
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qpos"
    }

    base_cmap = plt.get_cmap("plasma")

    # Normalize for color mapping
    max_t = max(all_logs[run_key]["x_traj"].shape[0] for run_key in run_keys)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_t - 1)

    # Loop through each qpos variable and make a figure
    for state_name, idx in qpos_plots.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for run_key in run_keys:
            x_traj = all_logs[run_key]["x_traj"]  # shape: (timesteps, horizon, state_dim)
            qpos = all_logs[run_key]["qpos"]      # shape: (timesteps, state_dim)
            num_timesteps, horizon = x_traj.shape[:2]

            # Subsample prediction lines
            for t in range(0, num_timesteps, tstep):
                traj = x_traj[t, ::hstep, :]  # Subsample within horizon
                color = base_cmap(norm(t))
                time_axis = np.arange(t, t + horizon)[::hstep]
                state_values = traj[:, idx]
                ax.plot(time_axis, state_values, color=color, label=run_key, alpha=0.1)

            # Plot full actual trajectory
            time_series = np.arange(len(qpos))
            true_state = qpos[:, idx]
            ax.plot(time_series, true_state, color="black", linewidth=1, label=run_key, alpha=0.5)

        ax.set_ylabel(state_name)
        ax.set_title(f"{state_name} over time\n(Predictions every {tstep} steps, horizon subsampled by {hstep})")
        ax.set_xlabel("Timestep")
        ax.grid(True)
        cursor = mplcursors.cursor(ax.lines, hover=False)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{state_name}_trajectory.png"))
    
    # === PLOT COST OVER TIME ===
    fig_cost, ax_cost = plt.subplots(figsize=(10, 4))
    for run_key in run_keys:
        cost = all_logs[run_key]["cost"]  # shape: (timesteps,)
        timesteps = np.arange(len(cost))
        ax_cost.plot(timesteps, cost, label=run_key, alpha=0.7)

    ax_cost.set_title("MPC Cost over Time")
    ax_cost.set_xlabel("Timestep")
    ax_cost.set_ylabel("Cost")
    ax_cost.grid(True)
    cursor = mplcursors.cursor(ax_cost.lines, hover=False)
    cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig_cost.tight_layout()
    fig_cost.savefig(os.path.join(save_dir, "mpc_cost_over_time.png"))

    # === PLOT INPUT OVER TIME ===
    fig_input, ax_input = plt.subplots(figsize=(10, 4))
    for run_key in run_keys:
        u_applied = all_logs[run_key]["u_applied"]  # shape: (timesteps, input_dim)
        timesteps = np.arange(u_applied.shape[0])
        for i in range(u_applied.shape[1]):
            ax_input.plot(timesteps, u_applied[:, i], label=f"{run_key} - u_applied {i}", alpha=0.7)

    ax_input.set_title("MPC Input over Time")
    ax_input.set_xlabel("Timestep")
    ax_input.set_ylabel("Input Value")
    ax_input.grid(True)
    cursor = mplcursors.cursor(ax_input.lines, hover=False)
    cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig_input.tight_layout()
    fig_input.savefig(os.path.join(save_dir, "mpc_input_over_time.png"))

    # Show all plots
    plt.show()

def plot_dist(
    all_logs,
    save_dir,
    samples=None,
    seed=44,
    config=None,
    run_filter=None,
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    run_keys = sorted(all_logs.keys())

    # Filter specific runs
    if run_filter:
        if run_filter in run_keys:
            run_keys = [run_filter]
        else:
            print(f"Run '{run_filter}' not found.")
            return
    elif samples is not None and samples < len(run_keys):
        random.seed(seed)
        run_keys = random.sample(run_keys, samples)

    # Get qpos indices from config
    qpos_plots = {
        plot_name: index
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qpos"
    }

    # === HISTOGRAMS for state values ===
    for state_name, idx in qpos_plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))

        all_values = []
        for run_key in run_keys:
            qpos = all_logs[run_key]["qpos"]  # shape: (timesteps, state_dim)
            all_values.append(qpos[:, idx])   # gather values for this axis

        all_values = np.concatenate(all_values)  # flatten across runs
        counts, bins, patches = ax.hist(all_values, bins=80, color="teal", alpha=0.75)
        ax.set_title(f"Distribution of {state_name}")
        ax.set_xlabel(state_name)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # --- Add count labels above bars ---
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                ax.text(
                    (bin_left + bin_right) / 2,  # center of the bar
                    count,                      # height position
                    f"{int(count)}",
                    ha="center", va="bottom",
                    fontsize=8,
                    rotation=90,                # vertical label for compactness
                )

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{state_name}_histogram.png"))

    # === HISTOGRAM for cost ===
    fig_cost, ax_cost = plt.subplots(figsize=(8, 4))
    all_costs = []
    for run_key in run_keys:
        cost = all_logs[run_key]["cost"]
        all_costs.append(cost)

    all_costs = np.concatenate(all_costs)
    counts, bins, patches = ax_cost.hist(all_costs, bins=80, color="crimson", alpha=0.7)
    ax_cost.set_title("Distribution of MPC Cost")
    ax_cost.set_xlabel("Cost")
    ax_cost.set_ylabel("Frequency")
    ax_cost.grid(True)

    # --- Add count labels above bars ---
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            ax_cost.text(
                (bin_left + bin_right) / 2,  # center of the bar
                count,                      # height position
                f"{int(count)}",
                ha="center", va="bottom",
                fontsize=8,
                rotation=90,                # vertical label for compactness
            )
    fig_cost.tight_layout()
    fig_cost.savefig(os.path.join(save_dir, "mpc_cost_histogram.png"))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data for a model")
    parser.add_argument("model", type=str, help="Model name to load from config.yaml")
    parser.add_argument("log_file", type=str, help="Base name of the logs npz file (without extension)")
    parser.add_argument("--run", type=str, default=None, help="Optional: specific run key (e.g., run_001)")
    parser.add_argument("--samples", type=int, default=None, help="Optional: number of samples to plot")
    args = parser.parse_args()

    # Call main with all args
    main(args.model, args.log_file, args.run, args.samples)