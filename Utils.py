import os
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import importlib

# ========== FILE MANAGEMENT HELPERS ==========

def get_next_filename(base_name, ext="txt", folder="outputs"):
    """
    Finds the next available filename with an incrementing number.
    """
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        filename = os.path.join(folder, f"{base_name}_{i}.{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

# ========== PLOTTING ==========

def plot_signals(time, logs, model, plots_config, yref, output_dir="outputs"):
    """
    Processes signals based on plots_config and plots them over time.

    Parameters
    ----------
    time : array-like
        Shared time axis.
    logs : dict
        Dictionary containing signal arrays (e.g., qpos, qvel, u_applied).
    model : object
        Object containing model info: nq, nv, nu.
    plots_config : dict
        {name: (source, idx, unit)} config to specify signals to plot.
    output_dir : str
        Directory to save plots.
    """
    signals = {}
    ylabel_units = {}

    yref_full = expand_yref_over_time(yref, time)

    # Offsets to access yref by type
    source_offsets = {
        "qpos": 0,
        "qvel": model.nq,
        "ctrl": model.nq + model.nv,
        "u_applied": model.nq + model.nv,  # treat alias like 'ctrl'
    }
    
    for name, (source, idx, unit) in plots_config.items():
        if source == "qpos":
            assert idx < model.nq, f"Index {idx} out of range for qpos (nq={model.nq})"
            signals[name] = logs["qpos"][:, idx]
        elif source == "qvel":
            assert idx < model.nv, f"Index {idx} out of range for qvel (nv={model.nv})"
            signals[name] = logs["qvel"][:, idx]
        elif source in ["ctrl", "u_applied"]:
            assert idx < model.nu, f"Index {idx} out of range for control (nu={model.nu})"
            signals[name] = logs["u_applied"][:, idx]
        elif source == "cost":
            # Cost is a 1D array, no indexing needed
            assert "cost" in logs, "Cost not found in logs"
            signals[name] = logs["cost"]
        else:
            raise ValueError(f"Invalid signal source '{source}' in plots config for '{name}'")

        ylabel_units[name] = unit

    # Now do the plotting
    os.makedirs(output_dir, exist_ok=True)
    filename = get_next_filename("plot", ext="png", folder=output_dir)

    n = len(signals)
    dpi = 120
    width, height = 800, 200 * n
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)
    if n == 1:
        ax = [ax]
    
    for i, (name, values) in enumerate(signals.items()):
        ax[i].plot(time, values, label="Actual")

        source, idx, _ = plots_config[name]
        yref_idx = None
        if source in source_offsets:
            yref_idx = source_offsets[source] + idx

        if yref_idx is not None and yref_idx < yref_full.shape[1]:
            ax[i].plot(time, yref_full[:, yref_idx], "--", label="Ref")

        ax[i].set_title(name)
        ax[i].set_ylabel(ylabel_units.get(name, ""))
        ax[i].legend(loc="best")

    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to: {os.path.abspath(filename)}")
    plt.show()

def expand_yref_over_time(yref, time):
    """
    Expand sparse yref definitions into a full yref array aligned with `time`.

    Parameters
    ----------
    yref_raw : np.ndarray
        Array of shape (T_ref, ny+1) where first column is time.
    time : np.ndarray
        Simulation time vector of shape (T_sim,).

    Returns
    -------
    yref_full : np.ndarray
        Array of shape (T_sim, ny), repeated using zero-order hold.
    """
    yref_times = yref[:, 0]
    yref_values = yref[:, 1:]

    yref_full = np.zeros((len(time), yref_values.shape[1]))
    current_index = 0

    for i, t in enumerate(time):
        # Move to the latest reference before or at time t
        while (current_index + 1 < len(yref_times)) and (yref_times[current_index + 1] <= t):
            current_index += 1
        yref_full[i] = yref_values[current_index]

    return yref_full

# ========== VIDEO SAVING ==========

def save_video(frames, output_dir="outputs", base_name="video", fps=30):
    """
    Saves simulation frames as a video file with a running number.

    Parameters
    ----------
    frames : list or np.ndarray
        Video frames.
    output_dir : str
        Directory to save the video.
    base_name : str
        Prefix for the filename.
    fps : int
        Frames per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = get_next_filename(base_name, ext="mp4", folder=output_dir)

    media.write_video(filename, frames, fps=fps)
    print(f"Video saved to: {os.path.abspath(filename)}")


# ========== SUMMARY SAVING ==========

def _format_value(val):
    """
    Format values into readable strings for summary output.
    """
    if isinstance(val, float):
        return f"{val:.6g}"
    elif isinstance(val, (list, tuple, np.ndarray)):
        return np.array(val).tolist()
    else:
        return str(val)

def save_summary(config, elapsed=None, config_path=None, output_dir="outputs", sub_name=None):
    """
    Save simulation config and details to a text file with running number.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    elapsed : float, optional
        Total runtime in seconds.
    config_path : str, optional
        Path to config/model file used.
    output_dir : str
        Directory to save summary file.
    sub_name : str, optional
        Prefix name for the summary file. If None, defaults to 'summary'.
    """
    os.makedirs(output_dir, exist_ok=True)

    prefix = sub_name if sub_name else "summary"
    filename = get_next_filename(prefix, ext="txt", folder=output_dir)

    with open(filename, "w") as f:
        f.write("Simulation Summary\n")
        f.write("=================\n\n")

        if config_path:
            f.write(f"Config/model file: {config_path}\n\n")

        for section, params in config.items():
            f.write(f"{section.capitalize()}:\n")
            if isinstance(params, dict):
                for key, val in params.items():
                    f.write(f"  {key}: {_format_value(val)}\n")
            else:
                f.write(f"  {section}: {_format_value(params)}\n")
            f.write("\n")

        if elapsed is not None:
            f.write(f"Total execution time: {elapsed:.2f} seconds\n")

    print(f"Summary saved to: {os.path.abspath(filename)}")


def load_yref(model_name):
    try:
        yref_module = importlib.import_module(f"yrefs.{model_name}_yref")
        return yref_module.yref
    except ModuleNotFoundError:
        raise ValueError(f"No yref file found for model '{model_name}'")
    
def randomise_x0(config):
    x0_range = config["mpc"]["x0_range"]
    x0 = config["mpc"]["x0"]
    sampling = config["mpc"]["x0_sampling"]

    if sampling == "uniform":
        for i in range(len(x0)):
            x0[i] = np.random.uniform(low=x0_range[i][0], high=x0_range[i][1])
    else:
        raise ValueError(f"Unsupported x0_sampling method: {sampling}")
    return x0

def load_x0(config):
    # Randomise inital state if specified
    if config["mpc"]["x0_random"]:
        x0 = randomise_x0(config)
        config["mpc"]["x0"] = x0
        print(f"Randomised initial state: {x0}")
    return config