import os
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import importlib
from robot_descriptions.loaders.mujoco import load_robot_description
import mujoco

# ========== PLOTTING ==========

def plot_signals(time, logs, model, plots_config, yref, output_dir, file_name="plot"):
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

    # Ensure logs entries are numpy arrays
    for key, val in logs.items():
        if isinstance(val, list):
            logs[key] = np.array(val)

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
    full_path = os.path.join(output_dir, f"{file_name}.jpg")

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
    plt.savefig(full_path)
    print(f"Plot saved to: {os.path.abspath(full_path)}")
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

def save_video(frames, output_dir, file_name="video", fps=30):
    """
    Saves simulation frames as a video file with a running number.

    Parameters
    ----------
    frames : list or np.ndarray
        Video frames.
    output_dir : str
        Directory to save the video.
    file_name : str
        Filename.
    fps : int
        Frames per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.mp4")

    media.write_video(full_path, frames, fps=fps)
    print(f"Video saved to: {os.path.abspath(full_path)}")

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

def save_summary(config, output_dir, elapsed=None, file_name="summary"):
    """
    Save simulation config and details to a text file with running number.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    elapsed : float, optional
        Total runtime in seconds.
    output_dir : str
        Directory to save summary file.
    file_name : str, optional
        Name for the summary file. If None, defaults to 'summary'.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.txt")

    with open(full_path, "w") as f:
        f.write("Simulation Summary\n")
        f.write("=================\n\n")

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

    print(f"Summary saved to: {os.path.abspath(full_path)}")


def load_yref(model_name):
    try:
        yref_module = importlib.import_module(f"yrefs.{model_name}_yref")
        return yref_module.yref
    except ModuleNotFoundError:
        raise ValueError(f"No yref file found for model '{model_name}'")
    
def load_collision_config(model_name):
    try:
        cfg_module = importlib.import_module(
            f"collision_config.{model_name}_collision_config"
        )
        return cfg_module.collision_config
    except ModuleNotFoundError:
        raise ValueError(f"No collision config found for model '{model_name}'")
    
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

def ocp_plot(simulator, output_dir, file_name="OCP_plot"):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.jpg")

    config = simulator.config
    logs = simulator.logs
    # import pdb; pdb.set_trace()
    qpos_traj = logs["qpos_traj"][0]
    qvel_traj = logs["qvel_traj"][0]
    u_traj = logs["u_traj"][0]
    yref = logs["yref"]
    nq = simulator.model.nq
    
    dt = config["mpc"]["mpc_timestep"]
    T = qpos_traj.shape[0]  # Total time steps (N+1)
    time = np.arange(T) * dt  # Time axis for states

    time_u = np.arange(u_traj.shape[0]) * dt  # Time axis for control inputs

    # Extract constant reference from first yref entry (ignore time)
    yref_qpos = np.tile(yref[0, : nq], (T, 1))       # shape (T, nq)
    yref_qvel = np.tile(yref[0, nq : 2 * nq], (T, 1))  # shape (T, nq)
    yref_u = np.tile(yref[0, 2 * nq :], (u_traj.shape[0], 1))  # shape (T-1, nu)
    # pdb.set_trace()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot positions
    axs[0].plot(time, qpos_traj, label="Trajectory")
    axs[0].plot(time, yref_qpos, "--", label="Reference")
    axs[0].set_ylabel("Positions")
    axs[0].legend([f"q{i}" for i in range(qpos_traj.shape[1])])
    axs[0].grid(True)

    # Plot velocities
    axs[1].plot(time, qvel_traj, label="Trajectory")
    axs[1].plot(time, yref_qvel, "--", label="Reference")
    axs[1].set_ylabel("Velocities")
    axs[1].legend([f"v{i}" for i in range(qvel_traj.shape[1])])
    axs[1].grid(True)

    # Plot control inputs
    axs[2].plot(time_u, u_traj, label="Control")
    axs[2].plot(time_u, yref_u, "--", label="Reference")
    axs[2].set_ylabel("Control Inputs")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend([f"u{i}" for i in range(u_traj.shape[1])])
    axs[2].grid(True)

    plt.tight_layout()

    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")

    plt.show()

# Load model from xml file
def load_model_from_xml(config):
    """Load a MuJoCo model and create associated data object."""
    base_dir = "models_xml"
    model_name = config["model"]["name"].lower()  # e.g., "two_dof_arm"
    filename = os.path.join(base_dir, f"{model_name}.xml")
    
    model = mujoco.MjModel.from_xml_path(filename)
    data = mujoco.MjData(model)

    return model, data

# Load model from robot descriptions
def load_model_from_robot_descriptions(description_name: str):
    """Load a MuJoCo model and data using robot_descriptions."""
    model = load_robot_description(description_name)  # Loads and parses the MJCF
    data = mujoco.MjData(model)
    
    return model, data

# Apply model config only if loaded from xml
def apply_model_config(config, model):
    try:
        for body_name in config["model"]["mass"].keys():
            mass_value = config["model"]["mass"][body_name] # Respective mass value
            inertia_value = config["model"]["inertia"][body_name] # Respective inertia value
            # Find the body ID you want to modify
            body_id = model.body(name=body_name)
            # Assign the new mass & inertia value
            body_id.mass = mass_value
            body_id.inertia = inertia_value

            # Print to verify
            print(f"Updated body '{body_name}': mass={body_id.mass}, inertia={body_id.inertia}")
    except KeyError:
        print("No custom mass or inertia values found in config; using model defaults.")

def load_model(config):
    # Load MuJoCo model from cml or URDF if available
    if config["mujoco"]["urdf_available"]:
        menagerie_name =  config["mujoco"]["menagerie_name"]
        model, data = load_model_from_robot_descriptions(menagerie_name)
    else:
        model, data = load_model_from_xml(config)
        # Update model parameters from config
        apply_model_config(config, model)
    
    if config["model"]["name"] == "iiwa14": # Converts iiwa14 from PD to torque control (ultimately modify the URDF)
        model.actuator_biastype = np.array([0, 0, 0, 0, 0, 0, 0]) # removes bias by setting it to "none"
        model.actuator_gainprm = np.ones((7,10))   # sets gain to 1
        model.actuator_ctrlrange = np.array([[-320, 320], [-320, 320], [-176,176], [-176,176], [-110,110], [-40,40], [-40,40]]) # sets control range
    
    return model, data

def init_scene_options():
    """Initialize visualization options for rendering."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    # scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
    return scene_option

def get_yref_at_time(t_now, yref):
    """
    Get the most recent reference (no interpolation) for the given time.

    Args:
        t_now (float): Current time in seconds.

    Returns:
        ref (np.ndarray): Reference state at or before time t_now.
    """
    times = yref[:, 0]
    states = yref[:, 1:]

    # If before first timestamp, return the first reference
    if t_now <= times[0]:
        return states[0]
    
    # Find the last index where time <= t_now
    idx = np.searchsorted(times, t_now, side='right') - 1
    return states[idx]