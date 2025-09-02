import matplotlib.pyplot as plt
import mediapy as media
import os

# def plot_signals(time, signals, ylabel_units=None):
#     """
#     Generalized plotter for time series signals.

#     Parameters
#     ----------
#     time : array-like
#         Shared time axis for all signals.
#     signals : dict
#         Dictionary of {name: values}, where name is the plot title and values are the data arrays.
#     ylabel_units : dict, optional
#         Dictionary of {name: unit string} to label the y-axis. Defaults to None.
#     """
#     n = len(signals)
#     dpi = 120
#     width, height = 800, 200 * n
#     figsize = (width / dpi, height / dpi)

#     fig, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)

#     # If only one signal, ax is not a list
#     if n == 1:
#         ax = [ax]

#     for i, (name, values) in enumerate(signals.items()):
#         ax[i].plot(time, values)
#         ax[i].set_title(name)
#         if ylabel_units and name in ylabel_units:
#             ax[i].set_ylabel(ylabel_units[name])
#         else:
#             ax[i].set_ylabel("")

#     ax[-1].set_xlabel("time (s)")
#     plt.tight_layout()
#     plt.show()

def plot_signals(time, signals, ylabel_units=None, save_dir="outputs"):
    """
    Generalized plotter for time series signals.

    Parameters
    ----------
    time : array-like
        Shared time axis for all signals.
    signals : dict
        Dictionary of {name: values}, where name is the plot title and values are the data arrays.
    ylabel_units : dict, optional
        Dictionary of {name: unit string} to label the y-axis. Defaults to None.
    save_dir : str, optional
        Directory where plots will be saved. Default = "plots"
    """
    # --- Create save directory if not exists ---
    os.makedirs(save_dir, exist_ok=True)

    # --- Find next running number ---
    existing = [f for f in os.listdir(save_dir) if f.startswith("plot_") and f.endswith(".png")]
    numbers = []
    for f in existing:
        try:
            num = int(f.replace("plot_", "").replace(".png", ""))
            numbers.append(num)
        except ValueError:
            pass
    next_num = max(numbers) + 1 if numbers else 1
    filename = os.path.join(save_dir, f"plot_{next_num}.png")

    # --- Plotting ---
    n = len(signals)
    dpi = 120
    width, height = 800, 200 * n
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)

    if n == 1:
        ax = [ax]

    for i, (name, values) in enumerate(signals.items()):
        ax[i].plot(time, values)
        ax[i].set_title(name)
        if ylabel_units and name in ylabel_units:
            ax[i].set_ylabel(ylabel_units[name])
        else:
            ax[i].set_ylabel("")

    ax[-1].set_xlabel("time (s)")
    plt.tight_layout()

    # --- Save the figure ---
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

    plt.show()

# def save_video(frames, path="video.mp4", fps=30):
#     """Save recorded frames to a video file."""
#     media.write_video(path, frames, fps=fps)

def save_video(frames, save_dir="outputs", base_name="video", fps=30):
    """
    Save recorded frames to a video file with running numbering.

    Parameters
    ----------
    frames : list or np.ndarray
        Recorded frames.
    save_dir : str, optional
        Directory where videos will be saved. Default = "videos".
    base_name : str, optional
        Base filename for saved videos. Default = "video".
    fps : int, optional
        Frames per second for the video.
    """
    # --- Create directory if not exists ---
    os.makedirs(save_dir, exist_ok=True)

    # --- Find existing video files ---
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith(base_name + "_") and f.endswith(".mp4")
    ]

    numbers = []
    for f in existing:
        try:
            num = int(f.replace(base_name + "_", "").replace(".mp4", ""))
            numbers.append(num)
        except ValueError:
            pass

    # --- Next number ---
    next_num = max(numbers) + 1 if numbers else 1
    filename = os.path.join(save_dir, f"{base_name}_{next_num}.mp4")

    # --- Save video ---
    media.write_video(filename, frames, fps=fps)
    print(f"Video saved to {filename}")