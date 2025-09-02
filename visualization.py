import matplotlib.pyplot as plt
import mediapy as media

def plot_signals(time, signals, ylabel_units=None):
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
    """
    n = len(signals)
    dpi = 120
    width, height = 800, 200 * n
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)

    # If only one signal, ax is not a list
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
    plt.show()

def save_video(frames, path="video.mp4", fps=30):
    """Save recorded frames to a video file."""
    media.write_video(path, frames, fps=fps)