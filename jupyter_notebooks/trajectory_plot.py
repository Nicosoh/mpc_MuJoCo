import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --------------------------------------------------
# 1. Utility Functions
# --------------------------------------------------

def set_equal_3d(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = abs(limits[:, 1] - limits[:, 0])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def compute_global_limits(datasets):
    all_x, all_y, all_z = [], [], []

    for data in datasets:
        traj = data["traj"]
        all_x.append(traj[:, :, 0].flatten())
        all_y.append(traj[:, :, 1].flatten())
        all_z.append(traj[:, :, 2].flatten())

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    return (
        (all_x.min(), all_x.max()),
        (all_y.min(), all_y.max()),
        (all_z.min(), all_z.max()),
    )


# --------------------------------------------------
# 2. Main Animation Function
# --------------------------------------------------

def animate_trajectories(datasets, T):

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectory Comparison")

    # --- Auto axis limits ---
    xlim, ylim, zlim = compute_global_limits(datasets)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    set_equal_3d(ax)

    ax.view_init(elev=25, azim=135)

    # --- Optional Start & Goal ---
    x0 = [0.6696806406948206, -0.23525342081204376, 0.11106397173129595]
    yref = [0.6045709856861988, 0.26660351212961686, 0.11348543272288131]

    ax.scatter(*x0, color="black", s=120, marker="o", label="Start")
    ax.scatter(*yref, color="gold", edgecolor="black", s=150, marker="*", label="Goal")

    # --- Create scatter plots and current-state markers ---
    scatters = []
    current_points = []

    for data in datasets:
        traj = data["traj"]
        H = traj.shape[1] - 1

        # Horizon points (discrete prediction)
        scatter = ax.scatter(
            [], [], [],
            color=data["color"],
            s=18,              # point size
            alpha=0.9,
            label=f'{data["name"]} (H={H})'
        )

        # Current state marker
        current, = ax.plot(
            [], [], [],
            marker=data["marker"],
            color=data["color"],
            markersize=8,
            linestyle="None"
        )

        scatters.append(scatter)
        current_points.append(current)

    ax.legend()

    # --- Update function ---
    def update(t):

        artists = []

        for i, data in enumerate(datasets):

            traj = data["traj"]

            x = traj[t, :, 0]
            y = traj[t, :, 1]
            z = traj[t, :, 2]

            # Update horizon points
            scatters[i]._offsets3d = (x, y, z)

            # Update current state marker (first point of horizon)
            current_points[i].set_data([x[0]], [y[0]])
            current_points[i].set_3d_properties([z[0]])

            artists.extend([scatters[i], current_points[i]])

        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=60, blit=False)

    plt.show()
    # To save instead:
    # ani.save("xyz_3d_animation.mp4", writer="ffmpeg", fps=30, dpi=300)


# --------------------------------------------------
# 3. Script Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    project_root = "/home/nsoh/Documents/mpc_MuJoCo"
    os.chdir(project_root)

    if project_root not in sys.path:
        sys.path.append(project_root)

    from data_collection.data_utils import load_npz

    # --- Load Data ---
    mpc_data_500_data = load_npz(
        "data/2026-03-04_11-43-56_iiwa14_500/2026-03-04_11-43-56_logs.npz"
    )

    mpc_data_5 = load_npz(
        "data/2026-03-04_10-59-06_iiwa14_5/2026-03-04_10-59-06_logs.npz"
    )

    mpc_data_50 = load_npz(
        "data/2026-03-04_11-12-07_iiwa14_50/2026-03-04_11-12-07_logs.npz"
    )

    mpc_data_200 = load_npz(
        "data/2026-03-04_11-24-12_iiwa14_200/2026-03-04_11-24-12_logs.npz"
    )

    # --- Define datasets (ONLY PLACE YOU MODIFY) ---
    datasets = [
        {"name": "MPC 5",   "traj": mpc_data_5["default"]["xyz_traj"],   "color": "purple", "marker": "s"},
        {"name": "MPC 50",  "traj": mpc_data_50["default"]["xyz_traj"],  "color": "green",  "marker": "^"},
        {"name": "MPC 200", "traj": mpc_data_200["default"]["xyz_traj"], "color": "blue",   "marker": "p"},
        {"name": "MPC 500", "traj": mpc_data_500_data["default"]["xyz_traj"], "color": "orange",   "marker": "v"},
    ]

    # --- Compute common time horizon ---
    T = min(data["traj"].shape[0] for data in datasets)

    animate_trajectories(datasets, T)