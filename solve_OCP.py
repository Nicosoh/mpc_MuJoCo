from controller import AcadosMPCController
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import load_yref
from datetime import datetime
import os

def main(model_name):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]
    
    output_dir = "outputs_OCP"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_OCP_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    x0 = np.array(config["mpc"]["x0"])

    # Load reference trajectory
    yref = load_yref(model_name)
    
    # Create MPC controller
    mpc = AcadosMPCController(config, yref)

    nq = int(mpc.nx/2)

    # Set intital conditions
    qpos = x0[:nq]
    qvel = x0[nq:]

    # Form state matrix to feed to the controller
    state = {
        "qpos": np.copy(qpos),
        "qvel": np.copy(qvel),
    }
    
    # Solve OCP and return full traj.
    x_traj, u_traj = mpc.get_full_OCP(state)

    dt = config["mpc"]["mpc_timestep"]
    T = x_traj.shape[0]  # Total time steps (N+1)
    time = np.arange(T) * dt  # Time axis for states

    time_u = np.arange(u_traj.shape[0]) * dt  # Time axis for control inputs

    # Split state into positions and velocities
    qpos_traj = x_traj[:, :nq]
    qvel_traj = x_traj[:, nq:]

    # Extract constant reference from first yref entry (ignore time)
    yref_qpos = np.tile(yref[0, 1 : 1 + nq], (T, 1))       # shape (T, nq)
    yref_qvel = np.tile(yref[0, 1 + nq : 1 + 2 * nq], (T, 1))  # shape (T, nq)
    yref_u = np.tile(yref[0, 1 + 2 * nq :], (u_traj.shape[0], 1))  # shape (T-1, nu)

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

    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)