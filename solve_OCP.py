from controller import AcadosMPCController
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(model_name):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]

    x0 = np.array(config["mpc"]["x0"])

    # Create MPC controller
    mpc = AcadosMPCController(config)

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

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot positions
    axs[0].plot(time, qpos_traj)
    axs[0].set_ylabel("Positions")
    axs[0].legend([f"q{i}" for i in range(qpos_traj.shape[1])])
    axs[0].grid(True)

    # Plot velocities
    axs[1].plot(time, qvel_traj)
    axs[1].set_ylabel("Velocities")
    axs[1].legend([f"v{i}" for i in range(qvel_traj.shape[1])])
    axs[1].grid(True)

    # Plot control inputs
    axs[2].plot(time_u, u_traj)
    axs[2].set_ylabel("Control Inputs")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend([f"u{i}" for i in range(u_traj.shape[1])])
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)