from simulator import load_model, run_simulation
from visualization import save_video, plot_signals
from controller import AcadosMPCController
import numpy as np


def main():
    model, data = load_model("models_xml/inverted_pendulum.xml")

    # Initial condition
    x0 = np.array([0.0, np.pi/6, 0.0, 0.0])

    # Create MPC controller
    mpc = AcadosMPCController(x0, Fmax=80, N_horizon=80, Tf=0.16, use_RTI=False)

    # Run MuJoCo simulation with MPC in the loop
    results, frames = run_simulation(
        model,
        data,
        duration=30.0,
        framerate=30,
        render=True,
        controller=mpc,
        verbose=False,
    )

    # Save video if frames were recorded
    if frames:
        save_video(frames, "video_mpc.mp4", fps=30)

    plot_signals(
        results["time"],
        {
            "Cart Position": results["cart_pos"],
            "Pendulum Angle": results["pend_angle"],
            "Cart Velocity": results["cart_vel"],
            "Pendulum Angular Velocity": results["pend_angvel"],
            "Control Input": results["u_applied"],
        },
        ylabel_units={
            "Cart Position": "m",
            "Cart Velocity": "m/s",
            "Pendulum Angle": "rad",
            "Pendulum Angular Velocity": "rad/s",
            "Control Input": "N",
        },
    )


if __name__ == "__main__":
    main()
