import sys
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from pin_models import *
import yaml
import argparse

def main(model_name):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]
    
    dt = 0.02

    # Create selected model
    if config["model"]["name"].lower() == "cartpole":
        robot_sys = CartpoleDynamics(timestep=dt, config=config)

    elif config["model"]["name"].lower() == "pendulum":
        robot_sys = PendulumDynamics(timestep=dt, config=config)

    elif config["model"]["name"].lower() == "iiwa14":
        robot_sys = iiwa14Dynamics(timestep=dt, config=config)
    
    elif config["model"]["name"].lower() == "double_pendulum":
        robot_sys = DoublePendulumDynamics(timestep=dt, config=config)

    elif config["model"]["name"].lower() == "cartpole_double_pendulum":
        robot_sys = CartpoleDoublePendulumDynamics(timestep=dt, config=config)
    
    elif config["model"]["name"].lower() == "two_dof_arm":
        robot_sys = TwoDOFArmDynamics(timestep=dt, config=config)
    
    else:
        raise ValueError(f"Unknown model name '{config['model']['name']}'. Add in elif statement for new models")

    nu = config["pin"]["nu"]
    model = robot_sys.model

    print(model)

    q0 = np.random.rand(model.nq)
    q0 = pin.normalize(model, q0)
    v = np.zeros(model.nv)
    u = np.zeros(nu)
    a0 = robot_sys.acc_func(q0, v, u)

    print("a0:", a0)

    x0 = np.append(q0, v)
    xnext = robot_sys.forward(x0, u)

    def integrate_no_control(x0, nsteps):
        states_ = [x0.copy()]
        for t in range(nsteps):
            u = np.zeros(nu)
            xnext = robot_sys.forward(states_[t], u).ravel()
            states_.append(xnext)
        return states_


    states_ = integrate_no_control(x0, nsteps=1000)
    states_ = np.stack(states_).T
    print("states_", states_)

    try:
        viz = MeshcatVisualizer(
            model=model,
            collision_model=robot_sys.collision_model,
            visual_model=robot_sys.visual_model,
        )

        viz.initViewer(open=True) # Set open=True to automatically open the Meshcat tab in your browser
        viz.loadViewerModel("pinocchio")

        viz.displayFrames(visibility=True) # Display all frames

        qs_ = states_[: model.nq, :].T

        viz.play(q_trajectory=qs_, dt=dt)
    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)