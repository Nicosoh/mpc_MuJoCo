import os
import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from pinocchio.robot_wrapper import RobotWrapper
from loop_rate_limiters import RateLimiter

def generate_reference_trajectory(yref, obstacles, config):

    def step_IK():
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
        print("velocity: ", velocity)
        # Integrate motion
        configuration.integrate_inplace(velocity, dt)
        print("error:", error)
        # Update ee frame visual
        viewer["ee_frame"].set_transform(configuration.get_transform_frame_to_world(tasks["ee"].frame).np)
        # Display
        viz.display(configuration.q)
        
        rate.sleep()

    # Load model again from XML
    base_dir = "models_xml"
    model_name = config["model"]["name"].lower()  # e.g., "two_dof_arm"
    filename = os.path.join(base_dir, f"{model_name}.xml")
    
    # IK parameters
    stop_thres = 1e-3

    try:
        robot = RobotWrapper.BuildFromMJCF(filename=filename)

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{filename}' does not exist. Check your models_xml folder.")

    # Manually set velocitylimit 
    robot.model.velocityLimit = np.array(config["model"]["velocitylimit"])

    # Launch Viewer
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer

    meshcat_shapes.frame(viewer["target_frame"], opacity=0.3) # Target frame
    meshcat_shapes.frame(viewer["ee_frame"], opacity=1.0) # End-effector frame

    # -------------------------------
    # Define tasks
    # -------------------------------
    tasks = {
        "ee": FrameTask(
            "attachment_site",
            position_cost=1.0, # Translation position
            orientation_cost=1e-3 # Pose of the end-effector
        ),
        "posture": PostureTask(cost=1e-2) # Regularisation of angles
    }

    # Initial configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)

    # First goal is the starting position(x0)
    x0 = np.array(config["mpc"]["x0"])

    # Initialize tasks from the current configuration
    for task in tasks.values():
        task.set_target_from_configuration(configuration)
    
    # Copy target format, set translation to x0
    T_target = tasks["ee"].transform_target_to_world.copy()
    T_target.translation = x0
    print("Setting first IK target to x0:", x0)

    # Set ee task target
    tasks["ee"].transform_target_to_world = T_target

    # Show target in Meshcat
    viewer["target_frame"].set_transform(T_target.np)
    viz.display(configuration.q)

    # Error
    error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)

    # Select QP solver
    solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

    rate = RateLimiter(frequency=1/config["mpc"]["mpc_timestep"], warn=False)
    dt = rate.period

    # -------------------------------
    # Moving to starting position
    # -------------------------------
    while True:
        step_IK()
        error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)
        if error < stop_thres:
            print("Starting position reached")
            break
    

    # Now set actual end goal
    T_target.translation = yref
    # Show goal position in Meshcat
    viewer["target_frame"].set_transform(T_target.np)

    # Empty list to store traj
    traj_q0 = []

    # Store starting position(x0) in joint space
    traj_q0.append(configuration.q)

    # -------------------------------
    # Moving to goal position
    # -------------------------------
    while True:
        step_IK()
        traj_q0.append(configuration.q)
        error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)

        if error < stop_thres:
            print("Target position reached")
            break
    
    return np.array(traj_q0)