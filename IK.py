import os
import sys
import numpy as np
import pinocchio as pin
import qpsolvers

import hppfcl as fcl
import meshcat_shapes
import pink

from pink.utils import process_collision_pairs
from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from pinocchio.robot_wrapper import RobotWrapper
from loop_rate_limiters import RateLimiter

def generate_reference_trajectory(yref, obstacles, config):
    def step_IK():
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver, barriers=barriers, safety_break=False)

        # Integrate motion
        configuration.integrate_inplace(velocity, dt)

        sys.stdout.write("\rvelocity: " + str(velocity) + " | error: {:.3e}".format(error))
        sys.stdout.flush()

        if visualize:
            # Update ee frame visual
            viewer["ee_frame"].set_transform(configuration.get_transform_frame_to_world(tasks["ee"].frame).np)
            # Display
            viz.display(configuration.q)
    
    def add_obstacle_capsules(robot, obstacles_dict):
        """
        Add obstacles as capsule geometries to robot's collision and visual models.
        
        obstacles_dict: 
            e.g. {"obs1": {"from": np.array([x0,y0,z0]), "to": np.array([x1,y1,z1]), "radius": r}}
        """
        for name, obs in obstacles_dict.items():
            p0 = obs["from"]
            p1 = obs["to"]
            radius = obs["radius"]

            # Compute capsule length and placement
            vec = p1 - p0
            length = np.linalg.norm(vec)
            if length < 1e-8:
                length = 1e-6  # avoid zero-length capsule

            # Compute the placement SE3
            midpoint = (p0 + p1) / 2
            # Default capsule axis in Pinocchio/fcl is along z, so we need rotation
            z_axis = vec / length
            # Arbitrary choice of x_axis
            x_axis = np.array([1.0, 0.0, 0.0])
            if np.allclose(z_axis, x_axis):
                x_axis = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])
            placement = pin.SE3(R, midpoint)

            # Create FCL capsule
            shape_pole = fcl.Capsule(radius, length)

            # Create GeometryObject
            geom = pin.GeometryObject(name, 0, placement, shape_pole)
            geom.meshColor = np.array([1.0, 0.0, 0.0, 1.0])  # red for visualization

            # Add to models
            robot.collision_model.addGeometryObject(geom)
            robot.visual_model.addGeometryObject(geom)

        # Reprocess collision pairs after adding obstacles
        robot.collision_data = process_collision_pairs(
            robot.model, robot.collision_model, "models_xml/two_dof_arm.srdf"
        )

    def pad_yref(yref, config):
        """Pad yref to match target_dim by adding zeros."""
        yref_vel = np.zeros_like(yref)
        yref_u = np.zeros((yref.shape[0], config["pin"]["nu"]))

        yref_pos_vel = np.hstack([yref, yref_vel])
        yref_full = np.hstack([yref, yref_vel, yref_u])

        return yref_full, yref_pos_vel
    
    # Numpy print settings
    np.set_printoptions(precision=3)
    
    # Extract IK config
    visualize = config["IK"]["visualize_IK"]
    max_iterations = config["IK"]["max_iterations"]

    # Load model again from XML
    base_dir = "models_xml"
    model_name = config["model"]["name"].lower()  # e.g., "two_dof_arm"
    filename = os.path.join(base_dir, f"{model_name}.xml")
    
    # IK parameters
    stop_thres = config["IK"]["stop_threshold"]

    try:
        robot = RobotWrapper.BuildFromMJCF(filename=filename)

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{filename}' does not exist. Check your models_xml folder.")

    # Manually set velocitylimit 
    robot.model.velocityLimit = np.array(config["model"]["velocitylimit"])

    # Add obstacles
    add_obstacle_capsules(robot, obstacles)

    if visualize:
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
    configuration = pink.Configuration(model=robot.model, data=robot.data, q=robot.q0, collision_model=robot.collision_model, collision_data=robot.collision_data)

    # Collision barriers between self and obstacles
    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=1.0,
        safe_displacement_gain=1.0,
        d_min=0.05, # safety distance for collision
    )

    barriers = [collision_barrier]

    # First goal is the starting position(x0)
    x0 = np.array(config["mpc"]["x0"])

    # Initialize tasks from the current configuration
    for task in tasks.values():
        task.set_target_from_configuration(configuration)
    
    # Copy target format, set translation to x0
    T_target = tasks["ee"].transform_target_to_world.copy()
    T_target.translation = x0
    sys.stdout.write("Setting first IK target to x0:" + str(x0) + "\n")

    # Set ee task target
    tasks["ee"].transform_target_to_world = T_target

    if visualize:
        # Show target in Meshcat
        viewer["target_frame"].set_transform(T_target.np)
        viz.display(configuration.q)

    # Error
    error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)

    # Select QP solver
    solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

    dt = config["mpc"]["mpc_timestep"]

    iterations = 0
    # -------------------------------
    # Moving to starting position
    # -------------------------------
    while True:
        step_IK()
        error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)

        if error < stop_thres:
            sys.stdout.write("\nStarting position reached\n")
            break

        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(f"IK did not converge within {max_iterations} steps. Current error: {error:.3e}")
    
    # Now set actual end goal
    T_target.translation = yref

    if visualize:
        # Show goal position in Meshcat
        viewer["target_frame"].set_transform(T_target.np)

    # Empty list to store traj
    traj_qp = []

    # Store starting position(x0) in joint space
    traj_qp.append(configuration.q)

    # -------------------------------
    # Moving to goal position
    # -------------------------------
    iterations = 0
    while True:
        step_IK()
        traj_qp.append(configuration.q)
        error = np.linalg.norm(T_target.translation - configuration.get_transform_frame_to_world(tasks["ee"].frame).translation)

        if error < stop_thres:
            sys.stdout.write("\nTarget position reached\n")
            break

        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(f"IK did not converge within {max_iterations} steps. Current error: {error:.3e}")
    
    traj_qp = np.array(traj_qp) # Convert to np.array (only positions without velocity)

    sys.stdout.write("Trajectory of length: " + str(traj_qp.shape) + "\n")

    traj_qx_u, traj_qx = pad_yref(traj_qp, config)

    config["mpc"]["x0_q"] = traj_qx[0].tolist() # Add a new field for x0 in joint space

    return traj_qx_u, config