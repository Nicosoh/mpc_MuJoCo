# import mujoco
# import mujoco.viewer  # this is the built-in viewer (as of MuJoCo 2.3+)

# # Load the model using robot_descriptions
# from robot_descriptions import iiwa14_mj_description
# model = mujoco.MjModel.from_xml_path(iiwa14_mj_description.MJCF_PATH)

# # Alternatively, load via utility (commented out if using above)
# # from robot_descriptions.loaders.mujoco import load_robot_description
# # model = load_robot_description("panda_mj_description")

# # Create MjData (state container)
# data = mujoco.MjData(model)

# # Open the viewer
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("Viewer is running. Close the window to exit.")
    
#     # Keep rendering until the viewer is closed
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()

# import os
# from pathlib import Path
# from sys import argv

# import pinocchio

# # # This path refers to Pinocchio source code but you can define your own directory here.
# # model_dir = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))

# # # You should change here to set up your own URDF file or just pass it as an argument of
# # # this example.
# # urdf_filename = (
# #     model_dir / "ur_description/urdf/ur5_robot.urdf" if len(argv) < 2 else argv[1]
# # )
# urdf_filename = "/Users/nicodemussoh/Documents/mpc_MuJoCo/urdf/iiwa14_spheres_dense_collision.urdf"
# # Load the urdf model
# model = pinocchio.buildModelFromUrdf(urdf_filename)
# print("model name: " + model.name)

# # Create data required by the algorithms
# data = model.createData()

# # Sample a random configuration
# q = pinocchio.randomConfiguration(model)
# print(f"q: {q.T}")

# # Perform the forward kinematics over the kinematic tree
# pinocchio.forwardKinematics(model, data, q)

# # Print out the placement of each joint of the kinematic tree
# for name, oMi in zip(model.names, data.oMi):
#     print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Closest distance functions ---

# def closest_distance_segment_segment(p1, q1, p2, q2):
#     d1 = q1 - p1
#     d2 = q2 - p2
#     r = p1 - p2
    
#     a = np.dot(d1, d1)
#     e = np.dot(d2, d2)
#     f = np.dot(d2, r)
    
#     if a <= 1e-6 and e <= 1e-6: 
#         return np.linalg.norm(p1 - p2), p1, p2
#     if a <= 1e-6: 
#         return point_segment_distance(p1, p2, q2)
#     if e <= 1e-6: 
#         return point_segment_distance(p2, p1, q1)

#     c = np.dot(d1, r)
#     b = np.dot(d1, d2)
#     denom = a * e - b * b
    
#     if denom != 0.0:
#         s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
#     else:
#         s = 0.0
        
#     t = (b * s + f) / e
    
#     if t < 0.0:
#         t = 0.0
#         s = np.clip(-c / a, 0.0, 1.0)
#     elif t > 1.0:
#         t = 1.0
#         s = np.clip((b - c) / a, 0.0, 1.0)
        
#     closest_on_1 = p1 + s * d1
#     closest_on_2 = p2 + t * d2
    
#     return np.linalg.norm(closest_on_1 - closest_on_2), closest_on_1, closest_on_2

# def point_segment_distance(pt, p_seg, q_seg):
#     segment = q_seg - p_seg
#     v = pt - p_seg
#     t = np.clip(np.dot(v, segment) / np.dot(segment, segment), 0.0, 1.0)
#     closest = p_seg + t * segment
#     return np.linalg.norm(pt - closest), closest, pt

# # --- 2D Example ---

# # Capsule 1
# cap1_start = np.array([0, 0])
# cap1_end   = np.array([0, 5])
# cap1_rad   = 0.5

# # Capsule 2
# cap2_start = np.array([2, 4])
# cap2_end   = np.array([3, 3])
# cap2_rad   = 0.3

# # Compute distance
# dist, p_closest, q_closest = closest_distance_segment_segment(
#     cap1_start, cap1_end, cap2_start, cap2_end
# )
# min_dist = dist - (cap1_rad + cap2_rad)

# print(f"Distance between segments: {dist:.3f}")
# print(f"Distance between capsules: {min_dist:.3f}")
# print(f"Closest point on capsule 1: {p_closest}")
# print(f"Closest point on capsule 2: {q_closest}")

# # --- Plotting ---
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_aspect('equal')

# # Draw capsule 1
# ax.plot([cap1_start[0], cap1_end[0]], [cap1_start[1], cap1_end[1]], 'b-', linewidth=5, alpha=0.5)
# circle1_start = plt.Circle(cap1_start, cap1_rad, color='b', alpha=0.3)
# circle1_end   = plt.Circle(cap1_end, cap1_rad, color='b', alpha=0.3)
# ax.add_patch(circle1_start)
# ax.add_patch(circle1_end)

# # Draw capsule 2
# ax.plot([cap2_start[0], cap2_end[0]], [cap2_start[1], cap2_end[1]], 'r-', linewidth=5, alpha=0.5)
# circle2_start = plt.Circle(cap2_start, cap2_rad, color='r', alpha=0.3)
# circle2_end   = plt.Circle(cap2_end, cap2_rad, color='r', alpha=0.3)
# ax.add_patch(circle2_start)
# ax.add_patch(circle2_end)

# # Draw line connecting closest points
# ax.plot([p_closest[0], q_closest[0]], [p_closest[1], q_closest[1]], 'g--', linewidth=2, label='Closest points')

# # Plot closest points
# ax.plot(p_closest[0], p_closest[1], 'bo', markersize=8)
# ax.plot(q_closest[0], q_closest[1], 'ro', markersize=8)

# ax.set_xlim(-1,5)
# ax.set_ylim(-1,6)
# ax.set_title('2D Capsule Distance')
# ax.legend(['Capsule 1','Capsule 2','Closest Points Line'])
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # --- Closest distance functions ---

# def closest_distance_segment_segment(p1, q1, p2, q2):
#     d1 = q1 - p1
#     d2 = q2 - p2
#     r = p1 - p2
    
#     a = np.dot(d1, d1)
#     e = np.dot(d2, d2)
#     f = np.dot(d2, r)
    
#     if a <= 1e-6 and e <= 1e-6: 
#         return np.linalg.norm(p1 - p2), p1, p2
#     if a <= 1e-6: 
#         return point_segment_distance(p1, p2, q2)
#     if e <= 1e-6: 
#         return point_segment_distance(p2, p1, q1)

#     c = np.dot(d1, r)
#     b = np.dot(d1, d2)
#     denom = a * e - b * b
    
#     if denom != 0.0:
#         s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
#     else:
#         s = 0.0
        
#     t = (b * s + f) / e
    
#     if t < 0.0:
#         t = 0.0
#         s = np.clip(-c / a, 0.0, 1.0)
#     elif t > 1.0:
#         t = 1.0
#         s = np.clip((b - c) / a, 0.0, 1.0)
        
#     closest_on_1 = p1 + s * d1
#     closest_on_2 = p2 + t * d2
    
#     return np.linalg.norm(closest_on_1 - closest_on_2), closest_on_1, closest_on_2

# def point_segment_distance(pt, p_seg, q_seg):
#     segment = q_seg - p_seg
#     v = pt - p_seg
#     t = np.clip(np.dot(v, segment) / np.dot(segment, segment), 0.0, 1.0)
#     closest = p_seg + t * segment
#     return np.linalg.norm(pt - closest), closest, pt

# # --- 3D Example ---

# # Capsule 1 (stationary)
# cap1_start = np.array([0, 0, 0])
# cap1_end   = np.array([0, 0, 5])
# cap1_rad   = 0.5

# # Capsule 2 (moving, e.g., robotic arm)
# cap2_start = np.array([1, 1, 2])
# cap2_end   = np.array([1.5, 0, 4])
# cap2_rad   = 0.3

# # Compute distance
# dist, p_closest, q_closest = closest_distance_segment_segment(
#     cap1_start, cap1_end, cap2_start, cap2_end
# )
# min_dist = dist - (cap1_rad + cap2_rad)

# print(f"Distance between segments: {dist:.3f}")
# print(f"Distance between capsules: {min_dist:.3f}")
# print(f"Closest point on capsule 1: {p_closest}")
# print(f"Closest point on capsule 2: {q_closest}")

# # --- Plotting ---
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot capsules as lines (we won't draw actual cylinder radii in 3D for simplicity)
# ax.plot([cap1_start[0], cap1_end[0]], 
#         [cap1_start[1], cap1_end[1]], 
#         [cap1_start[2], cap1_end[2]], 'b-', linewidth=5, alpha=0.5, label='Capsule 1')

# ax.plot([cap2_start[0], cap2_end[0]], 
#         [cap2_start[1], cap2_end[1]], 
#         [cap2_start[2], cap2_end[2]], 'r-', linewidth=5, alpha=0.5, label='Capsule 2')

# # Draw line connecting closest points
# ax.plot([p_closest[0], q_closest[0]],
#         [p_closest[1], q_closest[1]],
#         [p_closest[2], q_closest[2]],
#         'g--', linewidth=2, label='Closest points')

# # Plot closest points
# ax.scatter(*p_closest, color='b', s=50)
# ax.scatter(*q_closest, color='r', s=50)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Capsule Distance')
# ax.legend()
# ax.set_box_aspect([1,1,1])  # Equal aspect ratio

# plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------------------
# # 1. Define two random line segments in 3D
# # -----------------------------------------
# np.random.seed(1)
# p1 = np.array([0,0,0])
# q1 = np.array([1,1,0.2])

# p2 = np.array([0.2,1.0,0.5])
# q2 = np.array([1.0,0.2,0.8])

# u = q1 - p1
# v = q2 - p2
# w0 = p1 - p2

# # -----------------------------------------
# # Distance squared function f(s,t)
# # -----------------------------------------
# def dist_sq(s, t):
#     """
#     Squared distance between point p1 + s*u and p2 + t*v
#     """
#     w = w0 + s*u - t*v
#     return np.dot(w, w)


# # -----------------------------------------
# # 2. Evaluate over an S-T grid
# # -----------------------------------------
# N = 100
# S = np.linspace(0,1,N)
# T = np.linspace(0,1,N)
# SS, TT = np.meshgrid(S,T)
# Z = np.zeros_like(SS)

# for i in range(N):
#     for j in range(N):
#         Z[i,j] = dist_sq(SS[i,j], TT[i,j])


# # -----------------------------------------
# # 3. Plot 3D surface of f(s,t)
# # -----------------------------------------
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(12,5))

# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(SS, TT, Z, cmap='viridis', edgecolor='none')
# ax.set_title("Squared Distance Function $f(s,t)$")
# ax.set_xlabel("s")
# ax.set_ylabel("t")
# ax.set_zlabel("distance^2")

# # -----------------------------------------
# # 4. Plot contour map to show convexity
# # -----------------------------------------
# ax2 = fig.add_subplot(122)
# contours = ax2.contourf(SS, TT, Z, cmap='viridis', levels=30)
# plt.colorbar(contours, ax=ax2)
# ax2.set_title("Contour Plot of $f(s,t)$ (Convex)")
# ax2.set_xlabel("s")
# ax2.set_ylabel("t")

# plt.tight_layout()
# plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sys
# import numpy as np
# import pinocchio as pin
# from pinocchio.visualize import MeshcatVisualizer
# from pin_models import *
# import yaml
# import argparse
# import time
# from loop_rate_limiters import RateLimiter
# # Pink imports for velocity-based IK
# import pink
# from pink import solve_ik
# from pink.tasks import FrameTask, PostureTask
# from pink.visualization import start_meshcat_visualizer

# # Rate limiter
# from loop_rate_limiters import RateLimiter  # pip install ratelimiter


# def main(model_name):
#     # Load configuration
#     with open("config.yaml", "r") as f:
#         config = yaml.safe_load(f)[model_name]

#     dt = 0.02
#     nsteps = 100

#     # Create selected robot system
#     if config["model"]["name"].lower() == "two_dof_arm":
#         robot_sys = TwoDOFArmDynamics(timestep=dt, config=config)
#     else:
#         raise NotImplementedError("Only 'two_dof_arm' example implemented")

#     # Random initial configuration
#     q0 = np.random.rand(robot_sys.model.nq)
#     q0 = pin.normalize(robot_sys.model, q0)
#     v0 = np.zeros(robot_sys.model.nv)

#     # Initialize configuration object for Pink
#     configuration = pink.Configuration(robot_sys.model, robot_sys.data, q0.copy())

#     # Define tasks
#     ee_frame = "ee"
#     tasks = {
#         "tip": FrameTask(
#             frame=ee_frame,
#             position_cost=1.0,      # [cost]/[m]
#             orientation_cost=1e-3,  # [cost]/[rad]
#         ),
#         "posture": PostureTask(cost=1e-2)
#     }

#     # Set initial task targets from current configuration
#     for task in tasks.values():
#         task.set_target_from_configuration(configuration)

#     # Slightly offset target (example)
#     tasks["tip"].transform_target_to_world.translation[2] -= 0.1
#     tasks["tip"].transform_target_to_world.translation[1] -= 0.2

#     # Select QP solver for solve_ik
#     import qpsolvers
#     solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

#     # Store trajectory
#     q_traj = [configuration.q.copy()]

#     # Create ratelimiter (50 Hz)
#     limiter = RateLimiter(max_calls=50, period=1.0)

#     for step in range(nsteps):
#         with limiter:
#             # Compute joint velocity to move toward target
#             velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
#             # Integrate velocity into configuration
#             configuration.integrate_inplace(velocity, dt)

#             # Store trajectory
#             q_traj.append(configuration.q.copy())

#     # Save trajectory
#     q_traj = np.stack(q_traj)
#     np.save("q_trajectory.npy", q_traj)
#     print(f"Saved trajectory of shape {q_traj.shape} to q_trajectory.npy")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model", type=str, help="Model name (e.g., 'two_dof_arm')")
#     args = parser.parse_args()
#     main(args.model)



# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat
 
import sys
from pathlib import Path
 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
 
# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = "pin_models"
 
model_path = pinocchio_model_dir
mesh_dir = pinocchio_model_dir
# urdf_filename = "talos_reduced.urdf"
# urdf_model_path = join(join(model_path,"talos_data/robots"),urdf_filename)
urdf_filename = "solo.urdf"
urdf_model_path = model_path / "solo_description/robots" / urdf_filename
 
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in
# a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. "
        "It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
 
# Load the robot in the viewer.
viz.loadViewerModel()
 
# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)
 
# Create a convex shape from solo main body
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex
 
# Place the convex object on the scene and display it
if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    geometry = pin.GeometryObject("convex", 0, placement, convex)
    geometry.meshColor = np.ones(4)
    # Add a PhongMaterial to the convex object
    geometry.overrideMaterial = True
    geometry.meshMaterial = pin.GeometryPhongMaterial()
    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
    geometry.meshMaterial.meshShininess = 0.8
    visual_model.addGeometryObject(geometry)
    # After modifying the visual_model we must rebuild
    # associated data inside the visualizer
    viz.rebuildData()
 
# Display another robot.
viz2 = MeshcatVisualizer(model, collision_model, visual_model)
viz2.initViewer(viz.viewer)
viz2.loadViewerModel(rootNodeName="pinocchio2")
q = q0.copy()
q[1] = 1.0
viz2.display(q)
 
# standing config
q1 = np.array(
    [0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
)
 
v0 = np.random.randn(model.nv) * 2
data = viz.data
pin.forwardKinematics(model, data, q1, v0)
frame_id = model.getFrameId("HR_FOOT")
viz.display()
viz.drawFrameVelocities(frame_id=frame_id)
 
model.gravity.linear[:] = 0.0
dt = 0.01
 
 
def sim_loop():
    tau0 = np.zeros(model.nv)
    qs = [q1]
    vs = [v0]
    nsteps = 100
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        viz.drawFrameVelocities(frame_id=frame_id)
    return qs, vs
 
 
qs, vs = sim_loop()
 
fid2 = model.getFrameId("FL_FOOT")
 
 
def my_callback(i, *args):
    viz.drawFrameVelocities(frame_id)
    viz.drawFrameVelocities(fid2)
 
 
with viz.create_video_ctx("../leap.mp4"):
    viz.play(qs, dt, callback=my_callback)

from pinocchio.robot_wrapper import RobotWrapper