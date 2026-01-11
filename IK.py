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
from pink.limits import AccelerationLimit

class InverseKinematicsSolver:
    def __init__(self, config, collision_config=None):
        self.config = config
        self.collision_config = collision_config

        # Extract IK config
        self.visualize = self.config["IK"]["visualize_IK"]
        self.max_iterations = self.config["IK"]["max_iterations"]
        self.stop_thres = self.config["IK"]["stop_threshold"]
        self.dt = config["mpc"]["mpc_timestep"]
        self.velocity_thres = self.config["IK"]["velocity_threshold"]
        self.limit_accel = self.config["IK"]["limit_acceleration"]
        self.d_min = self.config["IK"]["safety_distance"]

        self.traj_q = []

        self.setup()

    def setup(self):
        # =========== Load model ===========
        self.load_model()

        # =========== Load x0 ===========
        self.x0_q = self.get_valid_q("x0_q", "x0_q_range")
        self.config["mpc"]["x0_q"] = self.x0_q.tolist()       # Save x0 in joint space to config for summary saving purpose
        self.traj_q.append(self.x0_q)                       # Record starting position
        
        # =========== Define configuration ===========
        self.configuration = pink.Configuration(model=self.robot.model, data=self.robot.data, q=self.x0_q, collision_model=self.robot.collision_model, collision_data=self.robot.collision_data)

        # =========== Define tasks ===========
        self.tasks = {
            "ee": FrameTask(
                "attachment_site",
                position_cost=1.0, # Translation position
                orientation_cost=1e-3 # Pose of the end-effector
            ),
            "posture": PostureTask(cost=1e-2)} # Regularisation of angles
        
        # Initialize tasks from the current configuration
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

        # Copy target format, set translation to x0
        self.T_target = self.tasks["ee"].transform_target_to_world.copy()

        # Set ee task target
        self.tasks["ee"].transform_target_to_world = self.T_target
        
        # Select QP solver
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

        if self.limit_accel:
            self.limits = (self.configuration.model.velocity_limit, self.accellimit)
        
        if self.visualize:
            self.start_viz()
            self.update_target_viz(self.T_target.np)  # currently only updating translation, if 6DOF then update
            self.update_robot_viz(self.configuration.q)
    
    def get_valid_q(self, q_name: str, q_range: str, max_attempts=100):
        q_range = np.array(self.config["mpc"][q_range])

        for attempt in range(max_attempts):
            q = self.load_q(q_name)
            min_dist = self.distance_check(q)

            if (not self.collision_check(q)
                and self.frame_within_bbox(q, "attachment_site", q_range)
                and min_dist > self.d_min
            ):
                print(
                    f"Loaded valid {q_name} configuration in {attempt+1} attempts\n"
                    f"  q:        {q}\n"
                    f"  q_range:  min={q_range[0]}, max={q_range[1]}\n"
                    f"  min_dist: {min_dist:.4f} (d_min={self.d_min})\n"
                )
                return q
            
            elif not self.config["mpc"][f"{q_name}_random"]:
                raise ValueError(f"Loaded {q_name} configuration is in collision or outside of {q_name}_range or has a minimum distance less than {self.d_min}. Please provide a valid {q_name} or set {q_name}_random to True.")

        raise ValueError(f"Failed to load valid {q_name} configuration in {max_attempts} attempts: ")
    
    def load_q(self, q_name: str):
        if self.config["mpc"][f"{q_name}_random"]:
            # Random initial configuration sampled from joint limits
            return np.random.uniform(low=self.robot.model.lowerPositionLimit,
                                        high=self.robot.model.upperPositionLimit)
        else:
            return np.array(self.config["mpc"][f"{q_name}"])
    
    def frame_within_bbox(self, q, frame_name, bbox):
        """
        Check if a frame is within a world-axis-aligned bounding box.
        """
        frame_id = self.robot.model.getFrameId(frame_name)

        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        pos = self.robot.data.oMf[frame_id].translation

        if np.all(pos >= bbox[0]) and np.all(pos <= bbox[1]):
            self.config["mpc"]["x0"] = pos.tolist()  # Save x0 in Cartesian space to config for summary saving purpose
            return True
        else:
            return False
    
    def collision_check(self, q):
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateGeometryPlacements(
            self.robot.model,
            self.robot.data,
            self.robot.collision_model,
            self.robot.collision_data,
            q
        )

        return pin.computeCollisions(
            self.robot.collision_model,
            self.robot.collision_data,
            stop_at_first_collision=True
        )
    
    def distance_check(self, q):
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateGeometryPlacements(
            self.robot.model,
            self.robot.data,
            self.robot.collision_model,
            self.robot.collision_data,
            q
        )
        idx = pin.computeDistances(self.robot.collision_model,self.robot.collision_data)

        return self.robot.collision_data.distanceResults[idx].min_distance

    def load_model(self):
        # Load model from XML
        base_dir = "models_xml"
        model_name = self.config["model"]["name"].lower()  # e.g., "two_dof_arm"
        filename = os.path.join(base_dir, f"{model_name}.xml")

        try:
            # Load from MJCF
            self.robot = RobotWrapper.BuildFromMJCF(filename=filename)
            # Set velocitylimit
            self.robot.model.velocityLimit = np.array(self.config["IK"]["velocitylimit"])
            if self.limit_accel:
                self.accellimit = AccelerationLimit(self.robot.model, np.array(self.config["IK"]["accelerationlimit"]))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{filename}' does not exist. Check your models_xml folder.")
        
        if self.collision_config is not None:
            self.add_obstacle_capsules()
            # Collision barriers between self and obstacles
            collision_barrier = SelfCollisionBarrier(
                n_collision_pairs=len(self.robot.collision_model.collisionPairs),
                gain=1.0,
                safe_displacement_gain=1.0,
                d_min=self.d_min, # safety distance for collision
            )

            self.barriers = [collision_barrier]
        else:
            self.barriers = None

    def add_obstacle_capsules(self):
        """
        Add obstacles as capsule geometries to robot's collision and visual models.
        
        obstacles_dict: 
            e.g. {"obs1": {"from": np.array([x0,y0,z0]), "to": np.array([x1,y1,z1]), "radius": r}}
        """
        obstacles_dict = self.collision_config["obstacles"]

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
            self.robot.collision_model.addGeometryObject(geom)
            self.robot.visual_model.addGeometryObject(geom)

        # Reprocess collision pairs after adding obstacles
        self.robot.collision_data = process_collision_pairs(
            self.robot.model, self.robot.collision_model, "models_xml/two_dof_arm.srdf"
        )

    def start_viz(self):
        if self.visualize:
            # Launch Viewer
            self.viz = start_meshcat_visualizer(self.robot)
            self.viewer = self.viz.viewer

            meshcat_shapes.frame(self.viewer["target_frame"], opacity=0.3) # Target frame
            meshcat_shapes.frame(self.viewer["ee_frame"], opacity=1.0) # End-effector frame

    def update_target_viz(self, T_target):
        # Show goal position in Meshcat
        self.viewer["target_frame"].set_transform(T_target) # currently only updating translation, if 6DOF then update
    
    def update_ee_viz(self, T_ee):
        # Show goal position in Meshcat
        self.viewer["ee_frame"].set_transform(T_ee) # currently only updating translation, if 6DOF then update

    def update_robot_viz(self, q):
        self.viz.display(q)

    def pad_yref(self):
        """Pad yref to match target_dim by adding zeros."""
        yref_vel = np.zeros_like(self.traj_q)
        yref_u = np.zeros((self.traj_q.shape[0], self.config["pin"]["nu"]))

        self.yref_pos_vel = np.hstack([self.traj_q, yref_vel])
        self.yref_full = np.hstack([self.traj_q, yref_vel, yref_u])

    def IK_to_XYZ(self, yref):
        # Now set actual end goal
        self.T_target.translation = yref

        if self.visualize:
            self.update_target_viz(self.T_target.np) # currently only updating translation, if 6DOF then update
            self.update_robot_viz(self.x0_q)

        # IK to goal
        self.IK_loop(record = True)
        
        self.traj_q = np.array(self.traj_q) # Convert to np.array (only positions without velocity)

        sys.stdout.write("Trajectory of length: " + str(self.traj_q.shape) + "\n")

        self.pad_yref()
        
        self.config["mpc"]["x0_q"] = self.yref_pos_vel[0].tolist()
        
        return self.yref_full, self.config
    
    def IK_loop(self, record: bool):
        iterations = 0
        vel_below_thresh = 0
        error = np.linalg.norm(self.T_target.translation - self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).translation)

        while error > self.stop_thres:
            # Integrate motion
            if self.limit_accel:
                velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver, barriers=self.barriers, limits=self.limits, safety_break=False)
                self.configuration.integrate_inplace(velocity, self.dt)
                self.accellimit.set_last_integration(velocity, self.dt)
            else:
                velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver, barriers=self.barriers, safety_break=False)
                self.configuration.integrate_inplace(velocity, self.dt)

            error = np.linalg.norm(self.T_target.translation - self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).translation)
            
            sys.stdout.write("\rvelocity: " + str(velocity) + " | error: {:.3e}".format(error))
            sys.stdout.flush()

            if self.visualize:
                # Update ee frame visual
                self.update_ee_viz(self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).np)
                # Display robot config
                self.update_robot_viz(self.configuration.q)

            if record:
                self.traj_q.append(self.configuration.q)

            iterations += 1
            if iterations > self.max_iterations:
                raise RuntimeError(f"IK did not converge within {self.max_iterations} steps. Current error: {error:.3e}")
            
            if np.all(np.abs(velocity) < self.velocity_thres):
                vel_below_thresh += 1
            else:
                vel_below_thresh = 0  # reset if condition breaks

            if vel_below_thresh >= 10:
                raise RuntimeError("Velocity below threshold for 10 consecutive iterations")

        sys.stdout.write("\nPosition reached\n")