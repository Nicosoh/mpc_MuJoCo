import mujoco
import numpy as np
from tqdm import tqdm
from robot_descriptions.loaders.mujoco import load_robot_description
from controller import BaseMPCController
from utils import load_x0, load_yref

# Load model from xml file
def load_model_from_xml(model_path: str):
    """Load a MuJoCo model and create associated data object."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data

# Load model from robot descriptions
def load_model_from_robot_descriptions(description_name: str):
    """Load a MuJoCo model and data using robot_descriptions."""
    model = load_robot_description(description_name)  # Loads and parses the MJCF
    data = mujoco.MjData(model)
    
    return model, data

# Apply model config only if loaded from xml
def apply_model_config(config, model):
    try:
        for body_name in config["model"]["mass"].keys():
            mass_value = config["model"]["mass"][body_name] # Respective mass value
            inertia_value = config["model"]["inertia"][body_name] # Respective inertia value
            # Find the body ID you want to modify
            body_id = model.body(name=body_name)
            # Assign the new mass & inertia value
            body_id.mass = mass_value
            body_id.inertia = inertia_value

            # Print to verify
            print(f"Updated body '{body_name}': mass={body_id.mass}, inertia={body_id.inertia}")
    except KeyError:
        print("No custom mass or inertia values found in config; using model defaults.")

def load_model(config):
    # Load MuJoCo model from cml or URDF if available
    if config["mujoco"]["urdf_available"]:
        menagerie_name =  config["mujoco"]["menagerie_name"]
        model, data = load_model_from_robot_descriptions(menagerie_name)
    else:
        path = config["mujoco"]["model_path"]
        model, data = load_model_from_xml(path)
        # Update model parameters from config
        apply_model_config(config, model)
    
    if config["model"]["name"] == "iiwa14": # Converts iiwa14 from PD to torque control
        model.actuator_biastype = np.array([0, 0, 0, 0, 0, 0, 0]) # removes bias by setting it to "none"
        model.actuator_gainprm = np.ones((7,10))   # sets gain to 1
        model.actuator_ctrlrange = np.array([[-320, 320], [-320, 320], [-176,176], [-176,176], [-110,110], [-40,40], [-40,40]]) # sets control range
    
    return model, data

def init_scene_options():
    """Initialize visualization options for rendering."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    # scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
    return scene_option

def get_yref_at_time(t_now, yref):
    """
    Get the most recent reference (no interpolation) for the given time.

    Args:
        t_now (float): Current time in seconds.

    Returns:
        ref (np.ndarray): Reference state at or before time t_now.
    """
    times = yref[:, 0]
    states = yref[:, 1:]

    # If before first timestamp, return the first reference
    if t_now <= times[0]:
        return states[0]
    
    # Find the last index where time <= t_now
    idx = np.searchsorted(times, t_now, side='right') - 1
    return states[idx]

class MuJoCoSimulator:
    def __init__(self, config):
        self.config = load_x0(config=config)                        # Load x0
        self.yref = load_yref(model_name=self.config["model"]["name"])   # Load yref
        self.model, self.data = load_model(self.config)                  # Create MuJoCo simulator object with loaded model
        self.controller = BaseMPCController(self.config, self.yref) # Create Controller

        self.sanity_check() # Sanity check between model, controller, yref, x0

        self.model.opt.timestep = self.config["mujoco"]["sim_timestep"]  # Set simulation timestep

        if self.config["mujoco"]["render"]: # Set render options and create renderer
            self.scene_option = init_scene_options()
            height, width = self.config["mujoco"]["resolution"]
            self.renderer = mujoco.Renderer(self.model, height, width)
        
        # Reset model
        mujoco.mj_resetData(self.model, self.data)
    
    def sanity_check(self):
        # --- Sanity checks ---
        nq, nv, nu = self.model.nq, self.model.nv, self.model.nu
        nx_model = nq + nv
        nx_ctrl = self.controller.nx
        nu_ctrl = self.controller.nu
        x0 = self.config["mpc"]["x0"]
        solve_ocp = self.config["mpc"]["solve_ocp"]
        full_traj = self.config["mpc"]["full_traj"]

        # Check that controller state dim matches model state dim
        if nx_ctrl != nx_model:
            raise ValueError(
                f"State dimension mismatch: controller.nx={nx_ctrl}, "
                f"but model.nq+model.nv={nx_model}"
            )

        # Check that x0 length matches controller and model states
        if len(x0) != nx_ctrl:
            raise ValueError(
                f"x0 length mismatch: len(x0)={len(x0)}, expected {nx_ctrl} "
                f"(must match controller.nx and model.nq+nv)"
            )

        # Check that yref dimension matches controller outputs
        # yref can be shape (N, ny) or (ny,)
        if self.yref.ndim == 1:
            ny_yref = self.yref.shape[0]
        elif self.yref.ndim == 2:
            ny_yref = self.yref.shape[1]
        else:
            raise ValueError(f"Unexpected yref shape: {self.yref.shape}")

        if hasattr(self.controller, "ny"):
            if ny_yref != nx_ctrl+nu_ctrl:
                raise ValueError(
                    f"yref dimension mismatch: yref has {ny_yref}, "
                    f"controller expects nx + nu ={nx_ctrl+nu_ctrl}"
                )

        # Check that input dimensions match
        if nu_ctrl != nu:
            raise ValueError(
                f"Input dimension mismatch: controller.nu={nu_ctrl}, model.nu={nu}"
            )
        
        if solve_ocp and not full_traj:
            raise ValueError(
                "Input mismatch: solve_ocp=True requires full_traj=True."
            )

        print("Sanity checks passed: model, controller, x0, and yref are consistent.")

    def run(self):
        # Extract config for easier reading
        mpc_config = self.config["mpc"]
        mujoco_config = self.config["mujoco"]

        # Extract parameters from config for simulator
        sim_duration = mujoco_config["sim_duration"]
        verbose = mujoco_config["verbose"]
        render = mujoco_config["render"]
        sim_framerate = mujoco_config["sim_framerate"]
        sim_timestep = mujoco_config["sim_timestep"]

        # Extract parameters from config for MPC
        self.mpc_timestep = mpc_config["mpc_timestep"]
        early_termination = mpc_config["early_termination"]
        termination_cost = np.array(mpc_config["termination_cost"])
        x0 = np.array(mpc_config["x0"])
        solve_ocp = mpc_config["solve_ocp"]

        # Init variables for counting and so on...
        self.last_u = np.zeros(self.model.nu)
        self.next_mpc_time = 0.0
        
        # Empty lists for recording
        self.frames = []
        self.logs = {
            "time": [],
            "qpos": [],
            "qvel": [],
            "yref": [],
            "u_applied": [],
            "cost": [],
            "qpos_traj": [],
            "qvel_traj": [],
            "u_traj": [],
        }

        # Set initial position and velocity from x0
        self.data.qpos[:] = x0[:self.model.nq]
        self.data.qvel[:] = x0[self.model.nq:]

        mujoco.mj_forward(self.model, self.data) # Step forward to compute derived quantities
        print(f"Model has {self.model.nq} DoFs (qpos), {self.model.nv} velocities (qvel), and {self.model.nu} actuators.")
        print("Initial qpos:", self.data.qpos)
        print("Initial qvel:", self.data.qvel)

        # Calculate number of steps (for tqdm total)
        steps = int(sim_duration / sim_timestep)
        pbar = tqdm(total=steps, desc="Simulating")

        # Main simulation loop
        while self.data.time < sim_duration:
            # Step simulation
            cost, qpos_traj, qvel_traj, u_traj = self.step_sim()

            # Only log data if MPC actually produced new control
            if cost is not None and qpos_traj is not None and qvel_traj is not None and u_traj is not None:
                self.logs["time"].append(np.copy(self.data.time))
                self.logs["qpos"].append(np.copy(self.data.qpos))
                self.logs["qvel"].append(np.copy(self.data.qvel))
                self.logs["u_applied"].append(np.copy(self.data.ctrl))
                self.logs["cost"].append(cost)
                self.logs["qpos_traj"].append(qpos_traj)
                self.logs["qvel_traj"].append(qvel_traj)
                self.logs["u_traj"].append(u_traj)

            if verbose:
                print(
                    f"t = {self.data.time:.3f}s | "
                    f"qpos = {np.round(self.data.qpos, 2)} | "
                    f"qvel = {np.round(self.data.qvel, 2)} | "
                    f"ctrl = {np.round(self.data.ctrl, 2)} | "
                    f"cost = {np.round(cost, 2)}"
                )
            
            # Render if enabled
            if render and len(self.frames) < self.data.time * sim_framerate:
                self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=-1)
                pixels = self.renderer.render()
                self.frames.append(pixels)
            
            if solve_ocp:
                pbar.write("Exiting simulation after one step.")
                break

            # Check for early termination condition
            if cost is not None and cost < termination_cost and early_termination:
                pbar.write(f"Terminating early at t = {self.data.time:.2f}s with cost = {cost:.2f}")
                break

            # Update progress bar
            pbar.update(1)

        pbar.close()
        
        # Convert logs to arrays
        for key in ["qpos", "qvel", "u_applied", "cost", "qpos_traj", "qvel_traj", "u_traj", "time", "yref"]:
            self.logs[key] = np.array(self.logs[key])

    def step_sim(self):
        x = np.concatenate([self.data.qpos, self.data.qvel])
        cost = None
        qpos_traj = None
        qvel_traj = None
        u_traj = None

        # Only update MPC if needed
        if self.data.time >= self.next_mpc_time:
            try:
                # Get reference trajectory at the current time
                yref_now = get_yref_at_time(self.data.time, self.yref)
                self.logs["yref"].append(yref_now)  # Keep track of the reference
                # Run MPC to compute control input, cost, and trajectory
                self.last_u, cost, qpos_traj, qvel_traj, u_traj = self.controller(x, yref_now, self.config["mpc"]["full_traj"])

                # += to next mpc time step
                self.next_mpc_time += self.mpc_timestep

            except RuntimeError as e:
                raise RuntimeError(
                    f"MPC solver failed at t={self.data.time:.3f}s"
                ) from e

        # Apply last computed control
        self.data.ctrl[:] = self.last_u

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        return cost, qpos_traj, qvel_traj, u_traj