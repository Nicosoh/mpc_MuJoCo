import mujoco
import numpy as np

def load_model(model_path: str):
    """Load a MuJoCo model and create associated data object."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data

def apply_model_config(config, model):
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

def init_scene_options():
    """Initialize visualization options for rendering."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    # scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
    return scene_option

def run_simulation(
    x0,
    model,
    data,
    sim_duration=30.0,
    mpc_timestep=0.002,
    sim_framerate=30,
    resolution=(480, 640),
    render=True,
    controller=None,
    verbose=True,
    ):

    last_u = np.zeros(model.nu)
    next_mpc_time = 0.0

    # Reset conditions
    mujoco.mj_resetData(model, data)
    
    nq = model.nq
    nv = model.nv

    # Check x0 length vs nq + nv
    if len(x0) != nq + nv:
        raise ValueError(f"x0 should have length {nq + nv} (qpos + qvel), got {len(x0)}")

    # Set initial position and velocity from x0
    data.qpos[:] = x0[:nq]
    data.qvel[:] = x0[nq:]

    mujoco.mj_forward(model, data) # Step forward to compute derived quantities
    print(f"Model has {nq} DoFs (qpos), {nv} velocities (qvel), and {model.nu} actuators.")
    print("Initial qpos:", data.qpos)
    print("Initial qvel:", data.qvel)

    # Prepare recording
    frames = []
    logs = {
        "time": [],
        "qpos": [],
        "qvel": [],
        "u_applied": [],
    }
    height, width = resolution

    # Use renderer only if True
    if render:
        scene_option = init_scene_options()
        renderer = mujoco.Renderer(model, height, width)

    # Main simulation loop
    while data.time < sim_duration:
        # Gather state
        state = {
            "qpos": np.copy(data.qpos),
            "qvel": np.copy(data.qvel),
            "time": data.time,
        }

        # Call MPC only at specified intervals
        if controller is not None and data.time >= next_mpc_time:
            try:
                last_u = controller(state)
            except RuntimeError as e:
                print(f"[ERROR] MPC solver failed at t={data.time:.3f}s: {e}")
                break
            next_mpc_time += mpc_timestep

        # Apply last computed control
        data.ctrl[:] = last_u

        # Step simulation
        mujoco.mj_step(model, data)

        # Log data
        logs["time"].append(data.time)
        logs["qpos"].append(np.copy(data.qpos))
        logs["qvel"].append(np.copy(data.qvel))
        logs["u_applied"].append(np.copy(data.ctrl))

        if verbose:
            print(
                f"t = {data.time:.3f}s | "
                f"qpos = {np.round(data.qpos, 3)} | "
                f"qvel = {np.round(data.qvel, 3)} | "
                f"ctrl = {np.round(data.ctrl, 3)}"
            )

        # Render if enabled
        if render and len(frames) < data.time * sim_framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    # Convert logs to arrays
    for key in ["qpos", "qvel", "u_applied"]:
        logs[key] = np.array(logs[key])
    logs["time"] = np.array(logs["time"])

    return logs, frames