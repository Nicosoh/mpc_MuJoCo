import mujoco
import numpy as np

def load_model(model_path: str):
    """Load a MuJoCo model and create associated data object."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def init_scene_options():
    """Initialize visualization options for rendering."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
    return scene_option

def run_simulation(
    model,
    data,
    duration=30.0,
    framerate=30,
    resolution=(480, 640),
    render=True,
    controller=None,
    verbose=True,
):
    """Run the simulation, optionally with rendering and control."""

    # Reset and set initial conditions
    mujoco.mj_resetData(model, data)
    data.qpos[1] = np.deg2rad(30)  # initial pendulum angle

    # Steps simulation forward by one step to verify starting conditions
    mujoco.mj_forward(model, data)
    print('Total number of DoFs in the model:', model.nv)
    print("Time step (dt):", model.opt.timestep)
    print('Initial positions', data.qpos)
    print('Initial velocities', data.qvel)

    # Prepare recording
    frames = []
    time, cart_pos, cart_vel, pend_angle, pend_angvel, u_applied = [], [], [], [], [], []
    height, width = resolution
    scene_option = init_scene_options() if render else None

    # Use renderer only if requested
    renderer = mujoco.Renderer(model, height, width) if render else None

    while data.time < duration:
        # Gather state
        state = {
            "qpos": np.copy(data.qpos),
            "qvel": np.copy(data.qvel),
            "time": data.time,
        }

        # compute control with exception handling
        try:
            u = controller(state) if controller is not None else 0.0
            data.ctrl[0] = u
            u_applied.append(np.copy(u))
        except RuntimeError as e:
            print(f"[ERROR] MPC solver failed at t={data.time:.3f}s: {e}")
            print("Stopping simulation and returning recorded data.")
            break  # exit the loop immediately


        # Step simulation
        mujoco.mj_step(model, data)

        # Save states
        time.append(data.time)
        cart_pos.append(data.qpos[0])
        pend_angle.append(data.qpos[1])
        cart_vel.append(data.qvel[0])
        pend_angvel.append(data.qvel[1])

        if verbose:
            print("Current time:", data.time)
            print("Current control input:", data.ctrl[0])
            print("Current state:", data.qpos, data.qvel)

        # Render if enabled
        if render and len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    results = {
        "time": time,
        "cart_pos": cart_pos,
        "cart_vel": cart_vel,
        "pend_angle": pend_angle,
        "pend_angvel": pend_angvel,
        "u_applied": u_applied,
    }

    return results, frames