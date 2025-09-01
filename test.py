import mujoco
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt

# Import XML model from path
model = mujoco.MjModel.from_xml_path('models_xml/inverted_pendulum.xml')
data = mujoco.MjData(model)

# Enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM

# Simulation settings
duration = 15.0  # seconds
framerate = 30  # Hz

# Resets data
frames = []
mujoco.mj_resetData(model, data)

# Set initial conditions
data.qpos[1] = np.deg2rad(30)   # hinge joint

# Steps simulation forward by one step to verify starting conditions
mujoco.mj_forward(model, data)
print('Total number of DoFs in the model:', model.nv)
print('Initial positions', data.qpos)
print('Initial velocities', data.qvel)

# Empty arrays for plotting (Probably needs control input as well)
time = []
cart_pos, cart_vel = [], []
pend_angle, pend_angvel = [], []

# Simulation loop
with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)

        # record time + states
        time.append(data.time)
        cart_pos.append(data.qpos[0])      # cart slide
        pend_angle.append(data.qpos[1])    # pendulum angle
        cart_vel.append(data.qvel[0])      # cart velocity
        pend_angvel.append(data.qvel[1])   # angular velocity

        # Append a frame only if the simulation time passes the next frame timestamp
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

# media.write_video("video.mp4", frames, fps=framerate)

# Plot results
dpi = 120
width, height = 800, 1000
figsize = (width / dpi, height / dpi)

fig, ax = plt.subplots(4, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(time, cart_pos)
ax[0].set_title('Cart Position')
ax[0].set_ylabel('meters')

ax[1].plot(time, cart_vel)
ax[1].set_title('Cart Velocity')
ax[1].set_ylabel('m/s')

ax[2].plot(time, pend_angle)
ax[2].set_title('Pendulum Angle')
ax[2].set_ylabel('radians')

ax[3].plot(time, pend_angvel)
ax[3].set_title('Pendulum Angular Velocity')
ax[3].set_ylabel('rad/s')
ax[3].set_xlabel('time (s)')

plt.tight_layout()
plt.show()