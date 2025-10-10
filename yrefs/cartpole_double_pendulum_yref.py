import numpy as np

# First index is time in seconds
# Second onwards is the reference state


### For the cartpole_double_pendulum (time, x, theta1, theta2, x_dot, theta1_dot, theta2_dot, u)
yref = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])