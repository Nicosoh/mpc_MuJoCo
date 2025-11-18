import numpy as np

# First index is time in seconds
# Second onwards is the reference state


### For the double_pendulum (time, theta1, theta2, theta1_dot, theta2_dot, u)
yref = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    # [3.0,  1.0,  1.0,  0.0,  0.0,  0.0,  0.0],
    [3.0,  2.0,  -1.0,  0.0,  0.0,  0.0,  0.0],
])