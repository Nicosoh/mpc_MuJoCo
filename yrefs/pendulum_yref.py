import numpy as np

# First index is time in seconds
# Second onwards is the reference state


### For the pendulum (time, theta, theta_dot, u)
yref = np.array([
    [0.0,  3.142,  0.0,  0.0],
    [1.0,  0.5,  0.0,  0.0],
    [2.0,  1.0,  0.0,  0.0],
])
