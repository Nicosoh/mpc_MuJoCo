import numpy as np

# First index is time in seconds
# Second onwards is the reference state

### For the cartpole (time, x, theta, x_dot, theta_dot, u)
# yref = np.array([
#     [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
#     [1.0,  0.5,  0.0,  0.0,  0.0, 0.0],
#     [2.0,  0.0,  3.142,  0.0,  0.0, 0.0],
# ])

### For the pendulum (time, theta, theta_dot, u)
# yref = np.array([
#     [0.0,  0.0,  0.0,  0.0],
#     [1.0,  0.5,  0.0,  0.0],
#     [2.0,  1.0,  0.0,  0.0],
# ])

### For the double_pendulum (time, theta1, theta2, theta1_dot, theta2_dot, u)
yref = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
    [1.5,  3.142,  0,  0.0,  0.0, 0.0],
    [3.5,  3.142,  3.142,  0.0,  0.0, 0.0],
])