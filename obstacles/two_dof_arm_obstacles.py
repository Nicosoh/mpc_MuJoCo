import numpy as np

# First index is time in seconds
# Second onwards is the from-to positions


### For the double_pendulum (time, x1, y1, z1, x2, y2, z2, radius)
obstacles = np.array([[
#   [time, x1,   y1,   z1,   x2,   y2,   z2,   radius],
    [0.0,  1.1,  0.0,  0.5,  1.1,  0.0,  1.0,  0.1],
    ]])

# print(obstacles.shape)