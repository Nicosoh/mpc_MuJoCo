import numpy as np

# First index is time in seconds
# Second onwards is the reference state


### For the iiwa14 (time, 
# theta1, theta2, theta3, theta4, theta5, theta6, theta7, 
# theta1_dot, theta2_dot, theta3_dot, theta4_dot, theta5_dot, theta6_dot, theta7_dot, 
# u1, u2, u3, u4, u5, u6, u7)

yref = np.array([
    [0.0,
     0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [2.0,  
     0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])