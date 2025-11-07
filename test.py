import mujoco
import mujoco.viewer  # this is the built-in viewer (as of MuJoCo 2.3+)

# Load the model using robot_descriptions
from robot_descriptions import iiwa14_mj_description
model = mujoco.MjModel.from_xml_path(iiwa14_mj_description.MJCF_PATH)

# Alternatively, load via utility (commented out if using above)
# from robot_descriptions.loaders.mujoco import load_robot_description
# model = load_robot_description("panda_mj_description")

# Create MjData (state container)
data = mujoco.MjData(model)

# Open the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer is running. Close the window to exit.")
    
    # Keep rendering until the viewer is closed
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

# import os
# from pathlib import Path
# from sys import argv

# import pinocchio

# # # This path refers to Pinocchio source code but you can define your own directory here.
# # model_dir = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))

# # # You should change here to set up your own URDF file or just pass it as an argument of
# # # this example.
# # urdf_filename = (
# #     model_dir / "ur_description/urdf/ur5_robot.urdf" if len(argv) < 2 else argv[1]
# # )
# urdf_filename = "/Users/nicodemussoh/Documents/mpc_MuJoCo/urdf/iiwa14_spheres_dense_collision.urdf"
# # Load the urdf model
# model = pinocchio.buildModelFromUrdf(urdf_filename)
# print("model name: " + model.name)

# # Create data required by the algorithms
# data = model.createData()

# # Sample a random configuration
# q = pinocchio.randomConfiguration(model)
# print(f"q: {q.T}")

# # Perform the forward kinematics over the kinematic tree
# pinocchio.forwardKinematics(model, data, q)

# # Print out the placement of each joint of the kinematic tree
# for name, oMi in zip(model.names, data.oMi):
#     print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))