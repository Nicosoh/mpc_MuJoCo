from pin_models.pin_base_class import PinocchioCasadi
# import pinocchio as pin
import numpy as np
# from pathlib import Path
from robot_descriptions.loaders.pinocchio import load_robot_description

def make_iiwa14(config):
    # # Full path to the URDF
    # urdf_filename = Path("urdf/iiwa14/iiwa14_spheres_dense_collision.urdf").resolve()
    
    # # The base path for resolving "package://" URDF mesh paths
    # package_dirs = [str(urdf_filename.parent.parent.parent)]  # points to your project root

    # # Build the model (with fixed base by default)
    # model = pin.buildModelFromUrdf(str(urdf_filename))
    # model.gravity.linear = np.array([0, 0, -9.81])

    # # Build geometry models
    # collision_model = pin.buildGeomFromUrdf(
    #     model,
    #     str(urdf_filename),
    #     pin.GeometryType.COLLISION,
    #     package_dirs=package_dirs
    # )

    # visual_model = pin.buildGeomFromUrdf(
    #     model,
    #     str(urdf_filename),
    #     pin.GeometryType.VISUAL,
    #     package_dirs=package_dirs
    # )

    # Load the full robot (model + geometry models) from the description
    robot = load_robot_description("iiwa14_description")

    # Gravity check
    assert np.allclose(robot.model.gravity.linear, np.array([0, 0, -9.81])), \
    f"Gravity is not set to [0, 0, -9.81], but is {robot.model.gravity.linear}"

    # Extract models
    model = robot.model
    collision_model = robot.collision_model
    visual_model = robot.visual_model

    return model, collision_model, visual_model
    
class iiwa14Dynamics(PinocchioCasadi):
    def __init__(self, timestep: float, config):
        model, collision_model, visual_model = make_iiwa14(config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep, config=config)