from pin_models.pin_base_class import PinocchioCasadiRobotWrapper
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

def make_iiwa14(config):
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
    
class iiwa14Dynamics(PinocchioCasadiRobotWrapper):
    def __init__(self, timestep: float, config):
        model, collision_model, visual_model = make_iiwa14(config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep, config=config)