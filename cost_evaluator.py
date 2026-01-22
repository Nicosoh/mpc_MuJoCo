from IK import InverseKinematicsSolver
import numpy as np
from utils import load_collision_config

class CostEvaluator:
    def __init__(self, controller1, controller2, config, collision_config):
        self.controller1 = controller1
        self.controller2 = controller2
        self.config = config

        self.sample_points = 5000

        self.logs = {
            "samples": self.sample_points,
            "q": [],
            "yref": [],
            "stage_cost_1": [],
            "terminal_cost_1": [],
            "total_cost_1": [],            
            "stage_cost_2": [],
            "terminal_cost_2": [],
            "total_cost_2": [],
            "squared_error" : [],
        }

    # Steps:
    # For every sample point:
        # Change x0, yref and obstacle.
        # Record stage, terminal and total cost

    # Save as an array
    # Compute MSE
 
    def run(self):

        for i in range(self.sample_points):
            # Load collision config and also randomize
            if self.config["collision"]["collision_avoidance"]:                                                      # If enabled in config
                collision_config, self.config = load_collision_config(self.config)                                        # Load obstacles
            else:
                collision_config = None

            # Find valid test configurations
            IK = InverseKinematicsSolver(self.config, collision_config)
            x0_q = IK.load_x0()
            yref = IK.load_yref()

            # Feed same position and yref to both controllers
            _, stage_cost_1, terminal_cost_1, total_cost_1, _, _, _ = self.controller1(x0_q, yref, self.config["mpc"]["full_traj"])
            _, stage_cost_2, terminal_cost_2, total_cost_2, _, _, _ = self.controller2(x0_q, yref, self.config["mpc"]["full_traj"])

            # Calculate squared error
            squared_error = (total_cost_1 - total_cost_2) ** 2

            # Log results
            self.logs["q"].append(x0_q)
            self.logs["yref"].append(yref)

            self.logs["stage_cost_1"].append(stage_cost_1)
            self.logs["terminal_cost_1"].append(terminal_cost_1)
            self.logs["total_cost_1"].append(total_cost_1)

            self.logs["stage_cost_2"].append(stage_cost_2)
            self.logs["terminal_cost_2"].append(terminal_cost_2)
            self.logs["total_cost_2"].append(total_cost_2)

            self.logs["squared_error"].append(squared_error)

        self.logs["MSE"] = np.mean(self.logs["squared_error"])

        # Convert logs to arrays
        for key in self.logs.keys():
            self.logs[key] = np.array(self.logs[key])