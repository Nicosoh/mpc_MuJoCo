import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_network.utils import run_scaling

class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))   

        # final output = 1/2 * ||z||^2
        x = self.fc4(x)
        # x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)

        return x
    
class PendulumModel(nn.Module):
    def __init__(self, train_config):
        super().__init__()

        # MLP
        self.fc0 = nn.Linear(2, 2)
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        # If scaling is required and is normalize
        self.apply_scaling = train_config.getboolean("DATA", "apply_scaling")
        if self.apply_scaling == True:
            self.scaling_type = train_config.get("DATA", "scaling_type")
            if self.scaling_type == "normalize":
                # Scaling ranges from config
                self.scaling_range_X = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_X")), dtype=torch.float32)
                self.scaling_range_y = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_y")), dtype=torch.float32)

                # Min/max for inputs and outputs
                X_min = ast.literal_eval(train_config.get("DATA", "X_min"))
                X_max = ast.literal_eval(train_config.get("DATA", "X_max"))
                y_min = ast.literal_eval(train_config.get("DATA", "y_min"))
                y_max = ast.literal_eval(train_config.get("DATA", "y_max"))

                # Store as dict
                self.scaling_params = {
                    "X_min": torch.tensor(X_min, dtype=torch.float32),
                    "X_max": torch.tensor(X_max, dtype=torch.float32),
                    "y_min": torch.tensor(y_min, dtype=torch.float32),
                    "y_max": torch.tensor(y_max, dtype=torch.float32)
                }
            else:
                raise NotImplementedError("Only 'normalize' scaling_type is implemented.")

    def forward(self, x):
        """
        If scaling_params is None:
            Training mode → model expects pre-scaled x
        
        If scaling_params is not None:
            ACADOS/Inference mode → model scales x internally, runs the MLP,
            then unscales the output.
        """

        # ---------------------------------------------------------
        # 1) Apply input scaling internally (ACADOS mode)
        # ---------------------------------------------------------
        if self.apply_scaling == True:
            # Only X is scaled; y is None
            x, _ = run_scaling(
                x, None,
                scaling_type=self.scaling_type,
                scaling_params=self.scaling_params,
                scaling_range_X=self.scaling_range_X,
                scaling_range_y=self.scaling_range_y,
                inverse=False
            )

        # ---------------------------------------------------------
        # 2) Neural network forward pass
        # ---------------------------------------------------------
        x = self.fc0(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        y = self.fc4(x)
        y = torch.abs(y)
        # y = 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)  # in [0,32] range

        # # ---------------------------------------------------------
        # # 3) Unscale output (ACADOS mode only)xs
        # # ---------------------------------------------------------
        # if self.apply_scaling == True:
        #     _, y = run_scaling(
        #         None, y,
        #         scaling_type=self.scaling_type,
        #         scaling_params=self.scaling_params,
        #         scaling_range_X=self.scaling_range_X,
        #         scaling_range_y=self.scaling_range_y,
        #         inverse=True
        #     )

        return y