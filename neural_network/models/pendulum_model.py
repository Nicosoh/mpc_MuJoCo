import torch
import torch.nn as nn
import torch.nn.functional as F

class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))   

        # final output = 1/2 * ||z||^2
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)

        return x

# class PendulumModelTruncated(PendulumModel):
#     def forward(self, x):
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = F.tanh(self.fc3(x))
#         x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)
#         return x  # return raw fc3 output
    
class PendulumModelTruncated(PendulumModel):
    def __init__(self):
        super().__init__()

        # Hard-coded min/max as torch tensors
        self.X_MIN = torch.tensor(
            [-3.140068292617798, -17.82215690612793],
            dtype=torch.float32
        )
        self.X_MAX = torch.tensor(
            [ 3.1285789012908936, 17.884841918945312],
            dtype=torch.float32
        )

        # Torch constants
        self.TWO      = torch.tensor(2.0,  dtype=torch.float32)
        self.MINUS_ONE = torch.tensor(-1.0, dtype=torch.float32)
        self.THIRTY_TWO = torch.tensor(32.0, dtype=torch.float32)
        self.EPS      = torch.tensor(1e-8, dtype=torch.float32)
        self.HALF     = torch.tensor(0.5, dtype=torch.float32)

        # Output scaling constants (torch)
        self.Y_MIN = torch.tensor([0.008676528930664062], dtype=torch.float32)
        self.Y_MAX = torch.tensor([1170.5596923828125],  dtype=torch.float32)

    def forward(self, x):
        # Move constants to correct device/dtype
        X_MIN = self.X_MIN.to(x.device).to(x.dtype)
        X_MAX = self.X_MAX.to(x.device).to(x.dtype)

        TWO       = self.TWO.to(x.device).to(x.dtype)
        EPS       = self.EPS.to(x.device).to(x.dtype)
        HALF      = self.HALF.to(x.device).to(x.dtype)
        MINUS_ONE = self.MINUS_ONE.to(x.device).to(x.dtype)

        Y_MIN = self.Y_MIN.to(x.device).to(x.dtype)
        Y_MAX = self.Y_MAX.to(x.device).to(x.dtype)

        # -----------------------------------
        # 1) Scale input from physical → [-1,1]
        # -----------------------------------
        x_scaled = TWO * (x - X_MIN) / (X_MAX - X_MIN + EPS) + MINUS_ONE

        # -----------------------------------
        # 2) Neural network forward pass
        # -----------------------------------
        h = torch.tanh(self.fc1(x_scaled))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))

        # Your energy-like output directly in [0,32]
        y_32 = HALF * torch.sum(h**2, dim=1, keepdim=True)

        # -----------------------------------
        # 3) Scale output [0,32] → [y_min, y_max]
        # -----------------------------------
        # y_phys = (y_32 / 32) * (ymax - ymin) + ymin
        y_phys = (y_32 / torch.tensor(32.0, device=x.device, dtype=x.dtype)) * (Y_MAX - Y_MIN) + Y_MIN

        return y_phys