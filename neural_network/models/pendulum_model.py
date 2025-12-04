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

class PendulumModelTruncated(PendulumModel):
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x  # return raw fc3 output