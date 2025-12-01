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
        x = F.tanh(self.fc3(x))    # z ∈ R^64

        # final output = 1/2 * ||z||^2
        x = 0.5 * torch.sum(x**2, dim=1, keepdim=True)

        return x