import torch
import torch.nn as nn

class StationaryLoss(nn.Module):
    def __init__(self, alpha):
        """
        alpha : weight for the stationary-point penalty
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred_main, y_batch, pred_stationary, y_stationary):
        """
        pred_main        : V(x_batch) predictions       (n x 1)
        y_batch          : MPC/Bellman targets          (n x 1)
        pred_stationary  : V(X_s) predictions           (m x 1)
        y_stationary     : V(X_s) targets             (m x 1)
        """
        # 1. Main Bellman / MPC loss
        loss1 = self.mse(pred_main, y_batch)

        # 2. Stationary-point loss: enforce V(x_s) ≈ 0
        loss2 = self.mse(pred_stationary, y_stationary)

        # 3. Combined loss
        total_loss = loss1 + self.alpha * loss2

        return total_loss, loss1, loss2