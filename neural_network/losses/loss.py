import torch
import torch.nn as nn

def weighted_mse_loss(y_pred, y_true):
    """
    Compute a weighted mean squared error loss to balance contributions from small and large target values.

    Args:
        y_pred (torch.Tensor): Predicted values, shape [batch_size, 1]
        y_true (torch.Tensor): True target values, shape [batch_size, 1]
        method (str): Weighting method. Options:
            - "inverse": weights = 1 / (y_true + eps)
        eps (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: scalar loss
    """
    eps=1e-12

    weights = 1.0 / (y_true + eps)
    # normalize weights to keep the same scale
    weights = weights / weights.sum() * y_true.shape[0]

    loss = weights * (y_pred - y_true) ** 2
    return loss.mean()
