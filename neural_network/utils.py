import matplotlib.pyplot as plt
import os
import torch

def plot_loss(train_losses, val_losses, run_dir):
    """
    Plots training and validation loss curves, distinguishing different learning rates.

    Args:
        train_losses (list of float): Loss per epoch for training
        val_losses (list of (epoch, loss, lr)): Validation loss entries
        run_dir (str): Directory to save the plot
    """
    save_path = os.path.join(run_dir, "loss_plot.jpg")
    plt.figure(figsize=(10, 6))

    # --- Plot training loss by learning rate ---
    if len(train_losses) > 0:
        unique_lrs = sorted(set(lr for _, _, lr in train_losses))
        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in train_losses if l == lr]
            lr_values = [v for (_, v, l) in train_losses if l == lr]
            plt.plot(lr_epochs, lr_values, linewidth=1, label=f'Train Loss (LR={lr:.1e})')
            plt.text(lr_epochs[-1], lr_values[-1], f'Train LR={lr:.1e}', fontsize=6, 
                     verticalalignment='bottom', horizontalalignment='left', rotation = 90)

    # --- Plot validation loss by learning rate ---
    if len(val_losses) > 0:
        # Get unique learning rates
        unique_lrs = sorted(set(lr for _, _, lr in val_losses))

        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in val_losses if l == lr]
            lr_values = [v for (_, v, l) in val_losses if l == lr]
            plt.plot(lr_epochs, lr_values, linewidth=1, label=f'Val Loss (LR={lr:.1e})')
            plt.text(lr_epochs[-1], lr_values[-1], f'Val LR={lr:.1e}', fontsize=6, 
                     verticalalignment='bottom', horizontalalignment='left', rotation = 90)

    # --- Labels and title ---
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title("Training & Validation Loss")
    plt.grid(True)

    # --- Save if requested ---
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    
    plt.show()

def run_scaling(X=None, y=None, scaling_type=None, scaling_params=None,
                scaling_range_X=None, scaling_range_y=None, inverse=False):
    """
    Scale X and/or y using MinMaxnormalization.
    Either X or y can be None if only one needs to be scaled/unscaled.

    Args:
        X (torch.Tensor, optional): Input features, shape (n_samples, n_features)
        y (torch.Tensor, optional): Targets, shape (n_samples, 1)
        scaling_type (str): "standardize" or "normalize"
        scaling_params (dict, optional): Precomputed statistics (mean/std or min/max)
        scaling_range_X (torch.Tensor or list, optional): Target range per feature for normalization
        scaling_range_y (torch.Tensor or list, optional): Target range for y
        inverse (bool): If True, performs inverse scaling

    Returns:
        X_scaled, y_scaled (torch.Tensor or None)
    """

    eps = 1e-8

    # Convert to tensors if provided
    if X is not None and not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if y is not None and not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    # Add batch dimension if 1D
    if X is not None and X.ndim == 1:
        X = X.unsqueeze(0)
    if y is not None:
        if y.ndim == 0:
            y = y.unsqueeze(0).unsqueeze(1)
        elif y.ndim == 1:
            y = y.unsqueeze(1)

    # Only normalize is implemented here
    if scaling_type == "normalize":
        if scaling_params is None:
            raise ValueError("scaling_params must be provided for normalization.")
        if (X is not None and scaling_range_X is None) or (y is not None and scaling_range_y is None):
            raise ValueError("scaling_range_X/y must be provided for normalization.")

        # Extract min/max
        X_min, X_max = scaling_params.get("X_min"), scaling_params.get("X_max")
        y_min, y_max = scaling_params.get("y_min"), scaling_params.get("y_max")

        # Forward scaling
        if not inverse:
            if X is not None:
                scaling_range_X = torch.tensor(scaling_range_X, dtype=X.dtype, device=X.device)
                X_scaled = (X - X_min) / (X_max - X_min + eps)
                X_scaled = X_scaled * (scaling_range_X[:, 1] - scaling_range_X[:, 0]) + scaling_range_X[:, 0]
            else:
                X_scaled = None

            if y is not None:
                scaling_range_y = torch.tensor(scaling_range_y, dtype=y.dtype, device=y.device)
                y_scaled = (y - y_min) / (y_max - y_min + eps)
                y_scaled = y_scaled * (scaling_range_y[1] - scaling_range_y[0]) + scaling_range_y[0]
            else:
                y_scaled = None

        # Inverse scaling
        else:
            if X is not None:
                scaling_range_X = torch.tensor(scaling_range_X, dtype=X.dtype, device=X.device)
                X_scaled = (X - scaling_range_X[:, 0]) / (scaling_range_X[:, 1] - scaling_range_X[:, 0] + eps)
                X_scaled = X_scaled * (X_max - X_min) + X_min
            else:
                X_scaled = None

            if y is not None:
                scaling_range_y = torch.tensor(scaling_range_y, dtype=y.dtype, device=y.device)
                y_scaled = (y - scaling_range_y[0]) / (scaling_range_y[1] - scaling_range_y[0] + eps)
                y_scaled = y_scaled * (y_max - y_min) + y_min
            else:
                y_scaled = None

    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}")

    return X_scaled, y_scaled