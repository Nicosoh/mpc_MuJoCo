import matplotlib.pyplot as plt
import os
import json
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

def import_scaling_params(checkpoint_dir):
    """
    Import scaling parameters from a checkpoint folder.
    Returns a dictionary with keys depending on type ("standardize" or "normalize")
    """
    file_path = os.path.join(checkpoint_dir, "normalization_stats.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No normalization_stats.json found in {checkpoint_dir}")

    with open(file_path, "r") as f:
        stats = json.load(f)

    tensors = {}
    if all(k in stats for k in ["X_mean", "X_std", "y_mean", "y_std"]):
        # Standardization
        tensors.update({k: torch.tensor(stats[k]) for k in ["X_mean", "X_std", "y_mean", "y_std"]})
        tensors["type"] = "standardize"
    elif all(k in stats for k in ["X_min", "X_max", "y_min", "y_max"]):
        # Normalization
        tensors.update({k: torch.tensor(stats[k]) for k in ["X_min", "X_max", "y_min", "y_max"]})
        tensors["type"] = "normalize"
    else:
        raise ValueError("Normalization file does not contain recognized keys.")

    return tensors

def run_scaling(X, y, scaling_type ,scaling_params, inverse=False):
    """
    Apply loaded scaling parameters to tensors X and y.

    Args:
        X (torch.Tensor): Input features
        y (torch.Tensor): Targets
        scaling_params (dict): Output from `load_scaling_params`

    Returns:
        tuple: scaled X, scaled y
    """
    eps = 1e-8
    if scaling_type == "standardize":
        if not inverse:
            X_scaled = (X - scaling_params["X_mean"]) / (scaling_params["X_std"] + eps)
            y_scaled = (y - scaling_params["y_mean"]) / (scaling_params["y_std"] + eps)
        else:
            X_scaled = X * (scaling_params["X_std"] + eps) + scaling_params["X_mean"]
            y_scaled = y * (scaling_params["y_std"] + eps) + scaling_params["y_mean"]

    elif scaling_type == "normalize":
        if not inverse:
            X_scaled = 2 * (X - scaling_params["X_min"]) / (scaling_params["X_max"] - scaling_params["X_min"] + eps) - 1
            # y_scaled = 2 * (y - scaling_params["y_min"]) / (scaling_params["y_max"] - scaling_params["y_min"] + eps) - 1
            y_scaled = y
        else:
            X_scaled = ((X + 1) / 2) * (scaling_params["X_max"] - scaling_params["X_min"] + eps) + scaling_params["X_min"]
            # y_scaled = ((y + 1) / 2) * (scaling_params["y_max"] - scaling_params["y_min"] + eps) + scaling_params["y_min"]
            y_scaled = y

    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")

    return X_scaled, y_scaled

def save_scaling_values(scaling_params, run_dir):
    """Save scaling parameters to JSON"""
    filename = {
        "standardize": "standardization_stats.json",
        "normalize": "normalization_stats.json"
    }.get(scaling_params["type"], "scaling_stats.json")

    serializable = {k: v.squeeze(0).tolist() if torch.is_tensor(v) else v for k, v in scaling_params.items()}
    with open(os.path.join(run_dir, filename), "w") as f:
        json.dump(serializable, f, indent=4)