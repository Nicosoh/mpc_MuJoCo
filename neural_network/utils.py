import matplotlib.pyplot as plt
import os

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
