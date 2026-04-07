import os
import ast
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import random
import numpy as np

from tqdm import tqdm
from utils import get_num_config
from torch.utils.data import DataLoader
from neural_network.utils import plot_loss
from neural_network.losses import StationaryLoss

from neural_network.models import MODEL_REGISTRY
from neural_network.datasets import DATASET_REGISTRY

def train_model(config, run_dir, data_path=None, seed=42):
    # === Set random seed ===
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Config ===
    # Training
    learning_rate = get_num_config("TRAINING", "learning_rate", config)
    batch_size    = config.getint("TRAINING", "batch_size")
    num_epochs    = config.getint("TRAINING", "num_epochs")
    patience      = config.getint("TRAINING", "patience")
    min_lr        = config.getfloat("TRAINING", "min_lr")

    # Overwrite data path if provided
    if data_path is not None:
        config.set("DATA", "data_path", data_path)

    # Data
    dataset_class = config.get("DATA", "dataset_class")

    # Model
    model_name = config.get("MODEL", "model_name")
    load_checkpoint = config.getboolean("MODEL", "load_checkpoint")
    if load_checkpoint:
        checkpoint_path = config.get("MODEL", "checkpoint_path")

    # Validation
    eval_interval = config.getint("VAL", "val_interval")

    # Loss
    alpha = torch.tensor(config.getfloat("LOSS", "alpha"), dtype=torch.float32)

    # === Create dataset + dataloader ===
    # Load dataset dynamically
    DatasetClass = DATASET_REGISTRY[dataset_class]
    dataset = DatasetClass(config=config, run_dir=run_dir, mode="train") # create dataset object
    train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

    # === Dataset Info ===
    num_train = len(dataset.train_dataset)
    num_val   = len(dataset.val_dataset)
    total     = num_train + num_val
    train_ratio = num_train / total
    val_ratio   = num_val / total

    # === Update train_config.ini with dataset statistics ===
    config.set("DATA", "num_train_samples", str(num_train))
    config.set("DATA", "num_val_samples", str(num_val))
    config.set("DATA", "train_ratio", f"{train_ratio:.4f}")
    config.set("DATA", "val_ratio", f"{val_ratio:.4f}")

    # Save updated config
    config_save_path = os.path.join(run_dir, "train_config.ini")
    with open(config_save_path, "w") as f:
        config.write(f)

    # === Model / optimizer / loss ===
    # Load model dynamically
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(config).to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from checkpoint: {checkpoint_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = StationaryLoss(alpha=alpha)

    # === Logging ===
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    stationary_ratios = []

    # === Variables ===
    vals_since_improvement = 0
    best_model_path = None

    # === Global epoch progress bar ===
    # epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    pbar = tqdm(total=num_epochs, desc="Training")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0

        for xb, xs, yb, ys in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds_main = model(xb)
            preds_stationary = model(xs.to(device))
            loss, loss1, loss2 = criterion(preds_main, yb, preds_stationary, ys.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_loss1 += loss1.item() * xb.size(0)
            total_loss2 += criterion.alpha * loss2.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_loss1 = total_loss1 / len(train_loader.dataset)
        avg_loss2 = total_loss2 / len(train_loader.dataset)
        stationary_ratio = avg_loss2 / (avg_loss1 + 1e-8)

        train_losses.append((epoch+1, avg_train_loss, optimizer.param_groups[0]['lr']))
        stationary_ratios.append(stationary_ratio)

        # Update the tqdm bar postfix with avg loss
        pbar.write(f"\n[Train @ epoch {epoch+1}]  Train Loss = {avg_train_loss}\n, Ratio = {stationary_ratio}\n")

        # === Validation every eval_interval ===
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, xs, yb, ys in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds_main = model(xb)
                    preds_stationary = model(xs.to(device))
                    loss, _, _ = criterion(preds_main, yb, preds_stationary, ys.to(device))
                    val_loss += loss.item() * xb.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append((epoch+1, avg_val_loss, optimizer.param_groups[0]['lr']))

            pbar.write(f"\n[Eval @ epoch {epoch+1}]  Val Loss = {avg_val_loss}\n")

            # === Save only the best model ===
            epoch_str = f"{epoch + 1}"
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch_str}.pt")

            if avg_val_loss < best_val_loss:
                # Remove previous best checkpoint (keep directory clean)
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                # Save new best model
                torch.save(model.state_dict(), ckpt_path)
                best_val_loss = avg_val_loss
                best_model_path = ckpt_path
                vals_since_improvement = 0

                print(f"New best model saved: {ckpt_path}")
            else:
                vals_since_improvement += 1

            if vals_since_improvement >= patience:
                lr_reduced = False

                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr / 10, min_lr)

                    if new_lr < old_lr:
                        param_group['lr'] = new_lr
                        lr_reduced = True

                if lr_reduced:
                    print(
                        f"Validation loss did not improve for {patience} validations. "
                        "Reducing learning rate and restoring best model weights."
                    )

                    if best_model_path is not None:
                        model.load_state_dict(torch.load(best_model_path))

                    print(
                        f"LR reduced from {old_lr:.2e} to {new_lr:.2e}"
                    )
                else:
                    print(
                        f"Validation loss did not improve for {patience} validations, "
                        f"but LR is already at minimum ({min_lr:.2e}). No reload performed."
                    )

                vals_since_improvement = 0
        pbar.update(1)

    pbar.close()

    # === Plot loss curves ===
    if data_path is not None:
        show_plot = False
    else:
        show_plot = True

    stationary_ratios_mean = float(np.mean(stationary_ratios))

    plot_loss(train_losses, val_losses, stationary_ratios, run_dir=run_dir, show_plot=show_plot)

    return train_losses[-1][1], stationary_ratios_mean # Return last epoch's train loss and mean stationary ratio