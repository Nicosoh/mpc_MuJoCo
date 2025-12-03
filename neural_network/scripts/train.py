import torch
import os
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from neural_network.utils import plot_loss
from neural_network.losses import weighted_mse_loss

from neural_network.models import MODEL_REGISTRY
from neural_network.datasets import DATASET_REGISTRY

def train_model(config, run_dir, seed=42):
    # === Set random seed ===
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Config ===
    # Training
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    batch_size    = config.getint("TRAINING", "batch_size")
    num_epochs    = config.getint("TRAINING", "num_epochs")
    patience      = config.getint("TRAINING", "patience")

    # Data
    dataset_class = config.get("DATA", "dataset_class")
    data_path = config.get("DATA", "data_path")
    apply_scaling = config.getboolean("DATA", "apply_scaling")
    scaling_type = config.get("DATA", "scaling_type")

    # Model
    model_name = config.get("MODEL", "model_name")

    # Validation
    eval_interval = config.getint("VAL", "val_interval")

    # === Create dataset + dataloader ===
    # Load dataset dynamically
    DatasetClass = DATASET_REGISTRY[dataset_class]
    dataset = DatasetClass(data_path=data_path, apply_scaling=apply_scaling, scaling_type=scaling_type, run_dir=run_dir, mode="train") # create dataset object
    train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=True)

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
    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # === Logging ===
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    # Track top-5 best checkpoints
    top_k = 5
    best_checkpoints = []   # list of tuples: (val_loss, path)

    # === Variables ===
    vals_since_improvement = 0
    best_model_path = None

    # === Global epoch progress bar ===
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append((epoch+1, avg_train_loss, optimizer.param_groups[0]['lr']))
        # Update the tqdm bar postfix with avg loss
        epoch_bar.set_postfix(Epoch_loss=avg_train_loss)

        # === Validation every eval_interval ===
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append((epoch+1, avg_val_loss, optimizer.param_groups[0]['lr']))

            tqdm.write(f"\n[Eval @ epoch {epoch+1}]  Val Loss = {avg_val_loss}\n")

            # === Top-5 checkpoint logic ===
            epoch_str = f"{epoch + 1}"
            ckpt_path = os.path.join(run_dir, f"model_epoch_{epoch_str}.pt")

            should_save = (
                len(best_checkpoints) < top_k or
                avg_val_loss < best_checkpoints[-1][0]
            )

            if should_save:
                torch.save(model.state_dict(), ckpt_path)

                best_checkpoints.append((avg_val_loss, ckpt_path))
                best_checkpoints.sort(key=lambda x: x[0])  # smaller loss = better

                # Remove extra checkpoints
                if len(best_checkpoints) > top_k:
                    worst_loss, worst_path = best_checkpoints.pop()
                    if os.path.exists(worst_path):
                        os.remove(worst_path)
                        tqdm.write(f"Removed old checkpoint: {worst_path}")

            # Update "best_val_loss" separately for LR scheduling logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = ckpt_path
                vals_since_improvement = 0
            else:
                vals_since_improvement += 1
            
            # If no improvement for a certain number of epochs, restore best weights and reduce learning rate
            if vals_since_improvement >= patience:
                tqdm.write(
                    f"Validation loss did not improve for {patience} validations. Reducing learning rate and restoring best model weights.")
                if best_model_path is not None:
                    model.load_state_dict(torch.load(best_model_path))
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                vals_since_improvement = 0

    plot_loss(train_losses, val_losses, run_dir=run_dir)