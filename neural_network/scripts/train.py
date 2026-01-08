import os
import ast
import torch
import random
import numpy as np

from tqdm import tqdm
from utils import get_num_config
from torch.utils.data import DataLoader
from neural_network.utils import plot_loss
from neural_network.losses import StationaryLoss

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
    learning_rate = get_num_config("TRAINING", "learning_rate", config)
    batch_size    = config.getint("TRAINING", "batch_size")
    num_epochs    = config.getint("TRAINING", "num_epochs")
    patience      = config.getint("TRAINING", "patience")

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

    # === Variables ===
    vals_since_improvement = 0
    best_model_path = None

    # === Global epoch progress bar ===
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        for xb, xs, yb, ys in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds_main = model(xb)
            preds_stationary = model(xs.to(device))
            loss = criterion(preds_main, yb, preds_stationary, ys.to(device))
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
                for xb, xs, yb, ys in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds_main = model(xb)
                    preds_stationary = model(xs.to(device))
                    loss = criterion(preds_main, yb, preds_stationary, ys.to(device))
                    val_loss += loss.item() * xb.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append((epoch+1, avg_val_loss, optimizer.param_groups[0]['lr']))

            tqdm.write(f"\n[Eval @ epoch {epoch+1}]  Val Loss = {avg_val_loss}\n")

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

                tqdm.write(f"New best model saved: {ckpt_path}")
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