import torch
import json
import os
from tqdm import tqdm
from configparser import ConfigParser
from neural_network.utils import import_standardize_params
from torch.utils.data import DataLoader

from neural_network.models import MODEL_REGISTRY
from neural_network.datasets import DATASET_REGISTRY

def evaluate_model(test_config_path, run_dir):
    # === Load val config ===
    test_config = ConfigParser()
    test_config.read(test_config_path)

    # Test config
    checkpoint_path = test_config.get("TEST", "checkpoint_path")
    test_data_path = test_config.get("TEST", "test_data_path")

    # === Load train config ===
    # Train config path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    train_config_path = os.path.join(checkpoint_dir, "train_config.ini")

    # Train config
    train_config = ConfigParser()
    train_config.read(train_config_path)
    
    dataset_class = train_config.get("DATA", "dataset_class")
    standardize = train_config.getboolean("DATA", "standardize")
    model_name = train_config.get("MODEL", "model_name")

    if standardize:
        X_mean, X_std, y_mean, y_std = import_standardize_params(checkpoint_dir)
    
    # === Save configs ===
    train_config_save_path = os.path.join(run_dir, "train_config.ini")
    with open(train_config_save_path, "w") as f:
        train_config.write(f)

    test_config_save_path = os.path.join(run_dir, "test_config.ini")
    with open(test_config_save_path, "w") as f:
        test_config.write(f)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DatasetClass = DATASET_REGISTRY[dataset_class]
    dataset = DatasetClass(data_path=test_data_path, 
                           standardize=False,           # During testing, this should always be false and if it was used 
                           run_dir=run_dir,             # during training then the values should be extracted from the train data.
                           mode = "test")              # Which is saved in the json

    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)

    # === Load saved model ===
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass().to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # === Prepare to collect results ===
    preds = []
    targets = []

    # === Evaluate ===
    for xb, yb in tqdm(test_loader, desc="Evaluating"):
        xb = xb.to(device)
        yb = yb.to(device)

        # Normalize X for inference
        xb = (xb - X_mean) / X_std

        with torch.no_grad():
            pred_norm = model(xb)

        # Unnormalize y
        pred = pred_norm * y_std + y_mean

        preds.append(pred.cpu())
        targets.append(yb.cpu())   # raw target values

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # === Compute errors ===
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()

    print("\n--- Evaluation Results ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    # === Combine predictions and targets ===
    pred_target_pairs = [[float(p.item()), float(t.item())] for p, t in zip(preds, targets)]

    # === Save results including paired predictions and targets ===
    results_path = os.path.join(run_dir, "evaluation_results.json")
    results_dict = {
        "MSE": mse,
        "MAE": mae,
        "pred_target_pairs": pred_target_pairs  # paired values
    }
    
    # === Save results ===
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Saved evaluation results → {results_path}")

    return preds, targets