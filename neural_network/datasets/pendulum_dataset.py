import torch
import json
import os
import ast

import numpy as np

from torch.utils.data import Dataset, random_split
from data_collection import load_npz
from neural_network.utils import save_scaling_values, run_scaling

class PendulumDataset(Dataset):
    """
    Dataset for Pendulum:
    Inputs X = [qpos, qvel]
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """
    def __init__(self, config, run_dir, mode, scaling_params=None, test_config=None):
        """
        Args:
            data_path (str): Path to the .npz file
        """
        data_path = config.get("DATA", "data_path")
        self.apply_scaling = config.getboolean("DATA", "apply_scaling")
        self.scaling_type = config.get("DATA", "scaling_type")
        self.run_dir = run_dir
        self.mode = mode
        self.scaling_params = scaling_params

        # Load scaling_range is using normalize
        if self.scaling_type == "normalize":
            self.scaling_range_X = ast.literal_eval(config.get("DATA", "scaling_range_X"))
            self.scaling_range_y = ast.literal_eval(config.get("DATA", "scaling_range_y"))
        else:
            self.scaling_range_X = None
            self.scaling_range_y = None
        
        # Replace data_path if in test mode with test dataset.
        if self.mode == "test" and test_config is not None:
            data_path = test_config.get("TEST", "test_data_path")

        # Load data
        data = load_npz(data_path)
        self.preprocess_data(data) # Process data to be in pytorch format

        if self.mode == "train":
            self.train_val_data() # Split data into train and val

        if self.apply_scaling:
            if self.mode == "train":
                self.compute_and_apply_scaling()
            elif self.mode =="test":
                if self.scaling_params is None:
                    raise ValueError("scaling_params must be provided for test/eval mode when apply_scaling=True")
                self._apply_scaling(self.scaling_params)

    def preprocess_data(self, data):
        X_list = []
        y_list = []

        for run_key in data.keys():  # iterate over each run
            run_data = data[run_key]
            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            cost = run_data["cost"]

            # Concatenate qpos and qvel
            X_run = np.concatenate([qpos, qvel], axis=1)
            X_list.append(X_run)

            # Ensure cost is 2D
            if len(cost.shape) == 1:
                y_run = cost[:, None]
            else:
                y_run = cost
            y_list.append(y_run)

        # Stack all runs together
        self.X = torch.from_numpy(np.vstack(X_list)).float()
        self.y = torch.from_numpy(np.vstack(y_list)).float()

    def compute_and_apply_scaling(self):
        train_idx = self.train_dataset.indices
        train_X = self.X[train_idx]
        train_y = self.y[train_idx]

        eps = 1e-8
        if self.scaling_type == "standardize":
            scaling_params = {
                "type": "standardize",
                "X_mean": train_X.mean(dim=0, keepdim=True),
                "X_std": train_X.std(dim=0, keepdim=True) + eps,
                "y_mean": train_y.mean(dim=0, keepdim=True),
                "y_std": train_y.std(dim=0, keepdim=True) + eps
            }
        elif self.scaling_type == "normalize":
            scaling_params = {
                "type": "normalize",
                "X_min": train_X.min(dim=0, keepdim=True)[0],
                "X_max": train_X.max(dim=0, keepdim=True)[0],
                "y_min": train_y.min(dim=0, keepdim=True)[0],
                "y_max": train_y.max(dim=0, keepdim=True)[0]
            }
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

        # Apply scaling
        self.X, self.y = run_scaling(self.X, self.y, self.scaling_type, scaling_params, self.scaling_range_X, self.scaling_range_y)
        save_scaling_values(scaling_params, self.run_dir)

    def _apply_scaling(self, scaling_params):
        self.X, self.y = run_scaling(self.X, self.y, self.scaling_type, scaling_params, self.scaling_range_X, self.scaling_range_y)
    
    def train_val_data(self, val_split=0.2, seed=42):
        dataset_size = len(self.X)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.train_dataset, self.val_dataset = random_split(
            self, [train_size, val_size], generator=generator
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
