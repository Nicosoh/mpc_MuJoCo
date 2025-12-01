import torch
from torch.utils.data import Dataset, random_split
from data_collection import load_npz
import numpy as np

class PendulumDataset(Dataset):
    """
    Dataset for Pendulum:
    Inputs X = [qpos, qvel]
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """
    def __init__(self, data_path, standardize=True):
        """
        Args:
            data_path (str): Path to the .npz file
        """
        self.standardize = standardize
        data = load_npz(data_path)
        self.preprocess_data(data) # Process data to be in pytorch format

        if self.standardize:
            self.compute_standardization()  # Compute mean/std and standardize y

        self.train_val_data() # Split data into train and val

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
    
    def compute_standardization(self):
        # Standardize X
        self.X_mean = self.X.mean(dim=0, keepdim=True)
        self.X_std = self.X.std(dim=0, keepdim=True) + 1e-8  # avoid div by 0
        self.X = (self.X - self.X_mean) / self.X_std

        # Standardize y
        self.y_mean = self.y.mean(dim=0, keepdim=True)
        self.y_std = self.y.std(dim=0, keepdim=True) + 1e-8
        self.y = (self.y - self.y_mean) / self.y_std
    
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
