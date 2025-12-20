# clients_tabular.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import SEED


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularClient:
    """
    Client for tabular FL baselines (FedAvg, FedProx, SCAFFOLD, Clustered).
    """
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray):
        self.client_id = client_id

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

        self.train_loader = DataLoader(
            TabularDataset(self.X_train, self.y_train),
            batch_size=32,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TabularDataset(self.X_val, self.y_val),
            batch_size=32,
            shuffle=False,
        )
