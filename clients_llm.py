# clients_llm.py
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config import TARGET_COL, DEVICE, SEED, FLConfig
from serialization import serialize_row
from embedder import LLMEmbedder
from models import SimpleNN
import torch.nn as nn


class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LLMClient:
    """
    Client for FedLLM-Align:
      - serializes its heterogeneous schema rows to text
      - uses a frozen LLM to embed them
      - trains a lightweight classifier locally
    """
    def __init__(
        self,
        client_id: int,
        df,
        serialization_format: str,
        embedder: LLMEmbedder,
    ):
        self.client_id = client_id
        self.df = df
        self.serialization_format = serialization_format
        self.embedder = embedder

        self.y = df[TARGET_COL].values.astype("float32")
        self.texts = self._serialize_df(df)

        self.embeddings = self.embedder.encode_texts(self.texts)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.embeddings,
            self.y,
            test_size=0.2,
            random_state=SEED,
            stratify=self.y,
        )

        self.train_loader = DataLoader(
            EmbeddingDataset(self.X_train, self.y_train),
            batch_size=32,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            EmbeddingDataset(self.X_val, self.y_val),
            batch_size=32,
            shuffle=False,
        )

    def _serialize_df(self, df) -> List[str]:
        texts = []
        for _, row in df.drop(columns=[TARGET_COL]).iterrows():
            texts.append(serialize_row(row, self.serialization_format))
        return texts

    def local_train(
        self,
        global_model: nn.Module,
        config: FLConfig,
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        model = SimpleNN(
            input_dim=self.embedder.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(DEVICE)

        model.load_state_dict(global_model.state_dict())

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for _ in range(config.local_epochs):
            model.train()
            for X, y in self.train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item() * len(X)
                all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
                all_labels.extend(y.cpu().numpy().astype(int).tolist())

        val_loss /= len(self.val_loader.dataset)
        f1 = f1_score(all_labels, all_preds)

        return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}, f1, val_loss
