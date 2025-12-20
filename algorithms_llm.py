# algorithms_llm.py
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader

from config import DEVICE, FLConfig
from models import SimpleNN
from clients_llm import LLMClient, EmbeddingDataset
from embedder import LLMEmbedder


def fedavg(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg = {}
    for k in weights_list[0].keys():
        avg[k] = sum(w[k] for w in weights_list) / len(weights_list)
    return avg


def federated_training_llm(
    clients: List[LLMClient],
    embedder: LLMEmbedder,
    config: FLConfig,
) -> Tuple[Dict[str, List[float]], SimpleNN]:
    global_model = SimpleNN(
        input_dim=embedder.embed_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(DEVICE)

    history = {
        "round": [],
        "global_val_f1_mean": [],
        "global_val_f1_std": [],
    }

    for rnd in range(1, config.num_rounds + 1):
        num_participants = max(1, int(len(clients) * config.participation_ratio))
        participating = np.random.choice(clients, size=num_participants, replace=False)

        local_weights = []
        local_f1s = []
        local_losses = []

        for client in participating:
            w, f1, loss = client.local_train(global_model, config)
            local_weights.append(w)
            local_f1s.append(f1)
            local_losses.append(loss)

        new_state = fedavg(local_weights)
        global_model.load_state_dict(new_state)

        history["round"].append(rnd)
        history["global_val_f1_mean"].append(float(np.mean(local_f1s)))
        history["global_val_f1_std"].append(float(np.std(local_f1s)))

        print(
            f"[FedLLM Round {rnd:02d}] "
            f"mean F1={history['global_val_f1_mean'][-1]:.4f} "
            f"std={history['global_val_f1_std'][-1]:.4f}"
        )

    return history, global_model


def evaluate_global_llm(
    clients: List[LLMClient],
    global_model: SimpleNN,
) -> Tuple[float, dict]:
    global_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for client in clients:
            ds = EmbeddingDataset(client.X_val, client.y_val)
            loader = DataLoader(ds, batch_size=64, shuffle=False)
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = global_model(X)
                all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
                all_labels.extend(y.cpu().numpy().astype(int).tolist())

    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return f1, report
