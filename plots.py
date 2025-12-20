# plots.py
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from models import SimpleNN
from config import TARGET_COL
from embedder import LLMEmbedder
from clients_llm import LLMClient


def plot_convergence(history: Dict[str, List[float]], label: str, save_path: Optional[str] = None):
    rounds = history["round"]
    if "global_val_f1_mean" in history:
        f1_mean = history["global_val_f1_mean"]
        f1_std = history["global_val_f1_std"]
    else:
        f1_mean = history["f1"]
        f1_std = [0.0] * len(f1_mean)

    plt.figure()
    plt.plot(rounds, f1_mean, label=label)
    lower = np.array(f1_mean) - np.array(f1_std)
    upper = np.array(f1_mean) + np.array(f1_std)
    plt.fill_between(rounds, lower, upper, alpha=0.2)
    plt.xlabel("Communication Rounds")
    plt.ylabel("F1-score")
    plt.title("Training Convergence")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def estimate_model_size_kb(model: SimpleNN) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / 1024.0  # float32


def plot_accuracy_vs_comm(
    points: Dict[str, tuple],
    save_path: Optional[str] = None,
):
    plt.figure()
    for label, (f1, comm) in points.items():
        plt.scatter([comm], [f1])
        plt.annotate(label, (comm, f1))
    plt.xlabel("Communication Cost per Round (KB)")
    plt.ylabel("F1-score")
    plt.title("Accuracy vs Communication Cost")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_tsne_embeddings(
    clients: List[LLMClient],
    embedder: LLMEmbedder,
    max_points_per_client: int = 300,
    save_path: Optional[str] = None,
):
    raw_feats = []
    emb_feats = []
    labels = []

    for cid, client in enumerate(clients):
        n = min(max_points_per_client, len(client.df))
        idxs = np.random.choice(len(client.df), size=n, replace=False)

        numeric_cols = [c for c in client.df.columns if c != TARGET_COL]
        raw_vals = client.df.iloc[idxs][numeric_cols].select_dtypes(include=[float, int]).values
        raw_feats.append(raw_vals)

        emb_vals = client.embeddings[idxs]
        emb_feats.append(emb_vals)

        labels.extend([cid] * n)

    raw_feats = np.vstack(raw_feats)
    emb_feats = np.vstack(emb_feats)
    labels = np.array(labels)

    tsne_raw = TSNE(n_components=2, random_state=42).fit_transform(raw_feats)
    tsne_emb = TSNE(n_components=2, random_state=42).fit_transform(emb_feats)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=labels, alpha=0.7)
    plt.title("Raw Features")
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, alpha=0.7)
    plt.title("LLM Embeddings")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
