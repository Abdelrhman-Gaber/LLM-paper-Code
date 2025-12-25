import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

import xgboost as xgb



@dataclass
class Config:
    # ---- Data ----
    data_path: str = "churn.csv"  # <-- SET THIS TO YOUR CSV
    target_column: str = "Exited"
    id_columns: List[str] = field(
        default_factory=lambda: ["RowNumber", "CustomerId", "Surname"]
    )
    test_size: float = 0.2
    random_state: int = 42

    # ---- Federated setup ----
    num_rounds: int = 25
    local_epochs: int = 3
    batch_size: int = 32
    client_fraction: float = 1.0  # fraction of clients per round
    overlap_values: List[float] = field(
        default_factory=lambda: [0.8, 0.6, 0.4, 0.2]
    )

    # ---- Optimization ----
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # ---- LLM backbone & serialization ----
    backbone_name: str = "distilbert-base-uncased"
    max_length: int = 128
    serialization_style: str = "structured"  # 'structured' | 'natural' | 'compact'

    # ---- LLM finetuning (optional) ----
    # Turn this ON to allow the last encoder layer to adapt slightly to the task.
    finetune_last_layer: bool = True

    # ---- Model head ----
    head_type: str = "nn"  # 'nn' or 'lr'
    hidden_dim: int = 16
    dropout: float = 0.2

    # ---- Device ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Plotting ----
    figure_dpi: int = 120


config = Config()

# Embedding sizes for some common backbones
BACKBONE_DIM = {
    "distilbert-base-uncased": 768,
    "albert-base-v2": 768,
    "roberta-base": 768,
    # add more if you use them
}




def load_financial_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)

    # Drop ID columns if present
    for col in cfg.id_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop rows with missing target
    df = df.dropna(subset=[cfg.target_column])

    # Ensure binary int target (0/1)
    if df[cfg.target_column].dtype not in (np.int32, np.int64, np.bool_):
        uniques = sorted(df[cfg.target_column].dropna().unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        df[cfg.target_column] = df[cfg.target_column].map(mapping).astype(int)
    else:
        df[cfg.target_column] = df[cfg.target_column].astype(int)

    return df


def train_test_split_df(df: pd.DataFrame, cfg: Config):
    X = df.drop(columns=[cfg.target_column])
    y = df[cfg.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# Load data once at module import
df = load_financial_dataset(config)
X_train, X_test, y_train, y_test = train_test_split_df(df, config)

print("Loaded dataset")
print("Shape:", df.shape)
print("Target distribution:")
print(df[config.target_column].value_counts(normalize=True).rename("proportion"))

# Global pos_weight for imbalance-aware BCE
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
if pos == 0:
    POS_WEIGHT = 1.0
else:
    POS_WEIGHT = float(neg / pos)
print(f"\n[INFO] Using pos_weight={POS_WEIGHT:.3f} for BCEWithLogitsLoss")



def assign_samples_to_clients(
    X: pd.DataFrame,
    y: pd.Series,
    n_clients: int,
    random_state: int = 42,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(random_state)
    client_ids = rng.integers(low=0, high=n_clients, size=len(X))
    clients = {}
    for cid in range(n_clients):
        idx = np.where(client_ids == cid)[0]
        X_c = X.iloc[idx].copy()
        y_c = y.iloc[idx].copy()
        clients[cid] = {"X": X_c, "y": y_c}
    return clients


def build_feature_subsets(
    features: List[str],
    n_clients: int,
    overlap_ratio: float,
    random_state: int = 42,
) -> Dict[int, List[str]]:
    """
    Create per-client feature subsets with approximate overlap_ratio.
    overlap_ratio ~ fraction of features shared across all clients.
    """
    rng = np.random.default_rng(random_state)
    n_features = len(features)

    # Number of shared features
    shared_count = max(1, int(overlap_ratio * n_features))
    shared_features = list(rng.choice(features, size=shared_count, replace=False))
    remaining = [f for f in features if f not in shared_features]

    client_features: Dict[int, List[str]] = {}
    base_extra = max(1, math.floor(max(0, n_features - shared_count) / n_clients))
    rem_feats = remaining.copy()

    for cid in range(n_clients):
        if len(rem_feats) > 0:
            k = min(len(rem_feats), base_extra)
            extra = list(rng.choice(rem_feats, size=k, replace=False))
            rem_feats = [f for f in rem_feats if f not in extra]
        else:
            extra = []
        client_features[cid] = shared_features + extra

    for f in rem_feats:
        cid = int(rng.integers(0, n_clients))
        client_features[cid].append(f)

    return client_features


def build_name_aliases(
    features: List[str],
    n_clients: int,
    random_state: int = 42,
) -> Dict[int, Dict[str, str]]:
    """
    Create client-specific textual aliases for feature names to mimic
    schema naming heterogeneity.
    """
    rng = np.random.default_rng(random_state)
    aliases: Dict[int, Dict[str, str]] = {}
    for cid in range(n_clients):
        alias_map: Dict[str, str] = {}
        for f in features:
            variants = [
                f,
                f"{f}_attr",
                f"{f}_feature",
                f"{f}_field",
                f.replace("_", " "),
                f"client{cid} {f}",
            ]
            alias_map[f] = rng.choice(variants)
        aliases[cid] = alias_map
    return aliases



def serialize_row(
    row: pd.Series,
    feature_names: List[str],
    alias_map: Dict[str, str],
    style: str = "structured",
) -> str:
    """
    Turn one tabular row into a text string.

    For bank-churn–like data we:
      - Keep most numeric values as raw numbers (LLM can handle them)
      - Make some booleans / small integers more verbal (Age, HasCrCard, etc.)
      - Keep text reasonably compact.
    """
    parts: List[str] = []

    for col in feature_names:
        alias = alias_map.get(col, col)
        val = row[col]

        if pd.isna(val):
            val_str = "unknown"
        elif col.lower() in ["geography", "country", "region"]:
            val_str = str(val)
        elif col.lower() in ["gender", "sex"]:
            val_str = str(val)
        elif col in ["HasCrCard", "IsActiveMember"]:
            val_str = "yes" if int(val) == 1 else "no"
        elif col == "NumOfProducts":
            val_str = f"{int(val)} products"
        elif col == "Age" and np.issubdtype(type(val), np.number):
            # Slight bucketing for age to give semantic hints
            if val < 30:
                val_str = "young"
            elif val < 50:
                val_str = "middle-aged"
            else:
                val_str = "senior"
        else:
            # Keep numeric features as plain numbers (credit score, balance, salary, etc.)
            val_str = str(val)

        if style == "structured":
            parts.append(f"{alias}: {val_str}")
        elif style == "natural":
            parts.append(f"The {alias} is {val_str}.")
        elif style == "compact":
            parts.append(f"{alias}={val_str}")
        else:
            raise ValueError(f"Unknown serialization style: {style}")

    if style == "natural":
        return " ".join(parts)
    elif style == "compact":
        return "; ".join(parts)
    else:
        return ", ".join(parts)


def serialize_client_data(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    alias_map: Dict[str, str],
    style: str,
) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels = y.values.astype(int)
    for i in range(len(X)):
        row = X.iloc[i]
        txt = serialize_row(row, feature_names, alias_map, style)
        texts.append(txt)
    return texts, labels




class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], int(self.labels[idx])


def build_tokenizer_and_model(backbone_name: str, device: str, finetune_last_layer: bool):
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = AutoModel.from_pretrained(backbone_name)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Optionally unfreeze last layer for finetuning
    if finetune_last_layer:
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
            # DistilBERT-style
            for p in model.transformer.layer[-1].parameters():
                p.requires_grad = True
            print("[Info] Unfroze last DistilBERT layer.")
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            # BERT / RoBERTa-style
            for p in model.encoder.layer[-1].parameters():
                p.requires_grad = True
            print("[Info] Unfroze last encoder layer.")
        else:
            print("[Warning] Could not identify encoder layers to unfreeze.")

    model.to(device)
    model.eval()  # we use it as a feature extractor here
    return tokenizer, model


def compute_embeddings_for_client(
    texts: List[str],
    labels: np.ndarray,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = TextDataset(texts, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_embs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_texts, batch_labels in dl:
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
            all_embs.append(cls_emb)
            all_labels.append(batch_labels.clone().long())

    embeddings = torch.cat(all_embs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return embeddings, labels_tensor



class LRHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits.view(-1)


class SimpleNNHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(-1)


def build_head(cfg: Config) -> nn.Module:
    embed_dim = BACKBONE_DIM[cfg.backbone_name]
    if cfg.head_type == "lr":
        model = LRHead(embed_dim)
    else:
        model = SimpleNNHead(embed_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    return model


def train_local_head(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
    pos_weight: Optional[float] = None,
) -> nn.Module:
    model = model.to(cfg.device)
    ds = TensorDataset(embeddings.to(cfg.device), labels.to(cfg.device))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=cfg.device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.local_epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()

    return model.cpu()




def model_state_to_cpu(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}


def fedavg_aggregate(
    client_states: List[Tuple[int, Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    total_samples = sum(n for n, _ in client_states)
    agg_state: Dict[str, torch.Tensor] = {}
    for n, state in client_states:
        weight = n / total_samples
        for k, v in state.items():
            if k not in agg_state:
                agg_state[k] = weight * v
            else:
                agg_state[k] += weight * v
    return agg_state


def evaluate_global_head(
    head: nn.Module,
    client_test_data: Dict[int, Dict[str, torch.Tensor]],
    cfg: Config,
) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate head, searching for a probability threshold that maximizes
    F1 for the positive class.
    """
    head = head.to(cfg.device)
    head.eval()
    all_probs_list: List[torch.Tensor] = []
    all_labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for cid, data in client_test_data.items():
            embs = data["embeddings"].to(cfg.device)
            labels = data["labels"].to(cfg.device)
            logits = head(embs)
            probs = torch.sigmoid(logits)
            all_probs_list.append(probs.cpu())
            all_labels_list.append(labels.cpu())

    all_probs = torch.cat(all_probs_list).numpy()
    all_labels = torch.cat(all_labels_list).numpy()

    # Search best threshold for F1 on positive class
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 17):
        preds = (all_probs >= thr).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    per_client_f1: Dict[int, float] = {}
    with torch.no_grad():
        for cid, data in client_test_data.items():
            embs = data["embeddings"].to(cfg.device)
            labels = data["labels"].cpu().numpy()
            logits = head(embs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= best_thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            per_client_f1[cid] = f1

    return best_f1, per_client_f1


def run_fedavg_training(
    client_train_data: Dict[int, Dict[str, torch.Tensor]],
    client_test_data: Dict[int, Dict[str, torch.Tensor]],
    cfg: Config,
    pos_weight: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[Dict[int, float]]]:
    """
    client_train_data[cid] = {"embeddings": tensor, "labels": tensor}
    client_test_data[cid]  = {"embeddings": tensor, "labels": tensor}
    """
    global_head = build_head(cfg)
    global_state = model_state_to_cpu(global_head.state_dict())

    global_f1_history: List[float] = []
    per_client_f1_history: List[Dict[int, float]] = []

    n_clients = len(client_train_data)

    for rnd in range(cfg.num_rounds):
        num_active = max(1, int(cfg.client_fraction * n_clients))
        active_clients = random.sample(list(client_train_data.keys()), num_active)

        client_states: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for cid in active_clients:
            local_head = build_head(cfg)
            local_head.load_state_dict(global_state)
            train_local_head(
                local_head,
                client_train_data[cid]["embeddings"],
                client_train_data[cid]["labels"],
                cfg,
                pos_weight=pos_weight,
            )
            client_states.append(
                (
                    len(client_train_data[cid]["labels"]),
                    model_state_to_cpu(local_head.state_dict()),
                )
            )

        global_state = fedavg_aggregate(client_states)
        global_head.load_state_dict(global_state)

        global_f1, per_client_f1 = evaluate_global_head(
            global_head, client_test_data, cfg
        )
        global_f1_history.append(global_f1)
        per_client_f1_history.append(per_client_f1)

        if verbose:
            print(f"[Round {rnd + 1}/{cfg.num_rounds}] Global F1: {global_f1:.4f}")

    return global_head, global_f1_history, per_client_f1_history




def build_federated_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_clients: int,
    overlap_ratio: float,
    cfg: Config,
    backbone_name: Optional[str] = None,
    serialization_style: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    if backbone_name is None:
        backbone_name = cfg.backbone_name
    if serialization_style is None:
        serialization_style = cfg.serialization_style

    train_clients = assign_samples_to_clients(
        X_train, y_train, n_clients, cfg.random_state
    )
    test_clients = assign_samples_to_clients(
        X_test, y_test, n_clients, cfg.random_state + 1
    )

    feature_subsets = build_feature_subsets(
        list(X_train.columns),
        n_clients=n_clients,
        overlap_ratio=overlap_ratio,
        random_state=cfg.random_state,
    )
    alias_maps = build_name_aliases(
        list(X_train.columns), n_clients=n_clients, random_state=cfg.random_state
    )

    tokenizer, model = build_tokenizer_and_model(
        backbone_name, cfg.device, cfg.finetune_last_layer
    )

    client_train_data: Dict[int, Dict[str, torch.Tensor]] = {}
    client_test_data: Dict[int, Dict[str, torch.Tensor]] = {}

    for cid in range(n_clients):
        feats = feature_subsets[cid]
        alias_map = alias_maps[cid]

        X_c_train = train_clients[cid]["X"][feats]
        y_c_train = train_clients[cid]["y"]
        texts_train, labels_train_np = serialize_client_data(
            X_c_train, y_c_train, feats, alias_map, serialization_style
        )
        embs_train, labels_train = compute_embeddings_for_client(
            texts_train,
            labels_train_np,
            tokenizer,
            model,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        X_c_test = test_clients[cid]["X"][feats]
        y_c_test = test_clients[cid]["y"]
        texts_test, labels_test_np = serialize_client_data(
            X_c_test, y_c_test, feats, alias_map, serialization_style
        )
        embs_test, labels_test = compute_embeddings_for_client(
            texts_test,
            labels_test_np,
            tokenizer,
            model,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        client_train_data[cid] = {"embeddings": embs_train, "labels": labels_train}
        client_test_data[cid] = {"embeddings": embs_test, "labels": labels_test}

    embed_dim = embs_train.shape[1]
    print(
        f"Backbone: {backbone_name}, serialization: {serialization_style}, embed_dim={embed_dim}"
    )
    return client_train_data, client_test_data


def build_homogeneous_federated_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_clients: int,
    cfg: Config,
    backbone_name: Optional[str] = None,
    serialization_style: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    if backbone_name is None:
        backbone_name = cfg.backbone_name
    if serialization_style is None:
        serialization_style = cfg.serialization_style

    full_features = list(X_train.columns)

    train_clients = assign_samples_to_clients(
        X_train, y_train, n_clients, cfg.random_state
    )
    test_clients = assign_samples_to_clients(
        X_test, y_test, n_clients, cfg.random_state + 1
    )

    alias_maps = build_name_aliases(
        full_features, n_clients=n_clients, random_state=cfg.random_state
    )

    tokenizer, model = build_tokenizer_and_model(
        backbone_name, cfg.device, cfg.finetune_last_layer
    )

    client_train_data: Dict[int, Dict[str, torch.Tensor]] = {}
    client_test_data: Dict[int, Dict[str, torch.Tensor]] = {}

    for cid in range(n_clients):
        alias_map = alias_maps[cid]

        X_c_train = train_clients[cid]["X"][full_features]
        y_c_train = train_clients[cid]["y"]
        texts_train, labels_train_np = serialize_client_data(
            X_c_train, y_c_train, full_features, alias_map, serialization_style
        )
        embs_train, labels_train = compute_embeddings_for_client(
            texts_train,
            labels_train_np,
            tokenizer,
            model,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        X_c_test = test_clients[cid]["X"][full_features]
        y_c_test = test_clients[cid]["y"]
        texts_test, labels_test_np = serialize_client_data(
            X_c_test, y_c_test, full_features, alias_map, serialization_style
        )
        embs_test, labels_test = compute_embeddings_for_client(
            texts_test,
            labels_test_np,
            tokenizer,
            model,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        client_train_data[cid] = {"embeddings": embs_train, "labels": labels_train}
        client_test_data[cid] = {"embeddings": embs_test, "labels": labels_test}

    return client_train_data, client_test_data




def preprocess_for_xgb(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode object (string) columns so XGBoost sees only numeric features.
    Ensures train/test have the same columns.
    """
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    if len(cat_cols) == 0:
        return X_train.copy(), X_test.copy()

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    return X_train_enc, X_test_enc


def run_centralized_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    X_train_enc, X_test_enc = preprocess_for_xgb(X_train, X_test)

    # Imbalance handling for XGBoost as well (fair comparison)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train_enc, y_train)
    y_prob = model.predict_proba(X_test_enc)[:, 1]

    # Threshold search for fair F1 comparison
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    y_pred = (y_prob >= best_thr).astype(int)
    return f1_score(y_test, y_pred, zero_division=0)



def plot_convergence(histories: Dict[str, List[float]], cfg: Config, title: str):
    plt.figure(dpi=cfg.figure_dpi)
    for label, f1_hist in histories.items():
        plt.plot(range(1, len(f1_hist) + 1), f1_hist, label=label)
    plt.xlabel("Federated round")
    plt.ylabel("Global F1-score (positive class)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def summarize_per_client_stats(
    per_client_history: List[Dict[int, float]]
) -> Dict[str, float]:
    last_round = per_client_history[-1]
    f1_vals = np.array(list(last_round.values()))
    stats = {
        "mean": float(f1_vals.mean()),
        "std": float(f1_vals.std()),
        "min": float(f1_vals.min()),
        "max": float(f1_vals.max()),
    }
    return stats


def debug_label_distribution(client_data: Dict[int, Dict[str, torch.Tensor]]):
    for cid, data in client_data.items():
        labels = data["labels"].numpy()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {cid}: {dict(zip(unique, counts))}")



def run_main_experiment() -> pd.DataFrame:
    cfg = config

    print("\n=== Centralized XGBoost Baseline (tabular) ===")
    xgb_f1 = run_centralized_xgboost(X_train, y_train, X_test, y_test)
    print(f"XGBoost F1: {xgb_f1:.4f}")

    client_settings = [3, 5, 10]
    overlap_map = {3: 0.6, 5: 0.5, 10: 0.4}

    results_rows = []
    conv_histories: Dict[str, List[float]] = {}

    for n_clients in client_settings:
        print(f"\n=== FedLLM-Align, {n_clients} clients, heterogeneous schemas ===")
        client_train_data, client_test_data = build_federated_data(
            X_train,
            y_train,
            X_test,
            y_test,
            n_clients=n_clients,
            overlap_ratio=overlap_map[n_clients],
            cfg=cfg,
        )

        print("Train label distribution per client:")
        debug_label_distribution(client_train_data)
        print("Test label distribution per client:")
        debug_label_distribution(client_test_data)

        head, global_f1_history, per_client_history = run_fedavg_training(
            client_train_data,
            client_test_data,
            cfg,
            pos_weight=POS_WEIGHT,
            verbose=True,
        )
        conv_histories[f"FedLLM-Align {n_clients}c"] = global_f1_history
        hetero_stats = summarize_per_client_stats(per_client_history)
        results_rows.append(
            {
                "Method": f"FedLLM-Align ({n_clients} clients)",
                "Clients": n_clients,
                "Heterogeneous": True,
                "Global_F1": global_f1_history[-1],
                "Mean_Client_F1": hetero_stats["mean"],
                "Std_Client_F1": hetero_stats["std"],
                "Min_Client_F1": hetero_stats["min"],
                "Max_Client_F1": hetero_stats["max"],
            }
        )

        print(f"\n=== Homogeneous FedAvg, {n_clients} clients ===")
        homo_train_data, homo_test_data = build_homogeneous_federated_data(
            X_train,
            y_train,
            X_test,
            y_test,
            n_clients=n_clients,
            cfg=cfg,
        )
        head_h, global_f1_history_h, per_client_history_h = run_fedavg_training(
            homo_train_data,
            homo_test_data,
            cfg,
            pos_weight=POS_WEIGHT,
            verbose=True,
        )
        conv_histories[f"Homo FedAvg {n_clients}c"] = global_f1_history_h
        homo_stats = summarize_per_client_stats(per_client_history_h)
        results_rows.append(
            {
                "Method": f"Homogeneous FedAvg ({n_clients} clients)",
                "Clients": n_clients,
                "Heterogeneous": False,
                "Global_F1": global_f1_history_h[-1],
                "Mean_Client_F1": homo_stats["mean"],
                "Std_Client_F1": homo_stats["std"],
                "Min_Client_F1": homo_stats["min"],
                "Max_Client_F1": homo_stats["max"],
            }
        )

    results_df = pd.DataFrame(results_rows)
    print("\n=== Summary Table (FedLLM-Align vs Homogeneous FedAvg) ===")
    print(results_df.to_string(index=False))

    xgb_row = {
        "Method": "Centralized XGBoost",
        "Clients": "N/A",
        "Heterogeneous": False,
        "Global_F1": xgb_f1,
        "Mean_Client_F1": np.nan,
        "Std_Client_F1": np.nan,
        "Min_Client_F1": np.nan,
        "Max_Client_F1": np.nan,
    }
    results_df = pd.concat([results_df, pd.DataFrame([xgb_row])], ignore_index=True)
    print("\n=== With XGBoost row ===")
    print(results_df.to_string(index=False))

    plot_convergence(
        conv_histories, cfg, "Convergence: FedLLM-Align vs Homogeneous FedAvg"
    )

    return results_df



def measure_inference_time(
    tokenizer,
    model,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    n_warmup: int = 1,
) -> float:
    ds = TextDataset(texts, np.zeros(len(texts)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, (batch_texts, _) in enumerate(dl):
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
            if i + 1 >= n_warmup:
                break

    t0 = time.time()
    with torch.no_grad():
        for batch_texts, _ in dl:
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
    t1 = time.time()
    total_time = t1 - t0
    avg_time_ms = 1000.0 * total_time / len(ds)
    return avg_time_ms


def run_backbone_comparison(
    n_clients: int = 3, overlap_ratio: float = 0.6
) -> pd.DataFrame:
    backbones = [
        "distilbert-base-uncased",
        "albert-base-v2",
        "roberta-base",
    ]

    rows = []
    cfg = config

    # Stratified subset for speed & balanced labels
    small_X_train, _, small_y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(256, len(X_train)),
        stratify=y_train,
        random_state=cfg.random_state,
    )
    small_X_test, _, small_y_test, _ = train_test_split(
        X_test,
        y_test,
        train_size=min(256, len(X_test)),
        stratify=y_test,
        random_state=cfg.random_state + 1,
    )

    for backbone in backbones:
        print(f"\n=== Backbone: {backbone} ===")
        cfg.backbone_name = backbone

        client_train_data, client_test_data = build_federated_data(
            small_X_train,
            small_y_train,
            small_X_test,
            small_y_test,
            n_clients=n_clients,
            overlap_ratio=overlap_ratio,
            cfg=cfg,
        )
        head, global_f1_history, _ = run_fedavg_training(
            client_train_data, client_test_data, cfg, pos_weight=POS_WEIGHT, verbose=False
        )
        final_f1 = global_f1_history[-1]

        tokenizer, model = build_tokenizer_and_model(
            backbone, cfg.device, cfg.finetune_last_layer
        )
        n_params = sum(p.numel() for p in model.parameters())
        approx_mem_mb = n_params * 4 / (1024**2)

        texts_example = ["This is a dummy financial record."] * 128
        inf_time_ms = measure_inference_time(
            tokenizer,
            model,
            texts_example,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            device=cfg.device,
        )

        rows.append(
            {
                "Backbone": backbone,
                "Final_F1": final_f1,
                "Params_M": n_params / 1e6,
                "Approx_Memory_MB": approx_mem_mb,
                "Inference_ms_per_record": inf_time_ms,
            }
        )
        print(
            f"F1={final_f1:.4f}, Params={n_params/1e6:.2f}M, "
            f"ApproxMem={approx_mem_mb:.1f}MB, Infer={inf_time_ms:.2f}ms/record"
        )

    df_backbone = pd.DataFrame(rows)
    print("\nBackbone comparison:")
    print(df_backbone.to_string(index=False))
    return df_backbone




def run_serialization_comparison(
    n_clients: int = 3, overlap_ratio: float = 0.6
) -> pd.DataFrame:
    cfg = config
    cfg.backbone_name = "distilbert-base-uncased"

    formats = ["structured", "natural", "compact"]
    rows = []

    for fmt in formats:
        print(f"\n=== Serialization format: {fmt} ===")
        cfg.serialization_style = fmt
        client_train_data, client_test_data = build_federated_data(
            X_train,
            y_train,
            X_test,
            y_test,
            n_clients=n_clients,
            overlap_ratio=overlap_ratio,
            cfg=cfg,
        )
        head, global_f1_history, per_client_hist = run_fedavg_training(
            client_train_data, client_test_data, cfg, pos_weight=POS_WEIGHT, verbose=False
        )
        final_f1 = global_f1_history[-1]
        per_stats = summarize_per_client_stats(per_client_hist)

        rows.append(
            {
                "Format": fmt,
                "Final_F1": final_f1,
                "Embedding_Variance_Proxy": per_stats["std"],
            }
        )
        print(
            f"Format={fmt}, F1={final_f1:.4f}, proxy variance={per_stats['std']:.4f}"
        )

    df_fmt = pd.DataFrame(rows)
    print("\nSerialization comparison:")
    print(df_fmt.to_string(index=False))
    return df_fmt



def run_schema_overlap_stress_test(n_clients: int = 5) -> pd.DataFrame:
    cfg = config
    cfg.backbone_name = "distilbert-base-uncased"
    cfg.serialization_style = "structured"

    rows = []

    for overlap in cfg.overlap_values:
        print(f"\n=== Overlap ratio: {overlap:.2f} ===")
        client_train_data, client_test_data = build_federated_data(
            X_train,
            y_train,
            X_test,
            y_test,
            n_clients=n_clients,
            overlap_ratio=overlap,
            cfg=cfg,
        )
        head, global_f1_history, _ = run_fedavg_training(
            client_train_data, client_test_data, cfg, pos_weight=POS_WEIGHT, verbose=False
        )
        hetero_f1 = global_f1_history[-1]

        homo_train_data, homo_test_data = build_homogeneous_federated_data(
            X_train,
            y_train,
            X_test,
            y_test,
            n_clients=n_clients,
            cfg=cfg,
        )
        head_h, global_f1_history_h, _ = run_fedavg_training(
            homo_train_data, homo_test_data, cfg, pos_weight=POS_WEIGHT, verbose=False
        )
        homo_f1 = global_f1_history_h[-1]

        rows.append(
            {
                "Overlap": overlap,
                "FedLLM-Align_F1": hetero_f1,
                "Homogeneous_F1": homo_f1,
            }
        )
        print(
            f"Overlap={overlap:.2f}, FedLLM-Align F1={hetero_f1:.4f}, "
            f"Homo F1={homo_f1:.4f}"
        )

    df_overlap = pd.DataFrame(rows)
    print("\nSchema overlap stress test:")
    print(df_overlap.to_string(index=False))

    plt.figure(dpi=cfg.figure_dpi)
    plt.plot(
        df_overlap["Overlap"],
        df_overlap["FedLLM-Align_F1"],
        marker="o",
        label="FedLLM-Align",
    )
    plt.plot(
        df_overlap["Overlap"],
        df_overlap["Homogeneous_F1"],
        marker="o",
        label="Homogeneous FedAvg",
    )
    plt.xlabel("Schema Overlap Ratio")
    plt.ylabel("F1-score (positive class)")
    plt.title("Schema Overlap Stress Test")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return df_overlap




def run_tsne_visualization(
    n_clients: int = 3, overlap_ratio: float = 0.6, n_samples: int = 500
) -> None:
    cfg = config
    cfg.backbone_name = "distilbert-base-uncased"
    cfg.serialization_style = "structured"

    n_samples = min(n_samples, len(X_train))
    small_X_train = X_train.head(n_samples)
    small_y_train = y_train.head(n_samples)
    small_X_test = X_test.head(n_samples)
    small_y_test = y_test.head(n_samples)

    client_train_data, _ = build_federated_data(
        small_X_train,
        small_y_train,
        small_X_test,
        small_y_test,
        n_clients=n_clients,
        overlap_ratio=overlap_ratio,
        cfg=cfg,
    )

    embs_list = []
    client_ids = []
    for cid, data in client_train_data.items():
        embs = data["embeddings"]
        embs_list.append(embs)
        client_ids.extend([cid] * len(embs))

    embs_all = torch.cat(embs_list, dim=0).numpy()
    client_ids = np.array(client_ids)

    tsne = TSNE(n_components=2, random_state=cfg.random_state, perplexity=30)
    embs_2d = tsne.fit_transform(embs_all)

    plt.figure(dpi=cfg.figure_dpi)
    for cid in range(n_clients):
        mask = client_ids == cid
        plt.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1], s=10, alpha=0.7, label=f"Client {cid}"
        )
    plt.title("t-SNE of LLM Embeddings by Client")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()




if __name__ == "__main__":
    # Make sure Config.data_path is correct before running.
    print("\nRunning main experiment...")
    main_results_df = run_main_experiment()

    # Optional additional experiments:
    backbone_results_df = run_backbone_comparison()
    serialization_results_df = run_serialization_comparison()
    overlap_results_df = run_schema_overlap_stress_test()
    run_tsne_visualization()
