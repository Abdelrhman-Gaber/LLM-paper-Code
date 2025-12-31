from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif

from config import DEVICE, FLConfig
from models import SimpleNN
from clients_tabular import TabularClient


def _evaluate_tabular_global(
    clients: List[TabularClient],
    global_model: nn.Module,
) -> Tuple[float, dict]:
    global_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for client in clients:
            for X, y in client.val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = global_model(X)
                all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
                all_labels.extend(y.cpu().numpy().astype(int).tolist())

    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return f1, report


def _fedavg(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg = {}
    for k in weights_list[0].keys():
        avg[k] = sum(w[k] for w in weights_list) / len(weights_list)
    return avg



def train_fedavg_tabular(
    clients: List[TabularClient],
    input_dim: int,
    config: FLConfig,
):
    global_model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)

    history = {"round": [], "f1": []}

    criterion = nn.BCELoss()

    for rnd in range(1, config.num_rounds + 1):
        local_weights = []

        for client in clients:
            model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
            model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            for _ in range(config.local_epochs):
                model.train()
                for X, y in client.train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()

            local_weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        new_state = _fedavg(local_weights)
        global_model.load_state_dict(new_state)

        f1, _ = _evaluate_tabular_global(clients, global_model)
        history["round"].append(rnd)
        history["f1"].append(f1)
        print(f"[FedAvg-Tab Round {rnd:02d}] F1={f1:.4f}")

    return history, global_model



def train_fedprox(
    clients: List[TabularClient],
    input_dim: int,
    config: FLConfig,
):
    """
    Simplified FedProx: local objective = cross-entropy + (mu/2)||w - w_global||^2
    """
    mu = config.mu
    global_model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
    criterion = nn.BCELoss()

    history = {"round": [], "f1": []}

    for rnd in range(1, config.num_rounds + 1):
        local_weights = []

        global_params = {k: v.detach().clone() for k, v in global_model.named_parameters()}

        for client in clients:
            model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
            model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            for _ in range(config.local_epochs):
                model.train()
                for X, y in client.train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)

                    # proximal term
                    prox = 0.0
                    for (name, param) in model.named_parameters():
                        prox += torch.norm(param - global_params[name].to(DEVICE)) ** 2
                    loss = loss + (mu / 2.0) * prox

                    loss.backward()
                    optimizer.step()

            local_weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        new_state = _fedavg(local_weights)
        global_model.load_state_dict(new_state)
        f1, _ = _evaluate_tabular_global(clients, global_model)
        history["round"].append(rnd)
        history["f1"].append(f1)
        print(f"[FedProx Round {rnd:02d}] F1={f1:.4f}")

    return history, global_model



def train_scaffold(
    clients: List[TabularClient],
    input_dim: int,
    config: FLConfig,
):
    """
    Simplified SCAFFOLD:
      - maintain global control variate c
      - per-client control variate c_i
    """
    K = len(clients)
    global_model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
    criterion = nn.BCELoss()

    c_global = {name: torch.zeros_like(param, device=DEVICE) for name, param in global_model.named_parameters()}
    c_clients = [
        {name: torch.zeros_like(param, device=DEVICE) for name, param in global_model.named_parameters()}
        for _ in range(K)
    ]

    history = {"round": [], "f1": []}

    for rnd in range(1, config.num_rounds + 1):
        local_weights = []
        new_c_clients = []

        for idx, client in enumerate(clients):
            model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
            model.load_state_dict(global_model.state_dict())

            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)  # SCAFFOLD often uses SGD

            c_i = c_clients[idx]

            for _ in range(config.local_epochs):
                model.train()
                for X, y in client.train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)
                    loss.backward()

                    # control variate correction
                    with torch.no_grad():
                        for (name, param) in model.named_parameters():
                            grad = param.grad
                            grad += c_global[name] - c_i[name]
                    optimizer.step()

            local_weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

            # update client control variate (simplified)
            new_c = {}
            for (name, param), (_, global_param) in zip(
                model.named_parameters(), global_model.named_parameters()
            ):
                new_c[name] = c_i[name] - (1.0 / (config.num_rounds)) * (param.detach() - global_param.detach())
            new_c_clients.append(new_c)

        # average
        new_state = _fedavg(local_weights)
        global_model.load_state_dict(new_state)
        c_global = {
            name: sum(c[name] for c in new_c_clients) / len(new_c_clients)
            for name in c_global.keys()
        }
        c_clients = new_c_clients

        f1, _ = _evaluate_tabular_global(clients, global_model)
        history["round"].append(rnd)
        history["f1"].append(f1)
        print(f"[SCAFFOLD Round {rnd:02d}] F1={f1:.4f}")

    return history, global_model



from sklearn.cluster import KMeans


def _flatten_weights(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    return torch.cat([p.view(-1) for p in state_dict.values()]).cpu().numpy()


def train_clustered_fl(
    clients: List[TabularClient],
    input_dim: int,
    config: FLConfig,
):
    """
    Very simplified Clustered FL:
      - warmup rounds with plain FedAvg
      - then cluster clients based on local model weights
      - train cluster-specific models
    """
    K = len(clients)
    num_clusters = min(config.num_clusters, K)

    global_model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
    criterion = nn.BCELoss()

    history = {"round": [], "f1": []}

    # Warmup: plain FedAvg
    warmup_rounds = min(config.cluster_warmup_rounds, config.num_rounds)
    for rnd in range(1, warmup_rounds + 1):
        local_weights = []
        for client in clients:
            model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
            model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            for _ in range(config.local_epochs):
                model.train()
                for X, y in client.train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()
            local_weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        new_state = _fedavg(local_weights)
        global_model.load_state_dict(new_state)
        f1, _ = _evaluate_tabular_global(clients, global_model)
        history["round"].append(rnd)
        history["f1"].append(f1)
        print(f"[Cluster-FL Warmup Round {rnd:02d}] F1={f1:.4f}")

    # Cluster based on local models from warmup
    client_weight_vecs = []
    for client in clients:
        model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
        model.load_state_dict(global_model.state_dict())
        client_weight_vecs.append(_flatten_weights(model.state_dict()))
    client_weight_vecs = np.vstack(client_weight_vecs)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    assignments = kmeans.fit_predict(client_weight_vecs)

    # Cluster-specific models
    cluster_models = [
        SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
        for _ in range(num_clusters)
    ]
    for m in cluster_models:
        m.load_state_dict(global_model.state_dict())

    # Continue rounds with cluster-specific aggregation
    for rnd in range(warmup_rounds + 1, config.num_rounds + 1):
        cluster_updates = [[] for _ in range(num_clusters)]

        for idx, client in enumerate(clients):
            c_idx = assignments[idx]
            model = SimpleNN(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(DEVICE)
            model.load_state_dict(cluster_models[c_idx].state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            for _ in range(config.local_epochs):
                model.train()
                for X, y in client.train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()

            cluster_updates[c_idx].append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        for c_idx in range(num_clusters):
            if cluster_updates[c_idx]:
                new_state = _fedavg(cluster_updates[c_idx])
                cluster_models[c_idx].load_state_dict(new_state)

        # Evaluate by assigning each client to its cluster model
        all_preds, all_labels = [], []
        with torch.no_grad():
            for idx, client in enumerate(clients):
                c_idx = assignments[idx]
                mdl = cluster_models[c_idx]
                mdl.eval()
                for X, y in client.val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    preds = mdl(X)
                    all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
                    all_labels.extend(y.cpu().numpy().astype(int).tolist())

        f1 = f1_score(all_labels, all_preds)
        history["round"].append(rnd)
        history["f1"].append(f1)
        print(f"[Cluster-FL Round {rnd:02d}] F1={f1:.4f}")

    # For simplicity, return the global_model as the last warmup model.
    return history, global_model


# ------------------------------------------------------------------
# FedXGBoost baseline (approximate)
# ------------------------------------------------------------------

def train_fedxgboost_central(
    X_clients: List[np.ndarray],
    y_clients: List[np.ndarray],
    max_depth: int = 3,
    n_estimators: int = 100,
):
    """
    Approximate FedXGBoost baseline:
      - centrally train XGBoost on concatenated client data.
      - This ignores secure aggregation but matches predictive behavior.

    You need: pip install xgboost
    """
    import xgboost as xgb

    X = np.vstack(X_clients)
    y = np.concatenate(y_clients).astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    import pickle
    buf = pickle.dumps(model)
    model_size_kb = len(buf) / 1024.0

    print(f"[FedXGBoost Central] F1={f1:.4f}, size~{model_size_kb:.1f} KB")

    return f1, report, model_size_kb



def train_mi_baseline(
    X_clients: List[np.ndarray],
    y_clients: List[np.ndarray],
    top_k: int = 10,
):
    """
    Approximate mutual-information–driven baseline:
      - compute MI between each feature and label on each client
      - aggregate MI scores
      - select top-k global features
      - centrally train a small NN on those features
    """
    X_all = np.vstack(X_clients)
    y_all = np.concatenate(y_clients).astype(int)

    mi_scores = mutual_info_classif(X_all, y_all, discrete_features=False)
    feature_indices = np.argsort(mi_scores)[::-1][:top_k]

    X_all_sel = X_all[:, feature_indices]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_sel, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    model = SimpleNN(input_dim=top_k, hidden_dim=8, dropout=0.1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train.astype("float32")).float().to(DEVICE)

    for _ in range(50):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
        y_val_t = torch.from_numpy(y_val.astype("float32")).float().to(DEVICE)
        preds = model(X_val_t)
        y_pred = (preds.cpu().numpy() >= 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)
        report = classification_report(y_val_t.cpu().numpy().astype(int), y_pred, output_dict=True)

    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024.0

    print(f"[MI Baseline] F1={f1:.4f}, size~{model_size_kb:.1f} KB, top_k={top_k}")

    return f1, report, model_size_kb
