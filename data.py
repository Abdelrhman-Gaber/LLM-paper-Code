import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from config import FRAMINGHAM_FEATURES, TARGET_COL, SEED
from serialization import random_alt_name


def load_framingham_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Try to normalize target column name if needed
    col_map = {}
    for col in df.columns:
        if col.strip().lower() in ["tenyearchd", "10-year risk of chd", "10_year_chd"]:
            col_map[col] = TARGET_COL
    df = df.rename(columns=col_map)

    missing_cols = [c for c in FRAMINGHAM_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    # simple imputation
    for col in FRAMINGHAM_FEATURES:
        if df[col].dtype == "O":
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def get_shared_unique_config(num_clients: int) -> Tuple[int, int]:
    if num_clients == 3:
        return 8, 3
    elif num_clients == 5:
        return 6, 4
    elif num_clients == 10:
        return 4, 6
    else:
        shared = max(3, len(FRAMINGHAM_FEATURES) // 2)
        unique = max(1, (len(FRAMINGHAM_FEATURES) - shared) // 2)
        return shared, unique


def simulate_clients(
    df: pd.DataFrame,
    num_clients: int,
    imbalance: bool = True,
) -> List[pd.DataFrame]:
    """
    Simulate heterogeneous schemas and label imbalance.
    Returns a list of client-specific DataFrames with alternate column names.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    shared_count, unique_per_client = get_shared_unique_config(num_clients)
    all_feats = FRAMINGHAM_FEATURES.copy()
    random.shuffle(all_feats)
    shared_feats = all_feats[:shared_count]
    remaining = all_feats[shared_count:]

    X = df[FRAMINGHAM_FEATURES]
    y = df[TARGET_COL].values

    client_dfs = []

    for i in range(num_clients):
        unique_feats = random.sample(remaining, k=min(unique_per_client, len(remaining)))
        client_feats = list(set(shared_feats + unique_feats))

        if imbalance:
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            pos_ratio = 0.10 + (0.15 * i / max(1, num_clients - 1))
            num_samples = len(df) // num_clients
            num_pos = int(num_samples * pos_ratio)
            num_neg = num_samples - num_pos

            pos_sample = np.random.choice(pos_indices, size=min(num_pos, len(pos_indices)), replace=True)
            neg_sample = np.random.choice(neg_indices, size=min(num_neg, len(neg_indices)), replace=True)
            indices = np.concatenate([pos_sample, neg_sample])
        else:
            indices = np.random.choice(len(df), size=len(df) // num_clients, replace=False)

        client_df = df.iloc[indices].copy()

        keep_cols = client_feats + [TARGET_COL]
        client_df = client_df[keep_cols].copy()

        rename_map: Dict[str, str] = {}
        for col in client_feats:
            rename_map[col] = random_alt_name(col)
        client_df = client_df.rename(columns=rename_map)

        client_dfs.append(client_df)

    return client_dfs


def build_global_feature_space(client_dfs: List[pd.DataFrame]):
    """
    Build union-of-columns feature space for tabular baselines.

    NOTE: This intentionally treats different aliases as distinct features,
    which simulates the schema-heterogeneity issue for traditional FL baselines.
    """
    all_cols = set()
    for df in client_dfs:
        for c in df.columns:
            if c != TARGET_COL:
                all_cols.add(c)
    all_cols = sorted(list(all_cols))

    X_clients = []
    y_clients = []

    for df in client_dfs:
        X = df.drop(columns=[TARGET_COL])
        # Add missing columns with zeros
        for col in all_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[all_cols]  # consistent ordering
        X_clients.append(X.values.astype("float32"))
        y_clients.append(df[TARGET_COL].values.astype("float32"))

    return all_cols, X_clients, y_clients
