# config.py
import random
import numpy as np
import torch
from dataclasses import dataclass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAMINGHAM_FEATURES = [
    "sex",
    "age",
    "is_smoking",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "heartRate",
    "glucose",
]

TARGET_COL = "TenYearCHD"  # adjust to your CSV if needed


@dataclass
class FLConfig:
    num_rounds: int = 25          # global rounds
    local_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    max_seq_len: int = 128
    num_clients: int = 3
    serialization_format: str = "structured"  # "structured" | "natural" | "compact"
    backbone: str = "distilbert"             # "distilbert" | "albert" | "roberta" | "clinical"
    hidden_dim: int = 16
    dropout: float = 0.2
    participation_ratio: float = 1.0  # fraction of clients per round

    # FedProx-specific
    mu: float = 0.01

    # Clustered FL
    num_clusters: int = 2
    cluster_warmup_rounds: int = 5
