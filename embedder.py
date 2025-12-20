# embedder.py
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    DistilBertTokenizerFast,
    DistilBertModel,
    AlbertTokenizerFast,
    AlbertModel,
)

from config import DEVICE


class LLMEmbedder(nn.Module):
    def __init__(self, backbone: str = "distilbert", max_len: int = 128):
        super().__init__()
        self.backbone_name = backbone
        self.max_len = max_len

        if backbone == "distilbert":
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif backbone == "albert":
            self.tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
            self.model = AlbertModel.from_pretrained("albert-base-v2")
        elif backbone == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModel.from_pretrained("roberta-base")
        elif backbone == "clinical":
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.embed_dim = self.model.config.hidden_size

        # freeze encoder
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            outputs = self.model(**enc)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_embeddings = outputs.pooler_output
            else:
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
        return np.vstack(embeddings)
