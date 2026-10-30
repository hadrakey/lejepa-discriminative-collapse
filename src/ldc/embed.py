"""Encoder embeddings h (mean of the feature tokens) and projector outputs z."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .model import load_checkpoint


def resolve_device(device: str) -> str:
    return device if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(X: np.ndarray, encoder, device: str, batch_size: int = 512) -> np.ndarray:
    encoder.eval()
    out = []
    for i in tqdm(range(0, len(X), batch_size), desc="embed", leave=False):
        tokens = encoder(torch.from_numpy(X[i : i + batch_size]).float().to(device))
        out.append(tokens[:, encoder.n_cls_tokens :].mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def project(H: np.ndarray, projector, device: str, batch_size: int = 2048) -> np.ndarray:
    projector.eval()
    out = [projector(torch.from_numpy(H[i : i + batch_size]).float().to(device)).cpu().numpy()
           for i in range(0, len(H), batch_size)]
    return np.concatenate(out, axis=0)


def embed_checkpoint(path, X: np.ndarray, device: str):
    """(h, encoder, projector) for a checkpoint; projector is None if not saved."""
    m = load_checkpoint(path, n_features=X.shape[1], device=device)
    return embed(X, m.encoder, device), m.encoder, m.projector
