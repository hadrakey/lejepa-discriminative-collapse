"""Save / load encoder checkpoints.

Checkpoints written by this package embed their training config, so loading
needs nothing else. Checkpoints from the T-JEPA repo (``epoch_50.pth`` under
``checkpoints/<DATASET>/<long-job-name>/``) are also accepted: the architecture
is inferred from the state dict, with ``num_heads=2`` (not recoverable from
weights; it is the value every LeJEPA run used).

Both formats share the keys ``context_encoder`` and ``lejepa_projector``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from .encoder import Encoder, ModelConfig
from .lejepa import LeJEPAProjector

FORMAT_VERSION = 2


@dataclass
class LoadedModel:
    encoder: Encoder
    projector: LeJEPAProjector | None
    model_cfg: ModelConfig
    path: Path
    train_config: dict | None = None  # full TrainConfig dict (new checkpoints only)


def save_checkpoint(path: Path, *, encoder, projector, optimizer, scaler, epoch: int,
                    train_loss: float, config: dict, n_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": config,
            "n_features": n_features,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "context_encoder": encoder.state_dict(),
            "lejepa_projector": projector.state_dict() if projector is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        path,
    )


def infer_legacy_model_config(state: dict, n_features: int) -> ModelConfig:
    if any(k.startswith("feature_type_embedding") for k in state):
        raise NotImplementedError("checkpoints with categorical feature-type embeddings")
    tok_w = state["tokenizer.weight"]
    n_layers = 1 + max(
        int(k.split(".")[3]) for k in state if k.startswith("encoder.transformer.layers.")
    )
    return ModelConfig(
        hidden_dim=int(tok_w.shape[1]),
        num_layers=n_layers,
        num_heads=2,
        dim_feedforward=int(state["encoder.transformer_layers.linear1.weight"].shape[0]),
        n_cls_tokens=max(int(tok_w.shape[0] - n_features), 1),
        feature_index_embedding=any(k.startswith("feature_index_embedding") for k in state),
    )


def _projector_from_state(state: dict) -> LeJEPAProjector:
    in_dim = state["net.0.weight"].shape[1]
    last_w = [v for k, v in state.items() if k.endswith(".weight") and v.ndim == 2][-1]
    hidden = state["net.0.weight"].shape[0]
    proj = LeJEPAProjector(in_dim=in_dim, proj_dim=int(last_w.shape[0]), hidden_dim=hidden)
    proj.load_state_dict(state)
    return proj


def load_checkpoint(path: str | Path, n_features: int, device: str = "cpu") -> LoadedModel:
    """Rebuild encoder (+ projector if saved) in eval mode on ``device``."""
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["context_encoder"]
    train_cfg = ckpt.get("config")
    if train_cfg is not None:
        known = {f.name for f in fields(ModelConfig)}
        model_cfg = ModelConfig(**{k: v for k, v in train_cfg["model"].items() if k in known})
    else:
        model_cfg = infer_legacy_model_config(state, n_features)

    encoder = Encoder(n_features, model_cfg)
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()

    projector = None
    if isinstance(ckpt.get("lejepa_projector"), dict):
        projector = _projector_from_state(ckpt["lejepa_projector"]).to(device).eval()
    return LoadedModel(encoder, projector, model_cfg, path, train_cfg)


def model_config_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)
