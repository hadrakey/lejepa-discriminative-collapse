"""Feature-as-token Transformer encoder for numeric tabular rows.

Each of the D features becomes one token (``x_i * W_i + b_i``), preceded by
``n_cls_tokens`` [CLS] tokens. Sinusoidal position encodings and a learned
feature-index embedding mark column identity. The output is one token per
input token; ``Encoder.embed`` pools the [CLS] tokens into the row embedding Z.

All corpora in this project are fed as numeric columns (categorical columns
are integer-coded upstream), so the categorical-embedding branch of the
original T-JEPA encoder is not carried over.

State-dict keys and parameter creation order match the T-JEPA ``Encoder`` so
that its checkpoints load with ``strict=True`` and seeded inits are identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    num_layers: int = 4
    num_heads: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    n_cls_tokens: int = 1
    feature_index_embedding: bool = True
    activation: str = "relu"


class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, d_token: int, n_cls_tokens: int = 1):
        super().__init__()
        self.n_cls_tokens = n_cls_tokens
        self.weight = nn.Parameter(torch.empty(d_numerical + n_cls_tokens, d_token))
        self.bias = nn.Parameter(torch.empty(d_numerical, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(x.size(0), self.n_cls_tokens, device=x.device)
        x = torch.cat([ones, x], dim=1)  # [CLS] tokens carry value 1
        tokens = self.weight[None] * x[:, :, None]
        bias = torch.cat(
            [torch.zeros(self.n_cls_tokens, self.bias.size(1), device=x.device), self.bias]
        )
        return tokens + bias[None]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class TabularTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.hidden_dim
        # `transformer_layers` is the prototype cloned by TransformerEncoder. It is
        # unused in forward but kept so the state-dict layout matches T-JEPA.
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation=cfg.activation,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layers, cfg.num_layers, enable_nested_tensor=False
        )
        self.fc = nn.Linear(h, h)
        self.layernorm1 = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm1(self.dropout1(self.transformer(x)))
        return self.layernorm2(self.dropout2(self.fc(x)))


def gather_tokens(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Keep the token indices of each mask; views are stacked on the batch axis.

    x: [B, N, H]; each mask: [B, K] -> output [len(masks) * B, K, H].
    """
    out = [torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, x.size(-1))) for m in masks]
    return torch.cat(out, dim=0)


class Encoder(nn.Module):
    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.n_cls_tokens = cfg.n_cls_tokens
        self.tokenizer = Tokenizer(n_features, cfg.hidden_dim, cfg.n_cls_tokens)
        if cfg.feature_index_embedding:
            self.feature_index_embedding = nn.Embedding(n_features, cfg.hidden_dim)
            self.register_buffer("feature_indices", torch.arange(n_features), persistent=False)
        else:
            self.feature_index_embedding = None
        self.encoder = TabularTransformer(cfg)
        self.pe = PositionalEncoding(cfg.hidden_dim)

    def tokens(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Embed + position-encode all tokens, then keep the masked ones."""
        out = self.pe(self.tokenizer(x))
        if self.feature_index_embedding is not None:
            idx = self.feature_index_embedding(self.feature_indices)
            cls = torch.zeros(self.n_cls_tokens, idx.size(1), device=out.device, dtype=out.dtype)
            out = out + torch.cat([cls, idx], dim=0)[None]
        if masks is not None:
            out = gather_tokens(out, masks)
        return out

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """x: [B, D] -> tokens [B, n_cls + D, H] (or [V*B, K, H] with masks).

        Masks hold *token* indices. The mask collator emits feature indices
        0..D-1 and, by historical default, they are used as token indices
        without the +n_cls offset (see ``LeJEPAConfig.fix_cls_mask``).
        """
        return self.encoder(self.tokens(x, masks))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Row embedding Z: mean of the [CLS] output tokens."""
        return self.forward(x)[:, : self.n_cls_tokens].mean(dim=1)


def clamp_gradients(module: nn.Module, value: float) -> None:
    """Element-wise gradient clamp via parameter hooks (T-JEPA behaviour).

    The hooks fire during backward, i.e. on *AMP-scaled* gradients when a
    GradScaler is used, so with AMP on the clamp is much tighter than ``value``
    in real units. Kept as-is so trainings stay comparable with old runs.
    """
    for p in module.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -value, value))
