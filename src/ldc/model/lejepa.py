"""LeJEPA objective for tabular encoders.

Follows Balestriero & LeCun (arXiv:2511.08544): invariance across views +
Sketched Isotropic Gaussian Regularization (SIGReg), without teacher–student /
EMA / stop-gradient heuristics.

SIGReg implementation adapted from the official minimal example:
https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Gaussian Regularization (Epps–Pulley style).

    Default target is isotropic N(0, I). Optional ``target_var`` of shape [D]
    replaces the per-slice characteristic function with that of N(0, Σ) where
    Σ = diag(target_var), so a unit slice a has φ_a(t) = exp(−σ_a² t² / 2)
    with σ_a² = Σ_i a_i² σ_i² (P1-C anisotropic ablation).
    """

    def __init__(
        self,
        knots: int = 17,
        num_slices: int = 256,
        t_max: float = 3.0,
        target_var: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # isotropic CF; unused if target_var set
        self.register_buffer("weights", weights * window)
        if target_var is None:
            self.target_var = None
        else:
            tv = torch.as_tensor(target_var, dtype=torch.float32).reshape(-1)
            self.register_buffer("target_var", tv)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: [V, B, D] or [B, D] projected embeddings.
        """
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        # Flatten views into the sample axis for CF estimation: [V*B, D]
        flat = proj.reshape(-1, proj.size(-1))
        d = flat.size(-1)
        a = torch.randn(
            d, self.num_slices, device=flat.device, dtype=flat.dtype
        )
        a = a.div_(a.norm(p=2, dim=0).clamp_min(1e-12))
        t = self.t.to(dtype=flat.dtype)
        x_t = (flat @ a).unsqueeze(-1) * t  # [N, S, K]
        if self.target_var is None:
            phi = self.phi.to(dtype=flat.dtype)  # [K]
            cos_err = (x_t.cos().mean(0) - phi).square()
        else:
            if self.target_var.numel() != d:
                raise ValueError(
                    f"SIGReg target_var dim {self.target_var.numel()} != proj dim {d}"
                )
            # σ_a² = a^T Σ a for Σ = diag(target_var); φ_a(t) = exp(−σ_a² t² / 2)
            var_a = (a.square() * self.target_var.to(dtype=flat.dtype).unsqueeze(1)).sum(0)
            phi = torch.exp(-0.5 * var_a.unsqueeze(1) * t.square())  # [S, K]
            cos_err = (x_t.cos().mean(0) - phi).square()
        err = cos_err + x_t.sin().mean(0).square()
        statistic = (err @ self.weights.to(dtype=flat.dtype)) * flat.size(0)
        return statistic.mean()


def make_sigreg_target_var(
    dim: int,
    kind: str = "isotropic",
    power: float = 1.0,
    min_var: float = 0.05,
) -> torch.Tensor | None:
    """Build diagonal target variances for SIGReg.

    kind:
      - isotropic: None (use built-in N(0,I) CF)
      - power: σ_i² ∝ i^(−power), renormalized so mean variance = 1
      - linear: linearly decaying from ~2 to min_var, mean 1
    """
    if kind in (None, "", "isotropic", "iso"):
        return None
    idx = torch.arange(1, dim + 1, dtype=torch.float32)
    if kind == "power":
        raw = idx.pow(-float(power))
    elif kind == "linear":
        raw = torch.linspace(2.0, float(min_var), dim)
    else:
        raise ValueError(f"Unknown SIGReg target kind: {kind}")
    raw = raw.clamp_min(float(min_var))
    return raw * (dim / raw.sum())  # mean variance = 1

class LeJEPAProjector(nn.Module):
    """Lightweight MLP projector head used by LeJEPA."""

    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or max(in_dim * 2, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.net(x)


def pool_tokens(
    tokens: torch.Tensor,
    n_cls_tokens: int = 1,
    mode: str = "cls",
) -> torch.Tensor:
    """Pool encoder tokens to a sample embedding.

    tokens: [B, N, D]
    mode:
      - cls: mean of CLS tokens (default probe readout)
      - mean_features: mean of non-CLS tokens when present, else all tokens
      - mean: mean of every token in the sequence
    """
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "mean_features":
        if n_cls_tokens > 0 and tokens.size(1) > n_cls_tokens:
            return tokens[:, n_cls_tokens:].mean(dim=1)
        return tokens.mean(dim=1)
    if n_cls_tokens > 0 and tokens.size(1) >= n_cls_tokens:
        return tokens[:, :n_cls_tokens].mean(dim=1)
    return tokens.mean(dim=1)


def offset_masks_features_only(
    masks: list[torch.Tensor], n_cls_tokens: int
) -> list[torch.Tensor]:
    """Map feature indices {0..D-1} → token indices, without keeping CLS.

    Encoder layout is [CLS×n_cls | feature_0 | …]. Used when the LeJEPA
    readout is mean-pool over visible feature tokens only.
    """
    if n_cls_tokens <= 0:
        return masks
    return [m + n_cls_tokens for m in masks]


def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    """View-invariance: each view close to the mean view. proj: [V, B, D]."""
    return (proj.mean(0) - proj).square().mean()


def _flat_z(z: torch.Tensor) -> torch.Tensor:
    """[V, B, D] or [B, D] → [N, D] float."""
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    return z.float()


def variance_hinge(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg-style hinge: encourage per-dim std of Z >= target_std."""
    std = _flat_z(z).std(dim=0, unbiased=False).clamp_min(1e-6)
    return torch.relu(target_std - std).mean()


def covariance_offdiag(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance: mean squared off-diagonal of batch cov(Z)."""
    z = _flat_z(z)
    z = z - z.mean(dim=0)
    n = z.size(0)
    cov = (z.T @ z) / max(n - 1, 1)
    d = cov.size(0)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off / (d * max(d - 1, 1))


def lejepa_loss(
    proj: torch.Tensor,
    sigreg: SIGReg,
    lamb: float,
    z: torch.Tensor | None = None,
    z_var_weight: float = 0.0,
    z_var_target: float = 1.0,
    z_cov_weight: float = 0.0,
    inv_weight: float | None = None,
    proj_sig: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, inv, sigreg, z_var, z_cov).

    total = α·inv + λ·sigreg + γ_v·var_hinge(Z) + γ_c·cov_offdiag(Z).

    If inv_weight (α) is None, use the historical default α = 1 − λ so that
    older checkpoints remain comparable. Set α explicitly to study the
    invariance / SIGReg trade-off independently of λ.

    ``proj_sig``: optional separate input for SIGReg, e.g. g(detach(h)) so that
    SIGReg trains the projector but does not backpropagate into the encoder.
    """
    inv = invariance_loss(proj)
    sig = sigreg(proj if proj_sig is None else proj_sig)
    zero = inv.new_zeros(())
    var = (
        variance_hinge(z, target_std=z_var_target)
        if (z is not None and z_var_weight > 0.0)
        else zero
    )
    cov = (
        covariance_offdiag(z)
        if (z is not None and z_cov_weight > 0.0)
        else zero
    )
    alpha = (1.0 - lamb) if inv_weight is None else float(inv_weight)
    total = (
        alpha * inv
        + lamb * sig
        + z_var_weight * var
        + z_cov_weight * cov
    )
    return total, inv, sig, var, cov


def offset_masks_keep_cls(
    masks: list[torch.Tensor], n_cls_tokens: int
) -> list[torch.Tensor]:
    """Map feature indices {0..D-1} → token indices, always keeping CLS.

    Encoder token layout is [CLS×n_cls | feature_0 | … | feature_{D-1}].
    MaskCollator emits feature indices; without +n_cls, gather(0) wrongly
    selects CLS as 'feature 0' and shifts every other feature.
    """
    if n_cls_tokens <= 0:
        return masks
    out = []
    for m in masks:
        # m: [B, K]
        b = m.size(0)
        cls = torch.arange(
            n_cls_tokens, device=m.device, dtype=m.dtype
        ).view(1, n_cls_tokens).expand(b, -1)
        out.append(torch.cat([cls, m + n_cls_tokens], dim=1))
    return out
