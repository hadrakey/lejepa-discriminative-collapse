"""Invertible affine control (section 9 of the post).

    ldc affine --data-root $DATA --init runs/s42/sup/encoder_init.pth --seed 42 --out results/affine_s42.json

The supervised encoder is frozen. Only z = A·h + b is learned, to minimize
SIGReg(z) with a −β·log|det A| penalty that keeps A invertible. A linear probe
on standardized z should then match the probe on h: lowering the SIGReg loss
this way cannot remove linear information, up to numerics.

h is standardized first (itself an invertible affine map). Without it
(``--raw-h``), an embedding with a tiny variance makes A grow until it no
longer behaves as invertible in float32.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data import load_probe_data
from ..embed import embed, embed_checkpoint, resolve_device
from ..metrics import class_gap, effective_rank, fit_w, participation_ratio, probe
from ..model.lejepa import SIGReg, make_sigreg_target_var

TARGETS = [("iso", "isotropic", 1.0), ("power2", "power", 2.0), ("linear", "linear", 1.0)]


class AffineMap(nn.Module):
    """z = A h + b with A initialized near the identity."""

    def __init__(self, dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A.T + self.b

    def logabsdet(self) -> torch.Tensor:
        return torch.linalg.slogdet(self.A).logabsdet


def add_args(ap) -> None:
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--init", required=True, help="supervised encoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--det-weight", type=float, default=0.01, help="β")
    ap.add_argument("--raw-h", action="store_true", help="do not standardize h first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    d = load_probe_data(args.data_root)
    Htr, enc, _ = embed_checkpoint(args.init, d.Xtr, device)
    Hte = embed(d.Xte, enc, device)
    if not args.raw_h:
        mu, sd = Htr.mean(0), Htr.std(0) + 1e-12
        Htr, Hte = (Htr - mu) / sd, (Hte - mu) / sd
    dim = Htr.shape[1]
    w = fit_w(Htr, d.ytr)
    base = {"probe": probe(Htr, d.ytr, Hte, d.yte), **class_gap(Htr, d.ytr, w),
            "eff_rank": effective_rank(Htr), "participation_ratio": participation_ratio(Htr)}

    Ht = torch.tensor(Htr, device=device, dtype=torch.float32)
    Hte_t = torch.tensor(Hte, device=device, dtype=torch.float32)
    runs = {}
    for tag, kind, power in TARGETS:
        sigreg = SIGReg(knots=17, num_slices=256,
                        target_var=make_sigreg_target_var(dim, kind, power)).to(device)
        affine = AffineMap(dim).to(device)
        opt = torch.optim.Adam(affine.parameters(), lr=args.lr)
        hist = []
        for ep in range(1, args.epochs + 1):
            perm = torch.randperm(Ht.size(0), device=device)
            losses = []
            for i in range(0, Ht.size(0), args.batch_size):
                loss = sigreg(affine(Ht[perm[i : i + args.batch_size]])) - args.det_weight * affine.logabsdet()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            if ep in (1, 10, 50, 100, 200) or ep == args.epochs:
                with torch.no_grad():
                    Ztr, Zte = affine(Ht).cpu().numpy(), affine(Hte_t).cpu().numpy()
                    lad = float(affine.logabsdet().cpu())
                    cond = float(torch.linalg.cond(affine.A).cpu())
                hist.append({
                    "epoch": ep, "loss_mean": float(np.mean(losses)), "logabsdet": lad, "cond_A": cond,
                    "probe_z": probe(Ztr, d.ytr, Zte, d.yte),
                    "fisher_z_refit": class_gap(Ztr, d.ytr, fit_w(Ztr, d.ytr))["fisher_on_w"],
                    "eff_rank_z": effective_rank(Ztr), "participation_ratio_z": participation_ratio(Ztr),
                })
                r = hist[-1]
                print(f"[affine {tag}] epoch {ep:3d} loss={r['loss_mean']:.4g} log|det A|={lad:.1f} "
                      f"roc_auc_z={r['probe_z']['roc_auc']:.3f} (h: {base['probe']['roc_auc']:.3f})")
        runs[tag] = {"target": kind, "power": power, "trajectory": hist, "final": hist[-1]}

    report = {"init": str(args.init), "standardized_h": not args.raw_h, "epochs": args.epochs,
              "det_weight": args.det_weight, "start": base, "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(args.out)
