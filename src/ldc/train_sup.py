"""Supervised reference encoder: the LeJEPA encoder + a linear head, trained
end-to-end on normal-vs-fault with a class-weighted cross-entropy.

    ldc train-sup --data-root $DATA --seed 42 --out runs/s42/sup

Writes ``encoder_init.pth`` (the starting point of every continuation run)
and ``summary.json`` (training history, probe and geometry of h).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_probe_data
from .embed import embed, resolve_device
from .metrics import binary_metrics, geometry, probe
from .model.checkpoint import save_checkpoint
from .model.encoder import Encoder, ModelConfig, clamp_gradients


class SupHead(nn.Module):
    def __init__(self, encoder: Encoder, n_classes: int = 2):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(encoder.cfg.hidden_dim, n_classes)

    def forward(self, x):
        tokens = self.enc(x)
        return self.head(tokens[:, self.enc.n_cls_tokens :].mean(dim=1))


def train_supervised(Xtr, ytr, Xte, yte, device, *, epochs=40, batch_size=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder(Xtr.shape[1], ModelConfig())
    clamp_gradients(enc, 1.0)
    model = SupHead(enc).to(device)
    counts = np.bincount(ytr, minlength=2).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X, Y = torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()
    Xt = torch.from_numpy(Xte).float()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(X[idx].to(device)), Y[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            proba = np.concatenate([
                F.softmax(model(Xt[i : i + batch_size].to(device)), dim=-1)[:, 1].cpu().numpy()
                for i in range(0, len(Xt), batch_size)
            ])
        hist.append({"epoch": ep, "loss": total / len(X),
                     **binary_metrics(yte, (proba >= 0.5).astype(int), proba)})
        if ep % 10 == 0 or ep == 1:
            print(f"[train-sup] epoch {ep}/{epochs} loss={total / len(X):.4f} "
                  f"roc_auc={hist[-1]['roc_auc']:.3f}")
    return model, hist


def add_args(ap) -> None:
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")


def main(args) -> None:
    device = resolve_device(args.device)
    d = load_probe_data(args.data_root)
    model, hist = train_supervised(d.Xtr, d.ytr, d.Xte, d.yte, device,
                                   epochs=args.epochs, seed=args.seed)
    Htr, Hte = embed(d.Xtr, model.enc, device), embed(d.Xte, model.enc, device)
    out = Path(args.out)
    init = out / "encoder_init.pth"
    save_checkpoint(init, encoder=model.enc, projector=None, optimizer=None, scaler=None,
                    epoch=0, train_loss=0.0, config={"model": asdict(model.enc.cfg)},
                    n_features=d.Xtr.shape[1])
    summary = {"seed": args.seed, "train_history": hist,
               "h": {"geometry": geometry(Htr), "probe": probe(Htr, d.ytr, Hte, d.yte)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    h = summary["h"]
    print(f"[train-sup] probe on h: acc={h['probe']['accuracy']:.3f} "
          f"roc_auc={h['probe']['roc_auc']:.3f} eff_rank={h['geometry']['eff_rank']:.2f}")
    print(init)
