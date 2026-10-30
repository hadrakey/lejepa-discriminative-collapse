"""LeJEPA training on Tennessee Eastman, optionally continued from an encoder.

    ldc train-ssl --init runs/s42/sup/encoder_init.pth --alpha 1 --lam 0.05 --out runs/s42/both
    ldc train-ssl --init ... --path detach_h --out runs/s42/detach_h

Loss (Balestriero & LeCun, arXiv:2511.08544), on the projector output z = g(h)
of several masked views of each window:

    L = α · invariance(z) + λ · SIGReg(z)

``--path`` chooses how SIGReg reaches the encoder (section 9 of the post):

- ``full``       SIGReg backpropagates through g into the encoder (LeJEPA).
- ``detach_h``   SIGReg sees g(detach(h)): it trains g only; the encoder is
                 trained by the invariance loss alone.
- ``freeze_enc`` encoder frozen; only g is trained.
- ``on_h``       no projector: both losses act on h directly.

Training is the T-JEPA code behind the post, minus the linear probe it ran
every 5 epochs for monitoring (verified weight for weight against T-JEPA with
that probe disabled). The probe drew from the random generator, so the post's
runs used different dropout draws: expect numbers close to, not equal to, the
post's. Two inherited quirks are kept: 2 × 10% of the rows are carved out and
unused, and rows are shuffled once, not every epoch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import N_SENSORS, N_STATS, load_split
from .model.checkpoint import save_checkpoint
from .model.encoder import Encoder, ModelConfig, clamp_gradients
from .model.lejepa import (
    LeJEPAProjector,
    SIGReg,
    lejepa_loss,
    make_sigreg_target_var,
    offset_masks_features_only,
    pool_tokens,
)
from .model.masking import MaskCollator, MaskConfig

PATHS = ("full", "detach_h", "freeze_enc", "on_h")


@dataclass
class SSLConfig:
    data_root: str = ""
    out_dir: str = ""
    init: str = ""                # encoder checkpoint to start from ("" = random)
    seed: int = 42
    device: str = "cuda:0"
    alpha: float = 1.0            # invariance weight
    lam: float = 0.05             # SIGReg weight
    path: str = "full"
    proj_dim: int = 16
    sigreg_target: str = "isotropic"  # isotropic | power | linear
    sigreg_power: float = 1.0
    shared_core: float = 0.0      # fraction of sensors visible in every view
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_value: float = 1.0  # element-wise, via parameter hooks
    save_epochs: tuple[int, ...] = (1, 5, 10, 25, 50)
    legacy_holdout_ratio: float = 0.1
    model: ModelConfig = field(default_factory=ModelConfig)


class _Rows(Dataset):
    def __init__(self, X: torch.Tensor):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i],)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ssl_rows(X: np.ndarray, holdout_ratio: float) -> torch.Tensor:
    """MinMax-scale on the full train split, then drop T-JEPA's unused 10% test
    and 10% val carve-outs (fixed random_state=42), which also shuffles once."""
    Xs = torch.from_numpy(MinMaxScaler().fit_transform(X).astype(np.float32))
    if holdout_ratio > 0:
        n_cut = int(Xs.shape[0] * holdout_ratio)
        Xs, _ = train_test_split(Xs, test_size=n_cut, random_state=42)
        Xs, _ = train_test_split(Xs, test_size=n_cut, random_state=42)
    return Xs


def build_optimizer(modules, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW; biases and 1-D params are excluded from weight decay."""
    groups = []
    for m in modules:
        decay, no_decay = [], []
        for name, p in m.named_parameters():
            (no_decay if ("bias" in name or p.ndim == 1) else decay).append(p)
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def mask_config(shared_core: float) -> MaskConfig:
    """Two context + two target views of 15-35% of the sensors each."""
    return MaskConfig(
        num_context=2, num_target=2,
        min_context_share=0.15, max_context_share=0.35,
        min_target_share=0.15, max_target_share=0.35,
        mode="sensor_block", n_sensors=N_SENSORS, n_stats=N_STATS,
        feature_layout="stat_major", shared_core=shared_core,
    )


class Trainer:
    def __init__(self, cfg: SSLConfig):
        if cfg.path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        seed_everything(cfg.seed)
        split = load_split(Path(cfg.data_root) / "train")
        self.n_features = split.X.shape[1]
        self.X = ssl_rows(split.X, cfg.legacy_holdout_ratio)
        print(f"[train-ssl] {len(self.X)} rows x {self.n_features} features, path={cfg.path}")

        # Construction order fixes the seeded init: encoder, then projector.
        self.encoder = Encoder(self.n_features, cfg.model).to(self.device)
        if cfg.grad_clip_value:
            clamp_gradients(self.encoder, cfg.grad_clip_value)
        if cfg.path == "on_h":
            self.projector, sig_dim = None, cfg.model.hidden_dim
        else:
            self.projector = LeJEPAProjector(cfg.model.hidden_dim, cfg.proj_dim).to(self.device)
            sig_dim = cfg.proj_dim
        target_var = make_sigreg_target_var(sig_dim, kind=cfg.sigreg_target, power=cfg.sigreg_power)
        self.sigreg = SIGReg(knots=17, num_slices=256, target_var=target_var).to(self.device)

        modules = [self.encoder] + ([self.projector] if self.projector is not None else [])
        self.optimizer = build_optimizer(modules, cfg.lr, cfg.weight_decay)
        if cfg.init:
            state = torch.load(cfg.init, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(state["context_encoder"])
            if self.projector is not None and isinstance(state.get("lejepa_projector"), dict):
                self.projector.load_state_dict(state["lejepa_projector"])
        if cfg.path == "freeze_enc":
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.collator = MaskCollator(mask_config(cfg.shared_core), self.n_features)
        self.loader = DataLoader(_Rows(self.X), batch_size=cfg.batch_size, shuffle=False,
                                 collate_fn=self.collator, drop_last=False,
                                 pin_memory=self.device.type == "cuda")

    def _loss(self, batch):
        rows, ctx, trg = batch
        rows = rows.to(self.device, non_blocking=True)
        n_cls = self.encoder.n_cls_tokens
        masks = offset_masks_features_only(
            [m.to(self.device, non_blocking=True) for m in list(ctx) + list(trg)], n_cls)
        pooled = [pool_tokens(self.encoder(rows, [m]), n_cls_tokens=0, mode="mean") for m in masks]
        if self.projector is None:
            proj, proj_sig = torch.stack(pooled), None
        else:
            proj = torch.stack([self.projector(p) for p in pooled])
            proj_sig = (torch.stack([self.projector(p.detach()) for p in pooled])
                        if self.cfg.path == "detach_h" else None)
        total, inv, sig, _, _ = lejepa_loss(proj, self.sigreg, self.cfg.lam,
                                            inv_weight=self.cfg.alpha, proj_sig=proj_sig)
        return total, inv, sig

    def train(self) -> Path:
        cfg = self.cfg
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        metrics = out / "metrics.jsonl"
        metrics.write_text("")
        self.encoder.train()
        if self.projector is not None:
            self.projector.train()
        for epoch in range(1, cfg.epochs + 1):
            t0, sums, n = time.time(), np.zeros(3), 0
            for batch in tqdm(self.loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
                loss, inv, sig = self._loss(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                sums += [loss.item(), inv.item(), sig.item()]
                n += 1
            rec = {"epoch": epoch, "loss": sums[0], "inv_mean": sums[1] / n,
                   "sigreg_mean": sums[2] / n, "seconds": round(time.time() - t0, 1)}
            with metrics.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[train-ssl] epoch {epoch:3d} loss={rec['loss']:.4f} "
                  f"inv={rec['inv_mean']:.4g} sigreg={rec['sigreg_mean']:.4g} ({rec['seconds']}s)")
            if epoch in cfg.save_epochs or epoch == cfg.epochs:
                save_checkpoint(out / f"epoch_{epoch}.pth", encoder=self.encoder,
                                projector=self.projector, optimizer=self.optimizer, scaler=None,
                                epoch=epoch, train_loss=sums[0], config=_config_dict(cfg),
                                n_features=self.n_features)
        return out / f"epoch_{cfg.epochs}.pth"


def _config_dict(cfg: SSLConfig) -> dict:
    """Checkpoint config; ``model`` is what ``load_checkpoint`` reads back."""
    return asdict(cfg)


def add_args(ap: argparse.ArgumentParser) -> None:
    d = SSLConfig()
    ap.add_argument("--data-root", required=True, help="folder with train/ and test/")
    ap.add_argument("--out", required=True, help="output folder for epoch_<E>.pth")
    ap.add_argument("--init", default="", help="encoder checkpoint to continue from")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--alpha", type=float, default=d.alpha, help="invariance weight α")
    ap.add_argument("--lam", type=float, default=d.lam, help="SIGReg weight λ")
    ap.add_argument("--path", choices=PATHS, default=d.path)
    ap.add_argument("--proj-dim", type=int, default=d.proj_dim)
    ap.add_argument("--sigreg-target", choices=["isotropic", "power", "linear"], default=d.sigreg_target)
    ap.add_argument("--sigreg-power", type=float, default=d.sigreg_power)
    ap.add_argument("--shared-core", type=float, default=d.shared_core)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--save-epochs", default=",".join(map(str, d.save_epochs)))


def main(args) -> None:
    cfg = SSLConfig(
        data_root=args.data_root, out_dir=args.out, init=args.init, seed=args.seed,
        device=args.device, alpha=args.alpha, lam=args.lam, path=args.path,
        proj_dim=args.proj_dim, sigreg_target=args.sigreg_target,
        sigreg_power=args.sigreg_power, shared_core=args.shared_core, epochs=args.epochs,
        save_epochs=tuple(int(e) for e in args.save_epochs.split(",") if e),
    )
    print(Trainer(cfg).train())
