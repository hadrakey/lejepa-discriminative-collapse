"""Local effect of SIGReg on the supervised discriminant (section 7 of the post).

    ldc grad --data-root $DATA --init runs/s42/sup/encoder_init.pth --out results/grad_iso.json

No training. On the supervised embeddings H of the training windows:

1. fit w, the discriminant direction of a balanced linear probe;
2. pass H through a freshly initialized projector g (as at the start of a
   continuation run) and take g_sig = ∇_H SIGReg(g(H)), averaged over
   ``--mc`` draws of the random SIGReg slices;
3. take g_inv = ∇_H invariance on two jittered copies of H;
4. report the projection of each gradient on w, overall and per class;
5. apply one hypothetical step H' = H − η·g and measure the Fisher criterion
   along w and the probe on H'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data import load_probe_data
from ..embed import embed, embed_checkpoint, resolve_device
from ..metrics import class_gap, fit_w, probe, r_disc
from ..model.lejepa import LeJEPAProjector, SIGReg, invariance_loss, make_sigreg_target_var


def projection_stats(g: np.ndarray, w: np.ndarray, y: np.ndarray) -> dict:
    w_u = w / (np.linalg.norm(w) + 1e-12)
    g_par = g @ w_u
    gn = np.linalg.norm(g, axis=1) + 1e-12
    return {
        "g_parallel_mean": float(g_par.mean()),
        "g_parallel_mean_normal": float(g_par[y == 0].mean()),
        "g_parallel_mean_fault": float(g_par[y == 1].mean()),
        "cos_to_w_mean": float((g_par / gn).mean()),
        "g_norm_mean": float(gn.mean()),
    }


def add_args(ap) -> None:
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--init", required=True, help="supervised encoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigreg-target", choices=["isotropic", "power", "linear"], default="isotropic")
    ap.add_argument("--sigreg-power", type=float, default=1.0)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--mc", type=int, default=8, help="SIGReg slice draws to average")
    ap.add_argument("--etas", default="0.01,0.05,0.2,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    d = load_probe_data(args.data_root)
    Htr, enc, _ = embed_checkpoint(args.init, d.Xtr, device)
    Hte = embed(d.Xte, enc, device)
    w = fit_w(Htr, d.ytr)
    base = {"probe": probe(Htr, d.ytr, Hte, d.yte), **class_gap(Htr, d.ytr, w), **r_disc(Htr, w)}

    proj = LeJEPAProjector(in_dim=Htr.shape[1], proj_dim=args.proj_dim).to(device).train()
    tv = make_sigreg_target_var(args.proj_dim, args.sigreg_target, args.sigreg_power)
    sigreg = SIGReg(knots=17, num_slices=256, target_var=tv).to(device)

    H = torch.tensor(Htr, device=device, dtype=torch.float32)
    g_sig = torch.zeros_like(H)
    for _ in range(args.mc):
        leaf = H.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(sigreg(proj(leaf)), leaf)
        g_sig += g.detach()
    g_sig = (g_sig / args.mc).cpu().numpy()

    # two nearby "views" of every window: H and H + small isotropic noise
    H1 = H.detach().requires_grad_(True)
    H2 = (H + 0.05 * float(Htr.std()) * torch.randn_like(H)).detach().requires_grad_(True)
    g1, g2 = torch.autograd.grad(invariance_loss(torch.stack([proj(H1), proj(H2)])), [H1, H2])
    g_inv = (0.5 * (g1 + g2)).detach().cpu().numpy()

    steps = {}
    for name, g in (("sigreg", g_sig), ("invariance", g_inv)):
        steps[name] = []
        for eta in (float(e) for e in args.etas.split(",")):
            Hp = Htr - eta * g
            steps[name].append({"eta": eta, **class_gap(Hp, d.ytr, w),
                                "probe": probe(Hp, d.ytr, Hte, d.yte)})

    report = {
        "init": str(args.init), "sigreg_target": args.sigreg_target,
        "sigreg_power": args.sigreg_power, "start": base,
        "sigreg_gradient": projection_stats(g_sig, w, d.ytr),
        "invariance_gradient": projection_stats(g_inv, w, d.ytr),
        "one_step": steps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    s = report["sigreg_gradient"]
    one = [r for r in steps["sigreg"] if r["eta"] == 1.0]
    print(f"[grad] cos(g_sig, w)={s['cos_to_w_mean']:.4f}  mean g∥: normal="
          f"{s['g_parallel_mean_normal']:.3f} fault={s['g_parallel_mean_fault']:.4f}")
    if one:
        print(f"[grad] one step η=1: fisher {base['fisher_on_w']:.2f} -> {one[0]['fisher_on_w']:.2f}, "
              f"accuracy {base['probe']['accuracy']:.3f} -> {one[0]['probe']['accuracy']:.3f}")
    print(args.out)
