"""Probe, geometry and class separation of a run's checkpoints.

    ldc diagnose --data-root $DATA --run runs/s42/both --init runs/s42/sup/encoder_init.pth

For every ``epoch_<E>.pth`` in the run (and the starting encoder ``--init``):
linear-probe metrics on h, effective rank and variance of h, and the Fisher
criterion along w, the discriminant direction of the *starting* encoder; plus
probe and geometry of the projector output z when the checkpoint has one.
Writes ``<run>/diagnostics.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..data import load_probe_data
from ..embed import embed, embed_checkpoint, project, resolve_device
from ..metrics import class_gap, fit_w, geometry, probe, r_disc


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    found = [(int(re.search(r"epoch_(\d+)", p.name).group(1)), p) for p in run.glob("epoch_*.pth")]
    return sorted(found)


def describe(Htr, Hte, d, w) -> dict:
    return {"probe": probe(Htr, d.ytr, Hte, d.yte), "geometry": geometry(Htr),
            **class_gap(Htr, d.ytr, w), **r_disc(Htr, w)}


def add_args(ap) -> None:
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True, help="folder with epoch_<E>.pth files")
    ap.add_argument("--init", required=True, help="starting encoder (defines w)")
    ap.add_argument("--device", default="cuda:0")


def main(args) -> None:
    device = resolve_device(args.device)
    d = load_probe_data(args.data_root)
    Htr0, enc0, _ = embed_checkpoint(args.init, d.Xtr, device)
    w = fit_w(Htr0, d.ytr)
    report = {"init": str(args.init), "start": describe(Htr0, embed(d.Xte, enc0, device), d, w),
              "trajectory": []}
    for epoch, path in checkpoints(Path(args.run)):
        Htr, enc, proj = embed_checkpoint(path, d.Xtr, device)
        Hte = embed(d.Xte, enc, device)
        row = {"epoch": epoch, **describe(Htr, Hte, d, w)}
        if proj is not None:
            Ztr, Zte = project(Htr, proj, device), project(Hte, proj, device)
            row["z"] = {"probe": probe(Ztr, d.ytr, Zte, d.yte), "geometry": geometry(Ztr)}
        report["trajectory"].append(row)
        print(f"[diagnose] epoch {epoch:3d} roc_auc={row['probe']['roc_auc']:.3f} "
              f"acc={row['probe']['accuracy']:.3f} eff_rank={row['geometry']['eff_rank']:.2f} "
              f"var={row['geometry']['var_mean']:.2g} fisher={row['fisher_on_w']:.3f}")
    out = Path(args.run) / "diagnostics.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)
