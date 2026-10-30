"""Rebuild the post's tables and figures from the run folders.

    ldc report --runs runs --out results

Expects the layout written by ``scripts/reproduce.sh``; anything missing is
skipped. Writes ``<out>/tables.md`` and ``<out>/fig{1,2,3}_*.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (42, 43, 44)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _final(diag: dict) -> dict:
    return diag["trajectory"][-1]


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header, rows) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def _sci(x: float) -> str:
    return f"{x:.2g}" if x < 0.01 else f"{x:.3g}"


class Runs:
    def __init__(self, root: Path):
        self.root = root

    def diag(self, name: str, seed: int = 42):
        return _load(self.root / f"s{seed}" / name / "diagnostics.json")

    def scratch(self, name: str):
        return _load(self.root / "scratch" / name / "diagnostics.json")

    def result(self, name: str):
        return _load(self.root / "results" / name)


def tables(r: Runs) -> list[str]:
    md = ["# Tables of the post", ""]
    ssl = r.scratch("share0")
    if ssl:
        st, fin = ssl["start"], _final(ssl)  # start = the supervised encoder passed as --init
        md += ["## Supervised vs self-supervised (seed 42)", ""]
        rows = [["Supervised encoder", st["probe"]["accuracy"], st["probe"]["roc_auc"], st["geometry"]["eff_rank"],
                 st["geometry"]["var_mean"]],
                ["Self-supervised encoder", fin["probe"]["accuracy"], fin["probe"]["roc_auc"],
                 fin["geometry"]["eff_rank"], fin["geometry"]["var_mean"]]]
        if "z" in fin:
            rows.append(["Self-supervised projector", fin["z"]["probe"]["accuracy"], fin["z"]["probe"]["roc_auc"],
                         fin["z"]["geometry"]["eff_rank"], fin["z"]["geometry"]["var_mean"]])
        md += _table(["Representation", "Accuracy", "ROC-AUC", "Effective rank", "Mean variance"],
                     [[n, f"{a:.3f}", f"{b:.3f}", f"{c:.2f}", _sci(v)] for n, a, b, c, v in rows])

    shares = [(s, r.scratch(f"share{s}")) for s in (0, 25, 50, 75)]
    if any(d for _, d in shares):
        md += ["## Shared core (from scratch, seed 42)", ""]
        md += _table(["Shared core", "Accuracy", "ROC-AUC", "Effective rank", "Mean variance"],
                     [[f"{s}%", f"{_final(d)['probe']['accuracy']:.3f}", f"{_final(d)['probe']['roc_auc']:.3f}",
                       f"{_final(d)['geometry']['eff_rank']:.2f}", _sci(_final(d)["geometry"]["var_mean"])]
                      for s, d in shares if d])

    audit = r.result("mask_audit.json")
    if audit:
        md += ["## Mask audit", "",
               f"Top-{audit['top_k']} label-correlated features visible in both views: "
               f"{audit['mean_shared_topk_frac']:.3f}. Probe accuracy: view 0 "
               f"{audit['probe_view0']['accuracy']:.3f}, view 1 {audit['probe_view1']['accuracy']:.3f}, "
               f"all features {audit['probe_full']['accuracy']:.3f}.", ""]

    both = r.diag("both")
    if both:
        md += ["## Continuation from the supervised encoder: trajectory (seed 42)", ""]
        md += _table(["Epoch", "Accuracy", "ROC-AUC", "Effective rank", "Mean variance"],
                     [[t["epoch"], f"{t['probe']['accuracy']:.3f}", f"{t['probe']['roc_auc']:.3f}",
                       f"{t['geometry']['eff_rank']:.2f}", _sci(t["geometry"]["var_mean"])]
                      for t in both["trajectory"]])

    abl = [(n, r.diag(k)) for n, k in (("Invariance only", "inv_only"), ("SIGReg only", "sig_only"),
                                       ("Invariance + SIGReg", "both"))]
    if any(d for _, d in abl):
        start = next(d for _, d in abl if d)["start"]["probe"]
        md += ["## Which loss term (seed 42, epoch 50)", ""]
        md += _table(["Training objective", "Accuracy", "ROC-AUC"],
                     [["*Supervised start*", f"*{start['accuracy']:.3f}*", f"*{start['roc_auc']:.3f}*"]]
                     + [[n, f"{_final(d)['probe']['accuracy']:.3f}", f"{_final(d)['probe']['roc_auc']:.3f}"]
                        for n, d in abl if d])

    seed_rows = []
    for name, key in (("Invariance only", "inv_only"), ("Invariance + SIGReg", "both"),
                      ("SIGReg blocked at h", "detach_h")):
        deltas = []
        for s in SEEDS:
            d = r.diag(key, s)
            deltas.append(f"{_final(d)['probe']['roc_auc'] - d['start']['probe']['roc_auc']:+.3f}" if d else "n/a")
        seed_rows.append([name, " / ".join(deltas)])
    md += ["## ROC-AUC change after 50 epochs, from each seed's own start", ""]
    md += _table(["Objective", "Seeds " + " / ".join(map(str, SEEDS))], seed_rows)
    starts = [r.diag("both", s) for s in SEEDS]
    md += ["Supervised starting points (ROC-AUC): "
           + ", ".join(f"seed {s}: {d['start']['probe']['roc_auc']:.3f}" for s, d in zip(SEEDS, starts) if d), ""]

    paths = [("Invariance only (no SIGReg)", "inv_only"), ("Full LeJEPA", "both"),
             ("SIGReg gradient blocked at h", "detach_h"), ("Encoder frozen", "freeze_enc"),
             ("SIGReg applied on h, no projector", "on_h")]
    rows = [(n, r.diag(k)) for n, k in paths]
    if any(d for _, d in rows):
        md += ["## Gradient paths (seed 42, epoch 50)", ""]
        md += _table(["Variant", "Accuracy", "ROC-AUC", "Effective rank of h", "Mean variance of h", "Fisher along w"],
                     [[n, f"{_final(d)['probe']['accuracy']:.3f}", f"{_final(d)['probe']['roc_auc']:.3f}",
                       f"{_final(d)['geometry']['eff_rank']:.2f}", _sci(_final(d)["geometry"]["var_mean"]),
                       _sci(_final(d)["fisher_on_w"])] for n, d in rows if d])

    targets = [("Isotropic", "sig_only"), ("Power, p=1", "sig_power1"), ("Power, p=2", "sig_power2"),
               ("Linear", "sig_linear")]
    rows = [(n, r.diag(k)) for n, k in targets]
    if any(d for _, d in rows):
        md += ["## SIGReg target covariance (SIGReg only, seed 42, epoch 50)", ""]
        md += _table(["SIGReg target", "Accuracy", "ROC-AUC", "Fisher along w"],
                     [[n, f"{_final(d)['probe']['accuracy']:.3f}", f"{_final(d)['probe']['roc_auc']:.3f}",
                       f"{_final(d)['fisher_on_w']:.2f}"] for n, d in rows if d])

    rows = [(k, r.diag(f"k{k}") if k != 16 else r.diag("both")) for k in (2, 4, 8, 16)]
    if any(d for _, d in rows):
        md += ["## Projector dimension (seed 42, epoch 50)", ""]
        md += _table(["k", "Accuracy", "ROC-AUC", "Fisher along w"],
                     [[k, f"{_final(d)['probe']['accuracy']:.3f}", f"{_final(d)['probe']['roc_auc']:.3f}",
                       _sci(_final(d)["fisher_on_w"])] for k, d in rows if d])

    grads = [(n, r.result(f"grad_{t}.json")) for n, t in (("Isotropic", "iso"), ("Power, p=1", "power1"),
                                                         ("Power, p=2", "power2"), ("Linear", "linear"))]
    if any(g for _, g in grads):
        md += ["## SIGReg gradient on the supervised embedding (seed 42)", ""]
        rows = []
        for n, g in grads:
            if not g:
                continue
            s = g["sigreg_gradient"]
            one = next(x for x in g["one_step"]["sigreg"] if x["eta"] == 1.0)
            rows.append([n, f"{s['cos_to_w_mean']:+.4f}", f"{s['g_parallel_mean_normal']:+.3f}",
                         f"{s['g_parallel_mean_fault']:+.4f}",
                         f"{g['start']['fisher_on_w']:.2f} → {one['fisher_on_w']:.2f}",
                         f"{g['start']['probe']['accuracy']:.3f} → {one['probe']['accuracy']:.3f}"])
        md += _table(["Target", "cos(g, w)", "mean g∥ (normal)", "mean g∥ (fault)", "Fisher, one step η=1",
                      "Accuracy, one step η=1"], rows)
        inv = next(g for _, g in grads if g)["invariance_gradient"]["g_norm_mean"]
        sig = next(g for _, g in grads if g)["sigreg_gradient"]["g_norm_mean"]
        md += [f"Mean gradient norm: SIGReg {sig:.3g}, invariance {inv:.2g}.", ""]

    rows = []
    for s in SEEDS:
        a = r.result(f"affine_s{s}.json")
        if not a:
            continue
        for tag, run in a["runs"].items():
            t0, t1 = run["trajectory"][0], run["final"]
            rows.append([s, tag, f"{a['start']['probe']['roc_auc']:.3f}", f"{t1['probe_z']['roc_auc']:.3f}",
                         f"{100 * (1 - t1['loss_mean'] / t0['loss_mean']):.0f}%",
                         f"{t1['eff_rank_z']:.2f}", f"{t1['participation_ratio_z']:.2f}"])
    if rows:
        md += ["## Invertible affine control (h standardized)", ""]
        md += _table(["Seed", "Target", "ROC-AUC on h", "ROC-AUC on z", "SIGReg loss reduction",
                      "Effective rank of z", "Participation ratio of z"], rows)
    return md


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


def figures(r: Runs, out: Path) -> list[Path]:
    _style()
    made = []
    ssl = r.scratch("share0")
    if ssl:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        st = ssl["start"]
        ax.scatter(st["geometry"]["eff_rank"], st["probe"]["roc_auc"], s=70, color=BLUE, edgecolor=SURF, lw=2,
                   zorder=3, label="supervised")
        ax.annotate("Supervised encoder", (st["geometry"]["eff_rank"], st["probe"]["roc_auc"]),
                    xytext=(10, -4), textcoords="offset points")
        pts = [("SSL encoder", _final(ssl)["geometry"]["eff_rank"], _final(ssl)["probe"]["roc_auc"])]
        if "z" in _final(ssl):
            pts.append(("SSL projector", _final(ssl)["z"]["geometry"]["eff_rank"], _final(ssl)["z"]["probe"]["roc_auc"]))
        for s in (25, 50, 75):
            d = r.scratch(f"share{s}")
            if d:
                pts.append((f"SSL, {s}% shared core", _final(d)["geometry"]["eff_rank"], _final(d)["probe"]["roc_auc"]))
        for i, (lab, x, y) in enumerate(pts):
            ax.scatter(x, y, s=70, color=ORANGE, edgecolor=SURF, lw=2, zorder=3,
                       label="self-supervised (LeJEPA)" if i == 0 else None)
            ax.annotate(lab, (x, y), xytext=(10, -4), textcoords="offset points")
        ax.legend(frameon=False, loc="lower left")
        ax.set_xlabel("Effective rank of the embedding")
        ax.set_ylabel("Linear-probe ROC-AUC")
        ax.set_xlim(0, 19)
        ax.set_title("More dimensions, less fault information", loc="left", fontweight="bold")
        made.append(_save(fig, out / "fig1_rank_vs_auc.png"))

    runs = [("Invariance only", "inv_only", BLUE), ("Invariance + SIGReg", "both", AQUA), ("SIGReg only", "sig_only", ORANGE)]
    if any(r.diag(k) for _, k, _ in runs):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        start = None
        for lab, key, color in runs:
            d = r.diag(key)
            if not d:
                continue
            start = d["start"]["probe"]["roc_auc"]
            ep = [0] + [t["epoch"] for t in d["trajectory"]]
            roc = [start] + [t["probe"]["roc_auc"] for t in d["trajectory"]]
            ax.plot(ep, roc, color=color, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
            ax.annotate(lab, (ep[-1], roc[-1]), xytext=(8, -4), textcoords="offset points")
        ax.axhline(start, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.legend(frameon=False, loc="lower left")
        ax.set_xlim(0, 66)
        ax.set_xlabel("Self-supervised epochs, starting from the supervised encoder")
        ax.set_ylabel("Linear-probe ROC-AUC")
        ax.set_title("SIGReg erodes a representation that was already useful", loc="left", fontweight="bold")
        made.append(_save(fig, out / "fig2_continuation.png"))

    targets = [("Isotropic", "sig_only"), ("Power law, p=1", "sig_power1"), ("Power law, p=2", "sig_power2"),
               ("Linear decay", "sig_linear")]
    rows = [(n, r.diag(k)) for n, k in targets]
    rows = [(n, d) for n, d in rows if d]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        y = list(range(len(rows)))[::-1]
        vals = [_final(d)["probe"]["roc_auc"] for _, d in rows]
        ax.hlines(y, 0.6, vals, color=GRID, lw=2)
        ax.scatter(vals, y, s=80, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(9, 0), textcoords="offset points", va="center")
        ref = rows[0][1]["start"]["probe"]["roc_auc"]
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(y, [n for n, _ in rows])
        ax.set_xlim(0.6, 0.9)
        ax.set_ylim(-0.8, len(rows) - 0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Linear-probe ROC-AUC after 50 epochs of SIGReg only")
        ax.set_title("Relaxing isotropy helps, but not enough", loc="left", fontweight="bold")
        made.append(_save(fig, out / "fig3_sigreg_targets.png"))
    return made


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def add_args(ap) -> None:
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")


def main(args) -> None:
    r, out = Runs(Path(args.runs)), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(tables(r)))
    print(out / "tables.md")
    for p in figures(r, out):
        print(p)
