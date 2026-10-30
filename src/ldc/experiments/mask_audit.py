"""Do the two LeJEPA views share the fault signal? (section 4 of the post)

    ldc mask-audit --data-root $DATA --out results/mask_audit.json

Draws two context views per window with the training masks (sensor blocks,
15-35% of the sensors each, disjoint) and reports the share of the top-k
label-correlated features visible in *both* views, and how well a linear probe
does on each view's visible features alone (invisible features zeroed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..data import load_probe_data
from ..model.masking import MaskCollator
from ..train_ssl import mask_config


def label_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y0 = (y == 0).astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    return (Xs * (y0[:, None] - y0.mean())).mean(axis=0) / (y0.std() + 1e-12)


def probe_visible(X, y, keep, train_idx, test_idx) -> dict:
    M = np.zeros_like(X)
    for i, idx in enumerate(keep):
        M[i, np.asarray(idx, dtype=int)] = 1.0
    Xv = X * M
    sc = StandardScaler().fit(Xv[train_idx])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(Xv[train_idx]), y[train_idx])
    proba = clf.predict_proba(sc.transform(Xv[test_idx]))[:, 1]
    return {"accuracy": float(accuracy_score(y[test_idx], (proba >= 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(y[test_idx], proba)),
            "mean_visible_frac": float(M.mean())}


def audit(X, y, collator: MaskCollator, top_k=32, n_eval=4000, seed=0) -> dict:
    np.random.seed(seed)  # the collator draws its sensors from the global RNG
    rng = np.random.RandomState(seed)
    n = min(n_eval, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs, ys, D = X[idx], y[idx], X.shape[1]
    corr = np.abs(label_corr(X, y))
    top = set(map(int, np.argsort(-corr)[:top_k]))

    with collator._itr_counter.get_lock():
        collator._itr_counter.value = seed + 1
    _, ctx, _ = collator([(np.zeros(D, dtype=np.float32),) for _ in range(n)])
    v0 = [ctx[0][i].numpy() for i in range(n)]
    v1 = [ctx[1][i].numpy() for i in range(n)]
    shared = [len(set(map(int, a)) & set(map(int, b)) & top) / top_k for a, b in zip(v0, v1)]

    perm = rng.permutation(n)
    te, tr = perm[: max(1, n // 5)], perm[max(1, n // 5) :]
    return {
        "n": n, "top_k": top_k,
        "mean_shared_topk_frac": float(np.mean(shared)),
        "mean_view_visible_frac": float(np.mean([len(a) / D for a in v0])),
        "probe_full": probe_visible(Xs, ys, [np.arange(D)] * n, tr, te),
        "probe_view0": probe_visible(Xs, ys, v0, tr, te),
        "probe_view1": probe_visible(Xs, ys, v1, tr, te),
    }


def add_args(ap) -> None:
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)


def main(args) -> None:
    d = load_probe_data(args.data_root)
    X, y = np.concatenate([d.Xtr, d.Xte]), np.concatenate([d.ytr, d.yte])
    report = audit(X, y, MaskCollator(mask_config(0.0), X.shape[1]), seed=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[mask-audit] top-{report['top_k']} features shared by both views: "
          f"{report['mean_shared_topk_frac']:.3f}; probe accuracy view0="
          f"{report['probe_view0']['accuracy']:.3f} view1={report['probe_view1']['accuracy']:.3f} "
          f"all features={report['probe_full']['accuracy']:.3f}")
    print(args.out)
