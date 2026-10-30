"""Everything the post measures on an embedding matrix H (one row per window).

- ``probe``: class-balanced logistic regression on standardized H (fit on train,
  scored on test). Its accuracy is only comparable between representations:
  always answering "fault" scores ~0.944 on the test split.
- ``geometry``: per-dimension variance and effective rank.
- ``fit_w``, ``class_gap``, ``r_disc``: the supervised discriminant direction w
  and the class separation along it (Fisher criterion).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def binary_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }


def probe(Htr, ytr, Hte, yte) -> dict:
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Htr), ytr)
    proba = clf.predict_proba(sc.transform(Hte))[:, 1]
    return binary_metrics(yte, (proba >= 0.5).astype(int), proba)


def effective_rank(H: np.ndarray, eps: float = 1e-12) -> float:
    """exp(entropy) of the normalized singular values (Roy & Vetterli, 2007).

    Scale-invariant: it describes the shape of the spectrum, not its size.
    """
    s = np.linalg.svd(H - H.mean(axis=0, keepdims=True), compute_uv=False)
    p = np.clip(s, eps, None)
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(H: np.ndarray) -> float:
    """(Σλ)² / Σλ² over covariance eigenvalues λ: another effective dimension."""
    lam = np.linalg.svd(H - H.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


def geometry(H: np.ndarray) -> dict:
    var = H.var(axis=0)
    s = np.linalg.svd(H - H.mean(0, keepdims=True), compute_uv=False)
    s = s / (s.sum() + 1e-12)
    return {
        "dim": int(H.shape[1]),
        "var_mean": float(var.mean()),
        "var_min": float(var.min()),
        "frac_var_lt_1e-3": float((var < 1e-3).mean()),
        "eff_rank": effective_rank(H),
        "participation_ratio": participation_ratio(H),
        "spectrum_top5": [float(x) for x in s[:5]],
    }


def fit_w(Htr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    """Discriminant direction: coefficients of the balanced probe (standardized H)."""
    Hs = StandardScaler().fit_transform(Htr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Hs, ytr)
    return clf.coef_.reshape(-1).astype(np.float64)


def class_gap(H: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Separation of the class means along w, and the Fisher criterion
    (mean difference)² / (average within-class variance)."""
    s = H @ (w / (np.linalg.norm(w) + 1e-12))
    m0, m1 = s[y == 0].mean(), s[y == 1].mean()
    pooled = 0.5 * (s[y == 0].var() + s[y == 1].var()) + 1e-12
    return {"mean_diff_on_w": float(m1 - m0), "fisher_on_w": float((m1 - m0) ** 2 / pooled)}


def r_disc(H: np.ndarray, w: np.ndarray) -> dict:
    """Share of the total variance that lies along w."""
    Hc = H - H.mean(axis=0, keepdims=True)
    n = max(H.shape[0] - 1, 1)
    along = float(((Hc @ w) ** 2).sum() / n)
    tr = float((Hc**2).sum() / n)
    return {
        "R_disc": along / float((w @ w) * tr + 1e-12),
        "var_along_w": along / float(w @ w + 1e-12),
        "tr_cov": tr,
    }
