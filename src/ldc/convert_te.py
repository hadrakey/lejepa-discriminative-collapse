"""Rieth multi-run TE → windowed statistical features (tabular, lit. recipe).

Corpus
------
Rieth et al. (2017) Harvard Dataverse doi:10.7910/DVN/6C3JR1
  TEP_FaultFree_{Training,Testing}.RData
  TEP_Faulty_{Training,Testing}.RData

Preprocessing (tabular, as in the literature):
  sliding window → per-sensor {mean, std, skew, kurtosis, min, max}
  → D = 6 × 52 = 312 features per window.
  Windows never cross simulationRun boundaries.
  Faulty window labelled faulty iff >50% of samples are post onset.

Layout per row:
  [id, f_0 .. f_{D-1}, attack_cat, label]
  id = simulationRun * 10000 + window_start   (for run-grouped splits)
  attack_cat ∈ {0..20}  (Rieth has faults 0..20; no IDV21)
  label = 0 if Normal else 1

Outputs
-------
  {out_root}/train/{Normal,IDV01,..,IDV20}.npy
  {out_root}/test/{...}.npy
  {out_root}/meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pyreadr
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install pyreadr") from e

# Rieth sampling: 3 min/sample
ONSET_TRAIN = 20  # 1 h into faulty training runs
ONSET_TEST = 160  # 8 h into faulty testing runs
N_SENSORS = 52
STAT_NAMES = ("mean", "std", "skew", "kurtosis", "min", "max")
ID_RUN_MULT = 10_000  # id = run * MULT + window_start


def _skew_kurt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column skew / excess kurtosis (population), no scipy required."""
    # x: (T, D)
    mu = x.mean(axis=0)
    xc = x - mu
    m2 = np.mean(xc**2, axis=0)
    m3 = np.mean(xc**3, axis=0)
    m4 = np.mean(xc**4, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(m2 > 0, m3 / (m2**1.5), 0.0)
        kurt = np.where(m2 > 0, m4 / (m2**2) - 3.0, 0.0)
    return skew.astype(np.float32), kurt.astype(np.float32)


def window_features(
    X: np.ndarray,
    *,
    window: int,
    stride: int,
    onset: int | None,
    fault_majority: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats [n_win, 312], keep_mask as faulty labels 0/1 for windows).

    If onset is None (normal run), all windows are label 0 and kept.
    If onset is set, only windows with fault fraction > fault_majority are kept
    (and labelled 1).
    """
    t, d = X.shape
    assert d == N_SENSORS
    starts = list(range(0, t - window + 1, stride))
    feats = []
    labels = []
    starts_kept = []
    for s in starts:
        seg = X[s : s + window]
        if onset is None:
            lab = 0
            keep = True
        else:
            post = np.arange(s, s + window) >= onset
            frac = float(post.mean())
            lab = 1 if frac > fault_majority else 0
            keep = lab == 1
        if not keep:
            continue
        mu = seg.mean(axis=0)
        sd = seg.std(axis=0)
        sk, ku = _skew_kurt(seg)
        mn = seg.min(axis=0)
        mx = seg.max(axis=0)
        feats.append(np.concatenate([mu, sd, sk, ku, mn, mx], axis=0))
        labels.append(lab)
        starts_kept.append(s)
    if not feats:
        return (
            np.zeros((0, N_SENSORS * len(STAT_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return np.stack(feats, axis=0).astype(np.float32), np.asarray(
        starts_kept, dtype=np.int32
    )


def _pack(feats: np.ndarray, run_ids: np.ndarray, starts: np.ndarray, class_id: int) -> np.ndarray:
    n, d = feats.shape
    out = np.zeros((n, d + 3), dtype=np.float32)
    out[:, 0] = (run_ids.astype(np.float64) * ID_RUN_MULT + starts).astype(np.float32)
    out[:, 1 : 1 + d] = feats
    out[:, 1 + d] = float(class_id)
    out[:, 2 + d] = 0.0 if class_id == 0 else 1.0
    return out


def _load_rdata(path: Path, key: str) -> np.ndarray:
    res = pyreadr.read_r(str(path))
    if key not in res:
        # tolerate alternate key names
        if len(res) == 1:
            df = next(iter(res.values()))
        else:
            raise KeyError(f"{path}: expected key {key}, got {list(res)}")
    else:
        df = res[key]
    return df


def _process_split(
    free_path: Path,
    faulty_path: Path,
    free_key: str,
    faulty_key: str,
    *,
    onset: int,
    window: int,
    stride: int,
    max_runs: int | None,
    rng: np.random.Generator,
) -> dict[int, list[np.ndarray]]:
    """Return class_id → list of packed window blocks."""
    print(f"  loading {free_path.name} …")
    free = _load_rdata(free_path, free_key)
    print(f"  loading {faulty_path.name} …")
    faulty = _load_rdata(faulty_path, faulty_key)

    # columns: faultNumber, simulationRun, sample, x1..x52
    free_runs = np.sort(free["simulationRun"].unique())
    faulty_runs = np.sort(faulty["simulationRun"].unique())
    if max_runs is not None:
        free_runs = rng.choice(free_runs, size=min(max_runs, len(free_runs)), replace=False)
        faulty_runs = rng.choice(
            faulty_runs, size=min(max_runs, len(faulty_runs)), replace=False
        )
        free_runs = np.sort(free_runs)
        faulty_runs = np.sort(faulty_runs)

    free = free.loc[free["simulationRun"].isin(free_runs)].copy()
    faulty = faulty.loc[faulty["simulationRun"].isin(faulty_runs)].copy()
    print(
        f"  kept runs free={len(free_runs)} faulty={len(faulty_runs)} "
        f"rows free={len(free)} faulty={len(faulty)}"
    )

    by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(21)}

    # Normal (fault 0)
    for run, block in free.groupby("simulationRun", sort=True):
        block = block.sort_values("sample")
        X = block.iloc[:, 3:].to_numpy(dtype=np.float32)
        feats, starts = window_features(X, window=window, stride=stride, onset=None)
        if len(feats) == 0:
            continue
        runs = np.full(len(feats), int(run), dtype=np.int32)
        by_class[0].append(_pack(feats, runs, starts, 0))

    # Faults 1..20
    for (fid, run), block in faulty.groupby(["faultNumber", "simulationRun"], sort=True):
        fid = int(fid)
        if fid < 1 or fid > 20:
            continue
        block = block.sort_values("sample")
        X = block.iloc[:, 3:].to_numpy(dtype=np.float32)
        feats, starts = window_features(
            X, window=window, stride=stride, onset=onset
        )
        if len(feats) == 0:
            continue
        runs = np.full(len(feats), int(run), dtype=np.int32)
        by_class[fid].append(_pack(feats, runs, starts, fid))

    return by_class


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ldc convert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder with the four .RData files")
    ap.add_argument("--out-root", type=Path, required=True, help="writes train/, test/, meta.json")
    ap.add_argument("--window", type=int, default=30, help="samples (~90 min @ 3min)")
    ap.add_argument("--stride", type=int, default=10, help="samples (~30 min)")
    ap.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="simulation runs kept per fault (0 = all 500); the post used 50",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    max_runs = None if args.max_runs in (None, 0) else args.max_runs
    rng = np.random.default_rng(args.seed)

    # Accept either renamed files or Dataverse download ids
    raw = args.raw
    name_map = {
        "fault_free_training": [
            "TEP_FaultFree_Training.RData",
            "f_3031241.RData",
        ],
        "fault_free_testing": [
            "TEP_FaultFree_Testing.RData",
            "f_3031240.RData",
        ],
        "faulty_training": [
            "TEP_Faulty_Training.RData",
            "f_3031242.RData",
        ],
        "faulty_testing": [
            "TEP_Faulty_Testing.RData",
            "f_3031243.RData",
        ],
    }

    def resolve(key: str) -> Path:
        for cand in name_map[key]:
            p = raw / cand
            if p.exists():
                return p
        raise FileNotFoundError(f"{key}: none of {name_map[key]} under {raw}")

    train_dir = args.out_root / "train"
    test_dir = args.out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Building TRAIN windows…")
    tr = _process_split(
        resolve("fault_free_training"),
        resolve("faulty_training"),
        "fault_free_training",
        "faulty_training",
        onset=ONSET_TRAIN,
        window=args.window,
        stride=args.stride,
        max_runs=max_runs,
        rng=rng,
    )
    print("Building TEST windows…")
    te = _process_split(
        resolve("fault_free_testing"),
        resolve("faulty_testing"),
        "fault_free_testing",
        "faulty_testing",
        onset=ONSET_TEST,
        window=args.window,
        stride=args.stride,
        max_runs=max_runs,
        rng=rng,
    )

    d_feat = N_SENSORS * len(STAT_NAMES)
    names = ["Normal"] + [f"IDV{i:02d}" for i in range(1, 21)]
    meta = {
        "source": "Rieth et al. 2017 Harvard Dataverse doi:10.7910/DVN/6C3JR1",
        "feature_recipe": "windowed_stats",
        "window": args.window,
        "stride": args.stride,
        "stats": list(STAT_NAMES),
        "n_features": d_feat,
        "n_sensors": N_SENSORS,
        "max_runs": max_runs,
        "seed": args.seed,
        "onset_train": ONSET_TRAIN,
        "onset_test": ONSET_TEST,
        "id_encoding": f"simulationRun*{ID_RUN_MULT}+window_start",
        "layout": f"[id, f0..f{d_feat - 1}, attack_cat, label]",
        "note": "No IDV21 in Rieth (faults 0..20). Tabular GeoSelect; not SOTA TS.",
        "classes": {},
    }

    for cid, name in enumerate(names):
        tr_blocks = tr.get(cid, [])
        te_blocks = te.get(cid, [])
        tr_arr = (
            np.concatenate(tr_blocks, axis=0)
            if tr_blocks
            else np.zeros((0, d_feat + 3), dtype=np.float32)
        )
        te_arr = (
            np.concatenate(te_blocks, axis=0)
            if te_blocks
            else np.zeros((0, d_feat + 3), dtype=np.float32)
        )
        np.save(train_dir / f"{name}.npy", tr_arr)
        np.save(test_dir / f"{name}.npy", te_arr)
        meta["classes"][name] = {
            "id": cid,
            "n_train": int(len(tr_arr)),
            "n_test": int(len(te_arr)),
        }
        print(f"{name:8s} train={len(tr_arr):6d} test={len(te_arr):6d}")

    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {train_dir} and {test_dir} (D={d_feat})")

