"""Tennessee Eastman (Rieth) windowed-statistics data, as written by ``ldc convert``.

A split folder holds ``Normal.npy`` and ``IDV01.npy`` .. ``IDV20.npy``; every
row is ``[id, f_0 .. f_311, attack_cat, label]`` with label 0 = normal, 1 = fault.
The file order fixes the row order, which seeded splits and shuffles depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

TE_CLASS_NAMES = ["Normal"] + [f"IDV{i:02d}" for i in range(1, 22)]
N_SENSORS, N_STATS = 52, 6


@dataclass
class Split:
    X: np.ndarray  # raw features, float32
    y: np.ndarray  # 0 = normal, 1 = fault
    ids: np.ndarray  # simulationRun * 10000 + window start


@lru_cache(maxsize=4)
def load_split(path: str | Path) -> Split:
    path = Path(path)
    parts = [np.load(path / f"{n}.npy") for n in TE_CLASS_NAMES if (path / f"{n}.npy").exists()]
    if not parts:
        raise FileNotFoundError(f"no class .npy files under {path} (run `ldc convert` first)")
    data = np.concatenate(parts, axis=0)
    return Split(X=data[:, 1:-2].astype(np.float32), y=data[:, -1].astype(int),
                 ids=data[:, 0].astype(np.float64))


@dataclass
class ProbeData:
    """Train and test rows MinMax-scaled on train, as the encoder saw them."""

    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


def load_probe_data(root: str | Path) -> ProbeData:
    tr, te = load_split(Path(root) / "train"), load_split(Path(root) / "test")
    scaler = MinMaxScaler().fit(tr.X)
    return ProbeData(
        Xtr=scaler.transform(tr.X).astype(np.float32), ytr=tr.y,
        Xte=scaler.transform(te.X).astype(np.float32), yte=te.y,
    )
