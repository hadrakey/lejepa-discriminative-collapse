"""Random feature masks that define the LeJEPA views of a row.

Per batch the collator draws a context size and a target size, then for every
row ``num_context`` disjoint context masks and ``num_target`` target masks
(targets may overlap each other, not the contexts). LeJEPA uses all of them as
views. Inspired by I-JEPA's multiblock masks.

Two sampling units:
- ``feature``: a unit is one feature column.
- ``sensor_block``: a unit is one sensor, i.e. all of its windowed statistics
  (Tennessee Eastman Rieth features, D = n_sensors * n_stats).

``shared_core > 0`` makes every view = shared core ∪ private units instead of
disjoint contexts.

Randomness: mask sizes come from a torch Generator seeded by a batch counter;
which units are picked comes from the global NumPy RNG (seed it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import torch


@dataclass
class MaskConfig:
    num_context: int = 2
    num_target: int = 2
    min_context_share: float = 0.15
    max_context_share: float = 0.35
    min_target_share: float = 0.15
    max_target_share: float = 0.35
    mode: str = "feature"  # feature | sensor_block
    n_sensors: int = 52
    n_stats: int = 6
    feature_layout: str = "stat_major"  # stat_major | sensor_major
    shared_core: float = 0.0
    # False reproduces T-JEPA, where all target masks of a row were identical
    # (array aliasing). True draws genuinely different target views.
    distinct_targets: bool = False


def sensor_feature_indices(sensor_ids, *, n_sensors: int, n_stats: int, layout: str) -> np.ndarray:
    """Expand sensor ids to flat feature indices for windowed-statistics features.

    stat_major:   [mean×S | std×S | ... | max×S]   (prepare_te_rieth layout)
    sensor_major: [s0_mean, s0_std, ..., s1_mean, ...]
    """
    sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
    if layout == "stat_major":
        out = np.concatenate([sensor_ids + s * n_sensors for s in range(n_stats)])
    elif layout == "sensor_major":
        out = np.concatenate([sensor_ids * n_stats + s for s in range(n_stats)])
    else:
        raise ValueError(f"unknown mask layout: {layout}")
    return np.unique(out)


class MaskCollator:
    def __init__(self, cfg: MaskConfig, num_features: int):
        assert cfg.min_context_share < cfg.max_context_share, "min context share must be < max"
        assert cfg.min_target_share < cfg.max_target_share, "min target share must be < max"
        assert 0.0 <= cfg.shared_core <= 1.0, "shared_core must be in [0, 1]"
        self.cfg = cfg
        self.num_features = num_features
        self._itr_counter = Value("i", -1)

        if cfg.mode == "feature":
            self.n_units = num_features
            self.unit_to_features = None
        elif cfg.mode == "sensor_block":
            expected = cfg.n_sensors * cfg.n_stats
            if num_features != expected:
                raise ValueError(
                    f"sensor_block expects D={expected} "
                    f"(n_sensors={cfg.n_sensors} x n_stats={cfg.n_stats}), got {num_features}"
                )
            self.n_units = cfg.n_sensors
            self.unit_to_features = {
                s: sensor_feature_indices(
                    [s], n_sensors=cfg.n_sensors, n_stats=cfg.n_stats, layout=cfg.feature_layout
                )
                for s in range(cfg.n_sensors)
            }
        else:
            raise ValueError(f"mask mode must be 'feature' or 'sensor_block', got {cfg.mode}")

        self.min_context = max(1, round(self.n_units * cfg.min_context_share))
        self.max_context = round(self.n_units * cfg.max_context_share)
        self.min_target = max(1, round(self.n_units * cfg.min_target_share))
        self.max_target = round(self.n_units * cfg.max_target_share)
        too_close = "max and min mask shares are too close for this number of features"
        assert self.max_context > self.min_context, too_close
        assert self.max_target > self.min_target, too_close

    @property
    def num_views(self) -> int:
        return self.cfg.num_context + self.cfg.num_target

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_size(self, gen: torch.Generator, lo: int, hi: int) -> int:
        n = math.inf
        while n > self.n_units:
            n = lo + np.round((hi - lo) * torch.rand(1, generator=gen).numpy())
        return int(n[0])

    def __call__(self, batch):
        rows = [b[0] for b in batch]
        gen = torch.Generator()
        gen.manual_seed(self.step())
        while True:
            n_ctx = self._sample_size(gen, self.min_context, self.max_context)
            n_trg = self._sample_size(gen, self.min_target, self.max_target)
            if self.cfg.shared_core > 0.0:
                break
            if self.cfg.num_context * n_ctx + n_trg <= self.n_units:
                break

        ctx, trg = zip(*(self._masks_for_row(n_ctx, n_trg) for _ in rows))
        collate = torch.utils.data.default_collate
        return collate(rows), collate(list(ctx)), collate(list(trg))

    def _expand(self, units: np.ndarray) -> np.ndarray:
        if self.unit_to_features is None:
            return np.asarray(units, dtype=np.int64)
        return np.unique(np.concatenate([self.unit_to_features[int(u)] for u in units]))

    def _masks_for_row(self, n_ctx: int, n_trg: int):
        if self.cfg.shared_core > 0.0:
            return self._shared_core_masks(n_ctx)
        units = np.arange(self.n_units)
        ctx = []
        for _ in range(self.cfg.num_context):
            np.random.shuffle(units)
            chosen = units[:n_ctx]
            ctx.append(self._expand(chosen).copy())
            units = np.setdiff1d(units, chosen)
        trg = []
        pool = units.copy()
        for _ in range(self.cfg.num_target):
            np.random.shuffle(pool)
            k = min(n_trg, len(pool))
            if k == 0:
                raise ValueError("no units left for a target mask")
            trg.append(self._expand(pool[:k]).copy())
        if not self.cfg.distinct_targets and self.unit_to_features is None:
            # T-JEPA kept views of `pool`, so every target mask ended up equal to
            # the last draw. Reproduce that explicitly (same RNG consumption).
            # Sensor-block masks were expanded into new arrays, so the aliasing
            # never happened there: their target masks are distinct.
            trg = [trg[-1].copy() for _ in trg]
        return ctx, trg

    def _shared_core_masks(self, n_view: int):
        """Each view = shared core ∪ private units, same size for all views."""
        shared_n = max(0, min(int(round(self.cfg.shared_core * n_view)), n_view))
        private_n = n_view - shared_n
        units = np.arange(self.n_units)
        np.random.shuffle(units)
        shared, remaining = units[:shared_n], units[shared_n:]
        replace = private_n * self.num_views > len(remaining)
        if not replace:
            pool = remaining.copy()
            np.random.shuffle(pool)
        views = []
        for _ in range(self.num_views):
            if private_n == 0:
                chosen = shared
            elif replace:
                np.random.shuffle(remaining)
                chosen = np.concatenate([shared, remaining[:private_n]])
            else:
                chosen = np.concatenate([shared, pool[:private_n]])
                pool = pool[private_n:]
            views.append(self._expand(np.unique(chosen)))
        return views[: self.cfg.num_context], views[self.cfg.num_context :]
