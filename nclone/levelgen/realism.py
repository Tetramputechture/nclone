"""How far a level sits from the manifold of human-authored levels.

Motivation: our procedural buckets drifted a long way from the maps the game
ships and players build, and nothing measured it. Difficulty was parameterised as
a scalar knob per generator (mine count, corridor width, tile-noise rate), so
making a level "harder" pushed it *off* the human manifold rather than along it
-- the hard tiers converge on uniform tile noise, which no human map resembles.

This module makes that drift a number.

**Why not a per-metric min/max box.** The obvious gate -- accept if every feature
lies within the corpus p1..p99 -- does not work, and measuring it is how we found
out. The human corpus has troll/art levels in its tails, so the marginal bounds
are nearly vacuous (``row_repeat`` p99 = 1.0, ``partial_entropy`` p99 = 4.63) and
admit our most degenerate output unchanged. What is actually unhuman is the
*combination*: a level that is 96% solid AND perfectly periodic AND maximum tile
entropy AND carries two entities is nowhere in 250k human levels, though each
term alone is unremarkable. So the score is a joint one -- distance to the
nearest human levels in a robustly-scaled feature space.

The threshold is calibrated on **held-out human levels** (their p99), never tuned
to reject our generators; that they are rejected is the finding, not the design.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

GRID_W = 42
GRID_H = 23
TILE_START = 184
TILE_END = 1150
ENT_START = 1235
ENT_STRIDE = 5
GOLD_TYPE = 2

#: Order is load-bearing -- it is the column order of the calibration file.
FEATURE_KEYS: Tuple[str, ...] = (
    "solid_frac",
    "partial_entropy",
    "row_repeat",
    "col_repeat",
    "log_entities",
    "log_gold",
)

#: Shipped calibration produced by ``scripts/levelgen/calibrate_realism.py``.
CALIBRATION_PATH = Path(__file__).with_name("realism_calibration.npz")

#: Fewer than this many non-trivial tiles and the entropy estimate is noise, so
#: report 0 rather than a spuriously low value off 2 samples.
_MIN_PARTIAL_TILES = 8


def level_features(map_data: Sequence[int]) -> List[float]:
    """The six raw features, in :data:`FEATURE_KEYS` order.

    Chosen because each one names a *specific* way our generators leave the
    manifold, so a failing score is directly actionable:

    * ``solid_frac`` -- our buckets are bimodal, either a cave bored through
      near-solid rock (~0.95) or platforms in a void (~0.0); humans sit ~0.5.
    * ``partial_entropy`` -- entropy over slope/curve tile ids. Humans use a
      small coherent palette (~2.3 bits, ~6 ids per map); uniform-random tile
      spray saturates near log2(36) = 5.17 and produces geometry with no
      continuous surface to carry momentum across.
    * ``row_repeat`` / ``col_repeat`` -- fraction of rows/columns that are exact
      duplicates. Lattice-regular output approaches 1.0; humans ~0.4/0.6.
    * ``log_entities`` / ``log_gold`` -- humans populate maps (median 111
      entities, gold in 99% of maps); our buckets ship 2-4 entities and no gold.
      Logged because the raw counts are heavy-tailed.
    """
    tiles = list(map_data[TILE_START:TILE_END])
    grid = [tiles[r * GRID_W : (r + 1) * GRID_W] for r in range(GRID_H)]

    solid = sum(1 for row in grid for v in row if v != 0)

    partial = Counter(v for row in grid for v in row if v >= 2)
    n_partial = sum(partial.values())
    if n_partial >= _MIN_PARTIAL_TILES:
        entropy = -sum(
            (v / n_partial) * math.log2(v / n_partial) for v in partial.values()
        )
    else:
        entropy = 0.0

    row_counts = Counter(tuple(row) for row in grid)
    col_counts = Counter(
        tuple(grid[y][x] for y in range(GRID_H)) for x in range(GRID_W)
    )

    ents: Counter = Counter()
    i = ENT_START
    n = len(map_data)
    while i + ENT_STRIDE <= n:
        t = map_data[i]
        if t == 0:
            break
        ents[t] += 1
        i += ENT_STRIDE

    return [
        solid / (GRID_W * GRID_H),
        entropy,
        sum(v for v in row_counts.values() if v > 1) / GRID_H,
        sum(v for v in col_counts.values() if v > 1) / GRID_W,
        math.log1p(sum(ents.values())),
        math.log1p(ents.get(GOLD_TYPE, 0)),
    ]


class RealismModel:
    """Nearest-neighbour distance to a reference set of human levels."""

    def __init__(self, center, scale, reference, threshold: float, k: int = 5):
        import numpy as np

        self.center = np.asarray(center, dtype="float64")
        self.scale = np.asarray(scale, dtype="float64")
        #: Already normalised, so scoring is one broadcast subtract.
        self.reference = np.asarray(reference, dtype="float64")
        self.threshold = float(threshold)
        self.k = int(k)

    def normalize(self, feats: Sequence[float]):
        import numpy as np

        return (np.asarray(feats, dtype="float64") - self.center) / self.scale

    def score(self, map_data: Sequence[int]) -> float:
        """Root-mean-square distance to the ``k`` nearest human levels.

        0 means "sits on top of real levels"; larger is further off-manifold.
        Compare against :attr:`threshold`.
        """
        return self.score_features(level_features(map_data))

    def score_features(self, feats: Sequence[float]) -> float:
        import numpy as np

        d2 = ((self.reference - self.normalize(feats)) ** 2).sum(axis=1)
        k = min(self.k, d2.shape[0])
        return float(math.sqrt(np.partition(d2, k - 1)[:k].mean()))

    def is_realistic(
        self, map_data: Sequence[int], threshold: Optional[float] = None
    ) -> bool:
        thr = self.threshold if threshold is None else threshold
        return self.score(map_data) <= thr

    def explain(self, map_data: Sequence[int]) -> Dict[str, object]:
        """Score plus the per-feature normalised deviation from the corpus centre.

        The largest-magnitude entries name which properties to change; a gate
        that only says "rejected" cannot be acted on.
        """
        feats = level_features(map_data)
        z = self.normalize(feats)
        return {
            "score": self.score_features(feats),
            "threshold": self.threshold,
            "realistic": self.score_features(feats) <= self.threshold,
            "features": dict(zip(FEATURE_KEYS, feats)),
            "deviation": dict(zip(FEATURE_KEYS, (float(v) for v in z))),
        }


@lru_cache(maxsize=1)
def load_default() -> RealismModel:
    """The shipped calibration. Raises if it has not been generated yet."""
    import numpy as np

    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"No realism calibration at {CALIBRATION_PATH}. Generate it with:\n"
            "  python -m scripts.levelgen.calibrate_realism "
            "--corpus datasets/levelgen/corpus_full"
        )
    z = np.load(CALIBRATION_PATH)
    return RealismModel(
        center=z["center"],
        scale=z["scale"],
        reference=z["reference"],
        threshold=float(z["threshold"]),
        k=int(z["k"]),
    )


def realism_score(map_data: Sequence[int]) -> float:
    return load_default().score(map_data)


def realism_ok(map_data: Sequence[int], threshold: Optional[float] = None) -> bool:
    return load_default().is_realistic(map_data, threshold)
