"""Spatial hash cache for O(1) airborne state cost-to-go lookup.

During edge enumeration, waypoints are sampled along every simulated arc
and deduplicated per-(cell, destination) into a ``CompactWaypoints`` bundle
(three compact NumPy arrays).  This module takes those compact waypoints
and a Dijkstra ``d_star`` vector, then produces a spatial hash that maps
discretized ``(x, y, vx, vy)`` cells to the minimum cost-to-go observed at
any arc waypoint passing through that cell.  At runtime, an airborne state
can be looked up in O(1) to get a precomputed cost estimate without forward
simulation.

Bin sizes (see ``edge_enumeration``):
  - Position: 6 px
  - Velocity: 0.5 px / frame
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .edge_enumeration import (
    CompactWaypoints,
    POSITION_BIN_SIZE,
    VELOCITY_BIN_SIZE,
    _INV_POS_BIN,
    _INV_VEL_BIN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
def _bin_key(x: float, y: float, vx: float, vy: float) -> Tuple[int, int, int, int]:
    """Quantize a state into a spatial-hash key."""
    return (
        int(math.floor(x * _INV_POS_BIN)),
        int(math.floor(y * _INV_POS_BIN)),
        int(math.floor(vx * _INV_VEL_BIN)),
        int(math.floor(vy * _INV_VEL_BIN)),
    )


# ---------------------------------------------------------------------------
# Arc cache data structure
# ---------------------------------------------------------------------------
class ArcCache:
    """Spatial hash mapping airborne (x, y, vx, vy) to minimum cost-to-go.

    After ``build()``, the cache stores, for each quantized state cell, the
    minimum value of ``frames_to_land + d_star[destination_node]`` across all
    arc waypoints that pass through that cell.  The build path is fully
    vectorised in NumPy (``np.unique`` + ``np.minimum.at``); the final dict
    is only populated with the reduced per-cell min values (typically a few
    tens of thousands of entries) so the lookup path remains O(1).

    Attributes:
        _data: Mapping from bin key to minimum cost (float).
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[int, int, int, int], float] = {}

    @property
    def n_entries(self) -> int:
        return len(self._data)

    # ---- construction ----------------------------------------------------
    def build(
        self,
        waypoints: Union[CompactWaypoints, List],
        d_star: np.ndarray,
    ) -> None:
        """Populate the cache from compact waypoints and a d_star array.

        For each unique ``(cell_key, dst_node)`` pair present in ``waypoints``,
        computes the cost-to-go as ``frames_to_land + d_star[dst_node]`` and
        stores the per-cell minimum in the underlying dict.  Invalid entries
        (out-of-range dst, non-finite d_star, negative frames_to_land) are
        filtered before the reduction.

        Args:
            waypoints: ``CompactWaypoints`` bundle from ``enumerate_edges``.
                (Legacy list-of-ArcWaypoint input is no longer supported.)
            d_star: Cost-to-go array from Dijkstra (float, length n_nodes).
        """
        self._data.clear()

        if not isinstance(waypoints, CompactWaypoints):
            raise TypeError(
                "ArcCache.build() now requires a CompactWaypoints bundle "
                "from enumerate_edges(); the legacy list-of-ArcWaypoint path "
                "has been removed."
            )

        if waypoints.n_entries == 0 or len(d_star) == 0:
            return

        keys = waypoints.keys   # int32[N, 4]
        dst = waypoints.dst     # int32[N]
        ftl = waypoints.ftl     # float32[N]

        # Filter invalid entries vectorised
        valid = (
            (dst >= 0)
            & (dst < len(d_star))
            & (ftl >= 0.0)
            & np.isfinite(ftl)
        )
        if not np.all(valid):
            keys = keys[valid]
            dst = dst[valid]
            ftl = ftl[valid]
        if keys.shape[0] == 0:
            return

        d_dst = np.asarray(d_star, dtype=np.float64)[dst]
        finite = np.isfinite(d_dst)
        if not np.all(finite):
            keys = keys[finite]
            ftl = ftl[finite]
            d_dst = d_dst[finite]
        if keys.shape[0] == 0:
            return

        costs = ftl.astype(np.float64) + d_dst

        # Group by 4-D cell key and take the min cost per group.
        # ``np.unique(axis=0)`` returns lexicographically sorted unique rows
        # plus an inverse map suitable for scatter-min via ``np.minimum.at``.
        unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        min_costs = np.full(unique_keys.shape[0], np.inf, dtype=np.float64)
        np.minimum.at(min_costs, inv, costs)

        valid_costs = np.isfinite(min_costs)
        unique_keys = unique_keys[valid_costs]
        min_costs = min_costs[valid_costs]

        # Populate dict (small: ~tens of thousands of cells after the min-reduce)
        for i in range(unique_keys.shape[0]):
            k = (
                int(unique_keys[i, 0]),
                int(unique_keys[i, 1]),
                int(unique_keys[i, 2]),
                int(unique_keys[i, 3]),
            )
            self._data[k] = float(min_costs[i])

    # ---- expert demo ingestion -------------------------------------------
    def ingest_demonstration(
        self,
        frames: List[Tuple[float, float, float, float, int]],
        d_star: np.ndarray,
        graph,
    ) -> int:
        """Ingest airborne states from an expert demonstration into the cache.

        Traces the demo through all airborne segments.  For each contiguous
        run of airborne frames (state_code in {3, 4}), the first subsequent
        surface frame is the landing.  The landing is mapped to the nearest
        graph node and d*[landing_node] gives the cost to go from there.

        For each airborne frame at offset ``k`` from the start of the arc:
            cost = (arc_length - k) + d_star[landing_node]

        Inserts into the spatial hash, keeping the minimum per cell.

        Args:
            frames: Per-frame states as [(x, y, vx, vy, state_code), ...].
                    state_code 3 = jumping, 4 = falling, 0/1/2 = surface,
                    5 = wall-sliding, 6/7 = dead.
            d_star: Cost-to-go array (float32, length graph.n_nodes).
            graph: SurfaceGraph — used to map landing positions to node ids.

        Returns:
            Number of new cells populated (including cells where a lower cost
            replaced a previously cached value).
        """
        from .edge_enumeration import _landing_to_node, _AIRBORNE_STATES

        n_inserted = 0
        n = len(frames)
        i = 0

        while i < n:
            x, y, vx, vy, sc = frames[i]
            if sc not in _AIRBORNE_STATES:
                i += 1
                continue

            arc_start = i
            landing_node = None
            arc_end = i

            for j in range(i, n):
                xj, yj, vxj, vyj, scj = frames[j]
                if scj not in _AIRBORNE_STATES:
                    landing = (xj, yj, vxj, vyj, scj)
                    landing_node = _landing_to_node(graph, landing)
                    arc_end = j
                    break
                arc_end = j + 1  # still airborne — extend arc

            if landing_node is None or landing_node >= len(d_star):
                i = arc_end
                continue

            d_landing = float(d_star[landing_node])
            if not np.isfinite(d_landing):
                i = arc_end
                continue

            arc_length = arc_end - arc_start

            for k, idx in enumerate(range(arc_start, arc_end)):
                xi, yi, vxi, vyi, _ = frames[idx]
                frames_to_land = arc_length - k
                cost = frames_to_land + d_landing
                key = _bin_key(xi, yi, vxi, vyi)
                existing = self._data.get(key)
                if existing is None or cost < existing:
                    self._data[key] = cost
                    n_inserted += 1

            i = arc_end

        return n_inserted

    # ---- lookup ----------------------------------------------------------
    def lookup(
        self, x: float, y: float, vx: float, vy: float,
    ) -> Optional[float]:
        """Look up the cached cost-to-go for an airborne state.

        Returns:
            Minimum cost-to-go if the cell is populated, else None.
        """
        key = _bin_key(x, y, vx, vy)
        return self._data.get(key)

    def lookup_neighborhood(
        self, x: float, y: float, vx: float, vy: float,
    ) -> Optional[float]:
        """Look up cost in the cell and its immediate neighbors.

        Useful when the state falls on a cell boundary.  Returns the
        minimum cost found across the 3^4 = 81 neighboring cells.
        """
        base = _bin_key(x, y, vx, vy)
        best: Optional[float] = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dvx in (-1, 0, 1):
                    for dvy in (-1, 0, 1):
                        key = (base[0] + dx, base[1] + dy,
                               base[2] + dvx, base[3] + dvy)
                        val = self._data.get(key)
                        if val is not None:
                            if best is None or val < best:
                                best = val
        return best

    # ---- serialization ---------------------------------------------------
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize to numpy arrays for .npz storage.

        Vectorised conversion from the internal dict.  Previous versions
        built two Python lists before calling ``np.array`` which copied
        each entry twice and caused a spike of ~500 B × n_cells overhead.

        Returns:
            (keys_array, values_array)
            - keys_array: int32 shape (N, 4)
            - values_array: float32 shape (N,)
        """
        n = len(self._data)
        if n == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        keys = np.empty((n, 4), dtype=np.int32)
        values = np.empty(n, dtype=np.float32)
        for i, (k, v) in enumerate(self._data.items()):
            keys[i, 0] = k[0]
            keys[i, 1] = k[1]
            keys[i, 2] = k[2]
            keys[i, 3] = k[3]
            values[i] = v
        return keys, values

    @classmethod
    def from_arrays(
        cls, keys: np.ndarray, values: np.ndarray,
    ) -> "ArcCache":
        """Reconstruct an ArcCache from serialized arrays."""
        cache = cls()
        for i in range(len(keys)):
            k = tuple(int(keys[i, j]) for j in range(4))
            cache._data[k] = float(values[i])
        return cache
