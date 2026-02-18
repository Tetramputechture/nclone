"""Spatial hash cache for O(1) airborne state cost-to-go lookup.

During edge enumeration, waypoints are sampled along every simulated arc.
This module builds a spatial hash that maps discretized (x, y, vx, vy) cells
to the minimum cost-to-go observed at any arc waypoint passing through that
cell.  At runtime, an airborne state can be looked up in O(1) to get a
precomputed cost estimate without forward simulation.

Bin sizes (from spec):
  - Position: 12 px
  - Velocity: 0.5 px/frame
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .edge_enumeration import ArcWaypoint

# ---------------------------------------------------------------------------
# Bin configuration
# ---------------------------------------------------------------------------
POSITION_BIN_SIZE = 12.0     # pixels
VELOCITY_BIN_SIZE = 0.5      # pixels / frame

_INV_POS_BIN = 1.0 / POSITION_BIN_SIZE
_INV_VEL_BIN = 1.0 / VELOCITY_BIN_SIZE


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

    The cache stores, for each quantized state cell, the minimum value of
    ``(total_frames - frame_offset) + d_star[destination_node]`` across all
    arc waypoints that pass through that cell.

    Attributes:
        _data: Mapping from bin key to minimum cost (float).
        _n_entries: Number of populated cells.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[int, int, int, int], float] = {}

    @property
    def n_entries(self) -> int:
        return len(self._data)

    # ---- construction ----------------------------------------------------
    def build(
        self,
        waypoints: List[ArcWaypoint],
        d_star: np.ndarray,
    ) -> None:
        """Populate the cache from arc waypoints and a d_star array.

        For each waypoint, computes the cost-to-go as
        ``(total_frames - frame_offset) + d_star[dst_node]`` and stores
        the minimum per spatial cell.

        Args:
            waypoints: Waypoints collected during edge enumeration.
            d_star: Cost-to-go array from Dijkstra (float32, n_nodes).
        """
        self._data.clear()
        for wp in waypoints:
            if wp.dst_node < 0 or wp.total_frames < 0:
                continue
            if wp.dst_node >= len(d_star):
                continue
            d_dst = float(d_star[wp.dst_node])
            if not np.isfinite(d_dst):
                continue
            cost = (wp.total_frames - wp.frame_offset) + d_dst
            key = _bin_key(wp.x, wp.y, wp.vx, wp.vy)
            existing = self._data.get(key)
            if existing is None or cost < existing:
                self._data[key] = cost

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

        Returns:
            (keys_array, values_array)
            - keys_array: int32 shape (N, 4)
            - values_array: float32 shape (N,)
        """
        if not self._data:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        keys = np.array(list(self._data.keys()), dtype=np.int32)
        values = np.array(list(self._data.values()), dtype=np.float32)
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
