"""Per-tile-type surface normals derived from the EXACT tile geometry.

A single source of truth for the outward (anti-solid, i.e. pointing into the
traversable space the ninja occupies) surface normal of every tile type, built
from the precise segment geometry in :mod:`nclone.tile_definitions`:

* diagonal slopes  (6-9, 18-33) -> ``TILE_SEGMENT_DIAG_MAP`` line normal,
* curved tiles     (10-17)      -> ``TILE_SEGMENT_CIRCULAR_MAP`` arc normal,
* flat / half / empty           -> ``(0, 0)`` (no slope/curve information; the
  raw tile-id embedding + binary solidity already capture these).

Used by the RL observation (a per-tile normal *field* baked into the feature
extractor as a geometric prior) and by the look-ahead landing-normal feature.

Coordinate convention matches the renderer / collision code: a 24px tile with
the origin at the top-left, x right, y DOWN.  ``ny < 0`` therefore means the
surface faces upward (a floor the ninja stands on top of).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from nclone.tile_definitions import (
    TILE_SEGMENT_CIRCULAR_MAP,
    TILE_SEGMENT_DIAG_MAP,
)
from nclone.graph.reachability.graph_builder import _check_subnode_validity_simple

NUM_TILE_TYPES = 38
_TILE = 24.0
_INV_SQRT2 = 1.0 / math.sqrt(2.0)

_CACHE: Dict[int, Tuple[float, float]] = {}


def _clamp(v: float) -> int:
    return int(max(0, min(23, round(v))))


def _orient_outward(tile_type: int, mx: float, my: float, nx: float, ny: float) -> Tuple[float, float]:
    """Flip the normal so it points toward the TRAVERSABLE side of the tile.

    Probes a sub-pixel step along +/-n against the canonical solidity test so
    every tile uses one consistent convention (normal points into open space).
    """
    step = 4.0
    px, py = _clamp(mx + step * nx), _clamp(my + step * ny)
    if _check_subnode_validity_simple(tile_type, px, py):
        return (nx, ny)  # +n lands in open space -> already outward
    return (-nx, -ny)


def tile_surface_normal(tile_type: int) -> Tuple[float, float]:
    """Return the outward unit surface normal for a slope/curve tile.

    ``(0, 0)`` for empty / flat / half tiles (no diagonal or curved surface).
    Result is cached per tile type.
    """
    if tile_type in _CACHE:
        return _CACHE[tile_type]

    result = (0.0, 0.0)

    seg = TILE_SEGMENT_DIAG_MAP.get(tile_type)
    if seg is not None:
        (x1, y1), (x2, y2) = seg
        px, py = x2 - x1, y2 - y1
        length = math.hypot(px, py)
        if length > 1e-9:
            nx, ny = -py / length, px / length
            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            result = _orient_outward(tile_type, mx, my, nx, ny)
    else:
        circ = TILE_SEGMENT_CIRCULAR_MAP.get(tile_type)
        if circ is not None:
            (cx, cy), (qx, qy), _convex = circ
            # Arc midpoint at 45 deg within the quadrant; radius == tile size.
            mx = cx + _TILE * qx * _INV_SQRT2
            my = cy + _TILE * qy * _INV_SQRT2
            # Radial direction (center -> arc point), normalized.
            rx, ry = qx * _INV_SQRT2, qy * _INV_SQRT2
            result = _orient_outward(tile_type, mx, my, rx, ry)

    _CACHE[tile_type] = result
    return result


def build_normal_table() -> np.ndarray:
    """Return an ``(NUM_TILE_TYPES, 2)`` float32 table of outward normals."""
    table = np.zeros((NUM_TILE_TYPES, 2), dtype=np.float32)
    for t in range(NUM_TILE_TYPES):
        table[t] = tile_surface_normal(t)
    return table
