"""Perpendicular mine-displacement morph — the geometry axis for the
``mines-extremely-difficult`` reverse curriculum (see
``docs/GEOMETRY_MORPHING_CURRICULUM.md``).

The morph primitive widens the threadable channel of a mine-saturated level by a
controllable amount ``delta_px`` while keeping everything else faithful:

* **Route topology is preserved.** Each deadly mine is displaced *perpendicular
  to the route centerline* (outward, away from the channel). The centerline is
  the steepest-descent path on the connected mine-threading cost-to-go field
  (``MineThreadingDistanceSource.centerline``), which lives in the mine-safe
  region, so every mine is well off it and its outward direction is unambiguous.
  Pushing every mine outward by ``delta_px`` relaxes the binding clearance
  constraint along the whole route by ~``delta_px`` without moving the route.
* **delta_px = 0 is the identity** (byte-identical map) — the curriculum's
  terminal rung is *exactly* the real level.
* **Mine physics stay faithful.** Coordinates are written back on the native 6px
  unit grid (``pixel = unit * 6``), so the morphed map loads byte-identically
  across the Python and Rust backends. Because mines snap to that grid, the
  effective displacement resolution is 6px — **stage deltas should be multiples
  of 6** (e.g. 6, 12, 24, 48). Sub-6px deltas round per-mine and may not move
  some mines at all.

This is intentionally generic: it works on any level for which
``MineThreadingDistanceSource`` resolves a connected spawn->switch->exit route,
not just ``mines-extremely-difficult``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
)
from ..value_function.mine_threading_distance import MineThreadingDistanceSource

logger = logging.getLogger(__name__)

# --- map_data entity-stream layout (see nclone-native/src/map/loader.rs) -------
ENTITY_START = 1235          # first entity record (Python writer convention)
ENTITY_STRIDE = 5            # [type, x, y, orientation, mode]
PX_PER_UNIT = 6              # stored coord is in 6px units; world_px = unit * 6
SPAWN_X_IDX, SPAWN_Y_IDX = 1231, 1232

DEADLY_MINE_TYPE = 1         # toggle mine, starts TOGGLED (deadly)
SAFE_MINE_TYPE = 21          # toggle mine, starts UNTOGGLED (safe)
MINE_TYPES = (DEADLY_MINE_TYPE, SAFE_MINE_TYPE)

# Playfield interior in world pixels: one tile of border on every side.
_BORDER = TILE_PIXEL_SIZE
_INNER_HI_X = (MAP_TILE_WIDTH + 1) * TILE_PIXEL_SIZE    # 43*24 = 1032
_INNER_HI_Y = (MAP_TILE_HEIGHT + 1) * TILE_PIXEL_SIZE   # 24*24 = 576


class MineMorphError(RuntimeError):
    """Raised when a map has no resolvable mine-threading route to morph along."""


def iter_mines(map_data: Sequence[int], *, deadly_only: bool = False):
    """Yield ``(record_index, entity_type, world_x, world_y)`` for each mine.

    ``record_index`` points at the type byte so callers can mutate ``+1``/``+2``.
    World coords are pixels (``unit * 6``).
    """
    for i in range(ENTITY_START, len(map_data) - (ENTITY_STRIDE - 1), ENTITY_STRIDE):
        t = map_data[i]
        if t not in MINE_TYPES:
            continue
        if deadly_only and t != DEADLY_MINE_TYPE:
            continue
        yield i, t, map_data[i + 1] * PX_PER_UNIT, map_data[i + 2] * PX_PER_UNIT


def spawn_world(map_data: Sequence[int]) -> Tuple[float, float]:
    """Ninja spawn in world pixels (``unit * 6``)."""
    return (map_data[SPAWN_X_IDX] * float(PX_PER_UNIT),
            map_data[SPAWN_Y_IDX] * float(PX_PER_UNIT))


def route_centerline(
    map_data: Sequence[int], *,
    mt_src: Optional[MineThreadingDistanceSource] = None,
) -> List[Tuple[float, float]]:
    """World-coord polyline spawn -> switch -> exit for ``map_data``.

    Builds (or reuses) a ``MineThreadingDistanceSource`` and traces its centerline.
    Raises ``MineMorphError`` if the source can't resolve a connected route.
    """
    md = list(map_data)
    if mt_src is None:
        mt_src = MineThreadingDistanceSource(md)
    if not mt_src.available:
        raise MineMorphError(
            "MineThreadingDistanceSource unavailable — no connected mine-threading "
            "route to define a centerline (map has no mines, or the route is severed)."
        )
    pts = mt_src.centerline(spawn_world(md))
    if len(pts) < 2:
        raise MineMorphError("mine-threading centerline collapsed to < 2 points")
    return pts


def morph_mines(
    map_data: Sequence[int],
    delta_px: float, *,
    mode: str = "carve",
    centerline: Optional[Sequence[Tuple[float, float]]] = None,
    mt_src: Optional[MineThreadingDistanceSource] = None,
    deadly_only: bool = True,
) -> List[int]:
    """Return a new ``map_data`` whose threadable channel is widened by ``delta_px``.

    ``delta_px <= 0`` returns an identical copy (the curriculum's terminal rung is
    *exactly* the real level).

    Two morph modes:

    * ``mode="carve"`` (**default, robust**): neutralize every deadly mine whose
      distance to the centerline is ``< base_clearance + delta_px``, where
      ``base_clearance`` is the original channel half-width. This guarantees the
      morphed channel half-width is ``>= base_clearance + delta_px`` — strictly
      monotonic in ``delta_px`` and exactly the identity at ``delta_px = 0`` (the
      closest mine sits at ``base_clearance``, so a strict ``<`` removes nothing).
      A mine is neutralized by zeroing its entity-type byte, which the loader
      skips (preserving the 5-byte stream alignment); the rest of the level is
      untouched, so the route *topology* is exactly preserved. This is robust on
      tightly-folded serpentines where directional displacement is not.

    * ``mode="displace"``: push each mine ``delta_px`` perpendicular-outward from
      its nearest centerline point, preserving mine count/density. Faithful only
      when route legs are far apart — on a folded serpentine, pushing a mine off
      one leg shoves it into the adjacent leg's channel and *narrows* it. Use only
      for sparsely-folded levels; prefer ``carve`` for robustness.

    ``deadly_only`` (default) acts only on type-1 mines — the ones that define the
    14px kill-disc channel; safe (type-21) mines don't affect clearance.
    """
    md = list(map_data)
    if delta_px <= 0:
        return md

    if centerline is None:
        centerline = route_centerline(md, mt_src=mt_src)
    pts = np.asarray(centerline, dtype=float)

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)

    if mode == "carve":
        base = min_clearance_along(md, centerline, deadly_only=deadly_only)
        thresh = base + delta_px
        removed = 0
        for i, _t, wx, wy in iter_mines(md, deadly_only=deadly_only):
            d, _ = tree.query((wx, wy))
            if d < thresh:
                md[i] = 0           # type 0 -> loader skips it (faithful removal)
                removed += 1
        logger.debug("morph_mines carve delta=%.1f base=%.1f thresh=%.1f removed=%d",
                     delta_px, base, thresh, removed)
        return md

    if mode == "displace":
        for i, _t, wx, wy in iter_mines(md, deadly_only=deadly_only):
            _dist, idx = tree.query((wx, wy))
            c = pts[idx]
            v = np.array((wx - c[0], wy - c[1]), dtype=float)
            n = float(np.hypot(v[0], v[1]))
            if n < 1e-6:
                continue
            v /= n
            nx = min(max(wx + delta_px * v[0], _BORDER), _INNER_HI_X)
            ny = min(max(wy + delta_px * v[1], _BORDER), _INNER_HI_Y)
            md[i + 1] = int(min(max(round(nx / PX_PER_UNIT), 0), 255))
            md[i + 2] = int(min(max(round(ny / PX_PER_UNIT), 0), 255))
        return md

    raise ValueError(f"unknown morph mode {mode!r} (expected 'carve' or 'displace')")


def min_clearance_along(
    map_data: Sequence[int],
    centerline: Sequence[Tuple[float, float]], *,
    deadly_only: bool = True,
) -> float:
    """Minimum px distance from any centerline point to the nearest deadly mine.

    The channel half-width: a faithful proxy for how threadable the level is, and
    the quantity the morph is designed to increase monotonically with ``delta_px``.
    Returns ``inf`` if there are no mines.
    """
    mines = np.array([(wx, wy) for _i, _t, wx, wy
                      in iter_mines(map_data, deadly_only=deadly_only)], dtype=float)
    if not len(mines):
        return float("inf")
    from scipy.spatial import cKDTree
    tree = cKDTree(mines)
    d, _ = tree.query(np.asarray(centerline, dtype=float))
    return float(np.min(d))
