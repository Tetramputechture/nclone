"""5-channel entity overlay grid for Layer 1 observations (ninja + goals + mines).

Builds ``(5, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)`` float32 in [0, 1] from simulator
state. The generalist agent's scope is exactly {toggle mines, one exit switch,
one exit door, one player spawn, any tiles}, so the overlay carries only those:

    0. ``ninja_pos``           - bilinear splat at the ninja's tile-space position
    1. ``switch_pos``          - bilinear splat at the exit-switch position iff not
                                 yet activated (drops out once the switch is hit)
    2. ``exit_pos``            - bilinear splat at the exit-door position
    3. ``mine_deadly_density`` - count of deadly mines per tile, /8 clamp to [0, 1]
    4. ``mine_safe_density``   - count of safe/toggling mines per tile, /8 clamp

Channels 1/2 use the SAME bilinear sub-tile splat as the ninja channel (not a
hard single-cell write): static entities are placed on a 6px grid, so a 24px tile
holds 16 distinguishable goal positions. The bilinear splat preserves that
sub-tile location in the existing channel at no extra cost, instead of collapsing
all 16 to one cell.

Out-of-scope mechanics (gold, locked doors, one-way platforms) were removed
entirely — they could never fire under the generalist scope and only added dead
(and, for the one-way normal channels, constant-0.5) input planes.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    TILE_PIXEL_SIZE,
)

# Toggle mine kill zone radius (matches STRATEGY_EPIC / mine overlay spec).
MINE_KILL_ZONE_RADIUS_PX = 14.0
MINE_DENSITY_NORMALIZER = 8.0

# Canonical toggle-mine state semantics (see `entities/entity_toggle_mine.py`).
# state == 0 is the lethal "toggled" phase; any non-zero state is safe or
# currently toggling. Both `TOGGLE_MINE` (type 1) and `TOGGLE_MINE_TOGGLED`
# (type 21) entities are classified by the live `state` field, not by entity
# type, so the overlay stays consistent with `_compute_hazard_map`.
MINE_STATE_DEADLY = 0


def classify_mines_by_state(
    mines: Iterable[Any],
) -> Tuple[List[Any], List[Any]]:
    """Split a mine iterable into ``(deadly, safe)`` by the live ``state`` field.

    Mirrors the filter used by the hazard map so the two observation views never
    disagree when a mine toggles mid-episode.
    """
    deadly: List[Any] = []
    safe: List[Any] = []
    for mine in mines:
        if getattr(mine, "state", 1) == MINE_STATE_DEADLY:
            deadly.append(mine)
        else:
            safe.append(mine)
    return deadly, safe


# Channel count for entity_overlays (ninja + goals + toggle-mines).
ENTITY_OVERLAY_CHANNELS = 5

# Named channel indices (kept stable for downstream extractors).
ENTITY_OVERLAY_CH_NINJA = 0
ENTITY_OVERLAY_CH_SWITCH = 1
ENTITY_OVERLAY_CH_EXIT = 2
ENTITY_OVERLAY_CH_MINE_DEADLY = 3
ENTITY_OVERLAY_CH_MINE_SAFE = 4


def _entity_pixel_xy(entity: Any) -> Tuple[float, float]:
    """Return entity center in simulator pixel coordinates."""
    if hasattr(entity, "xpos") and hasattr(entity, "ypos"):
        return (float(entity.xpos), float(entity.ypos))
    if hasattr(entity, "x") and hasattr(entity, "y"):
        return (float(entity.x), float(entity.y))
    raise TypeError(f"Entity has no xpos/ypos or x/y: {type(entity)!r}")


def _bilinear_splat(overlays: np.ndarray, channel: int, px: float, py: float) -> None:
    """Splat a unit point mass at pixel (px, py) onto ``channel`` with bilinear
    sub-tile weights over the inner playable grid (0..41, 0..22).

    The four surrounding inner-tile cells receive weights summing to 1.0, so the
    splat's center of mass is the exact sub-tile position — preserving the 6px-grid
    location that a single-cell write would quantize to 24px.
    """
    tx_f = float(px) / float(TILE_PIXEL_SIZE) - 1.0
    ty_f = float(py) / float(TILE_PIXEL_SIZE) - 1.0
    x0 = int(math.floor(tx_f))
    y0 = int(math.floor(ty_f))
    wx = tx_f - x0
    wy = ty_f - y0
    weights = (
        ((0, 0), (1.0 - wx) * (1.0 - wy)),
        ((1, 0), wx * (1.0 - wy)),
        ((0, 1), (1.0 - wx) * wy),
        ((1, 1), wx * wy),
    )
    for (dx, dy), w in weights:
        tx, ty = x0 + dx, y0 + dy
        if 0 <= tx < MAP_TILE_WIDTH and 0 <= ty < MAP_TILE_HEIGHT and w > 0.0:
            overlays[channel, tx, ty] += float(w)


def _px_to_inner_tile(px: float, py: float) -> Tuple[int, int]:
    """Map pixel position to inner playable tile indices (0..41, 0..22)."""
    tx = int(px / TILE_PIXEL_SIZE) - 1
    ty = int(py / TILE_PIXEL_SIZE) - 1
    return (
        max(0, min(MAP_TILE_WIDTH - 1, tx)),
        max(0, min(MAP_TILE_HEIGHT - 1, ty)),
    )


def _tile_pixel_bounds_inner(tx: int, ty: int) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds for inner tile (tx, ty) in pixel space."""
    tx_min = float((tx + 1) * TILE_PIXEL_SIZE)
    tx_max = tx_min + float(TILE_PIXEL_SIZE)
    ty_min = float((ty + 1) * TILE_PIXEL_SIZE)
    ty_max = ty_min + float(TILE_PIXEL_SIZE)
    return tx_min, tx_max, ty_min, ty_max


def _inner_tile_bounds_for_kill_zone(
    cx: float, cy: float, radius: float
) -> Tuple[int, int, int, int]:
    """Inclusive inner-tile index bounds that may intersect the mine kill-zone circle.

    The naive implementation iterated all ``MAP_TILE_WIDTH * MAP_TILE_HEIGHT`` tiles
    per mine (very slow in Python). The circle radius is small relative to the map,
    so we only need a local neighborhood (~few tiles per axis). We derive bounds
    from the circle's pixel AABB (same as ``_px_to_inner_tile`` on corners) and
    pad by one tile so edge cases stay bit-identical to the full-grid reference.
    """
    ts = float(TILE_PIXEL_SIZE)
    min_px = cx - radius
    max_px = cx + radius
    min_py = cy - radius
    max_py = cy + radius
    # Match ``_px_to_inner_tile`` for positive coordinates (playable area).
    tx_lo = int(min_px / ts) - 1
    tx_hi = int(max_px / ts) - 1
    ty_lo = int(min_py / ts) - 1
    ty_hi = int(max_py / ts) - 1
    tx_lo -= 1
    tx_hi += 1
    ty_lo -= 1
    ty_hi += 1
    if tx_lo > tx_hi or ty_lo > ty_hi:
        return 0, -1, 0, -1
    tx_min = max(0, min(tx_lo, tx_hi))
    tx_max = min(MAP_TILE_WIDTH - 1, max(tx_lo, tx_hi))
    ty_min = max(0, min(ty_lo, ty_hi))
    ty_max = min(MAP_TILE_HEIGHT - 1, max(ty_lo, ty_hi))
    return tx_min, tx_max, ty_min, ty_max


# Cache of each mine's kill-zone tile footprint, keyed by pixel position.
# A mine's footprint (the set of inner tiles its kill-zone circle overlaps) is a
# pure function of its STATIC position, so it is computed once and reused every
# step. Keyed by (mx, my) floats: positions are deterministic and repeat exactly
# each step, and the footprint depends only on position so it is safe to share
# across envs/levels.
_MINE_FOOTPRINT_CACHE: dict[Tuple[float, float], np.ndarray] = {}


def _mine_kill_zone_footprint(mx: float, my: float) -> np.ndarray:
    """Flat inner-tile indices (``tx * MAP_TILE_HEIGHT + ty``) a mine's kill zone
    overlaps. Computed once per position via the exact same overlap test as the
    reference, then cached. Indices within a single mine's footprint are unique.
    """
    key = (mx, my)
    fp = _MINE_FOOTPRINT_CACHE.get(key)
    if fp is not None:
        return fp
    r = MINE_KILL_ZONE_RADIUS_PX
    r2 = r * r
    ts = float(TILE_PIXEL_SIZE)
    tx_min, tx_max, ty_min, ty_max = _inner_tile_bounds_for_kill_zone(mx, my, r)
    if tx_min > tx_max or ty_min > ty_max:
        fp = np.empty(0, dtype=np.intp)
    else:
        txs = np.arange(tx_min, tx_max + 1, dtype=np.float64)
        tys = np.arange(ty_min, ty_max + 1, dtype=np.float64)
        tx_lo = (txs + 1.0) * ts
        tx_hi = tx_lo + ts - 1e-9
        ty_lo = (tys + 1.0) * ts
        ty_hi = ty_lo + ts - 1e-9
        cx = np.clip(mx, tx_lo[:, None], tx_hi[:, None])
        cy = np.clip(my, ty_lo[None, :], ty_hi[None, :])
        dx = mx - cx
        dy = my - cy
        overlap = (dx * dx + dy * dy) <= r2  # (ntx, nty)
        oi, oj = np.nonzero(overlap)
        gtx = tx_min + oi
        gty = ty_min + oj
        fp = (gtx * MAP_TILE_HEIGHT + gty).astype(np.intp)
    _MINE_FOOTPRINT_CACHE[key] = fp
    return fp


# Cache of the finished density grid keyed by the mine position multiset. The
# grid is a pure function of which mines are in the set; mine positions are
# static, so the set only changes when a mine toggles (deadly<->safe) — rare
# relative to the per-step cadence. Keyed by sorted positions (order-independent,
# preserves duplicates); capped to bound memory.
_MINE_DENSITY_GRID_CACHE: dict[Tuple[Tuple[float, float], ...], np.ndarray] = {}
_MINE_DENSITY_GRID_CACHE_CAP = 4096


def _accumulate_mine_density(
    mines: Sequence[Any],
    out: np.ndarray,
) -> None:
    """Fill ``out`` with mine counts / ``MINE_DENSITY_NORMALIZER``, clamped to [0, 1].

    Bit-identical to the per-mine overlap reference. Two layers of caching keep
    this off the hot path: the finished density grid is cached by the mine
    position multiset (recomputed only on a toggle), and on a cache miss each
    mine's kill-zone footprint is itself cached (positions are static).
    """
    positions = tuple(_entity_pixel_xy(m) for m in mines)
    key = tuple(sorted(positions))
    cached = _MINE_DENSITY_GRID_CACHE.get(key)
    if cached is not None:
        out[:] = cached
        return
    counts = np.zeros(MAP_TILE_WIDTH * MAP_TILE_HEIGHT, dtype=np.int32)
    for mx, my in positions:
        fp = _mine_kill_zone_footprint(mx, my)
        if fp.size:
            counts[fp] += 1  # footprint indices are unique within a mine
    grid = np.minimum(
        counts.reshape(MAP_TILE_WIDTH, MAP_TILE_HEIGHT).astype(np.float32)
        / MINE_DENSITY_NORMALIZER,
        1.0,
    )
    if len(_MINE_DENSITY_GRID_CACHE) < _MINE_DENSITY_GRID_CACHE_CAP:
        _MINE_DENSITY_GRID_CACHE[key] = grid
    out[:] = grid


def compute_entity_overlays(
    ninja_pos: Tuple[float, float],
    switch_pos: Optional[Tuple[float, float]],
    switch_activated: bool,
    exit_pos: Optional[Tuple[float, float]],
    deadly_mines: Sequence[Any],
    safe_mines: Sequence[Any],
    cached_mine_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the entity overlay grid (``ENTITY_OVERLAY_CHANNELS`` channels).

    Shape: ``(ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)``,
    float32 in [0, 1]. See module docstring for channel order. The ninja, switch,
    and exit channels are bilinear sub-tile splats (the switch is omitted once
    ``switch_activated``). ``cached_mine_channels`` is an optional pre-rasterized
    ``(2, W, H)`` deadly/safe block to skip the per-step mine rasterization on a
    non-toggling field.
    """
    overlays = np.zeros(
        (ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.float32
    )

    _bilinear_splat(
        overlays, ENTITY_OVERLAY_CH_NINJA, float(ninja_pos[0]), float(ninja_pos[1])
    )

    if not switch_activated and switch_pos is not None:
        _bilinear_splat(
            overlays, ENTITY_OVERLAY_CH_SWITCH,
            float(switch_pos[0]), float(switch_pos[1]),
        )

    if exit_pos is not None:
        _bilinear_splat(
            overlays, ENTITY_OVERLAY_CH_EXIT, float(exit_pos[0]), float(exit_pos[1])
        )

    # Mine density channels are STATIC on a non-toggling field, so the caller can
    # pass the previously-rasterized (2, W, H) deadly/safe block to skip the
    # per-step rasterization of ~1.7k mine positions (the dominant overlay cost).
    if cached_mine_channels is not None:
        overlays[ENTITY_OVERLAY_CH_MINE_DEADLY] = cached_mine_channels[0]
        overlays[ENTITY_OVERLAY_CH_MINE_SAFE] = cached_mine_channels[1]
    else:
        _accumulate_mine_density(deadly_mines, overlays[ENTITY_OVERLAY_CH_MINE_DEADLY])
        _accumulate_mine_density(safe_mines, overlays[ENTITY_OVERLAY_CH_MINE_SAFE])

    return overlays
