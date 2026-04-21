"""5-channel entity overlay grid for Layer 1 observations (NPP-38, toggle-mines scope).

Builds ``(5, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)`` float32 in [0, 1] from simulator
state. The Layer 1 policy sees only the toggle-mines-world entities:

    0. ``ninja_pos``            - bilinear splat at ninja tile-space position
    1. ``switch_pos``           - 1.0 at exit-switch tile iff not yet collected
    2. ``exit_pos``             - 1.0 at exit-door tile
    3. ``mine_deadly_density``  - count of deadly mines per tile, /8 clamp to [0, 1]
    4. ``mine_safe_density``    - count of safe/toggling mines per tile, /8 clamp

NPP-38 dropped the previous ``bounce_block``, ``thwump``, ``launch_pad_pos``
channels because their source mechanics are explicitly out of scope for the
toggle-mines Layer 1 chunk (see ``docs/LAYER1_MARKOV_AUDIT.md``). If any of
those mechanics come back into scope, extend the overlay tensor rather than
resurrecting these channels in place.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    TILE_PIXEL_SIZE,
)

# Toggle mine kill zone radius (matches STRATEGY_EPIC / mine overlay spec)
MINE_KILL_ZONE_RADIUS_PX = 14.0
MINE_DENSITY_NORMALIZER = 8.0

# NPP-38: channel count for entity_overlays after toggle-mines rescope.
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


def _circle_overlaps_tile_rect(
    px: float, py: float, radius: float, tx: int, ty: int
) -> bool:
    """True if a circle at (px, py) overlaps the inner tile rectangle."""
    tx_min, tx_max, ty_min, ty_max = _tile_pixel_bounds_inner(tx, ty)
    cx = max(tx_min, min(px, tx_max - 1e-9))
    cy = max(ty_min, min(py, ty_max - 1e-9))
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy <= radius * radius


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


def _accumulate_mine_density(
    mines: Sequence[Any],
    out: np.ndarray,
) -> None:
    """Fill ``out`` with mine counts / ``MINE_DENSITY_NORMALIZER``, clamped to [0, 1]."""
    counts = np.zeros((MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.int32)
    r = MINE_KILL_ZONE_RADIUS_PX
    for m in mines:
        mx, my = _entity_pixel_xy(m)
        tx_min, tx_max, ty_min, ty_max = _inner_tile_bounds_for_kill_zone(mx, my, r)
        if tx_min > tx_max or ty_min > ty_max:
            continue
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                if _circle_overlaps_tile_rect(mx, my, r, tx, ty):
                    counts[tx, ty] += 1
    out[:] = np.minimum(counts.astype(np.float32) / MINE_DENSITY_NORMALIZER, 1.0)


def compute_entity_overlays(
    ninja_pos: Tuple[float, float],
    switch_pos: Optional[Tuple[float, float]],
    switch_activated: bool,
    exit_pos: Optional[Tuple[float, float]],
    deadly_mines: Sequence[Any],
    safe_mines: Sequence[Any],
) -> np.ndarray:
    """Build the 5-channel entity overlay grid.

    Shape: ``(ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)``,
    float32 in [0, 1]. See module docstring for channel order.
    """
    overlays = np.zeros(
        (ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.float32
    )

    npx, npy = float(ninja_pos[0]), float(ninja_pos[1])
    tx_f = npx / float(TILE_PIXEL_SIZE) - 1.0
    ty_f = npy / float(TILE_PIXEL_SIZE) - 1.0
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
            overlays[ENTITY_OVERLAY_CH_NINJA, tx, ty] += float(w)

    if not switch_activated and switch_pos is not None:
        sx, sy = float(switch_pos[0]), float(switch_pos[1])
        stx, sty = _px_to_inner_tile(sx, sy)
        overlays[ENTITY_OVERLAY_CH_SWITCH, stx, sty] = 1.0

    if exit_pos is not None:
        ex, ey = float(exit_pos[0]), float(exit_pos[1])
        etx, ety = _px_to_inner_tile(ex, ey)
        overlays[ENTITY_OVERLAY_CH_EXIT, etx, ety] = 1.0

    _accumulate_mine_density(deadly_mines, overlays[ENTITY_OVERLAY_CH_MINE_DEADLY])
    _accumulate_mine_density(safe_mines, overlays[ENTITY_OVERLAY_CH_MINE_SAFE])

    return overlays
