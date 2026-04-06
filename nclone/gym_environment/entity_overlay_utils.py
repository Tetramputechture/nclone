"""8-channel entity overlay grid for Layer 1 observations (NPP-9).

Builds (8, MAP_TILE_WIDTH, MAP_TILE_HEIGHT) float32 in [0, 1] from simulator state.
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


def compute_entity_overlays(
    ninja_pos: Tuple[float, float],
    switch_pos: Optional[Tuple[float, float]],
    switch_activated: bool,
    exit_pos: Optional[Tuple[float, float]],
    deadly_mines: Sequence[Any],
    safe_mines: Sequence[Any],
    bounce_blocks: Sequence[Any],
    thwumps: Sequence[Any],
    launch_pads: Sequence[Any],
) -> np.ndarray:
    """Build 8-channel entity overlay grid (8, MAP_TILE_WIDTH, MAP_TILE_HEIGHT), float32 in [0, 1].

    Channel order matches STRATEGY_EPIC / NPP-9: ninja bilinear splat, switch, exit,
    deadly mine density, safe mine density, bounce block, thwump, launch pad.
    Mine densities count mines whose 14px kill zone overlaps each tile, normalized by 8.
    """
    overlays = np.zeros((8, MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.float32)

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
            overlays[0, tx, ty] += float(w)

    if not switch_activated and switch_pos is not None:
        sx, sy = float(switch_pos[0]), float(switch_pos[1])
        stx, sty = _px_to_inner_tile(sx, sy)
        overlays[1, stx, sty] = 1.0

    if exit_pos is not None:
        ex, ey = float(exit_pos[0]), float(exit_pos[1])
        etx, ety = _px_to_inner_tile(ex, ey)
        overlays[2, etx, ety] = 1.0

    for ty in range(MAP_TILE_HEIGHT):
        for tx in range(MAP_TILE_WIDTH):
            deadly_count = 0
            for m in deadly_mines:
                mx, my = _entity_pixel_xy(m)
                if _circle_overlaps_tile_rect(
                    mx, my, MINE_KILL_ZONE_RADIUS_PX, tx, ty
                ):
                    deadly_count += 1
            overlays[3, tx, ty] = min(
                1.0, float(deadly_count) / MINE_DENSITY_NORMALIZER
            )

    for ty in range(MAP_TILE_HEIGHT):
        for tx in range(MAP_TILE_WIDTH):
            safe_count = 0
            for m in safe_mines:
                mx, my = _entity_pixel_xy(m)
                if _circle_overlaps_tile_rect(
                    mx, my, MINE_KILL_ZONE_RADIUS_PX, tx, ty
                ):
                    safe_count += 1
            overlays[4, tx, ty] = min(1.0, float(safe_count) / MINE_DENSITY_NORMALIZER)

    for entities, ch in (
        (bounce_blocks, 5),
        (thwumps, 6),
        (launch_pads, 7),
    ):
        for e in entities:
            ex, ey = _entity_pixel_xy(e)
            etx, ety = _px_to_inner_tile(ex, ey)
            overlays[ch, etx, ety] = 1.0

    return overlays
