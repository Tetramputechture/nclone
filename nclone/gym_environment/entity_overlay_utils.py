"""8-channel entity overlay grid for Layer 1 observations (mines + gold + doors).

Builds ``(8, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)`` float32 in [0, 1] from simulator
state. The Layer 1 policy sees the toggle-mines-world entities plus gold and
locked doors:

    0. ``ninja_pos``             - bilinear splat at ninja tile-space position
    1. ``switch_pos``            - 1.0 at exit-switch tile iff not yet collected
    2. ``exit_pos``              - 1.0 at exit-door tile
    3. ``mine_deadly_density``   - count of deadly mines per tile, /8 clamp to [0, 1]
    4. ``mine_safe_density``     - count of safe/toggling mines per tile, /8 clamp
    5. ``gold_density``          - count of *remaining* (uncollected) gold per tile,
                                   /4 clamp to [0, 1]. Collected gold drops out of the
                                   caller's position list, so this channel *is* the
                                   remaining-gold state and is fixed-size at any N.
    6. ``locked_door_density``   - count of *closed* locked-door barriers per tile,
                                   /4 clamp. Opened doors drop out (live state).
    7. ``locked_switch_density`` - count of *uncollected* locked-door switches per
                                   tile, /4 clamp. N-invariant -> any door count.

NPP-38 dropped the previous ``bounce_block``, ``thwump``, ``launch_pad_pos``
channels because their source mechanics were out of scope for the toggle-mines
Layer 1 chunk (see ``docs/LAYER1_MARKOV_AUDIT.md``). The gold channel was added
for the gold-collection objective (see ``docs/GOLD_COLLECTION_HANDOFF.md``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    TILE_PIXEL_SIZE,
)

# Toggle mine kill zone radius (matches STRATEGY_EPIC / mine overlay spec)
MINE_KILL_ZONE_RADIUS_PX = 14.0
MINE_DENSITY_NORMALIZER = 8.0

# Gold density normalizer: a single gold in a tile reads as 0.25 (visible for the
# scattered-gold levels), 4+ gold saturates (dense clusters). Tunable.
GOLD_DENSITY_NORMALIZER = 4.0

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

    Mirrors the filter used by the 1-channel hazard map so the two observation
    views never disagree when a mine toggles mid-episode.
    """
    deadly: List[Any] = []
    safe: List[Any] = []
    for mine in mines:
        if getattr(mine, "state", 1) == MINE_STATE_DEADLY:
            deadly.append(mine)
        else:
            safe.append(mine)
    return deadly, safe

# Door density normalizer: a single closed door / uncollected switch in a tile
# reads as 0.25; 4+ saturates. Matches the gold-channel scale.
DOOR_DENSITY_NORMALIZER = 4.0

# Channel count for entity_overlays (toggle-mines + gold + locked doors).
ENTITY_OVERLAY_CHANNELS = 8

# Named channel indices (kept stable for downstream extractors).
ENTITY_OVERLAY_CH_NINJA = 0
ENTITY_OVERLAY_CH_SWITCH = 1
ENTITY_OVERLAY_CH_EXIT = 2
ENTITY_OVERLAY_CH_MINE_DEADLY = 3
ENTITY_OVERLAY_CH_MINE_SAFE = 4
ENTITY_OVERLAY_CH_GOLD = 5
# Locked doors (type 6): permutation/N-invariant density channels (same binning
# as gold), so the policy perceives any number of doors/switches at procgen scale.
ENTITY_OVERLAY_CH_LOCKED_DOOR = 6      # closed-door barriers per tile
ENTITY_OVERLAY_CH_LOCKED_SWITCH = 7    # uncollected locked-door switches per tile


def _entity_pixel_xy(entity: Any) -> Tuple[float, float]:
    """Return entity center in simulator pixel coordinates."""
    if hasattr(entity, "xpos") and hasattr(entity, "ypos"):
        return (float(entity.xpos), float(entity.ypos))
    if hasattr(entity, "x") and hasattr(entity, "y"):
        return (float(entity.x), float(entity.y))
    raise TypeError(f"Entity has no xpos/ypos or x/y: {type(entity)!r}")


def _door_barrier_xy(door: Any) -> Tuple[float, float]:
    """Return a locked door's BARRIER center (distinct from its switch).

    The barrier geometry lives on the door's ``segment`` (endpoints
    ``x1/y1/x2/y2``) on both the Python (GridSegmentLinear) and native
    (_DoorSegmentProxy) backends, so the midpoint is backend-agnostic. Falls back
    to the switch position (``xpos``/``ypos``) only if no segment is present.
    """
    seg = getattr(door, "segment", None)
    if seg is not None and hasattr(seg, "x1"):
        return (
            (float(seg.x1) + float(seg.x2)) * 0.5,
            (float(seg.y1) + float(seg.y2)) * 0.5,
        )
    return (float(getattr(door, "xpos", 0.0)), float(getattr(door, "ypos", 0.0)))


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


# Cache of each mine's kill-zone tile footprint, keyed by pixel position.
# A mine's footprint (the set of inner tiles its kill-zone circle overlaps) is a
# pure function of its STATIC position, so it is computed once and reused every
# step. This turns the per-step mine-density build from a per-mine numpy
# arange/clip/broadcast (the dominant env.step cost — ~74% in profiling) into a
# cached gather + in-place add, with bit-identical output. Keyed by (mx, my)
# floats: positions are deterministic and repeat exactly each step, and the
# footprint depends only on position so it is safe to share across envs/levels.
_MINE_FOOTPRINT_CACHE: Dict[Tuple[float, float], np.ndarray] = {}


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
_MINE_DENSITY_GRID_CACHE: Dict[Tuple[Tuple[float, float], ...], np.ndarray] = {}
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


def _accumulate_gold_density(
    gold_positions: Sequence[Tuple[float, float]],
    out: np.ndarray,
) -> None:
    """Fill ``out`` with remaining-gold counts / ``GOLD_DENSITY_NORMALIZER``, clamped.

    Gold is a point collectible; we bin each remaining gold center to its inner
    tile (same mapping as ``_px_to_inner_tile``) and normalize the per-tile count.
    """
    out.fill(0.0)
    if len(gold_positions) == 0:
        return
    pts = np.asarray(gold_positions, dtype=np.float64).reshape(-1, 2)
    ts = float(TILE_PIXEL_SIZE)
    tx = np.clip((pts[:, 0] / ts).astype(np.int64) - 1, 0, MAP_TILE_WIDTH - 1)
    ty = np.clip((pts[:, 1] / ts).astype(np.int64) - 1, 0, MAP_TILE_HEIGHT - 1)
    flat = tx * MAP_TILE_HEIGHT + ty
    counts = np.bincount(flat, minlength=MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
    counts = counts.reshape(MAP_TILE_WIDTH, MAP_TILE_HEIGHT)
    out[:] = np.minimum(counts.astype(np.float32) / GOLD_DENSITY_NORMALIZER, 1.0)


def _accumulate_density(
    positions: Sequence[Tuple[float, float]],
    out: np.ndarray,
    normalizer: float,
) -> None:
    """Bin point ``positions`` to inner tiles and write the per-tile count /
    ``normalizer`` (clamped to 1.0). Generalizes the gold-density binning."""
    out.fill(0.0)
    if len(positions) == 0:
        return
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    ts = float(TILE_PIXEL_SIZE)
    tx = np.clip((pts[:, 0] / ts).astype(np.int64) - 1, 0, MAP_TILE_WIDTH - 1)
    ty = np.clip((pts[:, 1] / ts).astype(np.int64) - 1, 0, MAP_TILE_HEIGHT - 1)
    flat = tx * MAP_TILE_HEIGHT + ty
    counts = np.bincount(flat, minlength=MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
    counts = counts.reshape(MAP_TILE_WIDTH, MAP_TILE_HEIGHT)
    out[:] = np.minimum(counts.astype(np.float32) / float(normalizer), 1.0)


def compute_entity_overlays(
    ninja_pos: Tuple[float, float],
    switch_pos: Optional[Tuple[float, float]],
    switch_activated: bool,
    exit_pos: Optional[Tuple[float, float]],
    deadly_mines: Sequence[Any],
    safe_mines: Sequence[Any],
    gold_positions: Sequence[Tuple[float, float]] = (),
    locked_doors: Sequence[Any] = (),
) -> np.ndarray:
    """Build the 8-channel entity overlay grid.

    Shape: ``(ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)``,
    float32 in [0, 1]. See module docstring for channel order. ``gold_positions``
    is the list of *remaining* (uncollected) gold centers in pixel coords.
    ``locked_doors`` is the list of locked-door entities; the BARRIER center comes
    from each door's ``segment`` (channel 6) and the SWITCH from ``xpos/ypos``
    (channel 7), gated by ``closed``/``active`` so each channel reflects the live
    remaining state. Both are N-invariant density channels.
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
    _accumulate_gold_density(gold_positions, overlays[ENTITY_OVERLAY_CH_GOLD])

    # Locked doors: closed barriers (channel 6) and uncollected switches
    # (channel 7). Both N-invariant density channels — the policy perceives any
    # number of doors. A door's barrier drops out once opened; its switch drops
    # out once collected, so each channel is the live remaining state.
    if locked_doors:
        closed_barriers: List[Tuple[float, float]] = []
        open_switches: List[Tuple[float, float]] = []
        for d in locked_doors:
            if bool(getattr(d, "closed", True)):
                # Channel 6 marks the BARRIER (door segment), not the switch.
                closed_barriers.append(_door_barrier_xy(d))
            if bool(getattr(d, "active", True)):  # switch uncollected
                # Channel 7 marks the SWITCH (entity xpos/ypos).
                open_switches.append(
                    (float(getattr(d, "xpos", 0.0)), float(getattr(d, "ypos", 0.0)))
                )
        _accumulate_density(
            closed_barriers, overlays[ENTITY_OVERLAY_CH_LOCKED_DOOR],
            DOOR_DENSITY_NORMALIZER,
        )
        _accumulate_density(
            open_switches, overlays[ENTITY_OVERLAY_CH_LOCKED_SWITCH],
            DOOR_DENSITY_NORMALIZER,
        )

    return overlays
