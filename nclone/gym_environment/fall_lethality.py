"""v26 W6a — static fall-lethality field.

Impact deaths are ~40% of v25's losses, and operator inspection of the failure
renders says most of them are *committed unrecoverable falls*: a precision jump
between platforms across a gap where the fall itself is fatal. The lethal
decision is made at **takeoff** — once the ninja is falling, no action recovers —
so every signal the agent currently gets about impacts (``lethal_landing_tau``,
the landing PBRS terms, the landing aux head) arrives far too late to matter.

What is missing is a signal that exists *before* the jump: a static property of
the level geometry saying "a fall from here is lethal". That is exactly what this
module computes.

Physics
-------
``ninja.py`` kills on contact when the velocity component INTO the surface normal
exceeds a normal-dependent threshold::

    impact_vel = -(n_x * xspeed_old + n_y * yspeed_old)
    lethal iff  impact_vel > MAX_SURVIVABLE_IMPACT - (4/3) * |n_y|

For a ninja falling from rest, ``(xspeed, yspeed) = (0, v)`` with ``v`` capped at
``TERMINAL_VELOCITY``, so ``impact_vel = |n_y| * v``. Both sides depend on the
landing surface's normal, and they pull in opposite directions:

===================  =======  ==========  ============  ============
landing surface      \\|n_y\\|  threshold   impact_vel    lethal drop
===================  =======  ==========  ============  ============
flat floor             1.000     4.667      1.000 * v      163 px
27 deg slope           0.891     4.812      0.891 * v      219 px
45 deg slope           0.707     5.057      0.707 * v      383 px
steep 63 deg           0.454     5.395      0.454 * v      >terminal
===================  =======  ==========  ============  ============

**A flat floor is the least forgiving surface** — ~6.8 tiles kills, not the ~11.3
a naive ``MAX_SURVIVABLE_IMPACT = 6`` reading gives. On a 23-tile level that is
under a third of the height, so lethal gaps are far more common than a
constant-threshold model suggests, and a "drop height > K" heuristic is wrong in
both directions. Hence the per-landing-site normal.

(The V26 plan's §1.8 table quotes 8.0 tiles for a 45 deg slope. That reading
compares the *fall speed* against the threshold directly, dropping the ``|n_y|``
projection that ``ninja.py`` applies to the incoming velocity. The projection is
in the sim, so it is applied here; slopes are correspondingly more forgiving than
the plan's table suggests, and flat floors are not.)

Output
------
A 2-channel field at 6 px resolution, co-registered with ``hazard_map`` /
``tile_grid`` / ``entity_overlays`` (the 1-tile sim border is stripped):

* ch0 ``margin`` — ``clip((threshold - impact_vel) / MAX_SURVIVABLE_IMPACT, -1, 1)``.
  Positive = a fall from this cell is survivable, negative = fatal, and the
  magnitude says by how much. Cells with no floor anywhere below read ``-1``.
* ch1 ``drop`` — ``clip(drop_px / MAX_DROP_NORM_PX, 0, 1)``, how far the fall is.
  Redundant with ch0 on flat ground but not on slopes, and it is the channel that
  survives pooling as "there is a pit here".

The field depends only on tile geometry, so it is computed once per level and
cached — the same treatment the mine SDF gets.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    GRAVITY_FALL,
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    MAX_SURVIVABLE_IMPACT,
    TERMINAL_VELOCITY,
    TILE_PIXEL_SIZE,
)
from ..value_function.surface_extraction import SURFACE_FLOOR, extract_surfaces
from .reflection import quantize_cell


def _canonical_tiles(tile_dict):
    """Pick a deterministic orientation for a ``{(col, row): type}`` tile map.

    Mirrors :func:`nclone.map_augmentation.transforms.canonical_map_data` but on
    the sim's tile dict, which is what this module is handed. Columns reflect
    about the bordered 44-tile width and every asymmetric tile ID swaps to its
    mirror variant -- flipping without the ID remap would face every slope the
    wrong way, which is geometrically plausible and physically wrong.

    Returns ``(canonical_tiles, was_mirrored)``.
    """
    from ..map_augmentation.transforms import _TILE_MIRROR

    last_col = int(MAP_TILE_WIDTH) + 1  # bordered grid is 0..MAP_TILE_WIDTH+1
    mirrored = {
        (last_col - c, r): _TILE_MIRROR.get(int(t), int(t))
        for (c, r), t in tile_dict.items()
    }
    key = lambda d: sorted(d.items())  # noqa: E731 - local, single use
    return (
        (dict(tile_dict), False)
        if key(tile_dict) <= key(mirrored)
        else (mirrored, True)
    )

# 6 px cells, matching hazard_map.
CELL_PX = TILE_PIXEL_SIZE / 4.0
# The sim frame carries a 1-tile border that the observation frame strips.
BORDER_CELLS = int(TILE_PIXEL_SIZE / CELL_PX)
# Normalizer for the drop channel: the full playable height. A fall longer than
# the level is tall is already saturated as "bottomless".
MAX_DROP_NORM_PX = float(MAP_TILE_HEIGHT * TILE_PIXEL_SIZE)
# Half-width of the column band a floor sample must fall in to count as "below"
# a given cell. extract_surfaces samples floors every 6 px (3 px near edges), so
# one cell width of tolerance always finds a sample on a real floor while not
# reaching across a genuine gap.
COLUMN_TOLERANCE_PX = CELL_PX

FALL_LETHALITY_CHANNELS = 2


def free_fall_speed(drop_px: float) -> float:
    """Speed after falling ``drop_px`` from rest, capped at terminal velocity.

    The sim integrates per frame (``v_n = n * g``, distance ``g*n*(n+1)/2``); the
    continuous ``sqrt(2*g*d)`` form agrees to within one frame of gravity and is
    what the V26 §1.8 drop table was derived from, so it is used here for an
    exact match with the documented thresholds.
    """
    if drop_px <= 0.0:
        return 0.0
    return min(math.sqrt(2.0 * GRAVITY_FALL * float(drop_px)), TERMINAL_VELOCITY)


def survivable_threshold(normal_y: float) -> float:
    """The impact speed at which ``ninja.py`` kills on this surface.

    Mirrors the sim exactly: ``MAX_SURVIVABLE_IMPACT - 4/3 * |n_y|``.
    """
    return MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * abs(float(normal_y))


def lethal_drop_px(normal_y: float) -> float:
    """Smallest fall (px) onto a surface with this normal that kills.

    ``inf`` when terminal velocity is reached before the threshold is (a steep
    enough slope is survivable from any height).
    """
    ny = abs(float(normal_y))
    thr = survivable_threshold(ny)
    if ny <= 1e-9:
        return float("inf")
    v_needed = thr / ny
    if v_needed >= TERMINAL_VELOCITY:
        return float("inf")
    # v > v_needed, so the drop must exceed v_needed^2 / (2g).
    return (v_needed * v_needed) / (2.0 * GRAVITY_FALL)


class _FloorColumns:
    """Floor sample points bucketed by 6 px column, sorted top-to-bottom."""

    __slots__ = ("ys", "nys", "starts")

    def __init__(self, ys: np.ndarray, nys: np.ndarray, starts: np.ndarray):
        self.ys = ys
        self.nys = nys
        self.starts = starts


def _bucket_floor_points(
    surface_points, width: int, center_registered: bool = False
) -> _FloorColumns:
    """Bucket floor SurfacePoints into 6 px columns of the OBSERVATION frame.

    Each point is widened to every column within ``COLUMN_TOLERANCE_PX`` so a
    cell whose center sits between two 6 px floor samples still finds the floor.

    ``center_registered`` (``mirror_equivariant_obs``) selects columns whose
    CENTRE lies within the tolerance, rather than flooring the tolerance band's
    endpoints onto column indices. The floored form is asymmetric exactly when
    ``p.x +- tol`` falls on a column boundary -- which, since surface samples and
    the tolerance are both multiples of the 6 px cell, is EVERY sample. The
    visible effect is that a platform's end column is present in a level and
    absent in its mirror, so the mirrored field reports a bottomless drop
    (-1.0 margin) exactly at a platform edge. The centre test is symmetric by
    construction and matches what the docstring above already claims.
    """
    xs: list[int] = []
    ys: list[float] = []
    nys: list[float] = []
    for p in surface_points:
        if p.surface_type != SURFACE_FLOOR:
            continue
        # World px -> observation-frame column, matching hazard_map's framing.
        if center_registered:
            # Column c's centre is at (c + BORDER_CELLS + 0.5) * CELL_PX.
            lo = (p.x - COLUMN_TOLERANCE_PX) / CELL_PX - BORDER_CELLS - 0.5
            hi = (p.x + COLUMN_TOLERANCE_PX) / CELL_PX - BORDER_CELLS - 0.5
            c_lo = int(math.ceil(lo))
            c_hi = int(math.floor(hi))
        else:
            c_lo = int(math.floor((p.x - COLUMN_TOLERANCE_PX) / CELL_PX)) - BORDER_CELLS
            c_hi = int(math.floor((p.x + COLUMN_TOLERANCE_PX) / CELL_PX)) - BORDER_CELLS
        for c in range(max(0, c_lo), min(width - 1, c_hi) + 1):
            xs.append(c)
            ys.append(float(p.y))
            nys.append(abs(float(p.normal_y)))

    if not xs:
        return _FloorColumns(
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(width + 1, dtype=np.int64),
        )

    xs_a = np.asarray(xs, dtype=np.int64)
    ys_a = np.asarray(ys, dtype=np.float32)
    nys_a = np.asarray(nys, dtype=np.float32)

    # Sort by (column, y) so each column's slice is already top-to-bottom and a
    # searchsorted inside the slice finds the first floor below a given y.
    order = np.lexsort((ys_a, xs_a))
    xs_a, ys_a, nys_a = xs_a[order], ys_a[order], nys_a[order]
    starts = np.searchsorted(xs_a, np.arange(width + 1), side="left")
    return _FloorColumns(ys_a, nys_a, starts.astype(np.int64))


def compute_fall_lethality_map(
    tile_dict: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    surface_points=None,
    center_registered: bool = False,
    canonical: bool = False,
) -> np.ndarray:
    """Build the (2, width, height) fall-lethality field for a level.

    Args:
        tile_dict: ``{(col, row): tile_type}`` from ``simulator.get_tile_dict()``.
        width, height: field dimensions in 6 px cells (``HAZARD_MAP_WIDTH/HEIGHT``).
        surface_points: optional pre-extracted surfaces, to avoid a second
            ``extract_surfaces`` pass when the caller already has them. Mine
            splitting MUST be off — a mine does not stop a fall, and splitting
            the floor at one would punch a phantom bottomless hole in the field.

    Returns:
        ``(2, width, height) float32`` — ch0 signed survivable margin in
        ``[-1, 1]``, ch1 normalized drop distance in ``[0, 1]``. Axis order
        matches ``tile_grid`` / ``hazard_map``.
    """
    if canonical:
        # Same lever as the free-space field: ``extract_surfaces`` samples floors
        # every 6 px but every 3 px NEAR SEGMENT ENDS, so the sample SET is not
        # symmetric and a platform's end column can be present in a level and
        # absent in its mirror -- reading as a bottomless drop exactly at the
        # edge where the fall decision is made. Building on the canonical
        # orientation and reflecting the field removes the asymmetry outright.
        tile_dict, mirrored = _canonical_tiles(tile_dict)
        if mirrored:
            field = compute_fall_lethality_map(
                tile_dict, width, height, surface_points, center_registered,
                canonical=False,
            )
            # The obs frame spans px [24, 1032), symmetric about the level's
            # axis, so a plain flip IS the reflection here (unlike the
            # free-space grid, which needs padding first).
            return np.ascontiguousarray(field[:, ::-1, :])

    out = np.empty((FALL_LETHALITY_CHANNELS, width, height), dtype=np.float32)
    # Default: nothing below -> a bottomless fall -> maximally lethal.
    out[0].fill(-1.0)
    out[1].fill(1.0)

    if surface_points is None:
        surface_points = extract_surfaces(tile_dict)
    cols = _bucket_floor_points(surface_points, width, center_registered)
    if cols.ys.size == 0:
        return out

    # World y of each cell's center, in the SIM frame (border included).
    cell_y = (np.arange(height, dtype=np.float32) + BORDER_CELLS + 0.5) * CELL_PX

    for cx in range(width):
        lo, hi = int(cols.starts[cx]), int(cols.starts[cx + 1])
        if hi <= lo:
            continue
        col_y = cols.ys[lo:hi]
        col_ny = cols.nys[lo:hi]
        # First floor sample at or below each cell center, with one cell of
        # tolerance ABOVE it. The field is quantized to 6 px, so a floor sample
        # within one cell of the center belongs to that cell — without the
        # tolerance, a ninja STANDING on a platform lands a pixel below its own
        # surface sample, the search skips past it, and the cell reports the drop
        # to the NEXT floor down. That reads "you are 8 tiles above a fatal
        # landing" while the ninja is standing still on solid ground.
        idx = np.searchsorted(col_y, cell_y - CELL_PX, side="right")
        has_floor = idx < col_y.size
        if not has_floor.any():
            continue
        safe_idx = np.where(has_floor, idx, 0)
        land_y = col_y[safe_idx]
        land_ny = col_ny[safe_idx]

        drop = np.maximum(land_y - cell_y, 0.0)
        v = np.minimum(
            np.sqrt(2.0 * GRAVITY_FALL * drop), float(TERMINAL_VELOCITY)
        )
        # The sim projects the incoming velocity onto the surface normal, and the
        # threshold itself relaxes on steeper surfaces — both matter, and they
        # move in opposite directions.
        impact = land_ny * v
        thr = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * land_ny
        margin = np.clip((thr - impact) / MAX_SURVIVABLE_IMPACT, -1.0, 1.0)
        drop_norm = np.clip(drop / MAX_DROP_NORM_PX, 0.0, 1.0)

        out[0, cx] = np.where(has_floor, margin, -1.0).astype(np.float32)
        out[1, cx] = np.where(has_floor, drop_norm, 1.0).astype(np.float32)

    return out


def crop_fall_lethality(
    field: Optional[np.ndarray],
    nx: float,
    ny: float,
    size: int,
    odd_dir: int = 1,
) -> np.ndarray:
    """Slice a ninja-centered ``(2, size, size)`` window out of the field.

    Off-grid cells fill ``(+1, 0)`` — "standing on solid ground, no drop" — which
    is correct at the level border (a wall, not a pit) and keeps a crop that
    overhangs the edge from reading as danger.

    ``odd_dir`` is the shared reflection-odd tie-break sign (see
    ``gym_environment/reflection.py``): at ``+1`` the window origin is the plain
    floor this always used, and only the exact-cell-boundary case differs.
    """
    out = np.empty((FALL_LETHALITY_CHANNELS, size, size), dtype=np.float32)
    out[0].fill(1.0)
    out[1].fill(0.0)
    if field is None:
        return out

    half = size // 2
    _, W, H = field.shape
    hx = quantize_cell(nx, CELL_PX, odd_dir) - BORDER_CELLS
    hy = int(ny / CELL_PX) - BORDER_CELLS
    sx0, sy0 = hx - half, hy - half
    gx0, gy0 = max(0, sx0), max(0, sy0)
    gx1, gy1 = min(W, sx0 + size), min(H, sy0 + size)
    if gx1 > gx0 and gy1 > gy0:
        out[:, gx0 - sx0 : gx1 - sx0, gy0 - sy0 : gy1 - sy0] = field[
            :, gx0:gx1, gy0:gy1
        ]
    return out
