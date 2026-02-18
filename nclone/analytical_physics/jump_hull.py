"""Precomputed jump hulls: geometry-independent reachability regions.

For each (surface_type, velocity_bin), precomputes the set of 12px grid cells
that the ninja can pass through during ANY single-jump trajectory.  These hulls
are level-independent and computed once at startup (~5 s).

Each trajectory is simulated until the ninja's downward velocity exceeds
``MAX_SURVIVABLE_IMPACT``.  This means hulls extend *below* launch height,
capturing all positions reachable during the descent phase of a jump -- critical
for slope jumps where the ninja can launch far from the slope and land on
platforms well below it.

At graph-build time the hull is translated to each source surface node and
intersected with the level's reachable node set (from flood-fill) to produce
exact jump edges -- no affine approximation, no speed-cap workaround.

Physics integration matches ninja.py exactly:
    integrate():
        vx *= DRAG_REGULAR
        vy *= DRAG_REGULAR
        vy += gravity               (GRAVITY_JUMP or GRAVITY_FALL)
        px += vx
        py += vy
    think():
        vx_new = vx + h * AIR_ACCEL
        if abs(vx_new) < MAX_HOR_SPEED:
            vx = vx_new             (rejection model)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nclone.constants.physics_constants import (
    AIR_ACCEL,
    DRAG_REGULAR,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    JUMP_FLOOR_Y,
    JUMP_SLOPE_DOWNHILL_X_MULTIPLIER,
    JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER,
    JUMP_SLOPE_UPHILL_DEFAULT_Y,
    JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER,
    JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = DRAG_REGULAR  # ~0.9933
GRID_CELL = 12  # Sub-node spacing in pixels

# Velocity bin configuration (mirrors surface_graph.py)
NUM_VELOCITY_BINS_FLOOR = 15
FLOOR_VEL_MIN = -MAX_HOR_SPEED
FLOOR_VEL_MAX = MAX_HOR_SPEED
_FLOOR_BIN_CENTERS = np.linspace(
    FLOOR_VEL_MIN, FLOOR_VEL_MAX, NUM_VELOCITY_BINS_FLOOR, dtype=np.float64,
)

# Wall velocity bins: vertical velocity while wall-attached (sliding)
NUM_VELOCITY_BINS_WALL = 12
WALL_VEL_MIN = -2.0   # wall-jump upward impulse with margin
WALL_VEL_MAX = 5.5     # approaching survivable impact limit
_WALL_BIN_CENTERS = np.linspace(
    WALL_VEL_MIN, WALL_VEL_MAX, NUM_VELOCITY_BINS_WALL, dtype=np.float64,
)

# Maximum simulation frames (safety bound)
_MAX_SIM_FRAMES = 500

# ---------------------------------------------------------------------------
# Surface type classification
# ---------------------------------------------------------------------------
# Flat ground
SURFACE_FLAT = 0
# Slopes: left/right refer to the sign of normal_x
# Mild slopes -- tiles 18-25  (rise/run = 12/24)
SURFACE_SLOPE_MILD_L = 1   # normal_x < 0
SURFACE_SLOPE_MILD_R = 2   # normal_x > 0
# 45-degree slopes -- tiles 6-9  (rise/run = 24/24)
SURFACE_SLOPE_45_L = 3     # normal_x < 0
SURFACE_SLOPE_45_R = 4     # normal_x > 0
# Steep slopes -- tiles 26-33  (rise/run = 24/12)
SURFACE_SLOPE_STEEP_L = 5  # normal_x < 0
SURFACE_SLOPE_STEEP_R = 6  # normal_x > 0

NUM_SURFACE_TYPES = 7

# Precomputed unit normals for each slope type  (nx, ny)
# These are the outward normals (away from solid, pointing "up" into free space).
# For floor surfaces ny < 0  (screen-space Y-down).
_SURFACE_NORMALS: Dict[int, Tuple[float, float]] = {
    SURFACE_FLAT: (0.0, -1.0),
    # Mild: direction (24, -12) -> normal = (12, 24)/|..| normalised
    # length = sqrt(12^2+24^2) = sqrt(720) = 26.833
    SURFACE_SLOPE_MILD_L: (-12.0 / math.sqrt(720), -24.0 / math.sqrt(720)),
    SURFACE_SLOPE_MILD_R: (12.0 / math.sqrt(720), -24.0 / math.sqrt(720)),
    # 45-deg: direction (24, -24) -> normal = (24, 24)/|..| normalised
    SURFACE_SLOPE_45_L: (-math.sqrt(2) / 2, -math.sqrt(2) / 2),
    SURFACE_SLOPE_45_R: (math.sqrt(2) / 2, -math.sqrt(2) / 2),
    # Steep: direction (12, -24) -> normal = (24, 12)/|..| normalised
    SURFACE_SLOPE_STEEP_L: (-24.0 / math.sqrt(720), -12.0 / math.sqrt(720)),
    SURFACE_SLOPE_STEEP_R: (24.0 / math.sqrt(720), -12.0 / math.sqrt(720)),
}

# Tolerance for flat floor detection
_FLAT_NORMAL_TOL = 0.01


def classify_surface(normal_x: float, normal_y: float) -> int:
    """Classify a floor surface by its normal into one of 7 types."""
    if abs(normal_x) < _FLAT_NORMAL_TOL:
        return SURFACE_FLAT

    # Determine slope angle from |nx|
    abs_nx = abs(normal_x)
    sign_right = normal_x > 0

    # Mild: |nx| ~ 0.447
    # 45-deg: |nx| ~ 0.707
    # Steep: |nx| ~ 0.894
    # Use midpoints as boundaries:  0.577 between mild/45,  0.800 between 45/steep
    if abs_nx < 0.577:
        return SURFACE_SLOPE_MILD_R if sign_right else SURFACE_SLOPE_MILD_L
    elif abs_nx < 0.800:
        return SURFACE_SLOPE_45_R if sign_right else SURFACE_SLOPE_45_L
    else:
        return SURFACE_SLOPE_STEEP_R if sign_right else SURFACE_SLOPE_STEEP_L


def surface_normal(surface_type: int) -> Tuple[float, float]:
    """Return the unit normal (nx, ny) for a surface type."""
    return _SURFACE_NORMALS[surface_type]


# ---------------------------------------------------------------------------
# Hull cell data
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HullCell:
    """One reachable grid cell in a jump hull.

    Attributes:
        min_frames: Minimum frame count (from jump start) to reach this cell.
        landing_vx: Horizontal velocity of the trajectory when it first
            reaches this cell at ``min_frames``.
        landing_vy: Vertical velocity at ``min_frames`` (positive = falling).
    """
    min_frames: int
    landing_vx: float
    landing_vy: float


# Type alias:  (cell_dx, cell_dy) -> HullCell
JumpHull = Dict[Tuple[int, int], HullCell]


# ---------------------------------------------------------------------------
# Slope impulse computation  (mirrors ninja.py floor_jump exactly)
# ---------------------------------------------------------------------------
def compute_slope_impulse(
    normal_x: float,
    normal_y: float,
    vx0: float,
    h1: int,
) -> Tuple[float, float, float, float]:
    """Compute jump impulse and post-impulse velocities for a floor jump.

    Mirrors ninja.py ``floor_jump`` (lines 539-578) exactly, including
    the pre-jump vertical velocity from slope movement.

    On slopes, the ninja's velocity is projected onto the slope tangent
    (ninja.py line 348-354), giving ``yspeed = xspeed * (-nx / ny)``.
    When ascending (yspeed < 0), this upward velocity is *preserved*
    across the jump (ninja.py only clamps positive yspeed to 0) and the
    jump impulse is added on top.

    Returns:
        (jx, jy, vx_after, vy_after):
            jx, jy -- impulse added to both velocity and position.
            vx_after -- horizontal velocity after impulse (accounts for
                the perpendicular-jump xspeed reset).
            vy_after -- vertical velocity after impulse, including the
                pre-jump slope velocity contribution.
    """
    # Compute pre-jump vy from slope geometry.  On the slope the velocity
    # is along the tangent: vy/vx = -nx/ny.  Must use the ORIGINAL vx0
    # because ninja.py's yspeed is set before the perp branch zeroes xspeed.
    if abs(normal_x) < _FLAT_NORMAL_TOL:
        # Flat ground: vy_pre = 0, standard impulse
        vy_after = JUMP_FLOOR_Y  # 0 + jy
        return 0.0, JUMP_FLOOR_Y, vx0, vy_after

    # Slope: derive pre-jump vy from slope tangent projection
    slope_ratio = -normal_x / normal_y  # vy/vx on slope surface
    vy_pre = vx0 * slope_ratio

    dx = normal_x
    dy = normal_y

    if vx0 * dx >= 0:
        # Moving downhill (or stationary on slope)
        if vx0 * h1 >= 0:
            # Downhill with input
            jx = JUMP_SLOPE_DOWNHILL_X_MULTIPLIER * dx
            jy = JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER * dy
        else:
            # Downhill against input
            jx = 0.0
            jy = JUMP_SLOPE_UPHILL_DEFAULT_Y
    else:
        # Moving uphill
        if vx0 * h1 > 0:
            # Uphill forward jump
            jx = 0.0
            jy = JUMP_SLOPE_UPHILL_DEFAULT_Y
        else:
            # Uphill perpendicular jump -- xspeed zeroed
            jx = JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER * dx
            jy = JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER * dy
            # ninja.py: self.xspeed = 0 before adding jx
            # (yspeed is NOT zeroed -- it retains the slope velocity)
            vx0 = 0.0

    # ninja.py: if self.yspeed > 0: self.yspeed = 0
    # Clamp positive (downward) pre-jump vy to 0; preserve negative (upward).
    vy_clamped = min(vy_pre, 0.0)

    vx_after = vx0 + jx
    vy_after = vy_clamped + jy
    return jx, jy, vx_after, vy_after


# ---------------------------------------------------------------------------
# Frame-by-frame trajectory simulation
# ---------------------------------------------------------------------------
def _simulate_trajectory(
    vx0: float,
    vy0: float,
    px0: float,
    py0: float,
    j: int,
    h1: int,
    h2: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Simulate a two-phase jump trajectory frame by frame.

    Phase 1 (frames 1..j):   GRAVITY_JUMP, horizontal input h1.
    Phase 2 (frames j+1..):  GRAVITY_FALL, horizontal input h2.

    Continues until the downward velocity exceeds MAX_SURVIVABLE_IMPACT
    (the ninja would die on landing), tracing the full arc including
    any descent below launch height.  This is critical for slope jumps
    where the ninja can reach platforms well below the launch point.

    Returns list of (frame, dx, dy, vx, vy) where dx/dy are offsets
    from (0, 0).
    """
    vx = vx0
    vy = vy0
    px = px0
    py = py0

    points: List[Tuple[int, float, float, float, float]] = []

    for frame in range(1, _MAX_SIM_FRAMES + 1):
        # --- integrate() ---
        vx *= D
        vy *= D
        gravity = GRAVITY_JUMP if frame <= j else GRAVITY_FALL
        vy += gravity
        px += vx
        py += vy

        # --- think() horizontal accel with cap ---
        h = h1 if frame <= j else h2
        vx_new = vx + h * AIR_ACCEL
        if abs(vx_new) < MAX_HOR_SPEED:
            vx = vx_new

        # Stop if fall velocity exceeds survivable impact threshold.
        # Any landing past this point would be lethal.
        if vy > MAX_SURVIVABLE_IMPACT:
            break

        points.append((frame, px, py, vx, vy))

    return points


# ---------------------------------------------------------------------------
# Build a single hull
# ---------------------------------------------------------------------------
def _build_hull(
    normal_x: float,
    normal_y: float,
    vx_bin_center: float,
) -> JumpHull:
    """Build one jump hull for a given surface normal and velocity bin.

    Sweeps all (j, h1, h2) control sequences, simulates each frame by frame,
    and records every 12px grid cell the ninja passes through.
    """
    hull: JumpHull = {}

    for j in range(0, MAX_JUMP_DURATION + 1):
        for h1 in [-1, 0, 1]:
            # Compute slope-aware impulse (includes pre-jump slope vy)
            jx, jy, vx_after, vy_after = compute_slope_impulse(
                normal_x, normal_y, vx_bin_center, h1,
            )

            for h2 in [-1, 0, 1]:
                points = _simulate_trajectory(
                    vx0=vx_after,
                    vy0=vy_after,
                    px0=jx,
                    py0=jy,
                    j=j,
                    h1=h1,
                    h2=h2,
                )

                for frame, dx, dy, vx_at, vy_at in points:
                    cx = round(dx / GRID_CELL)
                    cy = round(dy / GRID_CELL)
                    key = (cx, cy)
                    existing = hull.get(key)
                    if existing is None or frame < existing.min_frames:
                        hull[key] = HullCell(
                            min_frames=frame,
                            landing_vx=vx_at,
                            landing_vy=vy_at,
                        )

    return hull


# ---------------------------------------------------------------------------
# Hull collection
# ---------------------------------------------------------------------------
class JumpHullCollection:
    """Precomputed jump hulls for all surface types and velocity bins.

    Usage::

        collection = JumpHullCollection.build()   # once at startup
        hull = collection.get_hull(surface_type, velocity_bin)
    """

    def __init__(
        self,
        hulls: Dict[Tuple[int, int], JumpHull],
    ):
        self._hulls = hulls

    def get_hull(self, surface_type: int, velocity_bin: int) -> JumpHull:
        """Look up a hull by surface type and velocity bin index."""
        return self._hulls.get((surface_type, velocity_bin), {})

    @property
    def num_hulls(self) -> int:
        return len(self._hulls)

    @staticmethod
    def build() -> "JumpHullCollection":
        """Build all hulls for every (surface_type, velocity_bin) pair.

        Total: NUM_SURFACE_TYPES * NUM_VELOCITY_BINS_FLOOR hulls.
        """
        t0 = time.perf_counter()

        hulls: Dict[Tuple[int, int], JumpHull] = {}
        total_cells = 0

        for stype in range(NUM_SURFACE_TYPES):
            nx, ny = surface_normal(stype)

            for vb in range(NUM_VELOCITY_BINS_FLOOR):
                vx_center = float(_FLOOR_BIN_CENTERS[vb])
                hull = _build_hull(nx, ny, vx_center)
                hulls[(stype, vb)] = hull
                total_cells += len(hull)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Built {len(hulls)} jump hulls "
            f"({total_cells} total cells) in {elapsed_ms:.0f}ms"
        )

        return JumpHullCollection(hulls)


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------
_cached_collection: Optional[JumpHullCollection] = None


def get_or_build_hulls() -> JumpHullCollection:
    """Return the singleton hull collection, building on first call."""
    global _cached_collection
    if _cached_collection is None:
        _cached_collection = JumpHullCollection.build()
    return _cached_collection


# ---------------------------------------------------------------------------
# Wall jump impulse computation  (mirrors ninja.py wall_jump exactly)
# ---------------------------------------------------------------------------
def compute_wall_impulse(
    wall_normal_sign: int,
    jump_type: str,
    vy_pre: float,
) -> Tuple[float, float, float, float]:
    """Compute wall jump impulse and post-impulse velocities.

    Mirrors ninja.py ``wall_jump`` (lines 581-608) exactly.  Assumes
    ``xspeed_pre = 0`` because a wall-attached ninja always has zero
    horizontal velocity (contact with the wall prevents movement).

    The wall jump physics:
        1. If xspeed opposes wall normal -> zeroed (always true since 0).
        2. If yspeed > 0 (falling) -> clamped to 0.
        3. xspeed += jx * wall_normal; yspeed += jy.
        4. xpos += jx * wall_normal; ypos += jy.

    Args:
        wall_normal_sign: +1 for left wall (normal points right),
            -1 for right wall (normal points left).
        jump_type: ``"regular"`` or ``"slide"``.
        vy_pre: Vertical velocity before the jump (from wall sliding).

    Returns:
        (px0, py0, vx_after, vy_after):
            px0, py0 -- position offset from impulse.
            vx_after -- horizontal velocity after impulse.
            vy_after -- vertical velocity after impulse.
    """
    if jump_type == "regular":
        jx = JUMP_WALL_REGULAR_X_MULTIPLIER
        jy = JUMP_WALL_REGULAR_Y
    else:
        jx = JUMP_WALL_SLIDE_X_MULTIPLIER
        jy = JUMP_WALL_SLIDE_Y

    # xspeed assumed 0 (wall-attached ninja)
    # yspeed clamped: if vy_pre > 0 -> 0 (ninja.py: if self.yspeed > 0: self.yspeed = 0)
    vy_clamped = min(vy_pre, 0.0)

    vx_after = jx * wall_normal_sign
    vy_after = vy_clamped + jy

    px0 = jx * wall_normal_sign
    py0 = jy

    return px0, py0, vx_after, vy_after


# ---------------------------------------------------------------------------
# Build a single wall hull
# ---------------------------------------------------------------------------
def _build_wall_hull(
    wall_normal_sign: int,
    jump_type: str,
    vy_pre: float,
) -> JumpHull:
    """Build one wall jump hull for a given wall side, jump type, and vy.

    Sweeps all (j, h1, h2) control sequences, simulates each frame by frame,
    and records every 12px grid cell the ninja passes through.

    Unlike floor hulls where ``h1`` affects the impulse (via
    ``compute_slope_impulse``), wall impulse is independent of ``h1``.
    The impulse is computed once and ``h1`` only affects the airborne
    trajectory via horizontal acceleration each frame.  The full sweep
    ``h1 in [-1, 0, 1]`` captures all post-jump directional holds
    (toward wall, neutral, away from wall).
    """
    hull: JumpHull = {}

    # Wall impulse is fixed for all (j, h1, h2) -- compute once
    px0, py0, vx_after, vy_after = compute_wall_impulse(
        wall_normal_sign, jump_type, vy_pre,
    )

    for j in range(0, MAX_JUMP_DURATION + 1):
        for h1 in [-1, 0, 1]:
            for h2 in [-1, 0, 1]:
                points = _simulate_trajectory(
                    vx0=vx_after,
                    vy0=vy_after,
                    px0=px0,
                    py0=py0,
                    j=j,
                    h1=h1,
                    h2=h2,
                )

                for frame, dx, dy, vx_at, vy_at in points:
                    cx = round(dx / GRID_CELL)
                    cy = round(dy / GRID_CELL)
                    key = (cx, cy)
                    existing = hull.get(key)
                    if existing is None or frame < existing.min_frames:
                        hull[key] = HullCell(
                            min_frames=frame,
                            landing_vx=vx_at,
                            landing_vy=vy_at,
                        )

    return hull


# ---------------------------------------------------------------------------
# Wall hull collection
# ---------------------------------------------------------------------------
class WallHullCollection:
    """Precomputed wall jump hulls for all (wall_side, jump_type, vy_bin).

    Wall hulls are parameterized by pre-jump vertical velocity because
    the ninja's sliding velocity affects the trajectory.  ``vy_pre > 0``
    (falling) is clamped to 0 before the impulse, so all falling bins
    produce identical hulls.

    Total: 2 wall sides * 2 jump types * NUM_VELOCITY_BINS_WALL = 48 hulls.

    Usage::

        collection = WallHullCollection.build()
        hull = collection.get_hull(wall_normal_sign=1, jump_type="regular", vy_bin=0)
    """

    def __init__(
        self,
        hulls: Dict[Tuple[int, str, int], JumpHull],
    ):
        self._hulls = hulls

    def get_hull(
        self,
        wall_normal_sign: int,
        jump_type: str,
        vy_bin: int,
    ) -> JumpHull:
        """Look up a wall hull by wall side, jump type, and vy bin index."""
        return self._hulls.get((wall_normal_sign, jump_type, vy_bin), {})

    @property
    def num_hulls(self) -> int:
        return len(self._hulls)

    @staticmethod
    def build() -> "WallHullCollection":
        """Build all wall hulls for every (wall_side, jump_type, vy_bin).

        Total: 2 * 2 * NUM_VELOCITY_BINS_WALL hulls.
        """
        t0 = time.perf_counter()

        hulls: Dict[Tuple[int, str, int], JumpHull] = {}
        total_cells = 0

        for wall_sign in [1, -1]:
            for jtype in ["regular", "slide"]:
                for vb in range(NUM_VELOCITY_BINS_WALL):
                    vy_center = float(_WALL_BIN_CENTERS[vb])
                    hull = _build_wall_hull(wall_sign, jtype, vy_center)
                    hulls[(wall_sign, jtype, vb)] = hull
                    total_cells += len(hull)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Built {len(hulls)} wall jump hulls "
            f"({total_cells} total cells) in {elapsed_ms:.0f}ms"
        )

        return WallHullCollection(hulls)


# ---------------------------------------------------------------------------
# Wall hull singleton cache
# ---------------------------------------------------------------------------
_cached_wall_collection: Optional[WallHullCollection] = None


def get_or_build_wall_hulls() -> WallHullCollection:
    """Return the singleton wall hull collection, building on first call."""
    global _cached_wall_collection
    if _cached_wall_collection is None:
        _cached_wall_collection = WallHullCollection.build()
    return _cached_wall_collection
