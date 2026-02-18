"""Trajectory catalog: precomputed wall jump and airborne trajectory data.

Wall jump entries are fully precomputed constant offsets (no velocity
dependence).  Floor jump entries have been replaced by the hull-based
system in ``jump_hull.py`` -- the ``TrajectoryEntry`` dataclass is
retained for backward compatibility but ``floor_flat`` is always empty.
"""

import math
from dataclasses import dataclass
from typing import Optional

from nclone.constants import (
    DRAG_REGULAR,
    GRAVITY_JUMP,
    GRAVITY_FALL,
    AIR_ACCEL,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    JUMP_FLOOR_Y,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
)
from .analytical_trajectory import (
    S1,
    S2,
    Sx2,
    D,
    D_MINUS_1,
    _dn,
    velocity_at,
    position_y_at,
    solve_landing_frame,
    find_cap_frame,
)

# Mid-arc sampling interval in frames
ARC_SAMPLE_INTERVAL = 5


@dataclass(frozen=True)
class ArcPoint:
    """Intermediate arc point with affine coefficients.

    At frame `frame` of the trajectory, the ninja's position is:
        px = ninja_x + C * vx0 + Bx
        py = ninja_y + C * vy0 + By

    These are the same affine formulas as the landing point, evaluated
    at an intermediate frame rather than at k_land.

    Attributes:
        frame: Frame number from trajectory start.
        C: Velocity coefficient at this frame.
        Bx: x-bias at this frame (depends on h1, h2).
        By: y-bias at this frame (depends on gravity).
    """

    frame: int
    C: float
    Bx: float
    By: float


@dataclass(frozen=True)
class TrajectoryEntry:
    """Single precomputed trajectory with affine coefficients.

    Landing position for any continuous initial velocity (vx0, vy0):
        land_x = ninja_x + C * vx0 + Bx
        land_y = ninja_y + C * vy0 + By

    Attributes:
        j: Jump hold duration in frames [0, MAX_JUMP_DURATION].
        h1: Horizontal input during hold {-1, 0, +1}.
        h2: Horizontal input after release {-1, 0, +1}.
        k_land: Landing frame in phase 2.
        total_frames: j + k_land (total arc duration).
        C: Affine velocity coefficient: S1(j) + D^j * S1(k_land).
        Bx: x-axis position bias (corrected for think-after-integrate order).
        By: y-axis position bias.
        cap_threshold: |vx0| above which the horizontal speed cap binds.
        landing_vy: Vertical velocity at landing (for death check).
        C_vx_land: Landing vx affine coefficient: D^(j+k).
        Bx_vx_land: Landing vx bias.
        arc_points: Precomputed intermediate arc points sampled every
            ARC_SAMPLE_INTERVAL frames where the ninja is above launch height.
    """

    j: int
    h1: int
    h2: int
    k_land: int
    total_frames: int
    C: float
    Bx: float
    By: float
    cap_threshold: float
    landing_vy: float
    C_vx_land: float
    Bx_vx_land: float
    arc_points: tuple[ArcPoint, ...] = ()


@dataclass(frozen=True)
class WallTrajectoryEntry:
    """Wall jump trajectory with fully precomputed landing offsets.

    Wall jumps replace velocity entirely, so both vx0 and vy0 are fixed
    constants. No affine evaluation needed -- just add (dx, dy) to ninja position.

    Attributes:
        j: Jump hold duration.
        h1: Horizontal input during hold.
        h2: Horizontal input after release.
        total_frames: j + k_land.
        dx: Precomputed landing x offset from ninja position.
        dy: Precomputed landing y offset from ninja position.
        landing_vy: Vertical velocity at landing.
        arc_offsets: Precomputed intermediate (frame, dx, dy) offsets
            sampled every ARC_SAMPLE_INTERVAL frames where the ninja is
            above launch height.
    """

    j: int
    h1: int
    h2: int
    total_frames: int
    dx: float
    dy: float
    landing_vy: float
    arc_offsets: tuple[tuple[int, float, float], ...] = ()


@dataclass(frozen=True)
class AirborneTrajectoryEntry:
    """Airborne trajectory template (landing frame solved at runtime).

    For airborne states, the landing frame depends on vy0 which is a free
    variable. These entries store the horizontal input and release timing;
    at runtime, k_land is solved per-state and C/B are evaluated analytically.

    Attributes:
        h: Horizontal input {-1, 0, +1}.
        release_at: Frames to continue holding jump before release (0 = already released).
    """

    h: int
    release_at: int


class TrajectoryCatalog:
    """Precomputed trajectory catalog for all control sequences.

    Contains:
        floor_flat: ~380 TrajectoryEntry for flat ground jumps.
        wall: dict mapping (wall_normal_sign, jump_type) -> list[WallTrajectoryEntry].
        airborne_falling: 3 AirborneTrajectoryEntry (one per h).
        airborne_holding: ~30 AirborneTrajectoryEntry (release_at x h).
    """

    def __init__(
        self,
        floor_flat: list[TrajectoryEntry],
        wall: dict[tuple[int, str], list[WallTrajectoryEntry]],
        airborne_falling: list[AirborneTrajectoryEntry],
        airborne_holding: list[AirborneTrajectoryEntry],
    ):
        self.floor_flat = floor_flat
        self.wall = wall
        self.airborne_falling = airborne_falling
        self.airborne_holding = airborne_holding

    @staticmethod
    def build() -> "TrajectoryCatalog":
        """Build the trajectory catalog (wall + airborne entries).

        Floor jump entries are no longer built here -- they have been
        replaced by the hull-based system in ``jump_hull.py``.

        Returns:
            TrajectoryCatalog with wall and airborne entries.
        """
        wall = _build_wall_catalogs()
        airborne_falling = _build_airborne_falling()
        airborne_holding = _build_airborne_holding()

        return TrajectoryCatalog(
            floor_flat=[],  # Replaced by hull-masked jump edges
            wall=wall,
            airborne_falling=airborne_falling,
            airborne_holding=airborne_holding,
        )


def _compute_arc_offsets_wall(
    vx0: float,
    vy0: float,
    j: int,
    k_land: int,
    h1: int,
    h2: int,
    interval: int = ARC_SAMPLE_INTERVAL,
) -> tuple[tuple[int, float, float], ...]:
    """Compute intermediate arc offsets for a wall jump trajectory.

    Wall jumps have fixed vx0/vy0, so intermediate positions are fully
    precomputed as (frame, dx, dy) offsets from ninja position.

    Only includes points where the ninja is above launch height (dy < 0).

    Args:
        vx0: Fixed initial horizontal velocity (from wall impulse).
        vy0: Fixed initial vertical velocity (from wall impulse).
        j: Jump hold duration.
        k_land: Landing frame in phase 2.
        h1: Horizontal input during hold.
        h2: Horizontal input after release.
        interval: Sampling interval in frames.

    Returns:
        Tuple of (frame, dx, dy) for each sampled intermediate point.
    """
    from .analytical_trajectory import jump_trajectory

    total_frames = j + k_land
    offsets: list[tuple[int, float, float]] = []

    for frame in range(interval, total_frames, interval):
        if frame <= j:
            px, py, _, _ = jump_trajectory(
                0.0,
                0.0,
                vx0,
                vy0,
                h1,
                frame,
                h2,
                0,
            )
        else:
            k2 = frame - j
            px, py, _, _ = jump_trajectory(
                0.0,
                0.0,
                vx0,
                vy0,
                h1,
                j,
                h2,
                k2,
            )

        # Only include above-launch points
        if py >= 0:
            continue

        offsets.append((frame, px, py))

    return tuple(offsets)


def _build_wall_catalogs() -> dict[tuple[int, str], list[WallTrajectoryEntry]]:
    """Build wall jump trajectory catalogs for all wall types.

    Wall jumps replace velocity entirely, so all landing points are
    precomputed constants (no affine evaluation needed).
    Includes precomputed mid-arc offsets for each entry.

    4 wall types: (left_wall, regular), (left_wall, slide),
                  (right_wall, regular), (right_wall, slide)
    """
    from .analytical_trajectory import jump_trajectory

    wall_types = [
        # (wall_normal_sign, jump_type, vx0, vy0)
        (1, "regular", JUMP_WALL_REGULAR_X_MULTIPLIER * 1, JUMP_WALL_REGULAR_Y),
        (-1, "regular", JUMP_WALL_REGULAR_X_MULTIPLIER * -1, JUMP_WALL_REGULAR_Y),
        (1, "slide", JUMP_WALL_SLIDE_X_MULTIPLIER * 1, JUMP_WALL_SLIDE_Y),
        (-1, "slide", JUMP_WALL_SLIDE_X_MULTIPLIER * -1, JUMP_WALL_SLIDE_Y),
    ]

    catalogs: dict[tuple[int, str], list[WallTrajectoryEntry]] = {}

    for normal_sign, jump_type, vx0, vy0 in wall_types:
        entries: list[WallTrajectoryEntry] = []

        for j in range(0, MAX_JUMP_DURATION + 1):
            k_land = solve_landing_frame(vy0, j)
            if k_land is None:
                continue

            # Check landing velocity
            vy_j = velocity_at(vy0, GRAVITY_JUMP, j)
            landing_vy = velocity_at(vy_j, GRAVITY_FALL, k_land)
            if landing_vy > MAX_SURVIVABLE_IMPACT:
                continue

            for h1 in [-1, 0, 1]:
                for h2 in [-1, 0, 1]:
                    px, py, _, _ = jump_trajectory(
                        0.0, 0.0, vx0, vy0, h1, j, h2, k_land
                    )

                    # Precompute mid-arc offsets
                    arc_offs = _compute_arc_offsets_wall(
                        vx0,
                        vy0,
                        j,
                        k_land,
                        h1,
                        h2,
                    )

                    entries.append(
                        WallTrajectoryEntry(
                            j=j,
                            h1=h1,
                            h2=h2,
                            total_frames=j + k_land,
                            dx=px,
                            dy=py,
                            landing_vy=landing_vy,
                            arc_offsets=arc_offs,
                        )
                    )

        catalogs[(normal_sign, jump_type)] = entries

    return catalogs


def _build_airborne_falling() -> list[AirborneTrajectoryEntry]:
    """Build airborne (jump released) trajectory templates.

    3 entries, one per horizontal input. Landing frame is solved at runtime
    since it depends on the current vy.
    """
    return [AirborneTrajectoryEntry(h=h, release_at=0) for h in [-1, 0, 1]]


def _build_airborne_holding() -> list[AirborneTrajectoryEntry]:
    """Build airborne (jump still held) trajectory templates.

    Sweep release_at in ~10 values x 3 horizontal inputs = ~30 entries.
    At runtime, for each template, the remaining hold is used as phase 1
    duration and landing frame is solved from there.
    """
    entries: list[AirborneTrajectoryEntry] = []

    # Sample release times: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45
    release_values = list(range(0, MAX_JUMP_DURATION + 1, 5))

    for release_at in release_values:
        for h in [-1, 0, 1]:
            entries.append(AirborneTrajectoryEntry(h=h, release_at=release_at))

    return entries
