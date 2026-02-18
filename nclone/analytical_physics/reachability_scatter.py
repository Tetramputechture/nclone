"""
Scatter-and-collect reachability engine.

For each trajectory in the catalog, computes the landing point using
the affine formula and looks up nearby surface nodes via spatial hash.
The union of all nearby nodes is the reachable surface set.

Runtime cost: ~414 entries x (1 mul-add + 1 hash lookup) ~ 2-5 us per state.
"""

import math
from typing import Optional

from nclone.constants import (
    AIR_ACCEL,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    JUMP_FLOOR_Y,
)
from .analytical_trajectory import (
    S1,
    Sx2,
    S2,
    D,
    D_MINUS_1,
    _dn,
    velocity_at,
    position_x_at,
    position_y_at,
    jump_trajectory,
    fall_trajectory,
    solve_landing_frame,
    solve_fall_landing_frame,
)
from .trajectory_catalog import (
    TrajectoryEntry,
    WallTrajectoryEntry,
    AirborneTrajectoryEntry,
    TrajectoryCatalog,
)


# Contact type constants (matching ninja.py state machine)
CONTACT_FLOOR = 0
CONTACT_WALL = 1
CONTACT_AIRBORNE = 2

# Ninja states from ninja.py
STATE_IMMOBILE = 0
STATE_RUNNING = 1
STATE_GROUND_SLIDING = 2
STATE_JUMPING = 3
STATE_FALLING = 4
STATE_WALL_SLIDING = 5


class ReachabilityScatter:
    """Runtime scatter-and-collect reachability engine.

    Given the ninja's current state, evaluates all relevant trajectories
    from the precomputed catalog to determine which surface nodes are
    reachable via a single jump/fall.

    Attributes:
        catalog: Precomputed trajectory catalog.
        match_radius: How close a surface must be to a landing point (px).
    """

    def __init__(
        self,
        catalog: Optional[TrajectoryCatalog] = None,
        match_radius: float = 12.0,
    ):
        """Initialize the reachability scatter engine.

        Args:
            catalog: Precomputed trajectory catalog. If None, builds one.
            match_radius: Proximity threshold for surface matching (px).
        """
        if catalog is None:
            catalog = TrajectoryCatalog.build()
        self.catalog = catalog
        self.match_radius = match_radius

    def compute_reachable(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        state: int,
        floor_normal_x: float,
        floor_normal_y: float,
        walled: bool,
        wall_normal: float,
        jump_duration: int,
        spatial_hash,
    ) -> tuple[set, list[dict]]:
        """Compute reachable surfaces and landing points from current state.

        Args:
            ninja_x, ninja_y: Current ninja position.
            vx, vy: Current ninja velocity.
            state: Ninja movement state (0-5).
            floor_normal_x, floor_normal_y: Floor surface normal.
            walled: Whether ninja is touching a wall.
            wall_normal: Wall normal direction (1.0 or -1.0).
            jump_duration: Current jump hold frame count.
            spatial_hash: SpatialHash with surface node positions.

        Returns:
            Tuple of:
                - set of reachable node positions (tuples)
                - list of landing point dicts with metadata for visualization
        """
        # Determine contact type and select trajectory set
        contact_type = self._classify_contact(state, walled)
        landing_points = self._compute_landing_points(
            ninja_x,
            ninja_y,
            vx,
            vy,
            contact_type,
            state,
            floor_normal_x,
            floor_normal_y,
            walled,
            wall_normal,
            jump_duration,
        )

        # Collect nearby surfaces for each landing point
        reachable: set = set()
        if spatial_hash is not None:
            for lp in landing_points:
                nearby = spatial_hash.find_all_within_radius(
                    lp["x"], lp["y"], self.match_radius
                )
                for node_pos, _ in nearby:
                    reachable.add(node_pos)

        return reachable, landing_points

    def compute_reachable_filtered(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        state: int,
        floor_normal_x: float,
        floor_normal_y: float,
        walled: bool,
        wall_normal: float,
        jump_duration: int,
        spatial_hash,
        tile_dic: dict,
        deadly_mine_positions: list[tuple[float, float]],
        node_physics: dict,
        mine_kill_radius: float = 0.0,
    ) -> tuple[set, list[dict]]:
        """Compute reachable surfaces with geometry and mine filtering.

        Like compute_reachable() but additionally:
        1. Rejects landing points inside solid tiles
        2. Rejects landing points near deadly mines
        3. Annotates each reachable node with grounded/walled classification

        Args:
            ninja_x, ninja_y, vx, vy: Current ninja state.
            state: Ninja movement state (0-5).
            floor_normal_x, floor_normal_y: Floor surface normal.
            walled, wall_normal: Wall contact info.
            jump_duration: Current jump hold frame count.
            spatial_hash: SpatialHash with surface node positions.
            tile_dic: {(col, row): tile_type} from sim.
            deadly_mine_positions: Deadly mine (x, y) positions.
            node_physics: {(x,y): {"grounded": bool, "walled": bool}}.
            mine_kill_radius: Rejection radius around mines (px). 0 = auto.

        Returns:
            Tuple of:
                - set of reachable node positions (tuples)
                - list of landing point dicts with metadata + filter annotations
        """
        from .landing_filter import (
            is_position_in_traversable_tile,
            is_near_deadly_mine,
        )
        from nclone.constants.physics_constants import (
            NINJA_RADIUS,
            TOGGLE_MINE_RADII,
        )

        if mine_kill_radius <= 0.0:
            mine_kill_radius = NINJA_RADIUS + TOGGLE_MINE_RADII[0]

        # Compute raw landing points
        contact_type = self._classify_contact(state, walled)
        landing_points = self._compute_landing_points(
            ninja_x,
            ninja_y,
            vx,
            vy,
            contact_type,
            state,
            floor_normal_x,
            floor_normal_y,
            walled,
            wall_normal,
            jump_duration,
        )

        reachable: set = set()
        filtered_landing_points: list[dict] = []

        for lp in landing_points:
            lx, ly = lp["x"], lp["y"]

            # 1. Tile validity
            if not is_position_in_traversable_tile(lx, ly, tile_dic):
                lp["filtered"] = "tile"
                filtered_landing_points.append(lp)
                continue

            # 2. Mine proximity
            if is_near_deadly_mine(lx, ly, deadly_mine_positions, mine_kill_radius):
                lp["filtered"] = "mine"
                filtered_landing_points.append(lp)
                continue

            # 3. Surface matching
            lp["filtered"] = None
            if spatial_hash is not None:
                nearby = spatial_hash.find_all_within_radius(lx, ly, self.match_radius)
                matched = False
                for node_pos, dist in nearby:
                    phys = node_physics.get(node_pos)
                    if phys and (phys["grounded"] or phys["walled"]):
                        reachable.add(node_pos)
                        if not matched:
                            lp["matched_node"] = node_pos
                            lp["node_grounded"] = phys["grounded"]
                            lp["node_walled"] = phys["walled"]
                            matched = True
                if not matched:
                    lp["filtered"] = "no_surface"

            filtered_landing_points.append(lp)

        return reachable, filtered_landing_points

    def compute_landing_points_only(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        state: int,
        floor_normal_x: float,
        floor_normal_y: float,
        walled: bool,
        wall_normal: float,
        jump_duration: int,
    ) -> list[dict]:
        """Compute all landing points without spatial hash lookup.

        Useful for visualization when spatial hash is not available.

        Returns:
            List of landing point dicts: {x, y, j, h1, h2, total_frames, source}.
        """
        contact_type = self._classify_contact(state, walled)
        return self._compute_landing_points(
            ninja_x,
            ninja_y,
            vx,
            vy,
            contact_type,
            state,
            floor_normal_x,
            floor_normal_y,
            walled,
            wall_normal,
            jump_duration,
        )

    def _classify_contact(self, state: int, walled: bool) -> int:
        """Classify contact type from ninja state."""
        if state in (STATE_IMMOBILE, STATE_RUNNING, STATE_GROUND_SLIDING):
            return CONTACT_FLOOR
        if state == STATE_WALL_SLIDING or (
            walled and state in (STATE_JUMPING, STATE_FALLING)
        ):
            return CONTACT_WALL if walled else CONTACT_AIRBORNE
        return CONTACT_AIRBORNE

    def _compute_landing_points(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        contact_type: int,
        state: int,
        floor_normal_x: float,
        floor_normal_y: float,
        walled: bool,
        wall_normal: float,
        jump_duration: int,
    ) -> list[dict]:
        """Dispatch to the appropriate trajectory set based on contact type."""
        if contact_type == CONTACT_FLOOR:
            return self._scatter_floor(
                ninja_x, ninja_y, vx, vy, floor_normal_x, floor_normal_y
            )
        elif contact_type == CONTACT_WALL:
            return self._scatter_wall(ninja_x, ninja_y, vx, vy, wall_normal, state)
        else:
            return self._scatter_airborne(
                ninja_x, ninja_y, vx, vy, state, jump_duration
            )

    def _scatter_floor(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        floor_normal_x: float,
        floor_normal_y: float,
    ) -> list[dict]:
        """Scatter landing and mid-arc points for floor (grounded) state.

        For flat ground: vx0 = vx (preserved), vy0 = JUMP_FLOOR_Y (fixed).
        For slopes: impulse depends on slope normal.

        In addition to the landing point, emits intermediate arc points
        where the ninja is above launch height. These allow detection of
        goals and surfaces reachable mid-jump.
        """
        is_flat = abs(floor_normal_x) < 0.01

        if is_flat:
            vx0 = vx
            vy0 = JUMP_FLOOR_Y
        else:
            # Slope jump: use downhill impulse as a conservative estimate
            # Full slope jump mechanics are complex; use flat-ground catalog
            # with adjusted initial velocity for now
            vx0 = vx
            vy0 = JUMP_FLOOR_Y

        landing_points = []
        for traj in self.catalog.floor_flat:
            # Affine evaluation: land = ninja + C*v0 + B
            if abs(vx0) < traj.cap_threshold:
                land_x = ninja_x + traj.C * vx0 + traj.Bx
            else:
                # Capped regime: approximate by clamping vx contribution
                land_x = (
                    ninja_x
                    + traj.C * math.copysign(min(abs(vx0), MAX_HOR_SPEED), vx0)
                    + traj.Bx
                )

            land_y = ninja_y + traj.C * vy0 + traj.By

            landing_points.append(
                {
                    "x": land_x,
                    "y": land_y,
                    "j": traj.j,
                    "h1": traj.h1,
                    "h2": traj.h2,
                    "total_frames": traj.total_frames,
                    "source": "floor",
                }
            )

            # Mid-arc points (above launch height)
            for arc_pt in traj.arc_points:
                if abs(vx0) < traj.cap_threshold:
                    arc_x = ninja_x + arc_pt.C * vx0 + arc_pt.Bx
                else:
                    arc_x = (
                        ninja_x
                        + arc_pt.C * math.copysign(min(abs(vx0), MAX_HOR_SPEED), vx0)
                        + arc_pt.Bx
                    )

                arc_y = ninja_y + arc_pt.C * vy0 + arc_pt.By

                landing_points.append(
                    {
                        "x": arc_x,
                        "y": arc_y,
                        "j": traj.j,
                        "h1": traj.h1,
                        "h2": traj.h2,
                        "total_frames": arc_pt.frame,
                        "source": "floor_arc",
                    }
                )

        return landing_points

    def _scatter_wall(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        wall_normal: float,
        state: int,
    ) -> list[dict]:
        """Scatter landing and mid-arc points for wall state.

        Wall jumps replace velocity, so all landing points are precomputed
        constants. Just add (dx, dy) to ninja position.

        Also emits intermediate arc points where the ninja is above launch
        height, enabling detection of goals reachable mid-wall-jump.
        """
        normal_sign = 1 if wall_normal > 0 else -1
        jump_type = "slide" if state == STATE_WALL_SLIDING else "regular"

        key = (normal_sign, jump_type)
        wall_entries = self.catalog.wall.get(key, [])

        landing_points = []
        for entry in wall_entries:
            landing_points.append(
                {
                    "x": ninja_x + entry.dx,
                    "y": ninja_y + entry.dy,
                    "j": entry.j,
                    "h1": entry.h1,
                    "h2": entry.h2,
                    "total_frames": entry.total_frames,
                    "source": "wall",
                }
            )

            # Mid-arc offset points (above launch height)
            for arc_frame, arc_dx, arc_dy in entry.arc_offsets:
                landing_points.append(
                    {
                        "x": ninja_x + arc_dx,
                        "y": ninja_y + arc_dy,
                        "j": entry.j,
                        "h1": entry.h1,
                        "h2": entry.h2,
                        "total_frames": arc_frame,
                        "source": "wall_arc",
                    }
                )

        return landing_points

    def _scatter_airborne(
        self,
        ninja_x: float,
        ninja_y: float,
        vx: float,
        vy: float,
        state: int,
        jump_duration: int,
    ) -> list[dict]:
        """Scatter landing points for airborne state.

        For falling (jump released): solve landing frame at runtime, then
        evaluate 3 trajectories (one per h).

        For jump held: sweep release times, solve landing for each.
        """
        landing_points = []

        if state == STATE_JUMPING and jump_duration < MAX_JUMP_DURATION:
            # Jump still held -- sweep release times
            jump_remaining = MAX_JUMP_DURATION - jump_duration
            for entry in self.catalog.airborne_holding:
                # Clamp release_at to remaining frames
                release = min(entry.release_at, jump_remaining)
                h = entry.h

                a = h * AIR_ACCEL

                # Phase 1: continue holding for 'release' more frames
                vx_r = velocity_at(vx, a, release)
                vy_r = velocity_at(vy, GRAVITY_JUMP, release)
                px_r = position_x_at(ninja_x, vx, a, release)
                py_r = position_y_at(ninja_y, vy, GRAVITY_JUMP, release)

                # Phase 2: fall until landing
                # Solve landing frame relative to phase 2 start
                k_land = solve_fall_landing_frame(vy_r, target_dy=0.0)
                if k_land is None:
                    continue

                px_f, py_f, _, _ = fall_trajectory(px_r, py_r, vx_r, vy_r, h, k_land)

                landing_points.append(
                    {
                        "x": px_f,
                        "y": py_f,
                        "j": release,
                        "h1": h,
                        "h2": h,
                        "total_frames": release + k_land,
                        "source": "airborne_holding",
                    }
                )
        else:
            # Falling -- solve landing frame for current vy
            for entry in self.catalog.airborne_falling:
                h = entry.h

                # Landing frame depends on vy
                k_land = solve_fall_landing_frame(vy, target_dy=0.0)
                if k_land is None:
                    continue

                px_f, py_f, _, _ = fall_trajectory(ninja_x, ninja_y, vx, vy, h, k_land)

                landing_points.append(
                    {
                        "x": px_f,
                        "y": py_f,
                        "j": 0,
                        "h1": h,
                        "h2": h,
                        "total_frames": k_land,
                        "source": "airborne_falling",
                    }
                )

        return landing_points
