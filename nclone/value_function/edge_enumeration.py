"""Edge enumeration via physics simulation for the surface graph.

For every graph node (surface_point, velocity_bin), simulates physically
realisable transitions to discover which other nodes are reachable and at
what frame cost.  Supports multiprocessing for parallelism.

Memory-efficient waypoint collection
------------------------------------
Arc waypoints (intermediate (x, y, vx, vy) samples along every simulated arc)
are needed to build the arc cache used for airborne-state O(1) lookup at
runtime.  To keep peak RSS bounded on large levels, waypoints are collected
as a worker-local ``Dict[(gx, gy, gvx, gvy, dst_node), frames_to_land]`` that
performs per-cell-per-destination deduplication on the fly (min
``frames_to_land`` per unique tuple).  Workers return three compact NumPy
arrays (``keys[N,4]``, ``dst[N]``, ``ftl[N]``) instead of a list of millions
of Python dataclass instances.  The parent aggregator merges workers with the
same streaming-dict-min operation before the vectorized ``ArcCache.build``.

Mine-aware arc cost inflation
-----------------------------
When mine positions are supplied, edge weights are inflated by a smooth
inverse-linear penalty based on the minimum clearance to any mine along the
arc trajectory.  Arcs that literally enter the kill zone (clearance <= 0)
are pruned as before; arcs that pass nearby receive a cost multiplier
``1 + min(beta / clearance, MAX)`` that decays smoothly with distance.  This
replaces the old binary "discard or keep" approach with a continuous
potential that steers Dijkstra — and the resulting Φ gradient — smoothly
away from mine danger zones rather than hugging the pruning boundary.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAX_JUMP_DURATION, GROUND_STATES, AIR_STATES, WALL_STATES,
    NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED,
)
from .surface_extraction import (
    SurfacePoint,
    SURFACE_FLOOR,
    SURFACE_WALL,
    SURFACE_SLIDE,
    SURFACE_LAUNCH_PAD,
    is_wall_like,
)
from .surface_graph import (
    SurfaceGraph, node_id, velocity_bin_centers,
    velocity_to_bin, num_velocity_bins, find_nearest_point,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arc cache binning — kept here so arc_cache.py can import without cycles.
# ---------------------------------------------------------------------------
POSITION_BIN_SIZE = 6.0      # pixels
VELOCITY_BIN_SIZE = 0.5      # pixels / frame
_INV_POS_BIN = 1.0 / POSITION_BIN_SIZE
_INV_VEL_BIN = 1.0 / VELOCITY_BIN_SIZE

# Mine arc cost inflation constants
_MINE_KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px
_MINE_INFLATION_BETA_DEFAULT = 50.0
# Cap on the penalty multiplier to prevent mine-adjacent edges from producing
# astronomically large costs (e.g. beta=50, clearance=0.01 px → 500,000×).
# Without this cap those inflated costs propagate through Dijkstra and set
# d_max to ~10M frames, which collapses the PBRS phi normalization range to
# [-0.0001, 0] everywhere a normal agent can reach.
# Cap prevents near-zero clearance from producing arbitrarily large costs.
# Worst inflated cost: w_base * (1 + 500) ≈ 10,000 frames.
_MINE_INFLATION_MAX_FACTOR = 500.0
# CHOMP-style bounded penalty support radius ``eps`` (px).  The penalty is
# only active when clearance < eps; beyond eps it is exactly zero, so very
# far-away mines don't contribute to the edge weight at all.  Chosen large
# enough to cover a typical ninja jump-arc approach (~30 px of lateral
# displacement while the ninja turns around).
_MINE_CHOMP_EPSILON: float = 32.0

# Launch-pad boost magnitude (see EntityLaunchPad.BOOST = 36/7).  A launch
# pad applies the boost as (normal_x * BOOST, normal_y * BOOST); the
# simulator further multiplies by 2/3 and adds an upward-scaling term for
# normal_y < 0.  For the surface-graph kick edges we use the nominal
# pre-simulation boost magnitude since the simulator handles the finer
# details once the ninja is injected at the pad position.
_LAUNCH_PAD_BOOST_MAG: float = 36.0 / 7.0

# Module-level extra-primitives list populated by ``enumerate_edges`` when the
# caller passes an ``extra_primitives`` JSON payload (see
# ``tools/audit_completeness.load_extra_primitives``).  ``_enumerate_node_edges``
# reads this after the built-in sweep to emit additional arcs.  Leaving the
# global ``None`` keeps the fast path allocation-free when no extras exist.
_EXTRA_PRIMITIVES: Optional[List[dict]] = None


def _iter_extra_primitives_for(kind: str) -> List[dict]:
    """Return ``_EXTRA_PRIMITIVES`` entries matching the requested ``kind``.

    Kept as a helper so callers don't have to repeat the ``None``-guard and
    so future filtering (e.g. per-surface-type) can be added in one place.
    """
    if not _EXTRA_PRIMITIVES:
        return []
    return [p for p in _EXTRA_PRIMITIVES if p.get("kind") == kind]

# Simulation parameter sweeps
RUN_DURATIONS = [1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 60]
JUMP_HOLD_DURATIONS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45]
H_INPUTS = [-1, 0, 1]
POST_RELEASE_INPUTS = [-1, 0, 1]
MAX_ARC_FRAMES_DEFAULT = 500
# Default arc-waypoint sampling interval (1 = every frame).  Exposed via
# ``enumerate_edges(arc_waypoint_interval=...)`` so operators can trade
# arc-cache coverage for peak memory on small budgets.
ARC_WAYPOINT_INTERVAL_DEFAULT = 1

# Ninja state codes (0=Immobile, 1=Running, 2=GroundSlide, 3=Jumping, 4=Falling, 5=WallSlide, 6=Dead, 7=AwaitingDeath)
_STATE_WALL_SLIDING = 5
_STATE_DEAD = 6
_STATE_AWAITING_DEATH = 7
_DEATH_STATES = {_STATE_DEAD, _STATE_AWAITING_DEATH}
_AIRBORNE_STATES = AIR_STATES                  # {3, 4}
_SURFACE_STATES = GROUND_STATES | WALL_STATES  # {0, 1, 2, 5}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class EdgeResult:
    """A discovered directed edge in the surface graph."""
    src_node: int
    dst_node: int
    weight: float  # frames (may include mine-proximity penalty)
    jump_hold_frames: int = 0   # 0 for runs/slides; hold duration for jumps
    h_input_during: int = 0     # primary horizontal input (-1/0/1)
    # Note: only the primary (during-hold) air control is stored. Post-release
    # air control is learned reactively by the agent from spatial context.


@dataclass
class CompactWaypoints:
    """Per-(cell, destination) deduplicated waypoints in compact NumPy form.

    For each unique ``(gx, gy, gvx, gvy, dst_node)`` tuple observed across all
    simulated arcs, stores the *minimum* ``frames_to_land`` (i.e. remaining
    arc cost from that cell to the destination surface node).  The min-reduce
    is order-independent and associative, so workers can accumulate locally
    and the parent merges chunk results with the same per-key min operation.

    The ArcCache final build then computes, per unique cell key,
    ``cost = min over dst of (ftl[i] + d_star[dst[i]])``.
    """

    keys: np.ndarray  # int32 [N, 4]  — (gx, gy, gvx, gvy)
    dst: np.ndarray   # int32 [N]     — destination node id
    ftl: np.ndarray   # float32 [N]   — frames_to_land (already mine-inflated)

    @property
    def n_entries(self) -> int:
        return int(len(self.dst))

    def __len__(self) -> int:  # convenience for existing callers
        return self.n_entries

    @classmethod
    def empty(cls) -> "CompactWaypoints":
        return cls(
            keys=np.empty((0, 4), dtype=np.int32),
            dst=np.empty(0, dtype=np.int32),
            ftl=np.empty(0, dtype=np.float32),
        )

    @classmethod
    def from_dict(
        cls,
        accum: Dict[Tuple[int, int, int, int, int], float],
    ) -> "CompactWaypoints":
        """Flush a worker-local accumulator to contiguous NumPy arrays."""
        n = len(accum)
        if n == 0:
            return cls.empty()
        keys = np.empty((n, 4), dtype=np.int32)
        dst = np.empty(n, dtype=np.int32)
        ftl = np.empty(n, dtype=np.float32)
        for i, (k, v) in enumerate(accum.items()):
            keys[i, 0] = k[0]
            keys[i, 1] = k[1]
            keys[i, 2] = k[2]
            keys[i, 3] = k[3]
            dst[i] = k[4]
            ftl[i] = v
        return cls(keys=keys, dst=dst, ftl=ftl)


def _inflate_arc_cost(
    w_base: float,
    arc_wps: List[Tuple[int, float, float, float, float]],
    mine_positions: np.ndarray,
    beta: float,
    kill_radius: float,
    eps: float = _MINE_CHOMP_EPSILON,
) -> float:
    """CHOMP-style bounded obstacle penalty for mine proximity.

    We follow the CHOMP formulation (Ratliff et al., 2009) and penalise
    only the arc segments whose clearance to the nearest mine falls below
    a support radius ``eps``::

        penalty(c) = 0,                              c >= eps
        penalty(c) = beta * (1/c - 1/eps) ** 2,      0 < c < eps
        penalty(c) = inf,                            c <= 0   (kill zone)

    The key property we need is boundedness: because ``(1/c - 1/eps) ** 2``
    goes to ``(1/0+)^2 = inf`` only as ``c -> 0`` and is otherwise finite
    with an upper bound ``(1/kill_eps - 1/eps) ** 2`` when ``c > kill_eps``,
    a hard cap on the finite penalty keeps ``d_max_raw / d_max`` close to
    unity even on dense-mine levels.  With the previous inverse-linear
    formulation, near-zero clearances produced ~500x cost inflation, which
    collapsed the Φ normalisation range.

    Args:
        w_base: Original frame-count edge weight.
        arc_wps: Waypoints as ``(frame, x, y, vx, vy)`` tuples.
        mine_positions: Array of shape ``(N, 2)`` with mine centre
            coordinates.  Pass an empty array to disable the penalty.
        beta: Multiplicative weight on the CHOMP bowl.  Higher values
            produce stronger avoidance while still respecting the cap.
        kill_radius: Lethal contact distance (NINJA_RADIUS + mine_radius).
            Arcs whose closest point is inside this radius are pruned
            from the graph.
        eps: Support radius (pixels) beyond which the penalty is exactly
            zero.  Defaults to ``_MINE_CHOMP_EPSILON``.

    Returns:
        Inflated weight ``w_base * (1 + penalty)`` or ``inf`` when the
        arc enters the kill zone.
    """
    if len(mine_positions) == 0 or not arc_wps or eps <= 0.0:
        return w_base
    pts = np.array([(wx, wy) for (_, wx, wy, _, _) in arc_wps], dtype=np.float32)
    diffs = pts[:, None, :] - mine_positions[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))
    min_dist = float(dists.min())
    clearance = min_dist - kill_radius
    if clearance <= 0:
        return float("inf")
    if clearance >= eps:
        return w_base
    # CHOMP bowl in the (0, eps) region.
    inv_c = 1.0 / clearance
    inv_eps = 1.0 / eps
    penalty = beta * (inv_c - inv_eps) ** 2
    # Preserve the legacy hard cap to guarantee finite d_max propagation
    # even when eps is configured aggressively (e.g. eps -> 0).
    if penalty > _MINE_INFLATION_MAX_FACTOR:
        penalty = _MINE_INFLATION_MAX_FACTOR
    return w_base * (1.0 + penalty)


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------
def _simulate_arc(
    sim, h_input, jump_input, max_frames,
    collect_waypoints: bool = True,
    waypoint_interval: int = ARC_WAYPOINT_INTERVAL_DEFAULT,
):
    """Simulate with constant inputs until landing, death, or timeout.

    Returns (landing_state_or_None, frames, waypoints).
    """
    waypoints: List[Tuple[int, float, float, float, float]] = []
    was_airborne = False

    for frame in range(1, max_frames + 1):
        sim.tick_tiles_only(h_input, jump_input)
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, frame, waypoints

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()

        if collect_waypoints and frame % waypoint_interval == 0:
            waypoints.append((frame, pos[0], pos[1], vel[0], vel[1]))

        # Detect landing: was airborne, now on surface
        if state_code in _AIRBORNE_STATES:
            was_airborne = True
        elif was_airborne and state_code in _SURFACE_STATES:
            return (pos[0], pos[1], vel[0], vel[1], state_code), frame, waypoints

        # If started on surface and still on surface, also count as arrived
        if not was_airborne and state_code in _SURFACE_STATES and frame > 0:
            return (pos[0], pos[1], vel[0], vel[1], state_code), frame, waypoints

    # Timed out without landing
    return None, max_frames, waypoints


def _simulate_jump_arc(
    sim, h_input_during, hold_duration, h_input_after,
    collect_waypoints: bool = True,
    waypoint_interval: int = ARC_WAYPOINT_INTERVAL_DEFAULT,
    max_arc_frames: int = MAX_ARC_FRAMES_DEFAULT,
):
    """Jump arc: hold for hold_duration, then fall with h_input_after."""
    waypoints: List[Tuple[int, float, float, float, float]] = []
    total_frame = 0

    # Phase 1: jump hold
    for _ in range(hold_duration):
        total_frame += 1
        sim.tick_tiles_only(h_input_during, 1)  # jump held
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, total_frame, waypoints

        if collect_waypoints and total_frame % waypoint_interval == 0:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            waypoints.append((total_frame, pos[0], pos[1], vel[0], vel[1]))

    # Phase 2: post-release fall
    for _ in range(max(0, max_arc_frames - total_frame)):
        total_frame += 1
        sim.tick_tiles_only(h_input_after, 0)  # jump released
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, total_frame, waypoints

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()

        if collect_waypoints and total_frame % waypoint_interval == 0:
            waypoints.append((total_frame, pos[0], pos[1], vel[0], vel[1]))

        if state_code in _SURFACE_STATES:
            return (pos[0], pos[1], vel[0], vel[1], state_code), \
                total_frame, waypoints

    return None, total_frame, waypoints


def _landing_to_node(graph, landing, return_clip: bool = False):
    """Map a landing (x, y, vx, vy, state_code) to the nearest graph node.

    When ``return_clip=True`` additionally returns a boolean indicating that
    the landing's continuous velocity is outside the discrete velocity-bin
    grid (i.e. it clipped to the first or last bin).  Clipping is a proxy
    for *systematic approximation error*: if many edges land at a clip bin
    the chosen bin grid is too narrow to represent the trajectory correctly,
    and the resulting d_star values will underestimate the true cost-to-go.
    """
    x, y, vx, vy, state_code = landing

    if state_code == _STATE_WALL_SLIDING:
        stype = SURFACE_WALL
        vel = vy
    else:
        stype = SURFACE_FLOOR
        vel = vx

    pt_idx = find_nearest_point(graph, x, y, stype)
    if pt_idx is None:
        pt_idx = find_nearest_point(graph, x, y, surface_type=None)
        if pt_idx is None:
            return (None, False) if return_clip else None
        sp = graph.surface_points[pt_idx]
        stype = sp.surface_type
        vel = vy if is_wall_like(stype) else vx
    vel_bin = velocity_to_bin(vel, stype)
    node = node_id(pt_idx, vel_bin, graph.max_vel_bins)
    if return_clip:
        from .surface_graph import velocity_bin_centers as _vbc
        centers = _vbc(stype)
        clipped = bool(vel <= float(centers[0]) or vel >= float(centers[-1]))
        return node, clipped
    return node


# ---------------------------------------------------------------------------
# Per-node edge enumeration
# ---------------------------------------------------------------------------
def _enumerate_node_edges(
    sim,
    graph,
    pt_idx: int,
    vel_bin: int,
    wp_accum: Optional[Dict[Tuple[int, int, int, int, int], float]],
    mine_positions: Optional[np.ndarray] = None,
    mine_kill_radius: float = _MINE_KILL_RADIUS,
    mine_inflation_beta: float = _MINE_INFLATION_BETA_DEFAULT,
    waypoint_interval: int = ARC_WAYPOINT_INTERVAL_DEFAULT,
    max_arc_frames: int = MAX_ARC_FRAMES_DEFAULT,
) -> Tuple[List[EdgeResult], int, int]:
    """Enumerate all edges from a single graph node.

    Returns ``(edges, n_clipped_landings, n_total_landings)`` where the
    counters track how many landing velocities fell on the first/last
    velocity bin (i.e. were clipped by the discrete grid).  High clip ratios
    indicate the velocity bin extents are too narrow for the mechanics on
    this level and d_star will systematically underestimate true cost-to-go.

    Waypoints are written directly into ``wp_accum`` (worker-local dict) with
    per-(cell, dst) min-reduce, avoiding millions of Python dataclass
    allocations.  Pass ``wp_accum=None`` to skip waypoint collection entirely.

    Args:
        sim: Native simulator instance.
        graph: Surface graph.
        pt_idx: Surface point index.
        vel_bin: Velocity bin index.
        wp_accum: Accumulator dict mapping ``(gx, gy, gvx, gvy, dst_node)`` to
            minimum ``frames_to_land``.  ``None`` disables waypoint collection.
        mine_positions: ``(N, 2)`` mine centre coordinates, or ``None``.
        mine_kill_radius: Lethal contact distance in pixels (default 14 px).
        mine_inflation_beta: Inverse-linear proximity penalty strength.
        waypoint_interval: Sample a waypoint every N frames (default 1).
        max_arc_frames: Per-arc simulation frame cap (default 500).
    """
    sp = graph.surface_points[pt_idx]
    src = node_id(pt_idx, vel_bin, graph.max_vel_bins)
    centers = velocity_bin_centers(sp.surface_type)
    if vel_bin >= len(centers):
        return [], 0, 0

    vel = float(centers[vel_bin])

    # Compute initial velocity components
    if sp.surface_type == SURFACE_FLOOR:
        vx0 = vel
        vy0 = 0.0
    elif sp.surface_type == SURFACE_LAUNCH_PAD:
        # Launch-pad boost vector is stored in the point's normal fields
        # (normal_x / normal_y repurposed to carry the boost direction).
        vx0 = sp.normal_x * _LAUNCH_PAD_BOOST_MAG
        vy0 = sp.normal_y * _LAUNCH_PAD_BOOST_MAG
    else:
        # SURFACE_WALL or SURFACE_SLIDE: velocity bin holds vy.
        vx0 = 0.0
        vy0 = vel

    _mine_positions = (
        mine_positions if (mine_positions is not None and len(mine_positions) > 0)
        else None
    )
    collect_waypoints = wp_accum is not None

    edges: List[EdgeResult] = []
    # Counters hoisted into a one-element list so the inner closure can
    # mutate without the overhead of ``nonlocal`` rebinding on every call.
    clip_counters = [0, 0]  # [n_clipped_landings, n_total_landings]

    def _try_edge(
        landing, frames, arc_wps, jump_hold_frames: int = 0, h_input_during: int = 0,
    ):
        """Process a simulation result into an edge + waypoint updates."""
        if landing is None:
            return
        result = _landing_to_node(graph, landing, return_clip=True)
        dst, clipped = result
        if dst is None or dst == src:
            return
        clip_counters[1] += 1
        if clipped:
            clip_counters[0] += 1
        effective_weight = float(frames)
        if _mine_positions is not None:
            effective_weight = _inflate_arc_cost(
                effective_weight, arc_wps, _mine_positions,
                mine_inflation_beta, mine_kill_radius,
            )
            if not np.isfinite(effective_weight):
                return  # Lethal arc — prune from graph
        edges.append(
            EdgeResult(src, dst, effective_weight, jump_hold_frames, h_input_during)
        )
        if wp_accum is not None:
            # Inline the bin quantization for speed — this is the hot loop.
            for (f, wx, wy, wvx, wvy) in arc_wps:
                ftl = effective_weight - f
                if ftl < 0.0:
                    continue
                key = (
                    int(math.floor(wx * _INV_POS_BIN)),
                    int(math.floor(wy * _INV_POS_BIN)),
                    int(math.floor(wvx * _INV_VEL_BIN)),
                    int(math.floor(wvy * _INV_VEL_BIN)),
                    dst,
                )
                existing = wp_accum.get(key)
                if existing is None or ftl < existing:
                    wp_accum[key] = ftl

    snapshot = sim.save_ninja_state()

    # Build dynamic primitive sweep lists by merging the built-in constants
    # with any user-supplied extras for this surface kind.  The built-in
    # constants stay untouched so baseline behaviour is preserved.  Only
    # ``floor_jump`` extras are consumed today; remaining kinds are reserved
    # (see :func:`_iter_extra_primitives_for`).
    extra_floor_jumps = _iter_extra_primitives_for("floor_jump")

    # ----- FLOOR ACTIONS -----
    if sp.surface_type == SURFACE_FLOOR:
        # 5.1 Running along surface
        for h_input in H_INPUTS:
            for duration in RUN_DURATIONS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, duration,
                    collect_waypoints=collect_waypoints,
                    waypoint_interval=waypoint_interval,
                )
                _try_edge(landing, frames, wps,
                          jump_hold_frames=0, h_input_during=h_input)

        # 5.2 Floor jumps
        for h_during in H_INPUTS:
            for hold_dur in JUMP_HOLD_DURATIONS:
                for h_after in POST_RELEASE_INPUTS:
                    sim.restore_ninja_state(snapshot)
                    sim.inject_ninja_for_value_iteration(
                        sp.x, sp.y, vx0, vy0, 0)
                    # First frame: initiate jump
                    sim.tick_tiles_only(h_during, 1)
                    sc = sim.get_ninja_state_code()
                    if sc in _DEATH_STATES:
                        continue
                    # Continue jump arc
                    landing, frames, wps = _simulate_jump_arc(
                        sim, h_during, min(hold_dur - 1, MAX_JUMP_DURATION - 1),
                        h_after,
                        collect_waypoints=collect_waypoints,
                        waypoint_interval=waypoint_interval,
                        max_arc_frames=max_arc_frames,
                    )
                    if landing is not None:
                        _try_edge(landing, frames + 1, wps,
                                  jump_hold_frames=hold_dur, h_input_during=h_during)

        # 5.2b Extra user-supplied floor-jump primitives (e.g. off-grid holds).
        for prim in extra_floor_jumps:
            h_during = int(prim.get("h_during", 0))
            hold_dur = int(prim.get("hold", 0))
            h_after = int(prim.get("h_after", h_during))
            if hold_dur <= 0:
                continue
            sim.restore_ninja_state(snapshot)
            sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
            sim.tick_tiles_only(h_during, 1)
            sc = sim.get_ninja_state_code()
            if sc in _DEATH_STATES:
                continue
            landing, frames, wps = _simulate_jump_arc(
                sim, h_during, min(hold_dur - 1, MAX_JUMP_DURATION - 1),
                h_after,
                collect_waypoints=collect_waypoints,
                waypoint_interval=waypoint_interval,
                max_arc_frames=max_arc_frames,
            )
            if landing is not None:
                _try_edge(landing, frames + 1, wps,
                          jump_hold_frames=hold_dur, h_input_during=h_during)

        # 5.4 Edge falls (from is_edge nodes)
        if sp.is_edge:
            for h_input in H_INPUTS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, max_arc_frames,
                    collect_waypoints=collect_waypoints,
                    waypoint_interval=waypoint_interval,
                )
                _try_edge(landing, frames, wps,
                          jump_hold_frames=0, h_input_during=h_input)

    # ----- WALL / SLIDE ACTIONS -----
    elif sp.surface_type == SURFACE_WALL or sp.surface_type == SURFACE_SLIDE:
        # Slide-to-jump templates reuse the same wall-jump physics because the
        # underlying simulator mode for both is the wall-slide state.  The
        # only observable difference is the initial vy magnitude, which is
        # already baked into the per-class bin schedule.
        # 5.3 Wall jumps
        # (wall_normal_sign is informational; the physics sim handles the push)
        for jump_type in ("regular", "slide"):
            for h_input in H_INPUTS:
                for hold_dur in JUMP_HOLD_DURATIONS:
                    sim.restore_ninja_state(snapshot)
                    sim.inject_ninja_for_value_iteration(
                        sp.x, sp.y, vx0, vy0, 0)
                    sim.tick_tiles_only(h_input, 1)
                    sc = sim.get_ninja_state_code()
                    if sc in _DEATH_STATES:
                        continue
                    h_after = h_input
                    landing, frames, wps = _simulate_jump_arc(
                        sim, h_input,
                        min(hold_dur - 1, MAX_JUMP_DURATION - 1),
                        h_after,
                        collect_waypoints=collect_waypoints,
                        waypoint_interval=waypoint_interval,
                        max_arc_frames=max_arc_frames,
                    )
                    if landing is not None:
                        _try_edge(landing, frames + 1, wps,
                                  jump_hold_frames=hold_dur, h_input_during=h_input)

        # Wall slide (fall along wall)
        for h_input in H_INPUTS:
            for duration in RUN_DURATIONS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, duration,
                    collect_waypoints=collect_waypoints,
                    waypoint_interval=waypoint_interval,
                )
                _try_edge(landing, frames, wps,
                          jump_hold_frames=0, h_input_during=h_input)

    # ----- LAUNCH-PAD KICK -----
    elif sp.surface_type == SURFACE_LAUNCH_PAD:
        # A launch pad produces a single kick with a fixed boost velocity.
        # We model it as one airborne arc per (h_input, duration) tuple; the
        # simulator starts the ninja at the pad centre with (vx0, vy0) set
        # above from the stored boost vector.  There is no jump-hold control.
        for h_input in POST_RELEASE_INPUTS:
            for duration in RUN_DURATIONS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, duration,
                    collect_waypoints=collect_waypoints,
                    waypoint_interval=waypoint_interval,
                )
                _try_edge(landing, frames, wps,
                          jump_hold_frames=0, h_input_during=h_input)

    sim.restore_ninja_state(snapshot)
    return edges, clip_counters[0], clip_counters[1]


# ---------------------------------------------------------------------------
# Edge pruning
# ---------------------------------------------------------------------------
def _prune_edges(edges: List[EdgeResult]) -> List[EdgeResult]:
    """Keep only the minimum-weight edge for each (source, destination) pair."""
    best: Dict[Tuple[int, int], EdgeResult] = {}
    for e in edges:
        key = (e.src_node, e.dst_node)
        if key not in best or e.weight < best[key].weight:
            best[key] = e
    return list(best.values())


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------
def _worker_enumerate(args):
    """Worker function: enumerate edges for a chunk of nodes.

    Returns ``(edges_list, keys_arr, dst_arr, ftl_arr)`` where the waypoint
    arrays have been deduplicated per (cell, destination) inside this worker.
    """
    (map_data, sp_data, graph_params, node_indices, collect_waypoints,
     mine_positions, mine_kill_radius, mine_inflation_beta,
     waypoint_interval, max_arc_frames) = args

    # Reconstruct graph in worker process
    from nclone_native import Simulator as NativeSimulator
    sim = NativeSimulator.load(map_data)

    surface_points = _deserialize_surface_points(sp_data)
    graph = _reconstruct_graph(surface_points, graph_params)

    all_edges: List[EdgeResult] = []
    wp_accum: Optional[Dict[Tuple[int, int, int, int, int], float]] = (
        {} if collect_waypoints else None
    )
    n_clipped_total = 0
    n_landings_total = 0

    for pt_idx, vel_bin in node_indices:
        edges, n_clip, n_total = _enumerate_node_edges(
            sim, graph, pt_idx, vel_bin,
            wp_accum=wp_accum,
            mine_positions=mine_positions,
            mine_kill_radius=mine_kill_radius,
            mine_inflation_beta=mine_inflation_beta,
            waypoint_interval=waypoint_interval,
            max_arc_frames=max_arc_frames,
        )
        all_edges.extend(edges)
        n_clipped_total += n_clip
        n_landings_total += n_total

    if wp_accum is None:
        wps = CompactWaypoints.empty()
    else:
        wps = CompactWaypoints.from_dict(wp_accum)
    return (
        all_edges, wps.keys, wps.dst, wps.ftl,
        n_clipped_total, n_landings_total,
    )


def _serialize_surface_points(points: List[SurfacePoint]) -> dict:
    """Serialize surface points for cross-process transfer."""
    return {
        "x": np.array([p.x for p in points], dtype=np.float32),
        "y": np.array([p.y for p in points], dtype=np.float32),
        "nx": np.array([p.normal_x for p in points], dtype=np.float32),
        "ny": np.array([p.normal_y for p in points], dtype=np.float32),
        "stype": np.array([p.surface_type for p in points], dtype=np.uint8),
        "seg_id": np.array([p.segment_id for p in points], dtype=np.int32),
        "is_edge": np.array([p.is_edge for p in points], dtype=np.bool_),
    }


def _deserialize_surface_points(data: dict) -> List[SurfacePoint]:
    """Reconstruct surface points from serialized data."""
    n = len(data["x"])
    return [
        SurfacePoint(
            x=float(data["x"][i]),
            y=float(data["y"][i]),
            normal_x=float(data["nx"][i]),
            normal_y=float(data["ny"][i]),
            surface_type=int(data["stype"][i]),
            segment_id=int(data["seg_id"][i]),
            is_edge=bool(data["is_edge"][i]),
        )
        for i in range(n)
    ]


def _reconstruct_graph(
    surface_points: List[SurfacePoint], params: dict,
) -> SurfaceGraph:
    """Reconstruct a minimal SurfaceGraph from serialized parts."""
    from .surface_graph import build_graph
    return build_graph(surface_points)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _merge_worker_waypoints(
    chunk_keys: np.ndarray,
    chunk_dst: np.ndarray,
    chunk_ftl: np.ndarray,
    merged: Dict[Tuple[int, int, int, int, int], float],
) -> None:
    """Merge one worker's compact waypoint arrays into the running dict.

    Preserves the per-(cell, dst) min-reduce invariant so the result is
    independent of chunk order.
    """
    n = int(len(chunk_dst))
    if n == 0:
        return
    # Local aliases for inner-loop speed
    get = merged.get
    ks0 = chunk_keys[:, 0]
    ks1 = chunk_keys[:, 1]
    ks2 = chunk_keys[:, 2]
    ks3 = chunk_keys[:, 3]
    for i in range(n):
        key = (
            int(ks0[i]), int(ks1[i]), int(ks2[i]), int(ks3[i]), int(chunk_dst[i]),
        )
        v = float(chunk_ftl[i])
        existing = get(key)
        if existing is None or v < existing:
            merged[key] = v


def enumerate_edges(
    graph: SurfaceGraph,
    map_data: List[int],
    num_workers: int = 4,
    collect_waypoints: bool = True,
    verbose: bool = False,
    mine_positions: Optional[np.ndarray] = None,
    mine_kill_radius: float = _MINE_KILL_RADIUS,
    mine_inflation_beta: float = _MINE_INFLATION_BETA_DEFAULT,
    waypoint_interval: int = ARC_WAYPOINT_INTERVAL_DEFAULT,
    max_arc_frames: int = MAX_ARC_FRAMES_DEFAULT,
    mp_chunks_per_worker: int = 4,
    stats_out: Optional[dict] = None,
    tile_cache_dir: Optional[str] = None,
    tile_cache_num_shards: int = 64,
    extra_primitives: Optional[List[dict]] = None,
) -> Tuple[List[EdgeResult], CompactWaypoints]:
    """Enumerate all edges in the surface graph via physics simulation.

    For each node (surface_point, velocity_bin), simulates all applicable
    action categories and records viable transitions.  Supports parallel
    execution across multiple workers, each with its own simulator instance.

    Args:
        graph: Surface graph with nodes (edges will be populated after).
        map_data: Raw level bytes as list[int] for simulator construction.
        num_workers: Number of parallel worker processes.
        collect_waypoints: Whether to collect arc waypoints for cache.
        verbose: Print progress info.
        mine_positions: Array of shape ``(N, 2)`` with mine centre (x, y)
            coordinates for mine-aware arc cost inflation.  Pass ``None`` or
            an empty array when there are no mines.  Arcs that enter the kill
            zone are pruned; arcs that pass nearby receive an inflated weight.
        mine_kill_radius: Lethal contact distance in pixels.  Default 14 px.
        mine_inflation_beta: Inflation strength β for the inverse-linear
            proximity penalty (default 50.0).  Sweep [10, 50, 200] to tune.
        waypoint_interval: Sample one waypoint every N simulated frames.
            Higher values reduce memory linearly at the cost of arc-cache
            density (default 1 = every frame).
        max_arc_frames: Per-arc simulation cap in frames (default 500).
        mp_chunks_per_worker: Number of sub-chunks to split each worker's
            share into.  Higher values allow ``imap_unordered`` to release
            intermediate Python memory sooner, at the cost of a little extra
            IPC overhead.  Default 4 → total chunks = 4 × num_workers.
        stats_out: Optional dict.  When provided, the function populates the
            following keys before returning:
                ``n_clipped_landings``: int — total edges whose destination
                    velocity fell on the first or last velocity bin.
                ``n_total_landings``:   int — total successfully enumerated
                    edges (before dominated-pair pruning).
                ``pct_clipped``:        float — percentage clip ratio.
        extra_primitives: Optional list of additional arc-primitive template
            dicts (see :mod:`nclone.value_function.tools.audit_completeness`
            for the schema) that should be swept in addition to the built-in
            tables.  Absent or empty means the legacy fixed sweep is used.

    Returns:
        ``(edges, compact_waypoints)`` — edges after dominated-pair pruning,
        waypoints after global per-(cell, dst) min-reduce.
    """
    # Normalise mine_positions to a consistent array (may be empty)
    if mine_positions is None or len(mine_positions) == 0:
        _mine_pos: Optional[np.ndarray] = None
    else:
        _mine_pos = np.asarray(mine_positions, dtype=np.float32)

    # Extra primitives are validated here so the caller fails fast.  Actual
    # per-node ingestion is handled by ``_enumerate_node_edges`` via the
    # module-level ``_EXTRA_PRIMITIVES`` global (set below).  The global is
    # restored to ``None`` after the call so unrelated callers are unaffected.
    global _EXTRA_PRIMITIVES
    _EXTRA_PRIMITIVES = list(extra_primitives) if extra_primitives else None

    # Fast path: delegate to the cached three-phase pipeline when a tile
    # cache directory is supplied.  The cached path is single-process and
    # does not emit compact waypoints (the arc cache is only needed by
    # the legacy path - the tile cache's landing lookup plus ballistic
    # sweep supersedes it).
    if tile_cache_dir is not None:
        from .edge_enumeration_cached import enumerate_edges_cached
        edges = enumerate_edges_cached(
            graph, map_data,
            tile_cache_dir=tile_cache_dir,
            num_shards=tile_cache_num_shards,
            mine_positions=_mine_pos,
            mine_kill_radius=mine_kill_radius,
            mine_inflation_beta=mine_inflation_beta,
            max_arc_frames=max_arc_frames,
            verbose=verbose,
            stats_out=stats_out,
        )
        return edges, CompactWaypoints.empty()

    # Build list of all (pt_idx, vel_bin) pairs to enumerate
    node_indices: List[Tuple[int, int]] = []
    for pt_idx, sp in enumerate(graph.surface_points):
        n_bins = num_velocity_bins(sp.surface_type)
        for vb in range(n_bins):
            node_indices.append((pt_idx, vb))

    total_nodes = len(node_indices)
    if verbose:
        logger.info(f"Enumerating edges for {total_nodes:,} nodes "
                     f"({graph.n_points} points) with {num_workers} workers")

    sp_data = _serialize_surface_points(graph.surface_points)
    graph_params = {"max_vel_bins": graph.max_vel_bins}

    n_clipped_total = 0
    n_landings_total = 0

    if num_workers <= 1:
        # Single-process path — still collect via dict accumulator
        from nclone_native import Simulator as NativeSimulator
        sim = NativeSimulator.load(map_data)
        all_edges: List[EdgeResult] = []
        wp_accum: Optional[Dict[Tuple[int, int, int, int, int], float]] = (
            {} if collect_waypoints else None
        )
        for i, (pt_idx, vel_bin) in enumerate(node_indices):
            e, n_clip, n_land = _enumerate_node_edges(
                sim, graph, pt_idx, vel_bin,
                wp_accum=wp_accum,
                mine_positions=_mine_pos,
                mine_kill_radius=mine_kill_radius,
                mine_inflation_beta=mine_inflation_beta,
                waypoint_interval=waypoint_interval,
                max_arc_frames=max_arc_frames,
            )
            all_edges.extend(e)
            n_clipped_total += n_clip
            n_landings_total += n_land
            if verbose and (i + 1) % 1000 == 0:
                wp_count = len(wp_accum) if wp_accum is not None else 0
                logger.info(
                    f"  Processed {i+1:,}/{total_nodes:,} nodes, "
                    f"{len(all_edges):,} edges, {wp_count:,} unique wp cells"
                )
        compact = (
            CompactWaypoints.from_dict(wp_accum)
            if wp_accum is not None else CompactWaypoints.empty()
        )
    else:
        # Multi-process path — finer-grained chunking so imap_unordered can
        # release per-chunk memory as chunks are consumed.
        n_chunks = max(num_workers, num_workers * mp_chunks_per_worker)
        chunk_size = max(1, (total_nodes + n_chunks - 1) // n_chunks)
        chunks = [
            node_indices[i:i + chunk_size]
            for i in range(0, total_nodes, chunk_size)
        ]
        worker_args = [
            (map_data, sp_data, graph_params, chunk, collect_waypoints,
             _mine_pos, mine_kill_radius, mine_inflation_beta,
             waypoint_interval, max_arc_frames)
            for chunk in chunks
        ]

        if verbose:
            logger.info(f"  Distributing {len(chunks)} chunks "
                         f"across {num_workers} workers")

        all_edges = []
        merged: Dict[Tuple[int, int, int, int, int], float] = {} \
            if collect_waypoints else None  # type: ignore[assignment]
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_worker_enumerate, worker_args):
                (chunk_edges, chunk_keys, chunk_dst, chunk_ftl,
                 chunk_clip, chunk_land) = result
                all_edges.extend(chunk_edges)
                n_clipped_total += chunk_clip
                n_landings_total += chunk_land
                if merged is not None and len(chunk_dst) > 0:
                    _merge_worker_waypoints(
                        chunk_keys, chunk_dst, chunk_ftl, merged,
                    )
                # Drop references eagerly so the worker-side buffers can be
                # reclaimed before the next chunk is processed.
                del chunk_edges, chunk_keys, chunk_dst, chunk_ftl
                if verbose:
                    wp_count = len(merged) if merged is not None else 0
                    logger.info(
                        f"  Chunk done: total edges {len(all_edges):,}, "
                        f"unique wp cells {wp_count:,}"
                    )
        compact = (
            CompactWaypoints.from_dict(merged)
            if merged is not None else CompactWaypoints.empty()
        )

    # Prune dominated edges
    n_before = len(all_edges)
    all_edges = _prune_edges(all_edges)
    n_after = len(all_edges)

    pct_clipped = (
        100.0 * n_clipped_total / max(1, n_landings_total)
        if n_landings_total > 0 else 0.0
    )
    if verbose:
        logger.info(f"Edge enumeration complete: {n_after:,} edges "
                     f"(pruned {n_before - n_after:,} dominated)")
        logger.info(
            f"Arc waypoints (unique cells × destinations): {compact.n_entries:,}"
        )
        logger.info(
            f"Clip telemetry: {n_clipped_total:,}/{n_landings_total:,} "
            f"({pct_clipped:.2f}%) edges landed at a velocity-bin extreme"
        )

    if stats_out is not None:
        stats_out["n_clipped_landings"] = int(n_clipped_total)
        stats_out["n_total_landings"] = int(n_landings_total)
        stats_out["pct_clipped"] = float(pct_clipped)

    return all_edges, compact


def populate_graph_edges(
    graph: SurfaceGraph,
    edges: List[EdgeResult],
) -> None:
    """Write edge arrays into the graph structure.

    Args:
        graph: SurfaceGraph (modified in place).
        edges: List of EdgeResult from ``enumerate_edges``.
    """
    graph.edges_src = np.array([e.src_node for e in edges], dtype=np.int32)
    graph.edges_dst = np.array([e.dst_node for e in edges], dtype=np.int32)
    graph.edges_weight = np.array([e.weight for e in edges], dtype=np.float32)
    graph.edges_jump_hold = np.array(
        [e.jump_hold_frames for e in edges], dtype=np.int8
    )
    graph.edges_h_input = np.array(
        [e.h_input_during for e in edges], dtype=np.int8
    )
