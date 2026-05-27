"""Velocity-augmented surface graph with Dijkstra cost-to-go computation.

Builds a directed weighted graph whose nodes are (surface_point, velocity_bin)
pairs and whose edges are discovered by physics simulation.  Runs backward
Dijkstra from goal node sets to produce exact cost-to-go d*[node] arrays
suitable for PBRS potential evaluation.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

import numpy as np

from ..constants.physics_constants import (
    MAX_HOR_SPEED,
    NINJA_RADIUS,
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
)
from .surface_extraction import (
    SurfacePoint,
    SURFACE_FLOOR,
    SURFACE_WALL,
    SURFACE_SLIDE,
    SURFACE_LAUNCH_PAD,
    is_wall_like,
)

# ---------------------------------------------------------------------------
# Velocity bin configuration
# ---------------------------------------------------------------------------
# Schedule version identifier - persisted into the tile-cache shard headers
# so that bin-layout changes invalidate stale caches automatically.  Bump
# this any time ``_FLOOR_BIN_CENTERS`` or ``_WALL_BIN_CENTERS`` or the new
# slide / launch-pad schedules below change.
BIN_SCHEDULE_VERSION: int = 3

FLOOR_VEL_MIN = -MAX_HOR_SPEED
FLOOR_VEL_MAX = MAX_HOR_SPEED
WALL_VEL_MIN = -2.0   # wall-jump upward impulse with margin
WALL_VEL_MAX = 5.5     # approaching survivable impact limit

# Hand-tuned non-uniform velocity schedules.
#
# The linspace baselines discarded many common ninja velocities to the
# nearest-bin quantiser: zero (stationary) fell between bins, peak run speed
# (+/- MAX_HOR_SPEED ~= 3.33 px/f) sat adjacent to the outer bin without being
# exactly represented, and wall-slide terminal velocity (~5.5 px/f) only
# occupied one coarse bin at the end of a mostly-empty range.
#
# The new schedules are denser around:
#   * 0 px/f (stationary / on-ramp equilibria)
#   * +-1 px/f (walk / slow-run)
#   * +-MAX_HOR_SPEED (sprint, the dominant floor-jump velocity)
#   * wall downward terminal velocity (~4-5 px/f)
# and coarser in between to keep the total bin count small.  The total
# number of bins for each surface is preserved so existing .npz files keep
# loading; increments to the schedule force a cache rebuild via
# ``BIN_SCHEDULE_VERSION``.
_FLOOR_BIN_CENTERS = np.array([
    -MAX_HOR_SPEED,           # full-speed leftward run
    -MAX_HOR_SPEED * 0.75,
    -MAX_HOR_SPEED * 0.5,
    -MAX_HOR_SPEED * 0.3,
    -MAX_HOR_SPEED * 0.15,
    -0.1,
    0.0,                       # exact stationary bin (was between linspace bins)
    0.1,
    MAX_HOR_SPEED * 0.15,
    MAX_HOR_SPEED * 0.3,
    MAX_HOR_SPEED * 0.5,
    MAX_HOR_SPEED * 0.75,
    MAX_HOR_SPEED * 0.9,
    MAX_HOR_SPEED * 0.98,
    MAX_HOR_SPEED,             # full-speed rightward run (saturated)
], dtype=np.float32)
NUM_VELOCITY_BINS_FLOOR: int = int(_FLOOR_BIN_CENTERS.shape[0])

_WALL_BIN_CENTERS = np.array([
    WALL_VEL_MIN,              # wall-jump upward impulse
    -1.0,
    -0.4,
    0.0,
    0.4,                       # initial slow downward slide
    1.0,
    1.8,
    2.6,
    3.4,
    4.2,
    5.0,
    WALL_VEL_MAX,              # saturated wall-slide terminal velocity
], dtype=np.float32)
NUM_VELOCITY_BINS_WALL: int = int(_WALL_BIN_CENTERS.shape[0])
# Effective bin widths (mean spacing).  Kept for any legacy callers; code
# that needs exact per-bin widths should use the ``_FLOOR_BIN_CENTERS``
# array directly.
_FLOOR_BIN_WIDTH = float(
    (_FLOOR_BIN_CENTERS[-1] - _FLOOR_BIN_CENTERS[0])
    / max(1, NUM_VELOCITY_BINS_FLOOR - 1)
)
_WALL_BIN_WIDTH = float(
    (_WALL_BIN_CENTERS[-1] - _WALL_BIN_CENTERS[0])
    / max(1, NUM_VELOCITY_BINS_WALL - 1)
)

# ---------------------------------------------------------------------------
# SURFACE_SLIDE bins — a narrower, downward-dominant subset of the wall
# schedule.  A slide node represents the ninja already wall-sliding, so the
# zero / upward bins are dropped; 0.0 is retained so the "just entered
# slide" state (vy ~= 0) remains expressible.
_SLIDE_BIN_CENTERS = np.array([
    0.0,
    0.4,
    1.0,
    1.8,
    2.6,
    3.4,
    4.2,
    5.0,
    WALL_VEL_MAX,
], dtype=np.float32)
NUM_VELOCITY_BINS_SLIDE: int = int(_SLIDE_BIN_CENTERS.shape[0])

# ---------------------------------------------------------------------------
# SURFACE_LAUNCH_PAD bins — a launch pad emits a fixed boost velocity, so
# only one "bin" is ever meaningful.  The physical velocity is carried as
# edge metadata rather than a bin coordinate.
_LAUNCH_PAD_BIN_CENTERS = np.array([0.0], dtype=np.float32)
NUM_VELOCITY_BINS_LAUNCH_PAD: int = 1

# ---------------------------------------------------------------------------
# Spatial index cell size for nearest-node lookup (pixels)
# ---------------------------------------------------------------------------
_SPATIAL_CELL_SIZE = 6.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SurfaceGraph:
    """Velocity-augmented surface graph.

    Attributes:
        surface_points: List of SurfacePoint (position / normal info).
        n_points: Number of surface points.
        n_nodes: Total number of graph nodes (points x velocity bins).
        velocity_bins_floor: Bin center array for floor nodes.
        velocity_bins_wall: Bin center array for wall nodes.
        max_vel_bins: Max bin count (for node-id arithmetic).
        edges_src: Source node ids (int32).
        edges_dst: Destination node ids (int32).
        edges_weight: Edge frame-counts (float32).
        d_star_switch: Cost-to-go to switch for each node (float32).
        d_star_exit: Cost-to-go to exit for each node (float32).
        d_max_switch: Max finite d_star_switch.
        d_max_exit: Max finite d_star_exit.
        spatial_index: {(gx, gy)} -> list of surface-point indices.
    """
    surface_points: List[SurfacePoint]
    n_points: int
    n_nodes: int
    velocity_bins_floor: np.ndarray
    velocity_bins_wall: np.ndarray
    max_vel_bins: int

    # Edges (populated by edge_enumeration)
    edges_src: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    edges_dst: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    edges_weight: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    # Action metadata per edge (optional; None if loaded from legacy .npz)
    edges_jump_hold: Optional[np.ndarray] = None  # int8: hold duration (0=run/slide)
    edges_h_input: Optional[np.ndarray] = None    # int8: primary h-input (-1/0/1)
    edges_h_after: Optional[np.ndarray] = None    # int8: post-release h-input (-1/0/1)

    # Trick edges (e.g. Corner Double Jump): edge_index -> per-tick (h,j) tape.
    # These are composite edges the executor replays verbatim (not a single
    # continuous jump), letting the global d* route compose tricks anywhere.
    trick_tapes: Optional[Dict[int, list]] = None

    # Cost-to-go arrays (populated by dijkstra)
    d_star_switch: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32))
    d_star_exit: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32))
    d_max_switch: float = 0.0
    d_max_exit: float = 0.0

    # Spatial index for point lookup
    spatial_index: Dict[Tuple[int, int], List[int]] = field(
        default_factory=dict)


# ---------------------------------------------------------------------------
# Node-id helpers
# ---------------------------------------------------------------------------
def num_velocity_bins(surface_type: int) -> int:
    """Return the number of velocity bins for the given surface type."""
    if surface_type == SURFACE_FLOOR:
        return NUM_VELOCITY_BINS_FLOOR
    if surface_type == SURFACE_SLIDE:
        return NUM_VELOCITY_BINS_SLIDE
    if surface_type == SURFACE_LAUNCH_PAD:
        return NUM_VELOCITY_BINS_LAUNCH_PAD
    return NUM_VELOCITY_BINS_WALL


def velocity_bin_centers(surface_type: int) -> np.ndarray:
    """Return the velocity bin center array for the surface type."""
    if surface_type == SURFACE_FLOOR:
        return _FLOOR_BIN_CENTERS
    if surface_type == SURFACE_SLIDE:
        return _SLIDE_BIN_CENTERS
    if surface_type == SURFACE_LAUNCH_PAD:
        return _LAUNCH_PAD_BIN_CENTERS
    return _WALL_BIN_CENTERS


def velocity_to_bin(velocity: float, surface_type: int) -> int:
    """Quantize a velocity value to the nearest bin index."""
    centers = velocity_bin_centers(surface_type)
    idx = int(np.argmin(np.abs(centers - velocity)))
    return idx


def velocity_to_bin_fractional(
    velocity: float, surface_type: int,
) -> Tuple[int, int, float]:
    """Map a velocity to bracketing bin indices and interpolation alpha.

    Returns:
        (bin_lo, bin_hi, alpha) such that
        value ~= (1 - alpha) * d_star[bin_lo] + alpha * d_star[bin_hi].
    """
    centers = velocity_bin_centers(surface_type)
    n = len(centers)
    if n < 2:
        return (0, 0, 0.0)
    if velocity <= centers[0]:
        return (0, 0, 0.0)
    if velocity >= centers[-1]:
        return (n - 1, n - 1, 0.0)
    # Binary search for bracketing bins
    lo = int(np.searchsorted(centers, velocity, side="right")) - 1
    lo = max(0, min(lo, n - 2))
    hi = lo + 1
    width = centers[hi] - centers[lo]
    alpha = (velocity - centers[lo]) / width if width > 1e-9 else 0.0
    return (lo, hi, float(alpha))


def node_id(point_index: int, vel_bin: int, max_vel_bins: int) -> int:
    """Compute a unique integer node id."""
    return point_index * max_vel_bins + vel_bin


def node_id_to_parts(nid: int, max_vel_bins: int) -> Tuple[int, int]:
    """Recover (point_index, vel_bin) from a node id."""
    return divmod(nid, max_vel_bins)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(surface_points: List[SurfacePoint]) -> SurfaceGraph:
    """Build an empty velocity-augmented surface graph from surface points.

    Edges are populated later by ``edge_enumeration.enumerate_edges``.

    Args:
        surface_points: Output of ``surface_extraction.extract_surfaces``.

    Returns:
        SurfaceGraph with nodes, spatial index, and empty edge arrays.
    """
    n_points = len(surface_points)
    max_bins = max(
        NUM_VELOCITY_BINS_FLOOR,
        NUM_VELOCITY_BINS_WALL,
        NUM_VELOCITY_BINS_SLIDE,
        NUM_VELOCITY_BINS_LAUNCH_PAD,
    )
    n_nodes = n_points * max_bins

    # Build spatial index
    inv_cell = 1.0 / _SPATIAL_CELL_SIZE
    spatial_index: Dict[Tuple[int, int], List[int]] = {}
    for i, sp in enumerate(surface_points):
        gx = int(math.floor(sp.x * inv_cell))
        gy = int(math.floor(sp.y * inv_cell))
        key = (gx, gy)
        if key not in spatial_index:
            spatial_index[key] = []
        spatial_index[key].append(i)

    return SurfaceGraph(
        surface_points=surface_points,
        n_points=n_points,
        n_nodes=n_nodes,
        velocity_bins_floor=_FLOOR_BIN_CENTERS.copy(),
        velocity_bins_wall=_WALL_BIN_CENTERS.copy(),
        max_vel_bins=max_bins,
        spatial_index=spatial_index,
    )


# ---------------------------------------------------------------------------
# Nearest-node lookup
# ---------------------------------------------------------------------------
def find_nearest_point(
    graph: SurfaceGraph,
    x: float,
    y: float,
    surface_type: Optional[int] = None,
) -> Optional[int]:
    """Find the nearest surface-point index to (x, y).

    Args:
        graph: The surface graph.
        x, y: Query position (pixels, ninja center).
        surface_type: If given, restrict to points of this type.

    Returns:
        Index into graph.surface_points, or None if no point found.
    """
    inv_cell = 1.0 / _SPATIAL_CELL_SIZE
    gx = int(math.floor(x * inv_cell))
    gy = int(math.floor(y * inv_cell))

    best_idx: Optional[int] = None
    best_dist_sq = float("inf")
    search_radius = 3  # cells

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            key = (gx + dx, gy + dy)
            indices = graph.spatial_index.get(key)
            if indices is None:
                continue
            for idx in indices:
                sp = graph.surface_points[idx]
                if surface_type is not None and sp.surface_type != surface_type:
                    continue
                d2 = (sp.x - x) ** 2 + (sp.y - y) ** 2
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_idx = idx

    return best_idx


def find_nearest_node(
    graph: SurfaceGraph,
    x: float,
    y: float,
    velocity: float,
    surface_type: int,
) -> Optional[int]:
    """Find the nearest graph node to (x, y, velocity, surface_type).

    Returns:
        Node id, or None.
    """
    pt_idx = find_nearest_point(graph, x, y, surface_type)
    if pt_idx is None:
        return None
    vel_bin = velocity_to_bin(velocity, surface_type)
    return node_id(pt_idx, vel_bin, graph.max_vel_bins)


def lookup_cost_interpolated(
    graph: SurfaceGraph,
    d_star: np.ndarray,
    x: float,
    y: float,
    velocity: float,
    surface_type: int,
) -> float:
    """Look up cost-to-go with velocity interpolation.

    Returns the linearly interpolated cost between the two bracketing
    velocity bins at the nearest surface point.
    """
    pt_idx = find_nearest_point(graph, x, y, surface_type)
    if pt_idx is None:
        return float("inf")

    lo, hi, alpha = velocity_to_bin_fractional(velocity, surface_type)
    nid_lo = node_id(pt_idx, lo, graph.max_vel_bins)
    nid_hi = node_id(pt_idx, hi, graph.max_vel_bins)

    c_lo = d_star[nid_lo] if nid_lo < len(d_star) else float("inf")
    c_hi = d_star[nid_hi] if nid_hi < len(d_star) else float("inf")

    if not np.isfinite(c_lo) and not np.isfinite(c_hi):
        return float("inf")
    if not np.isfinite(c_lo):
        return float(c_hi)
    if not np.isfinite(c_hi):
        return float(c_lo)

    return float((1.0 - alpha) * c_lo + alpha * c_hi)


# ---------------------------------------------------------------------------
# Goal identification
# ---------------------------------------------------------------------------
def find_goal_nodes(
    graph: SurfaceGraph,
    goal_x: float,
    goal_y: float,
    goal_type: str,
) -> List[int]:
    """Find all graph nodes within contact range of a goal entity.

    Args:
        graph: The surface graph.
        goal_x, goal_y: Goal entity position (pixels).
        goal_type: ``"switch"`` or ``"exit"``.

    Returns:
        List of node ids that can contact the goal.
    """
    if goal_type == "switch":
        contact_range = NINJA_RADIUS + EXIT_SWITCH_RADIUS
    elif goal_type == "exit":
        contact_range = NINJA_RADIUS + EXIT_DOOR_RADIUS
    else:
        raise ValueError(f"Unknown goal_type: {goal_type!r}")

    contact_range_sq = contact_range * contact_range
    goal_nodes: List[int] = []

    for pt_idx, sp in enumerate(graph.surface_points):
        # Distance from ninja center to goal center
        d2 = (sp.x - goal_x) ** 2 + (sp.y - goal_y) ** 2
        if d2 <= contact_range_sq:
            n_bins = num_velocity_bins(sp.surface_type)
            for vb in range(n_bins):
                goal_nodes.append(node_id(pt_idx, vb, graph.max_vel_bins))

    return goal_nodes


# ---------------------------------------------------------------------------
# Backward Dijkstra
# ---------------------------------------------------------------------------
def _build_reverse_adjacency(
    graph: SurfaceGraph,
) -> Dict[int, List[Tuple[int, float, int]]]:
    """Build the reversed adjacency list used by every backward-Dijkstra pass.

    rev_adj[dst] is a list of ``(src, weight, original_edge_index)`` tuples
    representing edges ``src -> dst`` in the original graph.  Backward
    Dijkstra settles ``u = dst`` and relaxes each ``v = src`` neighbor, so
    an incoming original edge ``v -> u`` becomes an outgoing edge in the
    reversed graph.  The third element (``i``) lets the tree-building variant
    record ``best_edge_idx[v] = i`` without rescanning the edge list.

    Building this once and reusing it across both goals avoids redundant
    Python loops over every edge (O(|E|) × 2) and roughly halves the time
    spent on Dijkstra's preparation on large graphs.
    """
    rev_adj: Dict[int, List[Tuple[int, float, int]]] = {}
    src_arr = graph.edges_src
    dst_arr = graph.edges_dst
    w_arr = graph.edges_weight
    for i in range(len(src_arr)):
        dst = int(dst_arr[i])
        bucket = rev_adj.get(dst)
        if bucket is None:
            bucket = []
            rev_adj[dst] = bucket
        bucket.append((int(src_arr[i]), float(w_arr[i]), i))
    return rev_adj


def dijkstra_backward(
    graph: SurfaceGraph,
    goal_nodes: List[int],
    blocked_nodes: Optional[Set[int]] = None,
    rev_adj: Optional[Dict[int, List[Tuple[int, float, int]]]] = None,
) -> np.ndarray:
    """Run backward Dijkstra from goal nodes on the reversed edge set.

    Args:
        graph: Surface graph with populated edge arrays.
        goal_nodes: Node ids initialised at cost 0 (the goal).
        blocked_nodes: Optional set of node ids that are impassable (e.g. surface
            positions occupied by deadly toggle mines in the relevant phase).
            Blocked nodes are never assigned a finite d* value and are skipped
            as transit nodes, so no path can route through them.
        rev_adj: Optional prebuilt reversed adjacency list from
            ``_build_reverse_adjacency``.  Pass this in when running multiple
            Dijkstra calls on the same graph to avoid duplicate O(|E|) builds.

    Returns:
        d_star: float32 array of length ``graph.n_nodes``.
                d_star[node] = minimum frame-count from node to any goal node.
                Unreachable and blocked nodes receive ``inf``.
    """
    # Share the heavy rev_adj build with the tree-building variant.
    d_star, _pred, _best = dijkstra_backward_with_tree(
        graph, goal_nodes,
        blocked_nodes=blocked_nodes,
        rev_adj=rev_adj,
    )
    return d_star


def dijkstra_backward_with_tree(
    graph: SurfaceGraph,
    goal_nodes: List[int],
    blocked_nodes: Optional[Set[int]] = None,
    rev_adj: Optional[Dict[int, List[Tuple[int, float, int]]]] = None,
    goal_init_costs: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run backward Dijkstra and return the shortest-path tree.

    Identical to ``dijkstra_backward`` but also returns predecessor and
    best-edge-index arrays that encode the shortest-path tree.  These arrays
    enable forward route extraction: follow ``predecessor[v]`` repeatedly from
    any start node to trace the optimal path toward the goal.

    Predecessor convention (backward Dijkstra):
        ``predecessor[v] = u`` means: from node ``v`` the optimal next step
        toward the goal is to traverse original edge ``v -> u``.  Equivalently,
        ``d_star[u] + w(v->u) == d_star[v]`` for the chosen best edge.

        Invariant: ``d_star[predecessor[v]] < d_star[v]`` for all non-goal
        nodes ``v`` with finite d_star.  Goal nodes have
        ``predecessor[v] = -1``.

    Args:
        graph: Surface graph with populated edge arrays.
        goal_nodes: Node ids initialised at cost 0 (the goals).
        blocked_nodes: Optional set of impassable node ids.
        goal_init_costs: Optional ``{node: initial_cost}`` map.  Goal nodes are
            seeded with this cost instead of 0 — used for two-phase composition
            where each switch-contact node is seeded with its remaining cost to
            the exit (its phase-1 ``d_exit`` value).  Nodes in ``goal_nodes`` not
            present here default to 0.0.

    Returns:
        (d_star, predecessor, best_edge_idx)
            d_star: float32 array length ``n_nodes``.
            predecessor: int32 array; -1 for goals/unreachable.
            best_edge_idx: int32 array; index into graph.edges_* arrays.
                ``graph.edges_src[best_edge_idx[v]] == v`` and
                ``graph.edges_dst[best_edge_idx[v]] == predecessor[v]``
                for all non-goal nodes with finite d_star.
                -1 for goals/unreachable.
    """
    n = graph.n_nodes
    # Accumulate in float64 and cast to float32 only on return.  The heap keys
    # are Python floats (f64); storing the working distances as float32 would
    # truncate ``new_dist`` *down* for non-integer weights, after which the
    # popped f64 key tests ``dist_u > d_star[u]`` and the entry is wrongly
    # skipped as stale — its finite cost then never propagates, leaving
    # genuinely-reachable nodes at ``inf``.  Integer frame weights are exact in
    # float32 so this was latent until risk-reweighted (non-integer) costs were
    # introduced; float64 fixes it and is identical for integer weights.
    d_star = np.full(n, np.inf, dtype=np.float64)
    predecessor = np.full(n, -1, dtype=np.int32)
    best_edge_idx = np.full(n, -1, dtype=np.int32)
    _blocked = blocked_nodes or set()
    _seed = goal_init_costs or {}

    # rev_adj[dst] -> list of (src, weight, original_edge_index)
    # An entry (src, w, i) means original edge i: src -> dst exists.
    # When we settle dst (u) and relax src (v):
    #   predecessor[v] = u  (in original graph, v -> u is the next step toward goal)
    #   best_edge_idx[v] = i  (index of the v -> u edge in graph.edges_*)
    if rev_adj is None:
        rev_adj = _build_reverse_adjacency(graph)

    heap: List[Tuple[float, int]] = []
    for gn in goal_nodes:
        if 0 <= gn < n and gn not in _blocked:
            c = float(_seed.get(gn, 0.0))
            if c < d_star[gn]:  # keep the cheapest seed if duplicates
                d_star[gn] = c
                heapq.heappush(heap, (c, gn))

    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > d_star[u]:
            continue
        neighbors = rev_adj.get(u)
        if neighbors is None:
            continue
        for v, w, edge_i in neighbors:
            if v in _blocked:
                continue
            new_dist = dist_u + w
            if new_dist < d_star[v]:
                d_star[v] = new_dist
                predecessor[v] = u       # original edge v -> u is optimal
                best_edge_idx[v] = edge_i
                heapq.heappush(heap, (new_dist, v))

    return d_star.astype(np.float32), predecessor, best_edge_idx


def compute_cost_to_go_with_tree(
    graph: SurfaceGraph,
    goal_x: float,
    goal_y: float,
    goal_type: str,
    blocked_nodes: Optional[Set[int]] = None,
    rev_adj: Optional[Dict[int, List[Tuple[int, float, int]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cost-to-go and shortest-path tree for the given goal.

    Convenience wrapper around ``dijkstra_backward_with_tree``.  Pass
    ``rev_adj`` (from ``_build_reverse_adjacency``) to share the reversed
    adjacency list across multiple goals on the same graph.

    Returns:
        (d_star, predecessor, best_edge_idx) — see
        ``dijkstra_backward_with_tree`` for semantics.
    """
    goal_nodes = find_goal_nodes(graph, goal_x, goal_y, goal_type)
    if not goal_nodes:
        raise ValueError(
            f"No graph nodes within contact range of {goal_type} "
            f"at ({goal_x:.1f}, {goal_y:.1f}).  Check surface extraction."
        )
    return dijkstra_backward_with_tree(
        graph, goal_nodes, blocked_nodes=blocked_nodes, rev_adj=rev_adj,
    )


def compute_two_phase_cost_to_go(
    graph: SurfaceGraph,
    switch_x: float,
    switch_y: float,
    exit_x: float,
    exit_y: float,
    blocked_nodes: Optional[Set[int]] = None,
    rev_adj: Optional[Dict[int, List[Tuple[int, float, int]]]] = None,
    switch_air_contacts: Optional[Dict[int, float]] = None,
    exit_air_contacts: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jointly optimal switch-then-exit cost-to-go.

    The level is solved by touching the exit switch and then the exit door, so a
    node's true cost-to-go is ``cost(node -> nearest switch) + cost(that switch
    -> exit)`` minimised jointly.  Computing ``d_switch`` and ``d_exit``
    independently and summing the spawn's ``d_switch`` with one switch node's
    ``d_exit`` is NOT jointly optimal: the cheapest switch to *reach* is rarely
    the cheapest switch to *leave for the exit* (measured: long-jump-reachable
    235f separate vs 201f joint vs 198f spike — the 19% gap was almost entirely
    this composition error, not edge-set incompleteness).

    Correct composition: run phase-1 backward Dijkstra to the exit, then seed the
    phase-0 backward Dijkstra's switch-contact goal nodes with their phase-1
    ``d_exit`` value (instead of 0).  ``d_combined[node]`` is then the jointly
    optimal frames from ``node`` (pre-switch) to the exit via some switch.

    Route extraction: follow the ``predecessor_combined`` tree from the start node
    until a switch-contact node (``predecessor_combined == -1`` — a phase-0 seed),
    then continue along the ``predecessor_exit`` tree to the exit.

    Returns:
        ``(d_combined, pred_combined, best_combined, d_exit, pred_exit, best_exit)``.
    """
    if rev_adj is None:
        rev_adj = _build_reverse_adjacency(graph)
    _blocked = blocked_nodes or set()

    # --- Phase 1: cost-to-exit ---------------------------------------------
    # Exit goal seeds = surface-adjacent exit-contact nodes (cost 0) UNION
    # air-reachable exit contacts (cost = frames for the arc that passes through
    # the exit's contact zone — the exit may have no adjacent surface node, which
    # is valid on open levels).
    exit_seed: Dict[int, float] = {
        n: 0.0 for n in find_goal_nodes(graph, exit_x, exit_y, "exit")
    }
    # Air contacts are a FALLBACK only when the exit has NO surface-adjacent node
    # (a genuinely air-reachable exit).  When a surface node exists, prefer the
    # surface route: the executor reaches a goal by *standing on its node*, and a
    # fly-through arc that the d* finds cheaper is not (yet) executable as a
    # route arc — using it would mis-route otherwise-solvable levels.
    if exit_air_contacts and not exit_seed:
        for s, fr in exit_air_contacts.items():
            if s not in _blocked and (s not in exit_seed or fr < exit_seed[s]):
                exit_seed[s] = float(fr)
    if not exit_seed:
        raise ValueError(
            f"Exit at ({exit_x:.1f}, {exit_y:.1f}) is unreachable: no surface "
            f"contact node and no air-contact arc.  Check surface extraction / "
            f"pass goal_positions to enumerate_edges for air goals."
        )
    d_exit, pred_exit, best_exit = dijkstra_backward_with_tree(
        graph, list(exit_seed),
        blocked_nodes=blocked_nodes, rev_adj=rev_adj,
        goal_init_costs=exit_seed,
    )

    # --- Phase 0: jointly optimal cost spawn -> switch -> exit --------------
    # Surface-adjacent switch nodes are seeded with their d_exit (standing on the
    # switch, then go to the exit).  Air-reachable switch contacts: the arc from
    # source ``s`` touches the switch in ``fr`` frames; after that arc the ninja
    # continues to the exit, approximated by ``fr + d_exit[s]`` (the executor's
    # micro-search recovers the exact continuation — it targets the entities).
    surface_switch_nodes = find_goal_nodes(graph, switch_x, switch_y, "switch")
    goal_init: Dict[int, float] = {}
    for s in surface_switch_nodes:
        if s not in _blocked and np.isfinite(d_exit[s]):
            goal_init[s] = float(d_exit[s])
    # Air contacts are a FALLBACK only when the switch has no surface-adjacent
    # node (see the exit note above).
    if switch_air_contacts and not surface_switch_nodes:
        # Global min finite d_exit as a fallback continuation estimate.
        finite = d_exit[np.isfinite(d_exit)]
        min_dexit = float(finite.min()) if finite.size else 0.0
        for s, fr in switch_air_contacts.items():
            if s in _blocked:
                continue
            cont = float(d_exit[s]) if np.isfinite(d_exit[s]) else min_dexit
            c = float(fr) + cont
            if s not in goal_init or c < goal_init[s]:
                goal_init[s] = c
    if not goal_init:
        raise ValueError(
            f"Switch at ({switch_x:.1f}, {switch_y:.1f}) is unreachable: no "
            f"surface contact node and no air-contact arc.  Pass goal_positions "
            f"to enumerate_edges for air goals."
        )
    d_combined, pred_combined, best_combined = dijkstra_backward_with_tree(
        graph, list(goal_init),
        blocked_nodes=blocked_nodes, rev_adj=rev_adj,
        goal_init_costs=goal_init,
    )
    return (d_combined, pred_combined, best_combined,
            d_exit, pred_exit, best_exit)


def find_mine_blocked_nodes(
    graph: SurfaceGraph,
    mine_positions: List[Tuple[float, float]],
    kill_radius: float = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED,
) -> Set[int]:
    """Find all graph nodes whose surface position falls within mine kill range.

    Surface points store the ninja CENTER (offset NINJA_RADIUS from contact
    surface), so the kill condition is simply distance(ninja_center, mine) <
    kill_radius.  All velocity bins for a blocked surface point are blocked.

    Args:
        graph: The surface graph.
        mine_positions: List of (x, y) mine center positions in pixels.
        kill_radius: Distance threshold for death (default: NINJA_RADIUS +
            TOGGLE_MINE_RADIUS_TOGGLED = 14.0 px).

    Returns:
        Set of graph node ids that are impassable due to mine proximity.
    """
    if not mine_positions:
        return set()

    kill_sq = kill_radius * kill_radius
    blocked: Set[int] = set()

    for pt_idx, sp in enumerate(graph.surface_points):
        for mx, my in mine_positions:
            dx = sp.x - mx
            dy = sp.y - my
            if dx * dx + dy * dy <= kill_sq:
                n_bins = num_velocity_bins(sp.surface_type)
                for vb in range(n_bins):
                    blocked.add(node_id(pt_idx, vb, graph.max_vel_bins))
                break  # one mine match is enough to block this point

    return blocked


def compute_cost_to_go(
    graph: SurfaceGraph,
    goal_x: float,
    goal_y: float,
    goal_type: str,
    blocked_nodes: Optional[Set[int]] = None,
    rev_adj: Optional[Dict[int, List[Tuple[int, float, int]]]] = None,
) -> np.ndarray:
    """Compute cost-to-go from every graph node to the specified goal.

    Identifies goal nodes, runs backward Dijkstra and returns the raw result.
    Unreachable nodes (unused velocity bins, isolated nodes, blocked nodes)
    keep ``inf`` so that ``_robust_d_max`` / ``_safe_max`` can compute
    percentile-based normalization constants from truly reachable costs only.

    All downstream consumers (``lookup_cost_interpolated``, ``arc_cache.build``,
    ``_pessimistic_fallback``) already guard with ``np.isfinite``, so keeping
    ``inf`` is safe.

    Args:
        graph: Surface graph with populated edge arrays.
        goal_x, goal_y: Goal entity position (pixels).
        goal_type: ``"switch"`` or ``"exit"``.
        blocked_nodes: Optional set of impassable node ids (e.g. nodes near
            deadly toggle mines for this phase).  These nodes receive d* = inf.

    Returns:
        d_star array (float32, length n_nodes).  Unreachable nodes have inf.
    """
    goal_nodes = find_goal_nodes(graph, goal_x, goal_y, goal_type)
    if not goal_nodes:
        raise ValueError(
            f"No graph nodes within contact range of {goal_type} "
            f"at ({goal_x:.1f}, {goal_y:.1f}).  Check surface extraction."
        )

    return dijkstra_backward(
        graph, goal_nodes, blocked_nodes=blocked_nodes, rev_adj=rev_adj,
    )
