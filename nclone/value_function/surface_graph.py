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
from .surface_extraction import SurfacePoint, SURFACE_FLOOR, SURFACE_WALL

# ---------------------------------------------------------------------------
# Velocity bin configuration
# ---------------------------------------------------------------------------
NUM_VELOCITY_BINS_FLOOR = 15
NUM_VELOCITY_BINS_WALL = 12

FLOOR_VEL_MIN = -MAX_HOR_SPEED
FLOOR_VEL_MAX = MAX_HOR_SPEED
WALL_VEL_MIN = -2.0   # wall-jump upward impulse with margin
WALL_VEL_MAX = 5.5     # approaching survivable impact limit

# Pre-compute bin centers
_FLOOR_BIN_CENTERS = np.linspace(
    FLOOR_VEL_MIN, FLOOR_VEL_MAX, NUM_VELOCITY_BINS_FLOOR, dtype=np.float32,
)
_WALL_BIN_CENTERS = np.linspace(
    WALL_VEL_MIN, WALL_VEL_MAX, NUM_VELOCITY_BINS_WALL, dtype=np.float32,
)
_FLOOR_BIN_WIDTH = (
    _FLOOR_BIN_CENTERS[1] - _FLOOR_BIN_CENTERS[0]
    if NUM_VELOCITY_BINS_FLOOR > 1 else 1.0
)
_WALL_BIN_WIDTH = (
    _WALL_BIN_CENTERS[1] - _WALL_BIN_CENTERS[0]
    if NUM_VELOCITY_BINS_WALL > 1 else 1.0
)

# ---------------------------------------------------------------------------
# Spatial index cell size for nearest-node lookup (pixels)
# ---------------------------------------------------------------------------
_SPATIAL_CELL_SIZE = 12.0


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
    return NUM_VELOCITY_BINS_WALL


def velocity_bin_centers(surface_type: int) -> np.ndarray:
    """Return the velocity bin center array for the surface type."""
    if surface_type == SURFACE_FLOOR:
        return _FLOOR_BIN_CENTERS
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
    max_bins = max(NUM_VELOCITY_BINS_FLOOR, NUM_VELOCITY_BINS_WALL)
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
def dijkstra_backward(
    graph: SurfaceGraph,
    goal_nodes: List[int],
    blocked_nodes: Optional[Set[int]] = None,
) -> np.ndarray:
    """Run backward Dijkstra from goal nodes on the reversed edge set.

    Args:
        graph: Surface graph with populated edge arrays.
        goal_nodes: Node ids initialised at cost 0 (the goal).
        blocked_nodes: Optional set of node ids that are impassable (e.g. surface
            positions occupied by deadly toggle mines in the relevant phase).
            Blocked nodes are never assigned a finite d* value and are skipped
            as transit nodes, so no path can route through them.

    Returns:
        d_star: float32 array of length ``graph.n_nodes``.
                d_star[node] = minimum frame-count from node to any goal node.
                Unreachable and blocked nodes receive ``inf``.
    """
    n = graph.n_nodes
    d_star = np.full(n, np.inf, dtype=np.float32)
    _blocked = blocked_nodes or set()

    # Build reversed adjacency list from edge arrays
    # rev_adj[dst] -> list of (src, weight)   (in the reversed graph, dst is the start)
    rev_adj: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(len(graph.edges_src)):
        src = int(graph.edges_src[i])
        dst = int(graph.edges_dst[i])
        w = float(graph.edges_weight[i])
        if dst not in rev_adj:
            rev_adj[dst] = []
        rev_adj[dst].append((src, w))

    # Initialize priority queue with goal nodes at cost 0
    heap: List[Tuple[float, int]] = []
    for gn in goal_nodes:
        if 0 <= gn < n and gn not in _blocked:
            d_star[gn] = 0.0
            heapq.heappush(heap, (0.0, gn))

    # Standard Dijkstra — skip blocked nodes so no path routes through them
    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > d_star[u]:
            continue
        neighbors = rev_adj.get(u)
        if neighbors is None:
            continue
        for v, w in neighbors:
            if v in _blocked:
                continue
            new_dist = dist_u + w
            if new_dist < d_star[v]:
                d_star[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return d_star


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
) -> np.ndarray:
    """Compute cost-to-go from every graph node to the specified goal.

    Identifies goal nodes, runs backward Dijkstra, and fills unreachable
    nodes with ``d_max`` (the maximum finite cost).

    Args:
        graph: Surface graph with populated edge arrays.
        goal_x, goal_y: Goal entity position (pixels).
        goal_type: ``"switch"`` or ``"exit"``.
        blocked_nodes: Optional set of impassable node ids (e.g. nodes near
            deadly toggle mines for this phase).  These nodes receive d* = inf.

    Returns:
        d_star array (float32, length n_nodes).
    """
    goal_nodes = find_goal_nodes(graph, goal_x, goal_y, goal_type)
    if not goal_nodes:
        raise ValueError(
            f"No graph nodes within contact range of {goal_type} "
            f"at ({goal_x:.1f}, {goal_y:.1f}).  Check surface extraction."
        )

    d_star = dijkstra_backward(graph, goal_nodes, blocked_nodes=blocked_nodes)

    # Replace inf with d_max for unreachable nodes
    finite_mask = np.isfinite(d_star)
    if finite_mask.any():
        d_max = float(np.max(d_star[finite_mask]))
    else:
        d_max = 1.0

    d_star[~finite_mask] = d_max
    return d_star
