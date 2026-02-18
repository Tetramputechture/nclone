"""Velocity-augmented reachability graph with hull-masked jump edges.

Builds a directed weighted graph whose nodes are (surface_point_index, velocity_bin)
pairs.  Edges come from two sources:

1. **Walk edges**: from the existing graph adjacency (short-range, 12 px steps).
2. **Jump edges**: from precomputed jump hulls -- geometry-independent swept
   areas for each (surface_type, velocity_bin), intersected with the level's
   reachable node set.

Backward Dijkstra from goal node sets produces exact cost-to-go d_star arrays
suitable for PBRS potential evaluation.
"""

import heapq
import logging
import math
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
)
from nclone.value_function.surface_graph import (
    NUM_VELOCITY_BINS_FLOOR,
    NUM_VELOCITY_BINS_WALL,
    _FLOOR_BIN_CENTERS,
    _WALL_BIN_CENTERS,
)
from nclone.graph.reachability.graph_builder import SUB_NODE_SIZE
from .jump_hull import (
    JumpHullCollection,
    WallHullCollection,
    classify_surface,
    get_or_build_hulls,
    get_or_build_wall_hulls,
    GRID_CELL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_VEL_BINS = max(NUM_VELOCITY_BINS_FLOOR, NUM_VELOCITY_BINS_WALL)

# Walk edge cost: 1 frame per 12 px step (approximate, consistent with
# the graph adjacency structure where each step is 12 px).
WALK_EDGE_WEIGHT = 1.0

# Spatial cell size for the internal point lookup grid (px).
_SPATIAL_CELL_SIZE = 12.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class ReachabilityGraph:
    """Velocity-augmented reachability graph built from existing graph data.

    Attributes:
        point_positions: list of (x, y) tuples (surface point coordinates).
        point_is_floor: bool array, True if the point is grounded (floor).
        point_is_wall: bool array, True if the point is walled.
        n_points: number of surface points.
        n_nodes: total graph nodes = n_points * MAX_VEL_BINS.
        edges_src: source node ids (int32).
        edges_dst: destination node ids (int32).
        edges_weight: frame-cost weights (float32).
        d_star_switch: cost-to-go to switch (float32, length n_nodes).
        d_star_exit: cost-to-go to exit (float32, length n_nodes).
        d_max_switch: maximum finite switch cost (for normalization).
        d_max_exit: maximum finite exit cost (for normalization).
        spatial_index: {(gx, gy) -> list[point_index]} for nearest lookup.
    """

    __slots__ = (
        "point_positions",
        "point_is_floor",
        "point_is_wall",
        "n_points",
        "n_nodes",
        "edges_src",
        "edges_dst",
        "edges_weight",
        "d_star_switch",
        "d_star_exit",
        "d_max_switch",
        "d_max_exit",
        "spatial_index",
        "_fwd_adj",
        "_node_costs_cache",
    )

    def __init__(self):
        self.point_positions: list[tuple[int, int]] = []
        self.point_is_floor: np.ndarray = np.empty(0, dtype=bool)
        self.point_is_wall: np.ndarray = np.empty(0, dtype=bool)
        self.n_points: int = 0
        self.n_nodes: int = 0
        self.edges_src: np.ndarray = np.empty(0, dtype=np.int32)
        self.edges_dst: np.ndarray = np.empty(0, dtype=np.int32)
        self.edges_weight: np.ndarray = np.empty(0, dtype=np.float32)
        self.d_star_switch: np.ndarray = np.empty(0, dtype=np.float32)
        self.d_star_exit: np.ndarray = np.empty(0, dtype=np.float32)
        self.d_max_switch: float = 0.0
        self.d_max_exit: float = 0.0
        self.spatial_index: Dict[Tuple[int, int], List[int]] = {}
        # Lazy-built caches for visualization
        self._fwd_adj: Optional[Dict[int, List[Tuple[int, float]]]] = None
        self._node_costs_cache: Dict[int, np.ndarray] = {}  # id(d_star) -> array


# ---------------------------------------------------------------------------
# Node ID helpers (same scheme as surface_graph.py)
# ---------------------------------------------------------------------------
def node_id(point_index: int, vel_bin: int) -> int:
    """Unique integer node id = point_index * MAX_VEL_BINS + vel_bin."""
    return point_index * MAX_VEL_BINS + vel_bin


def node_id_to_parts(nid: int) -> Tuple[int, int]:
    """Recover (point_index, vel_bin) from a node id."""
    return divmod(nid, MAX_VEL_BINS)


def velocity_to_bin(velocity: float, is_floor: bool) -> int:
    """Quantize velocity to nearest bin index."""
    centers = _FLOOR_BIN_CENTERS if is_floor else _WALL_BIN_CENTERS
    return int(np.argmin(np.abs(centers - velocity)))


def velocity_to_bin_fractional(
    velocity: float, is_floor: bool,
) -> Tuple[int, int, float]:
    """Map velocity to bracketing bin indices + interpolation alpha.

    Returns (bin_lo, bin_hi, alpha) for linear interpolation.
    """
    centers = _FLOOR_BIN_CENTERS if is_floor else _WALL_BIN_CENTERS
    n = len(centers)
    if n < 2:
        return (0, 0, 0.0)
    if velocity <= centers[0]:
        return (0, 0, 0.0)
    if velocity >= centers[-1]:
        return (n - 1, n - 1, 0.0)
    lo = int(np.searchsorted(centers, velocity, side="right")) - 1
    lo = max(0, min(lo, n - 2))
    hi = lo + 1
    width = centers[hi] - centers[lo]
    alpha = (velocity - centers[lo]) / width if width > 1e-9 else 0.0
    return (lo, hi, float(alpha))


def num_velocity_bins(is_floor: bool) -> int:
    """Number of velocity bins for floor or wall surface."""
    return NUM_VELOCITY_BINS_FLOOR if is_floor else NUM_VELOCITY_BINS_WALL


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_reachability_graph(
    graph_data: dict,
    hull_collection: Optional[JumpHullCollection] = None,
    wall_hull_collection: Optional[WallHullCollection] = None,
) -> ReachabilityGraph:
    """Build the full velocity-augmented reachability graph.

    Combines walk edges from the existing graph adjacency with jump edges.
    Both floor and wall jump edges use precomputed hull masking with
    flood-fill obstruction filtering (exact frame-by-frame physics).

    Args:
        graph_data: Output of GraphBuilder.build_graph(). Must contain
            'adjacency', 'node_physics', 'spatial_hash'.
        hull_collection: Precomputed floor jump hulls.  If None, built
            on demand.
        wall_hull_collection: Precomputed wall jump hulls.  If None,
            built on demand.

    Returns:
        Populated ReachabilityGraph with edges and spatial index.
    """
    if hull_collection is None:
        hull_collection = get_or_build_hulls()

    if wall_hull_collection is None:
        wall_hull_collection = get_or_build_wall_hulls()

    node_physics: Dict[Tuple[int, int], Dict] = graph_data["node_physics"]
    adjacency = graph_data["adjacency"]

    # --- Extract surface points (floor + wall nodes) ---
    rg = ReachabilityGraph()
    pos_to_idx: Dict[Tuple[int, int], int] = {}
    positions: list[tuple[int, int]] = []
    is_floor_list: list[bool] = []
    is_wall_list: list[bool] = []

    for pos, phys in node_physics.items():
        grounded = phys["grounded"]
        walled = phys["walled"]
        if not grounded and not walled:
            continue  # Skip pure airborne nodes
        idx = len(positions)
        pos_to_idx[pos] = idx
        positions.append(pos)
        is_floor_list.append(grounded)
        is_wall_list.append(walled)

    rg.point_positions = positions
    rg.point_is_floor = np.array(is_floor_list, dtype=bool)
    rg.point_is_wall = np.array(is_wall_list, dtype=bool)
    rg.n_points = len(positions)
    rg.n_nodes = rg.n_points * MAX_VEL_BINS

    # Build spatial index for point lookup
    inv_cell = 1.0 / _SPATIAL_CELL_SIZE
    for i, (px, py) in enumerate(positions):
        gx = int(math.floor(px * inv_cell))
        gy = int(math.floor(py * inv_cell))
        key = (gx, gy)
        if key not in rg.spatial_index:
            rg.spatial_index[key] = []
        rg.spatial_index[key].append(i)

    # --- Collect edges ---
    src_list: list[int] = []
    dst_list: list[int] = []
    weight_list: list[float] = []

    # 1. Walk edges: for each surface node connected to another surface
    #    node via adjacency, add walk edges for all velocity bins.
    _add_walk_edges(
        adjacency, pos_to_idx, is_floor_list, is_wall_list,
        src_list, dst_list, weight_list,
    )

    # 2. Floor jump edges: hull-masked (exact physics).
    reachable_set = set(adjacency.keys())
    _add_jump_edges_floor_hull(
        positions, is_floor_list, pos_to_idx, node_physics,
        hull_collection, reachable_set,
        src_list, dst_list, weight_list,
    )

    # 3. Wall jump edges: hull-masked (exact physics).
    _add_jump_edges_wall(
        positions, is_wall_list, pos_to_idx, reachable_set,
        wall_hull_collection, is_floor_list,
        src_list, dst_list, weight_list,
    )

    rg.edges_src = np.array(src_list, dtype=np.int32)
    rg.edges_dst = np.array(dst_list, dtype=np.int32)
    rg.edges_weight = np.array(weight_list, dtype=np.float32)

    return rg


# ---------------------------------------------------------------------------
# Walk edges
# ---------------------------------------------------------------------------
def _add_walk_edges(
    adjacency: dict,
    pos_to_idx: Dict[Tuple[int, int], int],
    is_floor_list: list[bool],
    is_wall_list: list[bool],
    src_list: list[int],
    dst_list: list[int],
    weight_list: list[float],
) -> None:
    """Add short-range walk edges from existing graph adjacency.

    Walk edges are added for each velocity bin. Since walking conserves
    velocity approximately, the edge connects (src, vbin) -> (dst, vbin).
    """
    for pos, neighbors in adjacency.items():
        src_idx = pos_to_idx.get(pos)
        if src_idx is None:
            continue

        src_is_floor = is_floor_list[src_idx]
        n_bins = NUM_VELOCITY_BINS_FLOOR if src_is_floor else NUM_VELOCITY_BINS_WALL

        for neighbor_pos, geo_weight in neighbors:
            dst_idx = pos_to_idx.get(neighbor_pos)
            if dst_idx is None:
                continue

            # Weight: approximate frame cost for 12 px step
            w = max(WALK_EDGE_WEIGHT, geo_weight)

            for vb in range(n_bins):
                src_nid = node_id(src_idx, vb)
                dst_nid = node_id(dst_idx, vb)
                src_list.append(src_nid)
                dst_list.append(dst_nid)
                weight_list.append(w)


# ---------------------------------------------------------------------------
# Hull flood-fill filter for jump edges
# ---------------------------------------------------------------------------
_FLOOD_FILL_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # 4-connected


def _flood_fill_hull(
    hull: dict,
    sx: int,
    sy: int,
    open_set: set,
) -> set:
    """Return hull cell offsets reachable from the source through open space.

    Overlays the hull at source position ``(sx, sy)`` and flood-fills from
    the origin cell ``(0, 0)`` using 4-connectivity, only traversing hull
    cells whose absolute positions are in *open_set*.  Cells on the far
    side of solid geometry (gaps in *open_set*) are excluded.

    If ``(0, 0)`` is not in the hull (the jump impulse moves the ninja out
    of the origin cell on frame 1), the flood fill seeds from all hull
    cells with ``min_frames == 1`` instead.

    Args:
        hull: ``{(cdx, cdy): HullCell}`` -- precomputed jump hull.
        sx, sy: Source position in graph-space pixels.
        open_set: ``set(adjacency.keys())`` -- all reachable open-space
            positions on the 12px sub-node grid.

    Returns:
        Set of ``(cdx, cdy)`` hull cell offsets reachable from the source
        without crossing solid geometry.
    """
    # Determine seed cells
    origin = (0, 0)
    if origin in hull:
        seeds = [origin]
    else:
        seeds = [k for k, v in hull.items() if v.min_frames == 1]
    if not seeds:
        return set()

    # Filter seeds to those in open space
    valid_seeds = []
    for s in seeds:
        abs_pos = (sx + s[0] * GRID_CELL, sy + s[1] * GRID_CELL)
        if abs_pos in open_set:
            valid_seeds.append(s)
    if not valid_seeds:
        return set()

    visited = set(valid_seeds)
    queue = deque(valid_seeds)

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _FLOOD_FILL_NEIGHBORS:
            nk = (cx + dx, cy + dy)
            if nk in visited or nk not in hull:
                continue
            abs_pos = (sx + nk[0] * GRID_CELL, sy + nk[1] * GRID_CELL)
            if abs_pos not in open_set:
                continue
            visited.add(nk)
            queue.append(nk)

    return visited


# ---------------------------------------------------------------------------
# Floor jump edges (hull-masked)
# ---------------------------------------------------------------------------
def _add_jump_edges_floor_hull(
    positions: list[tuple[int, int]],
    is_floor_list: list[bool],
    pos_to_idx: Dict[Tuple[int, int], int],
    node_physics: Dict[Tuple[int, int], Dict],
    hull_collection: JumpHullCollection,
    reachable_set: set,
    src_list: list[int],
    dst_list: list[int],
    weight_list: list[float],
) -> None:
    """Add floor jump edges via hull flood-fill masking.

    For each floor source node, looks up its surface type (from the
    ``floor_normal`` in ``node_physics``), then for each velocity bin
    retrieves the precomputed hull and flood-fills it from the source
    position through cells in ``reachable_set``.  Only hull cells
    reachable by the flood fill (i.e. not blocked by solid geometry)
    become candidate edge destinations.

    This correctly handles arcs over short walls (flood fill routes
    through open hull cells above the wall) while rejecting edges that
    would pass through solid geometry (flood fill cannot cross gaps).
    """
    edge_count = 0

    for src_idx in range(len(positions)):
        if not is_floor_list[src_idx]:
            continue

        src_pos = positions[src_idx]
        phys = node_physics.get(src_pos)
        if phys is None:
            continue

        # Determine surface type from floor normal
        floor_normal = phys.get("floor_normal", (0.0, -1.0))
        surface_type = classify_surface(floor_normal[0], floor_normal[1])

        sx, sy = src_pos

        for vb in range(NUM_VELOCITY_BINS_FLOOR):
            hull = hull_collection.get_hull(surface_type, vb)
            if not hull:
                continue

            # Flood-fill: find hull cells reachable through open space
            reachable_hull = _flood_fill_hull(hull, sx, sy, reachable_set)

            src_nid = node_id(src_idx, vb)

            for (cdx, cdy) in reachable_hull:
                cell = hull[(cdx, cdy)]
                target = (sx + cdx * GRID_CELL, sy + cdy * GRID_CELL)

                # Destination must be a surface point (grounded or walled)
                dst_idx = pos_to_idx.get(target)
                if dst_idx is None or dst_idx == src_idx:
                    continue

                # Determine destination velocity bin from hull cell
                dst_is_floor = is_floor_list[dst_idx]
                dst_vb = velocity_to_bin(cell.landing_vx, dst_is_floor)
                dst_nid = node_id(dst_idx, dst_vb)

                src_list.append(src_nid)
                dst_list.append(dst_nid)
                weight_list.append(float(cell.min_frames))
                edge_count += 1

    logger.debug(f"Hull floor jump edges: {edge_count}")


# ---------------------------------------------------------------------------
# Wall jump edges
# ---------------------------------------------------------------------------
def _add_jump_edges_wall(
    positions: list[tuple[int, int]],
    is_wall_list: list[bool],
    pos_to_idx: Dict[Tuple[int, int], int],
    reachable_set: set,
    wall_hull_collection: WallHullCollection,
    is_floor_list: list[bool],
    src_list: list[int],
    dst_list: list[int],
    weight_list: list[float],
) -> None:
    """Add wall jump edges via hull flood-fill masking.

    For each walled source node, determines the wall normal sign, then
    for each (jump_type, vy_bin) retrieves the precomputed wall hull and
    flood-fills it through cells in ``reachable_set``.  Only hull cells
    reachable by the flood fill (i.e. not blocked by solid geometry)
    become candidate edge destinations.

    This correctly handles arcs over short walls and rejects edges that
    would pass through solid geometry, matching the floor hull approach.
    """
    edge_count = 0

    for src_idx in range(len(positions)):
        if not is_wall_list[src_idx]:
            continue

        sx, sy = positions[src_idx]

        # Determine wall normal: check which side is blocked
        left_pos = (sx - SUB_NODE_SIZE, sy)
        right_pos = (sx + SUB_NODE_SIZE, sy)

        # wall_normal_sign: +1 if left wall (normal points right),
        # -1 if right wall (normal points left)
        normals: list[int] = []
        if left_pos not in reachable_set and left_pos not in pos_to_idx:
            normals.append(1)   # Left wall -> normal points right (+1)
        if right_pos not in reachable_set and right_pos not in pos_to_idx:
            normals.append(-1)  # Right wall -> normal points left (-1)

        if not normals:
            normals.append(1)  # Fallback default

        for wall_normal_sign in normals:
            for jump_type in ["regular", "slide"]:
                for vb in range(NUM_VELOCITY_BINS_WALL):
                    hull = wall_hull_collection.get_hull(
                        wall_normal_sign, jump_type, vb,
                    )
                    if not hull:
                        continue

                    reachable_hull = _flood_fill_hull(
                        hull, sx, sy, reachable_set,
                    )

                    src_nid = node_id(src_idx, vb)

                    for (cdx, cdy) in reachable_hull:
                        cell = hull[(cdx, cdy)]
                        target = (sx + cdx * GRID_CELL, sy + cdy * GRID_CELL)

                        dst_idx = pos_to_idx.get(target)
                        if dst_idx is None or dst_idx == src_idx:
                            continue

                        # Determine destination velocity bin from hull cell:
                        # floor destinations use landing_vx (horizontal),
                        # wall destinations use landing_vy (vertical).
                        dst_is_floor = is_floor_list[dst_idx]
                        if dst_is_floor:
                            dst_vb = velocity_to_bin(
                                cell.landing_vx, True,
                            )
                        else:
                            dst_vb = velocity_to_bin(
                                cell.landing_vy, False,
                            )
                        dst_nid = node_id(dst_idx, dst_vb)

                        src_list.append(src_nid)
                        dst_list.append(dst_nid)
                        weight_list.append(float(cell.min_frames))
                        edge_count += 1

    logger.debug(f"Hull wall jump edges: {edge_count}")


# ---------------------------------------------------------------------------
# Nearest point lookup (public, for runtime queries)
# ---------------------------------------------------------------------------
def find_nearest_point(
    graph: ReachabilityGraph,
    x: float,
    y: float,
    surface_type: Optional[str] = None,
) -> Optional[int]:
    """Find the nearest surface point index to (x, y).

    Args:
        graph: The reachability graph.
        x, y: Query position (pixels).
        surface_type: "floor", "wall", or None (any).

    Returns:
        Point index, or None.
    """
    inv_cell = 1.0 / _SPATIAL_CELL_SIZE
    gx = int(math.floor(x * inv_cell))
    gy = int(math.floor(y * inv_cell))

    best_idx: Optional[int] = None
    best_dist_sq = float("inf")
    search_radius = 3

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            key = (gx + dx, gy + dy)
            indices = graph.spatial_index.get(key)
            if indices is None:
                continue
            for idx in indices:
                if surface_type == "floor" and not graph.point_is_floor[idx]:
                    continue
                if surface_type == "wall" and not graph.point_is_wall[idx]:
                    continue
                px, py = graph.point_positions[idx]
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_idx = idx

    return best_idx


def find_nearest_node(
    graph: ReachabilityGraph,
    x: float,
    y: float,
    velocity: float,
    is_floor: bool,
) -> Optional[int]:
    """Find the nearest graph node to (x, y, velocity).

    Returns node id, or None.
    """
    surface_type = "floor" if is_floor else "wall"
    pt_idx = find_nearest_point(graph, x, y, surface_type)
    if pt_idx is None:
        return None
    vb = velocity_to_bin(velocity, is_floor)
    return node_id(pt_idx, vb)


def lookup_cost_interpolated(
    graph: ReachabilityGraph,
    d_star: np.ndarray,
    x: float,
    y: float,
    velocity: float,
    is_floor: bool,
) -> float:
    """Look up cost-to-go with velocity interpolation.

    Linearly interpolates between the two bracketing velocity bins
    at the nearest surface point.
    """
    surface_type = "floor" if is_floor else "wall"
    pt_idx = find_nearest_point(graph, x, y, surface_type)
    if pt_idx is None:
        return float("inf")

    lo, hi, alpha = velocity_to_bin_fractional(velocity, is_floor)
    nid_lo = node_id(pt_idx, lo)
    nid_hi = node_id(pt_idx, hi)

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
    graph: ReachabilityGraph,
    goal_x: float,
    goal_y: float,
    goal_type: str,
) -> List[int]:
    """Find all graph nodes within contact range of a goal entity.

    Args:
        graph: The reachability graph.
        goal_x, goal_y: Goal entity position (pixels).
        goal_type: "switch" or "exit".

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

    for pt_idx, (px, py) in enumerate(graph.point_positions):
        d2 = (px - goal_x) ** 2 + (py - goal_y) ** 2
        if d2 <= contact_range_sq:
            is_floor = graph.point_is_floor[pt_idx]
            n_bins = num_velocity_bins(is_floor)
            for vb in range(n_bins):
                goal_nodes.append(node_id(pt_idx, vb))

    return goal_nodes


# ---------------------------------------------------------------------------
# Backward Dijkstra
# ---------------------------------------------------------------------------
def dijkstra_backward(
    graph: ReachabilityGraph,
    goal_nodes: List[int],
) -> np.ndarray:
    """Run backward Dijkstra from goal nodes on the reversed edge set.

    Returns:
        d_star: float32 array of length n_nodes.
                d_star[node] = minimum frame-count from node to any goal.
                Unreachable nodes receive inf.
    """
    n = graph.n_nodes
    d_star = np.full(n, np.inf, dtype=np.float32)

    # Build reversed adjacency list
    rev_adj: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(len(graph.edges_src)):
        src = int(graph.edges_src[i])
        dst = int(graph.edges_dst[i])
        w = float(graph.edges_weight[i])
        if dst not in rev_adj:
            rev_adj[dst] = []
        rev_adj[dst].append((src, w))

    # Initialize with goal nodes at cost 0
    heap: List[Tuple[float, int]] = []
    for gn in goal_nodes:
        if 0 <= gn < n:
            d_star[gn] = 0.0
            heapq.heappush(heap, (0.0, gn))

    # Standard Dijkstra
    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > d_star[u]:
            continue
        neighbors = rev_adj.get(u)
        if neighbors is None:
            continue
        for v, w in neighbors:
            new_dist = dist_u + w
            if new_dist < d_star[v]:
                d_star[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return d_star


def compute_cost_to_go(
    graph: ReachabilityGraph,
    goal_x: float,
    goal_y: float,
    goal_type: str,
) -> Tuple[np.ndarray, float]:
    """Compute cost-to-go from every graph node to a goal.

    Returns:
        (d_star, d_max) where d_max is the maximum finite cost.
    """
    goal_nodes = find_goal_nodes(graph, goal_x, goal_y, goal_type)
    if not goal_nodes:
        # No surface nodes near goal -- return uniform high cost
        d_star = np.full(graph.n_nodes, 1.0, dtype=np.float32)
        return d_star, 1.0

    d_star = dijkstra_backward(graph, goal_nodes)

    # Compute d_max for normalization
    finite_mask = np.isfinite(d_star)
    if finite_mask.any():
        d_max = float(np.max(d_star[finite_mask]))
    else:
        d_max = 1.0

    # Replace inf with d_max for unreachable nodes
    d_star[~finite_mask] = d_max

    return d_star, d_max


# ---------------------------------------------------------------------------
# Path tracing (for visualization)
# ---------------------------------------------------------------------------
def _ensure_fwd_adj(graph: ReachabilityGraph) -> Dict[int, List[Tuple[int, float]]]:
    """Build and cache forward adjacency list on the graph object.

    Returns fwd_adj: {src -> [(dst, weight), ...]}.
    """
    if graph._fwd_adj is not None:
        return graph._fwd_adj

    fwd_adj: Dict[int, List[Tuple[int, float]]] = {}
    src_arr = graph.edges_src
    dst_arr = graph.edges_dst
    w_arr = graph.edges_weight
    for i in range(len(src_arr)):
        src = int(src_arr[i])
        dst = int(dst_arr[i])
        w = float(w_arr[i])
        if src not in fwd_adj:
            fwd_adj[src] = []
        fwd_adj[src].append((dst, w))

    graph._fwd_adj = fwd_adj
    return fwd_adj


def trace_shortest_path(
    graph: ReachabilityGraph,
    d_star: np.ndarray,
    start_nid: int,
    max_hops: int = 200,
) -> List[dict]:
    """Trace the greedy shortest path from start_nid toward the goal.

    Follows forward edges greedily: at each node, pick the neighbor
    that minimizes (edge_weight + d_star[neighbor]).

    Each step is returned as a dict with:
        - src_point: (x, y) of source surface point
        - dst_point: (x, y) of destination surface point
        - src_nid: source node id
        - dst_nid: destination node id
        - weight: edge weight (frame cost)
        - edge_type: "walk" if weight <= WALK_EDGE_WEIGHT + 0.5, else "jump"
        - d_star_src: cost-to-go at source
        - d_star_dst: cost-to-go at destination

    Args:
        graph: The reachability graph.
        d_star: Cost-to-go array from backward Dijkstra.
        start_nid: Starting node id.
        max_hops: Safety limit to prevent infinite loops.

    Returns:
        List of edge dicts forming the path, empty if unreachable.
    """
    if start_nid < 0 or start_nid >= graph.n_nodes:
        return []
    if d_star[start_nid] <= 0 or not np.isfinite(d_star[start_nid]):
        return []  # Already at goal or unreachable

    fwd_adj = _ensure_fwd_adj(graph)

    path: List[dict] = []
    visited: set = set()
    current = start_nid
    walk_threshold = WALK_EDGE_WEIGHT + 0.5

    for _ in range(max_hops):
        if current in visited:
            break  # Cycle detected
        visited.add(current)

        neighbors = fwd_adj.get(current)
        if not neighbors:
            break

        # Greedy: pick neighbor minimizing w + d_star[dst]
        best_dst = -1
        best_w = float("inf")
        best_cost = float("inf")

        for dst, w in neighbors:
            total = w + d_star[dst]
            if total < best_cost:
                best_cost = total
                best_dst = dst
                best_w = w

        if best_dst < 0:
            break

        src_pt_idx = current // MAX_VEL_BINS
        dst_pt_idx = best_dst // MAX_VEL_BINS

        path.append({
            "src_point": graph.point_positions[src_pt_idx],
            "dst_point": graph.point_positions[dst_pt_idx],
            "src_nid": current,
            "dst_nid": best_dst,
            "weight": best_w,
            "edge_type": "walk" if best_w <= walk_threshold else "jump",
            "d_star_src": float(d_star[current]),
            "d_star_dst": float(d_star[best_dst]),
        })

        # Stop if we reached the goal (d_star ~ 0)
        if d_star[best_dst] <= 0:
            break

        current = best_dst

    return path


def get_node_costs_for_visualization(
    graph: ReachabilityGraph,
    d_star: np.ndarray,
    d_max: float,
) -> np.ndarray:
    """Get per-surface-point normalized cost-to-go for heatmap rendering.

    Uses vectorized numpy operations. Result is cached on the graph object
    for repeated calls with the same d_star array.

    Returns:
        float32 array of shape (n_points,) with values in [0, 1].
        0 = at goal, 1 = max cost.
    """
    # Fast cache check using d_star identity
    cache_key = id(d_star)
    cached = graph._node_costs_cache.get(cache_key)
    if cached is not None:
        return cached

    if d_max <= 0:
        d_max = 1.0

    n_pts = graph.n_points
    if n_pts == 0:
        result = np.empty(0, dtype=np.float32)
        graph._node_costs_cache[cache_key] = result
        return result

    # Reshape d_star into (n_points, MAX_VEL_BINS) and take min across bins.
    # d_star has length n_nodes = n_points * MAX_VEL_BINS.
    d_star_2d = d_star[:n_pts * MAX_VEL_BINS].reshape(n_pts, MAX_VEL_BINS)

    # For floor points use floor bins, for wall points use wall bins.
    # Taking min across ALL MAX_VEL_BINS columns is fine -- unused bins
    # have inf cost from Dijkstra, so they don't affect the min.
    best_costs = np.min(d_star_2d, axis=1)

    # Replace inf with d_max
    inf_mask = ~np.isfinite(best_costs)
    best_costs[inf_mask] = d_max

    # Normalize to [0, 1]
    result = np.clip(best_costs / d_max, 0.0, 1.0).astype(np.float32)

    # Cache (keep only last 2 entries to prevent leak)
    if len(graph._node_costs_cache) > 2:
        graph._node_costs_cache.clear()
    graph._node_costs_cache[cache_key] = result
    return result
