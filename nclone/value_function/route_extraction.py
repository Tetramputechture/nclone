"""Route extraction from the Dijkstra shortest-path tree.

Provides:
  - ArcTarget: One segment of the optimal route (launch point, landing point,
    action parameters, cost metadata).
  - extract_route: Trace the predecessor array from a start node to produce a
    concrete ordered list of ArcTargets.
  - find_nearest_start_node: Map continuous agent state to the nearest graph
    node, with airborne landing prediction fallback.
  - _predict_ballistic_landing: Simple ballistic simulator for predicting where
    an airborne agent will land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import GRAVITY_FALL, MAX_JUMP_DURATION
from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    SurfaceGraph,
    find_nearest_point,
    find_nearest_node,
    node_id_to_parts,
    velocity_bin_centers,
    velocity_to_bin,
    node_id,
)

# Max frames to simulate in the ballistic landing predictor before giving up.
_BALLISTIC_MAX_FRAMES = 300
# Pixel tolerance for "near" check (arc advancement / replanning).
_NEAR_THRESHOLD_PX = 6.0


@dataclass
class ArcTarget:
    """One segment of the optimal route through the level.

    Captures both the routing metadata (where to go, at what speed) and the
    action metadata (how to get there: jump hold, horizontal input).

    Attributes:
        launch_segment_id: Segment ID of the surface at launch.
        launch_x: Ninja-center x at the launch point (pixels).
        launch_y: Ninja-center y at the launch point (pixels).
        launch_vx: Required horizontal velocity at launch (floor: vx; wall: 0).
        surface_normal_x: Surface normal x at launch point.
        surface_normal_y: Surface normal y at launch point.
        jump_hold_frames: Frames to hold the jump button (0 = run/slide, no jump).
        h_input_during: Primary horizontal input (-1/0/1) during the arc.
            Post-release air control is not stored; the agent learns it
            reactively from spatial context (tile grid).
        landing_x: Ninja-center x at the landing point (pixels).
        landing_y: Ninja-center y at the landing point (pixels).
        landing_vx: Expected horizontal velocity at landing.
        arc_cost_frames: Expected frame count for this arc (used to clamp the
            frame counter; actual execution may differ).
        cumulative_cost: d_star of the destination node — total expected frames
            from the landing point to the goal.  Monotonically decreases along
            the route (goal has cumulative_cost = 0).
    """
    launch_segment_id: int
    launch_x: float
    launch_y: float
    launch_vx: float
    surface_normal_x: float
    surface_normal_y: float
    jump_hold_frames: int
    h_input_during: int
    landing_x: float
    landing_y: float
    landing_vx: float
    arc_cost_frames: float
    cumulative_cost: float


def extract_route(
    graph: SurfaceGraph,
    d_star: np.ndarray,
    predecessor: np.ndarray,
    best_edge_idx: np.ndarray,
    start_node: int,
) -> List[ArcTarget]:
    """Trace the Dijkstra shortest-path tree from start_node toward the goal.

    Follows ``predecessor[node]`` until a goal node (predecessor == -1) is
    reached, building an ordered list of ArcTargets.

    Args:
        graph: Surface graph with populated edge arrays and surface_points.
        d_star: Cost-to-go array (float32, length n_nodes).
        predecessor: predecessor[v] = u means v -> u is the optimal next edge
            toward the goal.  -1 for goal/unreachable nodes.
        best_edge_idx: best_edge_idx[v] = index of the v -> u edge in
            graph.edges_* arrays.
        start_node: Node id to start tracing from.

    Returns:
        Ordered list of ArcTargets from start toward the goal.
        Empty list if start_node is unreachable (d_star[start_node] == inf)
        or start_node already at the goal (predecessor == -1).
    """
    if start_node < 0 or start_node >= graph.n_nodes:
        return []
    if not np.isfinite(d_star[start_node]):
        return []

    route: List[ArcTarget] = []
    node = start_node
    visited = set()  # cycle guard (should never trigger for a valid tree)

    while predecessor[node] != -1:
        if node in visited:
            break
        visited.add(node)

        next_node = int(predecessor[node])
        edge_i = int(best_edge_idx[node])

        # Decompose source node to surface point + velocity bin
        src_pt_idx, src_vel_bin = node_id_to_parts(node, graph.max_vel_bins)
        src_sp = graph.surface_points[src_pt_idx]

        # Decompose destination node
        dst_pt_idx, dst_vel_bin = node_id_to_parts(next_node, graph.max_vel_bins)
        dst_sp = graph.surface_points[dst_pt_idx]

        # Recover velocity values from bin indices
        src_centers = velocity_bin_centers(src_sp.surface_type)
        if src_vel_bin < len(src_centers):
            launch_vel_raw = float(src_centers[src_vel_bin])
        else:
            launch_vel_raw = 0.0
        launch_vx = launch_vel_raw if src_sp.surface_type == SURFACE_FLOOR else 0.0

        dst_centers = velocity_bin_centers(dst_sp.surface_type)
        if dst_vel_bin < len(dst_centers):
            landing_vel_raw = float(dst_centers[dst_vel_bin])
        else:
            landing_vel_raw = 0.0
        landing_vx = landing_vel_raw if dst_sp.surface_type == SURFACE_FLOOR else 0.0

        # Action metadata (may be None for legacy graphs without these arrays)
        jh = int(graph.edges_jump_hold[edge_i]) if graph.edges_jump_hold is not None else 0
        hi = int(graph.edges_h_input[edge_i]) if graph.edges_h_input is not None else 0

        arc = ArcTarget(
            launch_segment_id=src_sp.segment_id,
            launch_x=src_sp.x,
            launch_y=src_sp.y,
            launch_vx=launch_vx,
            surface_normal_x=src_sp.normal_x,
            surface_normal_y=src_sp.normal_y,
            jump_hold_frames=jh,
            h_input_during=hi,
            landing_x=dst_sp.x,
            landing_y=dst_sp.y,
            landing_vx=landing_vx,
            arc_cost_frames=float(graph.edges_weight[edge_i]),
            cumulative_cost=float(d_star[next_node]),
        )
        route.append(arc)
        node = next_node

    return route


def merge_running_arcs(route: List[ArcTarget]) -> List[ArcTarget]:
    """Collapse consecutive same-segment ground-run arcs into a single arc.

    Post-processes a raw Dijkstra route to merge adjacent running arcs that
    share the same surface segment and direction into one arc. This reduces
    routes on long flat floors from dozens of micro-arcs (one per velocity bin
    transition) to a single arc per continuous run, eliminating observation
    jitter and unnecessary arc-advancement events.

    Arcs are merged when all of the following hold:
      - Both have jump_hold_frames == 0 (neither arc is a jump).
      - Both share the same launch_segment_id (same surface segment).
      - The h_input signs are compatible (same direction, or next is neutral).

    The merged arc uses the launch metadata of the first arc and the landing
    metadata of the last arc, with costs summed.

    Args:
        route: Ordered list of ArcTargets from extract_route.

    Returns:
        New list with consecutive mergeable arcs collapsed.
    """
    if not route:
        return route

    def _sign(v: int) -> int:
        return (v > 0) - (v < 0)

    merged: List[ArcTarget] = []
    current = route[0]

    for next_arc in route[1:]:
        same_segment = current.launch_segment_id == next_arc.launch_segment_id
        both_running = current.jump_hold_frames == 0 and next_arc.jump_hold_frames == 0
        same_direction = (
            _sign(current.h_input_during) == _sign(next_arc.h_input_during)
            or next_arc.h_input_during == 0
        )

        if same_segment and both_running and same_direction:
            current = ArcTarget(
                launch_segment_id=current.launch_segment_id,
                launch_x=current.launch_x,
                launch_y=current.launch_y,
                launch_vx=current.launch_vx,
                surface_normal_x=current.surface_normal_x,
                surface_normal_y=current.surface_normal_y,
                jump_hold_frames=0,
                h_input_during=current.h_input_during,
                landing_x=next_arc.landing_x,
                landing_y=next_arc.landing_y,
                landing_vx=next_arc.landing_vx,
                arc_cost_frames=current.arc_cost_frames + next_arc.arc_cost_frames,
                cumulative_cost=next_arc.cumulative_cost,
            )
        else:
            merged.append(current)
            current = next_arc

    merged.append(current)
    return merged


def extract_and_simplify_route(
    graph: SurfaceGraph,
    d_star: np.ndarray,
    predecessor: np.ndarray,
    best_edge_idx: np.ndarray,
    start_node: int,
) -> List[ArcTarget]:
    """Extract the Dijkstra route and merge consecutive same-segment running arcs.

    Convenience wrapper combining extract_route and merge_running_arcs. Use
    this as the default route-extraction entry point; reserve bare
    extract_route for contexts that need the raw unmerged arc list (e.g. tests
    that verify Dijkstra tree structure).

    Args:
        graph: Surface graph with populated edge arrays and surface_points.
        d_star: Cost-to-go array (float32, length n_nodes).
        predecessor: predecessor[v] = u means v -> u is the optimal next edge.
        best_edge_idx: Index of the v -> u edge for each node v.
        start_node: Node id to start tracing from.

    Returns:
        Ordered list of ArcTargets with consecutive running arcs merged.
    """
    return merge_running_arcs(
        extract_route(graph, d_star, predecessor, best_edge_idx, start_node)
    )


def find_nearest_start_node(
    graph: SurfaceGraph,
    x: float,
    y: float,
    vx: float,
    vy: float,
    contact_state: Optional[int],
    d_star: np.ndarray,
) -> Optional[int]:
    """Map continuous agent state to the nearest reachable graph node.

    Selects the appropriate surface type from ``contact_state``:
      - Wall-sliding (state code 5): use wall nodes with ``vy`` as velocity.
      - Otherwise grounded or airborne: use floor nodes with ``vx``.

    For airborne agents, predicts the landing point via a simple ballistic
    trajectory and finds the nearest node at the predicted landing.  Falls
    back to the nearest surface node below if prediction gives no match.

    Args:
        graph: Surface graph.
        x, y: Current ninja-center position (pixels).
        vx, vy: Current ninja velocity (pixels/frame).
        contact_state: Ninja state code (0=Immobile, 1=Running, 2=GroundSlide,
            3=Jumping, 4=Falling, 5=WallSlide, 6=Dead).  None treated as
            grounded.
        d_star: Cost-to-go array; nodes with d_star == inf are filtered out.

    Returns:
        Node id with finite d_star, or None if none found.
    """
    _WALL_SLIDE_STATE = 5
    _AIRBORNE_STATES = {3, 4}

    is_airborne = contact_state in _AIRBORNE_STATES
    is_wall = contact_state == _WALL_SLIDE_STATE

    if is_airborne:
        landing = _predict_ballistic_landing(graph, x, y, vx, vy)
        if landing is not None:
            lx, ly, lvx = landing
            node = _nearest_reachable_node(graph, lx, ly, lvx, SURFACE_FLOOR, d_star)
            if node is not None:
                return node
        # Fallback: find nearest floor node below current position
        node = _nearest_reachable_node(graph, x, y, vx, SURFACE_FLOOR, d_star)
        if node is not None:
            return node
        # Last resort: any surface type
        return _nearest_reachable_node(graph, x, y, vx, None, d_star)

    if is_wall:
        node = _nearest_reachable_node(graph, x, y, vy, SURFACE_WALL, d_star)
        if node is not None:
            return node
        return _nearest_reachable_node(graph, x, y, vx, SURFACE_FLOOR, d_star)

    # On floor or default
    node = _nearest_reachable_node(graph, x, y, vx, SURFACE_FLOOR, d_star)
    if node is not None:
        return node
    return _nearest_reachable_node(graph, x, y, vy, SURFACE_WALL, d_star)


def _nearest_reachable_node(
    graph: SurfaceGraph,
    x: float,
    y: float,
    velocity: float,
    surface_type: Optional[int],
    d_star: np.ndarray,
) -> Optional[int]:
    """Find the nearest graph node with finite d_star to (x, y, velocity).

    If the exact velocity bin gives an unreachable node, tries all bins at
    the nearest surface point.
    """
    if surface_type is not None:
        pt_idx = find_nearest_point(graph, x, y, surface_type)
    else:
        pt_idx = find_nearest_point(graph, x, y)

    if pt_idx is None:
        return None

    sp = graph.surface_points[pt_idx]
    stype = sp.surface_type
    vel_bin = velocity_to_bin(velocity, stype)
    nid = node_id(pt_idx, vel_bin, graph.max_vel_bins)

    if np.isfinite(d_star[nid]):
        return nid

    # Preferred bin is unreachable; try all bins at this point
    from .surface_graph import num_velocity_bins
    n_bins = num_velocity_bins(stype)
    best_nid = None
    best_cost = float("inf")
    for vb in range(n_bins):
        candidate = node_id(pt_idx, vb, graph.max_vel_bins)
        cost = float(d_star[candidate])
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_nid = candidate

    return best_nid


def _predict_ballistic_landing(
    graph: SurfaceGraph,
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> Optional[Tuple[float, float, float]]:
    """Predict landing position for an airborne agent using simple ballistics.

    Integrates a simplified trajectory (constant vx, vy += GRAVITY_FALL each
    frame) until the predicted position is within 24px of any surface point
    below the starting y.  This is a cheap heuristic, not a physics replay.

    Args:
        graph: Surface graph (used for surface-point lookup).
        x, y: Starting position (pixels).
        vx, vy: Starting velocity (pixels/frame).

    Returns:
        (land_x, land_y, land_vx) at the predicted landing, or None if no
        surface point found within ``_BALLISTIC_MAX_FRAMES`` frames.
    """
    cx, cy = x, y
    cvx, cvy = vx, vy

    for _ in range(_BALLISTIC_MAX_FRAMES):
        cx += cvx
        cvy += GRAVITY_FALL
        cy += cvy

        # Check whether any floor point is close (downward motion or near-floor)
        pt_idx = find_nearest_point(graph, cx, cy, SURFACE_FLOOR)
        if pt_idx is not None:
            sp = graph.surface_points[pt_idx]
            dist_sq = (sp.x - cx) ** 2 + (sp.y - cy) ** 2
            if dist_sq < (24.0 * 24.0):
                return (sp.x, sp.y, cvx)

    return None
