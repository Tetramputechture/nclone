"""Edge enumeration via physics simulation for the surface graph.

For every graph node (surface_point, velocity_bin), simulates physically
realisable transitions to discover which other nodes are reachable and at
what frame cost.  Supports multiprocessing for parallelism.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

from ..constants.physics_constants import MAX_JUMP_DURATION
from .surface_extraction import SurfacePoint, SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    SurfaceGraph, node_id, velocity_bin_centers,
    velocity_to_bin, num_velocity_bins, find_nearest_point,
)

logger = logging.getLogger(__name__)

# Simulation parameter sweeps
RUN_DURATIONS = [1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 60]
JUMP_HOLD_DURATIONS = [1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45]
H_INPUTS = [-1, 0, 1]
POST_RELEASE_INPUTS = [-1, 0, 1]
MAX_ARC_FRAMES = 500
ARC_WAYPOINT_INTERVAL = 5

# Ninja state codes
_STATE_GROUNDED, _STATE_JUMPING, _STATE_FALLING = 0, 1, 2
_STATE_WALL_SLIDING, _STATE_DEAD, _STATE_AWAITING_DEATH = 5, 6, 7
_DEATH_STATES = {_STATE_DEAD, _STATE_AWAITING_DEATH}
_AIRBORNE_STATES = {_STATE_JUMPING, _STATE_FALLING}
_SURFACE_STATES = {_STATE_GROUNDED, _STATE_WALL_SLIDING}


@dataclass
class EdgeResult:
    """A discovered directed edge in the surface graph."""
    src_node: int
    dst_node: int
    weight: int


@dataclass
class ArcWaypoint:
    """A sampled intermediate state along a simulated trajectory."""
    frame_offset: int
    x: float
    y: float
    vx: float
    vy: float
    src_node: int
    dst_node: int
    total_frames: int


def _simulate_arc(sim, h_input, jump_input, max_frames, collect_waypoints=True):
    """Simulate with constant inputs until landing, death, or timeout.

    Returns (landing_state_or_None, frames, waypoints).
    """
    waypoints: List[Tuple[int, float, float, float, float]] = []
    was_airborne = False

    for frame in range(1, max_frames + 1):
        sim.tick_physics_only(h_input, jump_input)
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, frame, waypoints

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()

        if collect_waypoints and frame % ARC_WAYPOINT_INTERVAL == 0:
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


def _simulate_jump_arc(sim, h_input_during, hold_duration, h_input_after,
                       collect_waypoints=True):
    """Jump arc: hold for hold_duration, then fall with h_input_after."""
    waypoints: List[Tuple[int, float, float, float, float]] = []
    total_frame = 0

    # Phase 1: jump hold
    for _ in range(hold_duration):
        total_frame += 1
        sim.tick_physics_only(h_input_during, 1)  # jump held
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, total_frame, waypoints

        if collect_waypoints and total_frame % ARC_WAYPOINT_INTERVAL == 0:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            waypoints.append((total_frame, pos[0], pos[1], vel[0], vel[1]))

    # Phase 2: post-release fall
    for _ in range(max(0, MAX_ARC_FRAMES - total_frame)):
        total_frame += 1
        sim.tick_physics_only(h_input_after, 0)  # jump released
        state_code = sim.get_ninja_state_code()

        if state_code in _DEATH_STATES:
            return None, total_frame, waypoints

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()

        if collect_waypoints and total_frame % ARC_WAYPOINT_INTERVAL == 0:
            waypoints.append((total_frame, pos[0], pos[1], vel[0], vel[1]))

        if state_code in _SURFACE_STATES:
            return (pos[0], pos[1], vel[0], vel[1], state_code), \
                total_frame, waypoints

    return None, total_frame, waypoints


def _landing_to_node(graph, landing):
    """Map a landing (x, y, vx, vy, state_code) to the nearest graph node."""
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
            return None
        sp = graph.surface_points[pt_idx]
        stype = sp.surface_type
        vel = vy if stype == SURFACE_WALL else vx
    vel_bin = velocity_to_bin(vel, stype)
    return node_id(pt_idx, vel_bin, graph.max_vel_bins)


def _enumerate_node_edges(sim, graph, pt_idx, vel_bin, collect_waypoints=True):
    """Enumerate all edges from a single graph node."""
    sp = graph.surface_points[pt_idx]
    src = node_id(pt_idx, vel_bin, graph.max_vel_bins)
    centers = velocity_bin_centers(sp.surface_type)
    if vel_bin >= len(centers):
        return [], []

    vel = float(centers[vel_bin])

    # Compute initial velocity components
    if sp.surface_type == SURFACE_FLOOR:
        vx0 = vel
        vy0 = 0.0
    else:
        vx0 = 0.0
        vy0 = vel

    edges: List[EdgeResult] = []
    waypoints: List[ArcWaypoint] = []

    def _try_edge(landing, frames, arc_wps):
        """Process a simulation result into an edge + waypoints."""
        if landing is None:
            return
        dst = _landing_to_node(graph, landing)
        if dst is None or dst == src:
            return
        edges.append(EdgeResult(src, dst, frames))
        if collect_waypoints:
            for (f, wx, wy, wvx, wvy) in arc_wps:
                waypoints.append(ArcWaypoint(
                    frame_offset=f, x=wx, y=wy, vx=wvx, vy=wvy,
                    src_node=src, dst_node=dst, total_frames=frames,
                ))

    snapshot = sim.save_ninja_state()

    # ----- FLOOR ACTIONS -----
    if sp.surface_type == SURFACE_FLOOR:
        # 5.1 Running along surface
        for h_input in H_INPUTS:
            for duration in RUN_DURATIONS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, duration, collect_waypoints)
                _try_edge(landing, frames, wps)

        # 5.2 Floor jumps
        for h_during in H_INPUTS:
            for hold_dur in JUMP_HOLD_DURATIONS:
                for h_after in POST_RELEASE_INPUTS:
                    sim.restore_ninja_state(snapshot)
                    sim.inject_ninja_for_value_iteration(
                        sp.x, sp.y, vx0, vy0, 0)
                    # First frame: initiate jump
                    sim.tick_physics_only(h_during, 1)
                    sc = sim.get_ninja_state_code()
                    if sc in _DEATH_STATES:
                        continue
                    # Continue jump arc
                    landing, frames, wps = _simulate_jump_arc(
                        sim, h_during, min(hold_dur - 1, MAX_JUMP_DURATION - 1),
                        h_after, collect_waypoints)
                    if landing is not None:
                        _try_edge(landing, frames + 1, wps)

        # 5.4 Edge falls (from is_edge nodes)
        if sp.is_edge:
            for h_input in H_INPUTS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, MAX_ARC_FRAMES, collect_waypoints)
                _try_edge(landing, frames, wps)

    # ----- WALL ACTIONS -----
    elif sp.surface_type == SURFACE_WALL:
        # 5.3 Wall jumps
        # Determine wall normal direction from surface normal
        wall_normal_sign = 1 if sp.normal_x > 0 else -1

        for jump_type in ("regular", "slide"):
            for h_input in H_INPUTS:
                for hold_dur in JUMP_HOLD_DURATIONS:
                    sim.restore_ninja_state(snapshot)
                    sim.inject_ninja_for_value_iteration(
                        sp.x, sp.y, vx0, vy0, 0)
                    # Initiate wall jump
                    sim.tick_physics_only(h_input, 1)
                    sc = sim.get_ninja_state_code()
                    if sc in _DEATH_STATES:
                        continue
                    # Continue arc: hold jump then fall
                    h_after = h_input  # same input for simplicity
                    landing, frames, wps = _simulate_jump_arc(
                        sim, h_input,
                        min(hold_dur - 1, MAX_JUMP_DURATION - 1),
                        h_after, collect_waypoints)
                    if landing is not None:
                        _try_edge(landing, frames + 1, wps)

        # Wall slide (fall along wall)
        for h_input in H_INPUTS:
            for duration in RUN_DURATIONS:
                sim.restore_ninja_state(snapshot)
                sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
                landing, frames, wps = _simulate_arc(
                    sim, h_input, 0, duration, collect_waypoints)
                _try_edge(landing, frames, wps)

    sim.restore_ninja_state(snapshot)
    return edges, waypoints


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

    Args:
        args: (map_data, surface_points_data, graph_params, node_indices,
               collect_waypoints)

    Returns:
        (edges_list, waypoints_list)
    """
    (map_data, sp_data, graph_params, node_indices, collect_waypoints) = args

    # Reconstruct graph in worker process
    from nclone_native import Simulator as NativeSimulator
    sim = NativeSimulator.load(map_data)

    surface_points = _deserialize_surface_points(sp_data)
    graph = _reconstruct_graph(surface_points, graph_params)

    all_edges: List[EdgeResult] = []
    all_waypoints: List[ArcWaypoint] = []

    for pt_idx, vel_bin in node_indices:
        edges, waypoints = _enumerate_node_edges(
            sim, graph, pt_idx, vel_bin, collect_waypoints)
        all_edges.extend(edges)
        all_waypoints.extend(waypoints)

    return all_edges, all_waypoints


def _serialize_surface_points(points: List[SurfacePoint]) -> dict:
    """Serialize surface points for cross-process transfer."""
    n = len(points)
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
def enumerate_edges(
    graph: SurfaceGraph,
    map_data: List[int],
    num_workers: int = 4,
    collect_waypoints: bool = True,
    verbose: bool = False,
) -> Tuple[List[EdgeResult], List[ArcWaypoint]]:
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

    Returns:
        (all_edges, all_waypoints) after pruning dominated edges.
    """
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

    if num_workers <= 1:
        # Single-process path
        from nclone_native import Simulator as NativeSimulator
        sim = NativeSimulator.load(map_data)
        all_edges: List[EdgeResult] = []
        all_waypoints: List[ArcWaypoint] = []
        for i, (pt_idx, vel_bin) in enumerate(node_indices):
            e, w = _enumerate_node_edges(
                sim, graph, pt_idx, vel_bin, collect_waypoints)
            all_edges.extend(e)
            all_waypoints.extend(w)
            if verbose and (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1:,}/{total_nodes:,} nodes, "
                             f"{len(all_edges):,} edges so far")
    else:
        # Multi-process path
        chunk_size = max(1, total_nodes // num_workers)
        chunks = [
            node_indices[i:i + chunk_size]
            for i in range(0, total_nodes, chunk_size)
        ]
        worker_args = [
            (map_data, sp_data, graph_params, chunk, collect_waypoints)
            for chunk in chunks
        ]

        if verbose:
            logger.info(f"  Distributing {len(chunks)} chunks "
                         f"across {num_workers} workers")

        all_edges = []
        all_waypoints = []
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_worker_enumerate, worker_args):
                chunk_edges, chunk_wps = result
                all_edges.extend(chunk_edges)
                all_waypoints.extend(chunk_wps)
                if verbose:
                    logger.info(f"  Chunk done: +{len(chunk_edges):,} edges, "
                                 f"total {len(all_edges):,}")

    # Prune dominated edges
    n_before = len(all_edges)
    all_edges = _prune_edges(all_edges)
    n_after = len(all_edges)

    if verbose:
        logger.info(f"Edge enumeration complete: {n_after:,} edges "
                     f"(pruned {n_before - n_after:,} dominated)")
        logger.info(f"Arc waypoints collected: {len(all_waypoints):,}")

    return all_edges, all_waypoints


def populate_graph_edges(
    graph: SurfaceGraph,
    edges: List[EdgeResult],
) -> None:
    """Write edge arrays into the graph structure.

    Args:
        graph: SurfaceGraph (modified in place).
        edges: List of EdgeResult from ``enumerate_edges``.
    """
    n = len(edges)
    graph.edges_src = np.array([e.src_node for e in edges], dtype=np.int32)
    graph.edges_dst = np.array([e.dst_node for e in edges], dtype=np.int32)
    graph.edges_weight = np.array([e.weight for e in edges], dtype=np.float32)
