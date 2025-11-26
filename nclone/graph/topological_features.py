"""
Topological feature computation for graph nodes.

This module provides functions to compute graph-theoretic features that help
the agent understand the structure and importance of nodes in the level graph.

MEMORY OPTIMIZATION (Phase 6): Only computes objective-relative features.
Degree and betweenness centrality functions are kept for backward compatibility
but are not used in the default pipeline.

All computations are optimized for performance with caching where appropriate.
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import deque

from ..constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


def compute_node_degrees(
    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute in-degree and out-degree for all nodes.

    Args:
        adjacency: Dict mapping node position -> list of reachable neighbor positions

    Returns:
        Tuple of (in_degrees, out_degrees) arrays [num_nodes] normalized to [0, 1]
    """
    if not adjacency:
        return np.array([]), np.array([])

    # Create position to index mapping
    positions = sorted(adjacency.keys())
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
    num_nodes = len(positions)

    # Initialize degree arrays
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_degree = np.zeros(num_nodes, dtype=np.float32)

    # Count degrees
    for src_pos, neighbors in adjacency.items():
        src_idx = pos_to_idx[src_pos]
        out_degree[src_idx] = len(neighbors)

        for tgt_pos in neighbors:
            if tgt_pos in pos_to_idx:
                tgt_idx = pos_to_idx[tgt_pos]
                in_degree[tgt_idx] += 1

    # Normalize degrees to [0, 1]
    max_degree = max(np.max(out_degree), np.max(in_degree), 1.0)
    out_degree_norm = out_degree / max_degree
    in_degree_norm = in_degree / max_degree

    return in_degree_norm, out_degree_norm


def compute_objective_relative_positions(
    node_positions: np.ndarray,
    objective_pos: Tuple[float, float],
    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute objective-relative features for all nodes via BFS from objective.

    Features:
    - dx_norm: Normalized x-distance to objective [-1, 1]
    - dy_norm: Normalized y-distance to objective [-1, 1]
    - graph_hops: Normalized shortest path hops to objective [0, 1]

    Args:
        node_positions: Array of node positions [num_nodes, 2]
        objective_pos: Target objective position (x, y)
        adjacency: Dict mapping node position -> list of reachable neighbor positions

    Returns:
        Tuple of (dx_norm, dy_norm, graph_hops_norm) arrays [num_nodes]
    """
    import logging
    logger = logging.getLogger(__name__)
    
    num_nodes = node_positions.shape[0]

    if num_nodes == 0 or not adjacency:
        return np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)
    
    # DEFENSIVE: Check for valid objective_pos
    if objective_pos is None or not isinstance(objective_pos, tuple) or len(objective_pos) != 2:
        logger.warning(f"Invalid objective_pos: {objective_pos}, using zeros for topological features")
        return np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)
    
    # DEFENSIVE: Check for NaN/Inf in objective_pos
    if not np.isfinite(objective_pos[0]) or not np.isfinite(objective_pos[1]):
        logger.warning(f"NaN/Inf in objective_pos: {objective_pos}, using zeros for topological features")
        return np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)

    # Compute geometric distances (always available, even if no graph path)
    dx = node_positions[:, 0] - objective_pos[0]
    dy = node_positions[:, 1] - objective_pos[1]
    dx_norm = np.clip(dx / FULL_MAP_WIDTH_PX, -1.0, 1.0)
    dy_norm = np.clip(dy / FULL_MAP_HEIGHT_PX, -1.0, 1.0)

    # Compute graph hops via BFS from objective
    # Create position to index mapping
    pos_to_tuple = {
        (float(pos[0]), float(pos[1])): idx for idx, pos in enumerate(node_positions)
    }

    # Initialize hop distances (inf = unreachable)
    hop_distances = np.full(num_nodes, np.inf, dtype=np.float32)

    # Find objective node in adjacency (closest node to objective position)
    objective_tuple = (float(objective_pos[0]), float(objective_pos[1]))

    # If objective not in graph, find closest node
    if objective_tuple not in adjacency:
        # Find closest node in adjacency
        min_dist = float("inf")
        closest_pos = None
        for pos in adjacency.keys():
            dist = np.sqrt(
                (pos[0] - objective_pos[0]) ** 2 + (pos[1] - objective_pos[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest_pos = pos
        if closest_pos:
            objective_tuple = closest_pos

    # BFS from objective (backward search to find distances TO objective)
    if objective_tuple in adjacency and objective_tuple in pos_to_tuple:
        obj_idx = pos_to_tuple[objective_tuple]
        hop_distances[obj_idx] = 0.0

        # Build reverse adjacency for backward BFS
        reverse_adj: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
        for src, neighbors in adjacency.items():
            for tgt in neighbors:
                if tgt not in reverse_adj:
                    reverse_adj[tgt] = []
                reverse_adj[tgt].append(src)

        # BFS from objective using reverse adjacency
        queue = deque([objective_tuple])
        visited = {objective_tuple}

        while queue:
            current_pos = queue.popleft()
            current_idx = pos_to_tuple.get(current_pos, None)
            if current_idx is None:
                continue
            current_hops = hop_distances[current_idx]

            # Visit all predecessors (nodes that can reach current)
            for neighbor_pos in reverse_adj.get(current_pos, []):
                if neighbor_pos not in visited and neighbor_pos in pos_to_tuple:
                    visited.add(neighbor_pos)
                    neighbor_idx = pos_to_tuple[neighbor_pos]
                    hop_distances[neighbor_idx] = current_hops + 1
                    queue.append(neighbor_pos)

    # Normalize hop distances to [0, 1]
    # Unreachable nodes (inf) will be set to 1.0
    max_hops = (
        np.max(hop_distances[hop_distances != np.inf])
        if np.any(hop_distances != np.inf)
        else 1.0
    )
    hop_distances_norm = np.where(
        hop_distances == np.inf,
        1.0,  # Unreachable nodes
        hop_distances / max(max_hops, 1.0),
    )
    
    # DEFENSIVE: Check for NaN/Inf in output features
    if np.isnan(dx_norm).any() or np.isinf(dx_norm).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("NaN/Inf detected in dx_norm after computation")
        logger.warning(f"  objective_pos: {objective_pos}")
        logger.warning(f"  node_positions sample: {node_positions[:5]}")
        dx_norm = np.nan_to_num(dx_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(dy_norm).any() or np.isinf(dy_norm).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("NaN/Inf detected in dy_norm after computation")
        dy_norm = np.nan_to_num(dy_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(hop_distances_norm).any() or np.isinf(hop_distances_norm).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("NaN/Inf detected in hop_distances_norm after computation")
        logger.warning(f"  max_hops: {max_hops}")
        logger.warning(f"  hop_distances sample: {hop_distances[:10]}")
        hop_distances_norm = np.nan_to_num(hop_distances_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return dx_norm, dy_norm, hop_distances_norm


def compute_betweenness_centrality(
    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]],
    sample_size: int = 100,
) -> np.ndarray:
    """
    Compute approximate betweenness centrality using sampling.

    Betweenness centrality measures how often a node lies on shortest paths
    between other nodes. Higher values indicate "bottleneck" or "junction" nodes.

    Uses sampling for performance: O(sample_size * V) instead of exact O(V^3).

    Args:
        adjacency: Dict mapping node position -> list of reachable neighbor positions
        sample_size: Number of source nodes to sample for approximation

    Returns:
        Normalized centrality scores [num_nodes] in range [0, 1]
    """
    if not adjacency:
        return np.array([])

    positions = sorted(adjacency.keys())
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
    num_nodes = len(positions)

    # Initialize centrality scores
    centrality = np.zeros(num_nodes, dtype=np.float32)

    # Sample source nodes (or use all if num_nodes < sample_size)
    sample_count = min(sample_size, num_nodes)
    if sample_count == num_nodes:
        sample_indices = list(range(num_nodes))
    else:
        # Deterministic sampling using positions as seed
        rng = np.random.RandomState(seed=hash(tuple(sorted(positions)[:10])) % (2**32))
        sample_indices = rng.choice(num_nodes, size=sample_count, replace=False)

    # For each sampled source, compute shortest paths to all targets
    for src_idx in sample_indices:
        src_pos = positions[src_idx]

        # BFS to find shortest paths and count paths through each node
        # Using Brandes' algorithm (simplified version)
        predecessors: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        num_paths: Dict[int, int] = {i: 0 for i in range(num_nodes)}
        distances: Dict[int, int] = {i: -1 for i in range(num_nodes)}

        # BFS initialization
        queue = deque([src_pos])
        distances[src_idx] = 0
        num_paths[src_idx] = 1
        stack = []  # For backtracking

        while queue:
            current_pos = queue.popleft()
            current_idx = pos_to_idx[current_pos]
            stack.append(current_idx)

            for neighbor_pos in adjacency.get(current_pos, []):
                if neighbor_pos not in pos_to_idx:
                    continue
                neighbor_idx = pos_to_idx[neighbor_pos]

                # First time visiting this node
                if distances[neighbor_idx] < 0:
                    distances[neighbor_idx] = distances[current_idx] + 1
                    queue.append(neighbor_pos)

                # Shortest path to neighbor goes through current
                if distances[neighbor_idx] == distances[current_idx] + 1:
                    num_paths[neighbor_idx] += num_paths[current_idx]
                    predecessors[neighbor_idx].append(current_idx)

        # Accumulate centrality by backtracking (Brandes' algorithm)
        dependency = np.zeros(num_nodes, dtype=np.float32)

        # Process nodes in reverse BFS order
        while stack:
            node_idx = stack.pop()
            for pred_idx in predecessors[node_idx]:
                # Fraction of paths through predecessor
                path_fraction = num_paths[pred_idx] / max(num_paths[node_idx], 1)
                dependency[pred_idx] += path_fraction * (1 + dependency[node_idx])

        # Add dependency to centrality (excluding source)
        for i in range(num_nodes):
            if i != src_idx:
                centrality[i] += dependency[i]

    # Normalize centrality to [0, 1]
    if centrality.max() > 0:
        centrality = centrality / centrality.max()

    return centrality


def compute_batch_topological_features(
    node_positions: np.ndarray,
    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]],
    objective_pos: Tuple[float, float],
    sample_size: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute topological features for nodes in one call.
    
    MEMORY OPTIMIZATION (Phase 6): Removed degree and betweenness computation.
    Only computes objective-relative features essential for navigation.

    Args:
        node_positions: Array of node positions [num_nodes, 2]
        adjacency: Dict mapping node position -> list of reachable neighbor positions
        objective_pos: Target objective position (x, y)
        sample_size: DEPRECATED - no longer used (kept for API compatibility)

    Returns:
        Dict with keys (all arrays of shape [num_nodes]):
        - 'objective_dx': Normalized x-distance to objective [-1, 1]
        - 'objective_dy': Normalized y-distance to objective [-1, 1]
        - 'objective_hops': Normalized graph hops to objective [0, 1]
        
    Removed features (not critical for shortest-path navigation):
        - 'in_degree', 'out_degree': Network topology features
        - 'betweenness': Expensive centrality calculation
    """
    # Compute only objective-relative features (essential for navigation)
    objective_dx, objective_dy, objective_hops = compute_objective_relative_positions(
        node_positions, objective_pos, adjacency
    )

    return {
        "objective_dx": objective_dx,
        "objective_dy": objective_dy,
        "objective_hops": objective_hops,
    }
