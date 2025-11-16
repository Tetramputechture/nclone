"""
Mine proximity and danger computation for graph edges.

This module provides vectorized functions to compute mine danger metrics for edges,
enabling the agent to understand which paths are safe vs dangerous. Critical for
mine navigation and puzzle solving.

All computations are vectorized using numpy for performance.
"""

import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import cdist

from ..constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


def point_to_line_segment_distance(
    points: np.ndarray, line_starts: np.ndarray, line_ends: np.ndarray
) -> np.ndarray:
    """
    Compute minimum distance from points to line segments (vectorized).
    
    Uses geometric formula: project point onto line, clamp to segment bounds.
    
    Args:
        points: Points to test [num_points, 2] in pixels
        line_starts: Line segment start positions [num_lines, 2] in pixels
        line_ends: Line segment end positions [num_lines, 2] in pixels
    
    Returns:
        Distance matrix [num_points, num_lines] with minimum distances
    """
    # Broadcast shapes: points [P, 1, 2], line_starts [1, L, 2], line_ends [1, L, 2]
    P = points.shape[0]
    L = line_starts.shape[0]
    
    # Reshape for broadcasting
    points_exp = points[:, np.newaxis, :]  # [P, 1, 2]
    line_starts_exp = line_starts[np.newaxis, :, :]  # [1, L, 2]
    line_ends_exp = line_ends[np.newaxis, :, :]  # [1, L, 2]
    
    # Line vectors
    line_vec = line_ends_exp - line_starts_exp  # [1, L, 2]
    line_len_sq = np.sum(line_vec ** 2, axis=2, keepdims=True)  # [1, L, 1]
    
    # Avoid division by zero for zero-length segments
    line_len_sq = np.maximum(line_len_sq, 1e-10)
    
    # Vector from line start to point
    point_vec = points_exp - line_starts_exp  # [P, L, 2]
    
    # Project point onto line (parameter t in [0, 1])
    t = np.sum(point_vec * line_vec, axis=2, keepdims=True) / line_len_sq  # [P, L, 1]
    t = np.clip(t, 0.0, 1.0)
    
    # Closest point on line segment
    closest = line_starts_exp + t * line_vec  # [P, L, 2]
    
    # Distance from point to closest point
    dist = np.linalg.norm(points_exp - closest, axis=2)  # [P, L]
    
    return dist


def compute_edge_mine_features_vectorized(
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    mine_positions: np.ndarray,
    mine_radii: np.ndarray,
    mine_states: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute mine danger features for all edges (vectorized for performance).
    
    Args:
        edge_sources: Source positions [num_edges, 2] in pixels
        edge_targets: Target positions [num_edges, 2] in pixels
        mine_positions: Mine positions [num_mines, 2] in pixels
        mine_radii: Mine collision radii [num_mines] in pixels
        mine_states: Mine states [num_mines] (-1=deadly, 0=transitioning, +1=safe)
    
    Returns:
        Dict with keys (all arrays of shape [num_edges]):
        - nearest_mine_distance: Minimum distance to any mine [0, 1]
        - passes_deadly_mine: Binary flag if edge passes through deadly mine
        - mine_threat_level: Aggregate danger score [0, 1]
        - num_mines_nearby: Count of mines near edge [0, 1]
    """
    num_edges = edge_sources.shape[0]
    num_mines = mine_positions.shape[0]
    
    # Initialize output arrays
    nearest_mine_distance = np.ones(num_edges, dtype=np.float32)
    passes_deadly_mine = np.zeros(num_edges, dtype=np.float32)
    mine_threat_level = np.zeros(num_edges, dtype=np.float32)
    num_mines_nearby = np.zeros(num_edges, dtype=np.float32)
    
    if num_mines == 0:
        # No mines in level - all edges are safe
        return {
            'nearest_mine_distance': nearest_mine_distance,
            'passes_deadly_mine': passes_deadly_mine,
            'mine_threat_level': mine_threat_level,
            'num_mines_nearby': num_mines_nearby,
        }
    
    # Compute edge midpoints for proximity checks
    edge_midpoints = (edge_sources + edge_targets) / 2.0  # [num_edges, 2]
    
    # Compute distances from edge midpoints to all mines
    midpoint_dists = cdist(edge_midpoints, mine_positions)  # [num_edges, num_mines]
    
    # Compute distances from edge line segments to all mine positions
    # This is more accurate than midpoint distance for long edges
    line_dists = point_to_line_segment_distance(
        mine_positions, edge_sources, edge_targets
    ).T  # [num_edges, num_mines]
    
    # Identify deadly mines (state == -1)
    deadly_mask = mine_states == -1.0  # [num_mines]
    
    # For each edge, compute features
    map_diagonal = np.sqrt(FULL_MAP_WIDTH_PX**2 + FULL_MAP_HEIGHT_PX**2)
    
    for e in range(num_edges):
        # Get distances from this edge to all mines
        edge_line_dists = line_dists[e]  # [num_mines]
        edge_mid_dists = midpoint_dists[e]  # [num_mines]
        
        # Nearest mine distance (normalized)
        min_dist = np.min(edge_line_dists)
        nearest_mine_distance[e] = min(min_dist / map_diagonal, 1.0)
        
        # Check if edge passes through any deadly mine
        # Edge passes through mine if line distance <= mine radius
        deadly_line_dists = edge_line_dists[deadly_mask]
        deadly_radii = mine_radii[deadly_mask]
        
        if len(deadly_line_dists) > 0:
            passes_through = deadly_line_dists <= deadly_radii
            if np.any(passes_through):
                passes_deadly_mine[e] = 1.0
        
        # Threat level: sum of (mine_radius / distance) for nearby deadly mines
        # Higher threat = mines are closer and have larger radii
        deadly_mid_dists = edge_mid_dists[deadly_mask]
        if len(deadly_mid_dists) > 0:
            # Only consider mines within 2x their radius of the edge midpoint
            threat_mask = deadly_mid_dists < (deadly_radii * 2.0)
            if np.any(threat_mask):
                nearby_dists = deadly_mid_dists[threat_mask]
                nearby_radii = deadly_radii[threat_mask]
                # Avoid division by zero
                nearby_dists = np.maximum(nearby_dists, 1.0)
                threats = nearby_radii / nearby_dists
                mine_threat_level[e] = min(np.sum(threats) / 5.0, 1.0)  # Normalize
        
        # Count mines nearby (within 2x their radius of edge midpoint)
        nearby_mask = edge_mid_dists < (mine_radii * 2.0)
        num_mines_nearby[e] = min(np.sum(nearby_mask) / 5.0, 1.0)  # Normalize
    
    return {
        'nearest_mine_distance': nearest_mine_distance,
        'passes_deadly_mine': passes_deadly_mine,
        'mine_threat_level': mine_threat_level,
        'num_mines_nearby': num_mines_nearby,
    }


def batch_compute_mine_features(
    edges: List[Any],
    mine_nodes: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Compute mine features for a batch of edges efficiently.
    
    Args:
        edges: List of Edge objects with .source and .target attributes
        mine_nodes: List of dicts with keys:
            - 'position': (x, y) tuple in pixels
            - 'radius': collision radius in pixels
            - 'state': -1.0 (deadly), 0.0 (transitioning), or +1.0 (safe)
    
    Returns:
        Dict with keys (all arrays of shape [num_edges]):
        - nearest_mine_distance: [0, 1]
        - passes_deadly_mine: binary
        - mine_threat_level: [0, 1]
        - num_mines_nearby: [0, 1]
    """
    num_edges = len(edges)
    num_mines = len(mine_nodes)
    
    if num_edges == 0 or num_mines == 0:
        # No edges or no mines - return safe defaults
        return {
            'nearest_mine_distance': np.ones(num_edges, dtype=np.float32),
            'passes_deadly_mine': np.zeros(num_edges, dtype=np.float32),
            'mine_threat_level': np.zeros(num_edges, dtype=np.float32),
            'num_mines_nearby': np.zeros(num_edges, dtype=np.float32),
        }
    
    # Extract edge positions
    edge_sources = np.array([edge.source for edge in edges], dtype=np.float32)
    edge_targets = np.array([edge.target for edge in edges], dtype=np.float32)
    
    # Extract mine data
    mine_positions = np.array([m['position'] for m in mine_nodes], dtype=np.float32)
    mine_radii = np.array([m['radius'] for m in mine_nodes], dtype=np.float32)
    mine_states = np.array([m['state'] for m in mine_nodes], dtype=np.float32)
    
    # Compute features vectorized
    return compute_edge_mine_features_vectorized(
        edge_sources,
        edge_targets,
        mine_positions,
        mine_radii,
        mine_states,
    )

