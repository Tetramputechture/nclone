"""
Geometric feature computation for graph edges.

This module provides vectorized functions to compute geometric properties
of edges, including direction, distance, and movement category. These features
help the agent understand spatial relationships and navigation requirements
without requiring expensive physics simulation.

All computations are vectorized using numpy for performance.
"""

import numpy as np
from typing import Tuple

from ..constants.physics_constants import FULL_MAP_WIDTH_PX, FULL_MAP_HEIGHT_PX


def compute_edge_direction(
    src_positions: np.ndarray, tgt_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized direction vectors for edges.

    Args:
        src_positions: Source positions [num_edges, 2] in pixels
        tgt_positions: Target positions [num_edges, 2] in pixels

    Returns:
        Tuple of (dx_normalized, dy_normalized) arrays [num_edges]
        Each component is in range [-1, 1]
    """
    # Compute direction vector
    delta = tgt_positions - src_positions  # [num_edges, 2]

    # Normalize by map dimensions to get values in [-1, 1]
    dx_norm = delta[:, 0] / FULL_MAP_WIDTH_PX
    dy_norm = delta[:, 1] / FULL_MAP_HEIGHT_PX

    # Clip to ensure range
    dx_norm = np.clip(dx_norm, -1.0, 1.0)
    dy_norm = np.clip(dy_norm, -1.0, 1.0)

    return dx_norm, dy_norm


def compute_edge_distance(
    src_positions: np.ndarray, tgt_positions: np.ndarray
) -> np.ndarray:
    """
    Compute normalized Euclidean distances for edges.

    Args:
        src_positions: Source positions [num_edges, 2] in pixels
        tgt_positions: Target positions [num_edges, 2] in pixels

    Returns:
        Normalized distances [num_edges] in range [0, 1]
        Normalized by map diagonal length
    """
    # Compute Euclidean distance
    delta = tgt_positions - src_positions
    distances = np.linalg.norm(delta, axis=1)  # [num_edges]

    # Normalize by map diagonal (max possible distance)
    map_diagonal = np.sqrt(FULL_MAP_WIDTH_PX**2 + FULL_MAP_HEIGHT_PX**2)
    distances_norm = distances / map_diagonal

    # Clip to ensure range [0, 1]
    distances_norm = np.clip(distances_norm, 0.0, 1.0)

    return distances_norm


def compute_batch_geometric_features(
    src_positions: np.ndarray, tgt_positions: np.ndarray
) -> dict:
    """
    Compute all geometric features for a batch of edges in one call.

    Args:
        src_positions: Source positions [num_edges, 2] in pixels
        tgt_positions: Target positions [num_edges, 2] in pixels

    Returns:
        Dict with keys:
        - 'dx_norm': [num_edges] normalized x-direction
        - 'dy_norm': [num_edges] normalized y-direction
        - 'distance': [num_edges] normalized Euclidean distance
    """
    # Compute direction
    dx_norm, dy_norm = compute_edge_direction(src_positions, tgt_positions)

    # Compute distance
    distance = compute_edge_distance(src_positions, tgt_positions)

    return {
        "dx_norm": dx_norm,
        "dy_norm": dy_norm,
        "distance": distance,
    }
