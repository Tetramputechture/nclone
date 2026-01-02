"""Directional connectivity features for blind jump verification.

This module provides explicit geometric features that answer: "How far is the nearest
grounded platform in direction X?" This solves the blind jump problem where landing
platforms are beyond spatial context visibility (>192px).

Complements learned graph embeddings with interpretable geometric measurements.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# 8 compass directions (E, NE, N, NW, W, SW, S, SE)
COMPASS_DIRECTIONS = [
    (1, 0),  # East
    (1, -1),  # Northeast
    (0, -1),  # North
    (-1, -1),  # Northwest
    (-1, 0),  # West
    (-1, 1),  # Southwest
    (0, 1),  # South
    (1, 1),  # Southeast
]


def find_platform_in_direction(
    start_pos: Tuple[int, int],
    direction: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List],
    physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
    spatial_hash: Optional[Any] = None,
    max_distance: float = 500.0,
    step_size: int = 12,
) -> float:
    """Find distance to nearest grounded platform in specified direction.

    Raycasts from start_pos in direction until finding grounded node or hitting max_distance.
    Uses physics_cache to identify grounded nodes (has solid surface below).

    Args:
        start_pos: Starting position (x, y) in tile data space
        direction: Direction vector (dx, dy) - one of COMPASS_DIRECTIONS
        adjacency: Graph adjacency structure
        physics_cache: Precomputed physics properties {node_pos: {"grounded": bool, "walled": bool}}
        spatial_hash: Optional spatial hash for O(1) node lookup
        max_distance: Maximum raycast distance in pixels
        step_size: Step size for raycast (12px = sub-node spacing)

    Returns:
        Distance to nearest grounded platform in pixels, or max_distance if none found
    """
    dx, dy = direction

    # Normalize direction for non-axis-aligned directions
    direction_mag = (dx * dx + dy * dy) ** 0.5
    if direction_mag == 0:
        return max_distance

    dx_norm = dx / direction_mag
    dy_norm = dy / direction_mag

    # Raycast from start_pos in direction
    distance = 0.0
    while distance < max_distance:
        distance += step_size

        # Current position in raycast
        ray_x = start_pos[0] + int(dx_norm * distance)
        ray_y = start_pos[1] + int(dy_norm * distance)

        # Find nearest graph node to ray position
        nearest_node = None
        if spatial_hash is not None:
            # Fast O(1) spatial hash lookup
            try:
                candidates = spatial_hash.query(ray_x, ray_y, radius=step_size * 2)
                min_dist_sq = (step_size * 2) ** 2

                for candidate in candidates:
                    if candidate in adjacency:
                        cdist_sq = (candidate[0] - ray_x) ** 2 + (
                            candidate[1] - ray_y
                        ) ** 2
                        if cdist_sq < min_dist_sq:
                            min_dist_sq = cdist_sq
                            nearest_node = candidate
            except Exception:
                # Fallback to grid snapping if spatial hash fails
                pass

        if nearest_node is None:
            # Fallback: snap to nearest sub-node grid
            snapped_x = (ray_x // step_size) * step_size
            snapped_y = (ray_y // step_size) * step_size
            nearest_node = (
                (snapped_x, snapped_y) if (snapped_x, snapped_y) in adjacency else None
            )

        if nearest_node is not None and nearest_node in physics_cache:
            # Check if this node is grounded
            if physics_cache[nearest_node]["grounded"]:
                # Found grounded platform!
                return distance

    # No grounded platform found within max_distance
    return max_distance


def compute_directional_platform_distances(
    ninja_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List],
    physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
    spatial_hash: Optional[Any] = None,
    max_distance: float = 500.0,
) -> np.ndarray:
    """Compute distance to nearest grounded platform in all 8 compass directions.

    For each direction, finds first grounded node within max_distance.
    Returns normalized distances [0, 1] where:
    - 0.0 = platform immediately adjacent
    - 1.0 = no platform within 500px (unreachable)

    This enables blind jump verification: agent can check if landing platform
    exists before committing to jumps beyond spatial context visibility (192px).

    Args:
        ninja_pos: Ninja position (x, y) in tile data space
        adjacency: Graph adjacency structure
        physics_cache: Precomputed physics properties for grounded detection
        spatial_hash: Optional spatial hash for O(1) lookups
        max_distance: Maximum search distance in pixels

    Returns:
        8-dimensional array [E, NE, N, NW, W, SW, S, SE] with normalized distances
    """
    distances = np.zeros(8, dtype=np.float32)

    for i, direction in enumerate(COMPASS_DIRECTIONS):
        dist = find_platform_in_direction(
            ninja_pos,
            direction,
            adjacency,
            physics_cache,
            spatial_hash,
            max_distance,
        )
        # Normalize to [0, 1]
        distances[i] = min(dist / max_distance, 1.0)

    return distances
