"""Dynamic truncation limit calculator based on level complexity."""

import numpy as np
from .constants import MAX_TIME_IN_FRAMES

# Heuristic parameters (tunable)
BASE_TIME_PER_NODE = 15.0  # frames per sqrt(node) (was 10.0)
TIME_PER_MINE = (
    75.0  # frames per reachable mine (increased for toggle mine state complexity)
)
TRUNCATION_MULTIPLIER = (
    20  # generous multiplier for learning (was 15, increased for complex levels)
)
MIN_TRUNCATION_FRAMES = 600  # minimum for tiny levels


def calculate_truncation_limit(surface_area: float, reachable_mine_count: int) -> int:
    """
    Calculate dynamic truncation limit based on level complexity.

    Formula:
        base_time = sqrt(surface_area) * BASE_TIME_PER_NODE
        mine_time = reachable_mine_count * TIME_PER_MINE
        truncation = (base_time + mine_time) * TRUNCATION_MULTIPLIER

    Args:
        surface_area: Number of reachable nodes (from PBRS)
        reachable_mine_count: Number of toggle mines reachable from spawn

    Returns:
        Truncation limit in frames, clamped to [MIN, MAX]

    Examples:
        Small level (100 nodes, 0 mines):
            (sqrt(100)*15.0 + 0*75) * 20 = 3000 frames
        Medium level (400 nodes, 5 mines):
            (sqrt(400)*15.0 + 5*75) * 20 = 13500 frames
        Large level (900 nodes, 15 mines):
            (sqrt(900)*15.0 + 15*75) * 20 = 31500 frames → clamped to 18000
    """
    # Base time from traversal complexity (surface area)
    base_time = np.sqrt(surface_area) * BASE_TIME_PER_NODE

    # Additional time for mine navigation
    mine_time = reachable_mine_count * TIME_PER_MINE

    # Apply generous multiplier for learning phase
    truncation_limit = (base_time + mine_time) * TRUNCATION_MULTIPLIER

    # Clamp to reasonable range
    truncation_limit = int(
        np.clip(truncation_limit, MIN_TRUNCATION_FRAMES, MAX_TIME_IN_FRAMES)
    )

    return truncation_limit
