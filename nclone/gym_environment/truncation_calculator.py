"""Dynamic truncation limit calculator based on level complexity."""

import numpy as np

# Heuristic parameters (tunable)
BASE_TIME_PER_NODE = 5.0  # frames per sqrt(node)
TIME_PER_MINE = 50.0  # frames per reachable mine
TRUNCATION_MULTIPLIER = 10  # generous multiplier for learning
MIN_TRUNCATION_FRAMES = 300  # minimum for tiny levels
MAX_TRUNCATION_FRAMES = 4000  # cap for extremely large/complex levels


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
            (sqrt(100)*3.0 + 0*50) * 3.5 = 105 frames
        Medium level (400 nodes, 5 mines):
            (sqrt(400)*3.0 + 5*50) * 3.5 = 1085 frames
        Large level (900 nodes, 15 mines):
            (sqrt(900)*3.0 + 15*50) * 3.5 = 2940 frames
    """
    # Base time from traversal complexity (surface area)
    base_time = np.sqrt(surface_area) * BASE_TIME_PER_NODE

    # Additional time for mine navigation
    mine_time = reachable_mine_count * TIME_PER_MINE

    # Apply generous multiplier for learning phase
    truncation_limit = (base_time + mine_time) * TRUNCATION_MULTIPLIER

    # Clamp to reasonable range
    truncation_limit = int(
        np.clip(truncation_limit, MIN_TRUNCATION_FRAMES, MAX_TRUNCATION_FRAMES)
    )

    return truncation_limit
