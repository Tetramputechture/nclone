"""Path analysis utilities for action masking and navigation.

Provides functions to analyze path properties such as monotonicity,
path length, and waypoint relationships for distance-based action masking.
"""

from typing import List, Tuple, Optional
import math


def is_path_monotonic_horizontal(
    path: List[Tuple[int, int]],
) -> Tuple[bool, Optional[int]]:
    """Check if path moves consistently in one horizontal direction.

    A path is monotonic if all x-coordinate changes have the same sign.
    This identifies simple paths where backtracking is never optimal.

    Args:
        path: List of (x, y) waypoints

    Returns:
        (is_monotonic, direction) where:
        - is_monotonic: True if all horizontal movement is in same direction
        - direction: 1 (right), -1 (left), or None if no horizontal movement
    """
    if not path or len(path) < 2:
        return False, None

    # Track first non-zero horizontal direction
    reference_direction = None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]

        # Skip zero movements
        if dx == 0:
            continue

        # Determine direction of this movement
        current_direction = 1 if dx > 0 else -1

        # Set reference on first non-zero movement
        if reference_direction is None:
            reference_direction = current_direction
        # Check if direction changed
        elif current_direction != reference_direction:
            return False, None  # Found direction reversal, not monotonic

    # If we never found horizontal movement, path is vertical only
    if reference_direction is None:
        return False, None

    # All horizontal movements were in same direction
    return True, reference_direction


def is_path_monotonic_vertical(
    path: List[Tuple[int, int]],
) -> Tuple[bool, Optional[int], float]:
    """Check if path moves consistently in one vertical direction.

    A path is monotonic vertically if all y-coordinate changes have the same sign
    and horizontal deviation is minimal.

    Args:
        path: List of (x, y) waypoints

    Returns:
        (is_monotonic, direction, max_horizontal_deviation) where:
        - is_monotonic: True if all vertical movement is in same direction
        - direction: 1 (down), -1 (up), or None if no vertical movement
        - max_horizontal_deviation: Maximum horizontal distance from straight vertical line (pixels)
    """
    if not path or len(path) < 2:
        return False, None, 0.0

    # Track first non-zero vertical direction
    reference_direction = None

    # Calculate the vertical reference line (x-coordinate of first waypoint)
    reference_x = path[0][0]
    max_horizontal_deviation = 0.0

    for i in range(1, len(path)):
        dy = path[i][1] - path[i - 1][1]

        # Track maximum horizontal deviation from vertical line
        horizontal_offset = abs(path[i][0] - reference_x)
        max_horizontal_deviation = max(max_horizontal_deviation, horizontal_offset)

        # Skip zero vertical movements
        if dy == 0:
            continue

        # Determine direction of this movement (1 = down, -1 = up)
        current_direction = 1 if dy > 0 else -1

        # Set reference on first non-zero movement
        if reference_direction is None:
            reference_direction = current_direction
        # Check if direction changed
        elif current_direction != reference_direction:
            return False, None, max_horizontal_deviation

    # If we never found vertical movement, path is horizontal only
    if reference_direction is None:
        return False, None, max_horizontal_deviation

    # All vertical movements were in same direction
    return True, reference_direction, max_horizontal_deviation


def calculate_horizontal_offset_from_path(
    ninja_pos: Tuple[float, float], path: List[Tuple[int, int]]
) -> float:
    """Calculate horizontal offset of ninja from vertical path line.

    Args:
        ninja_pos: Current ninja position (x, y)
        path: List of waypoints

    Returns:
        Horizontal distance from vertical reference line (pixels)
    """
    if not path:
        return 0.0

    # Use first waypoint x-coordinate as vertical reference line
    reference_x = path[0][0]
    return abs(ninja_pos[0] - reference_x)


def get_immediate_path_direction(
    ninja_pos: Tuple[float, float],
    path: List[Tuple[int, int]],
    lookahead: int = 1,
) -> Optional[Tuple[int, int]]:
    """Get normalized direction to next waypoint(s) on path.

    Finds the direction from ninja's current position to the next
    waypoint(s) on the path, useful for determining which actions
    move toward vs away from the path goal.

    Args:
        ninja_pos: Current ninja position (continuous, pixels)
        path: List of waypoints (node positions)
        lookahead: Number of waypoints to look ahead (default 1)

    Returns:
        (dx_sign, dy_sign) where each is -1, 0, or 1, or None if no path
    """
    if not path or len(path) == 0:
        return None

    # Find target waypoint (lookahead steps into path, clamped to path length)
    target_idx = min(lookahead, len(path) - 1)
    target_waypoint = path[target_idx]

    # Calculate direction vector
    dx = target_waypoint[0] - ninja_pos[0]
    dy = target_waypoint[1] - ninja_pos[1]

    # Handle case where ninja is already at target waypoint
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        # Look further ahead if possible
        if target_idx < len(path) - 1:
            target_waypoint = path[target_idx + 1]
            dx = target_waypoint[0] - ninja_pos[0]
            dy = target_waypoint[1] - ninja_pos[1]
        else:
            # Already at final waypoint
            return (0, 0)

    # Normalize to sign (-1, 0, 1)
    dx_sign = 1 if dx > 1.0 else (-1 if dx < -1.0 else 0)
    dy_sign = 1 if dy > 1.0 else (-1 if dy < -1.0 else 0)

    return (dx_sign, dy_sign)


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    """Calculate total Euclidean length of a path.

    Args:
        path: List of (x, y) waypoints

    Returns:
        Total path length in pixels
    """
    if not path or len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total_length += math.sqrt(dx * dx + dy * dy)

    return total_length


def find_closest_waypoint_index(
    position: Tuple[float, float], path: List[Tuple[int, int]]
) -> int:
    """Find index of closest waypoint to given position.

    Args:
        position: Current position (x, y)
        path: List of waypoints

    Returns:
        Index of closest waypoint in path
    """
    if not path:
        return -1

    min_dist_sq = float("inf")
    closest_idx = 0

    for i, waypoint in enumerate(path):
        dx = waypoint[0] - position[0]
        dy = waypoint[1] - position[1]
        dist_sq = dx * dx + dy * dy

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_idx = i

    return closest_idx
