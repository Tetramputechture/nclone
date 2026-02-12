"""
Coordinate constants and grounding utilities for graph traversal.

Used by pathfinding (find_nearest_stable_position).
"""

from typing import Dict, List, Tuple

# Node coordinate offset for world coordinate conversion
# Nodes are in tile data space, entity positions in world space differ by 24px
NODE_WORLD_COORD_OFFSET = 24

# Sub-node spacing in pixels
SUB_NODE_SIZE = 12


def _is_node_grounded_util(
    node_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if a node is grounded (has solid surface below).

    Uses base_adjacency (pre-entity-mask) to determine actual level geometry.
    A node is grounded if there's a solid surface preventing downward movement.
    In screen coordinates: y=0 is top, y increases going DOWN.

    Args:
        node_pos: Node position (x, y) in pixels
        base_adjacency: Base graph adjacency (pre-entity-mask) for grounding checks

    Returns:
        True if node is grounded (on a surface), False otherwise (mid-air)
    """
    x, y = node_pos
    below_pos = (x, y + SUB_NODE_SIZE)

    if below_pos not in base_adjacency:
        return True

    if node_pos in base_adjacency:
        for neighbor_pos, _ in base_adjacency[node_pos]:
            if neighbor_pos == below_pos:
                return False

    return True
