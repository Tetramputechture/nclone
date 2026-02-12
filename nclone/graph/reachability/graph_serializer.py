"""
Serialize graph data for Rust FFI.

Flattens Python adjacency dict and physics cache into lists that can be
passed to nclone_native.LevelCache.build_from_graph().

Coordinates: Node positions and goals are in TILE DATA SPACE (same as adjacency keys).
World space = tile_data + 24. Goals from extract_goal_positions are in world space;
we convert to tile space for Rust.
"""

from typing import Any, Dict, List, Tuple

NODE_WORLD_COORD_OFFSET = 24


def serialize_adjacency_for_rust(
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
    goals: List[Tuple[Tuple[int, int], str]],
    mine_proximity: Dict[Tuple[int, int], float],
) -> Dict[str, Any]:
    """
    Serialize graph data for Rust LevelCache.build_from_graph().

    Args:
        adjacency: Dict mapping node_pos -> [(neighbor_pos, edge_weight), ...]
        physics_cache: Dict mapping node_pos -> {"grounded": bool, "walled": bool}
        goals: List of (goal_pos, goal_id) tuples
        mine_proximity: Dict mapping node_pos -> cost multiplier (1.0 = no penalty)

    Returns:
        Dict with keys:
        - node_positions: List[(x, y)]
        - edges: List[(from_idx, to_idx, cost)]
        - physics_grounded: List[bool]
        - goals: List[(x, y, goal_id)]
        - mine_proximity: List[(x, y, multiplier)]
    """
    # Build node index mapping (use sorted order for determinism)
    nodes = sorted(adjacency.keys())
    node_to_idx = {pos: i for i, pos in enumerate(nodes)}

    # Flatten edges: (from_idx, to_idx, geometric_cost)
    # Use geometric cost (Euclidean distance) for level cache - matches Python use_geometric_costs=True
    edges: List[Tuple[int, int, float]] = []
    for from_pos, neighbors in adjacency.items():
        from_idx = node_to_idx[from_pos]
        fx, fy = from_pos
        for to_pos, _ in neighbors:
            if to_pos in node_to_idx:
                to_idx = node_to_idx[to_pos]
                tx, ty = to_pos
                dx, dy = tx - fx, ty - fy
                cost = (dx * dx + dy * dy) ** 0.5
                edges.append((from_idx, to_idx, cost))

    # Physics grounded (indexed by node)
    physics_grounded = []
    for pos in nodes:
        props = physics_cache.get(pos, {})
        grounded = props.get("grounded", True)
        physics_grounded.append(grounded)

    # Goals as (x, y, goal_id) - convert from world to tile data space
    goals_flat = []
    for (gx, gy), goal_id in goals:
        tile_x = gx - NODE_WORLD_COORD_OFFSET
        tile_y = gy - NODE_WORLD_COORD_OFFSET
        goals_flat.append((tile_x, tile_y, goal_id))

    # Mine proximity as (x, y, multiplier)
    mine_proximity_flat = [
        (x, y, mult) for (x, y), mult in mine_proximity.items()
    ]

    return {
        "node_positions": nodes,
        "edges": edges,
        "physics_grounded": physics_grounded,
        "goals": goals_flat,
        "mine_proximity": mine_proximity_flat,
    }
