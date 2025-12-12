"""
Utility to build SharedLevelCache for single-level training.

Precomputes all level-based caches (path distances, mine proximity, SDF)
in the main process before spawning workers. Workers then access the shared
cache via copy-on-write (Linux fork), avoiding redundant computation.
"""

import logging
from typing import Optional, List, Tuple
import numpy as np

from .shared_level_cache import SharedLevelCache
from ...constants.entity_types import EntityType

logger = logging.getLogger(__name__)


def build_shared_level_cache(
    level_path: str,
    waypoints: Optional[List[Tuple[int, int]]] = None,
) -> SharedLevelCache:
    """Build shared level cache for single-level training.

    This should be called ONCE in the main process before spawning workers.
    The returned SharedLevelCache can be passed to workers, who will access
    it via copy-on-write shared memory (on Linux with fork).

    Args:
        level_path: Path to level file (.json or .txt)
        waypoints: Optional list of waypoint positions to include in cache

    Returns:
        SharedLevelCache with all precomputed data
    """
    from ...nplay_headless import NPlayHeadless
    from .graph_builder import GraphBuilder
    from .pathfinding_utils import (
        bfs_distance_from_start,
        find_closest_node_to_position,
        extract_spatial_lookups_from_graph_data,
    )
    from .level_data_helpers import extract_goal_positions
    from ...gym_environment.reward_calculation.reward_constants import (
        MINE_HAZARD_RADIUS,
        MINE_HAZARD_COST_MULTIPLIER,
        MINE_PENALIZE_DEADLY_ONLY,
    )
    from ..level_data import LevelData, extract_start_position_from_map_data
    from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

    logger.info(f"Building shared level cache for: {level_path}")

    # Step 1: Load level using minimal NPlayHeadless instance
    logger.info("  [1/5] Loading level...")
    nplay = NPlayHeadless(
        render_mode=None,
        enable_animation=False,
        enable_logging=False,
        enable_debug_overlay=False,
    )
    nplay.load_map(level_path)

    # Step 1b: Extract level_data from nplay_headless
    # Build tiles array from tile dictionary
    tile_dic = nplay.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            tiles[inner_y, inner_x] = int(tile_id)

    # Extract entities (simplified - no entity extractor needed for graph building)
    entities = []
    if hasattr(nplay.sim, "entity_dic"):
        for entity_type, entity_list in nplay.sim.entity_dic.items():
            for entity in entity_list:
                if hasattr(entity, "x") and hasattr(entity, "y"):
                    entity_dict = {
                        "type": entity_type,
                        "x": entity.x,
                        "y": entity.y,
                    }
                    # Add entity-specific attributes
                    if hasattr(entity, "state"):
                        entity_dict["state"] = entity.state
                    if hasattr(entity, "open"):
                        entity_dict["open"] = entity.open
                    if hasattr(entity, "activated"):
                        entity_dict["activated"] = entity.activated
                    entities.append(entity_dict)

    # Extract start position
    start_position = extract_start_position_from_map_data(nplay.sim.map_data)

    # Create LevelData object
    level_data = LevelData(
        start_position=start_position,
        tiles=tiles,
        entities=entities,
        switch_states={},
    )

    if level_data is None:
        raise ValueError(f"Failed to load level from {level_path}")

    # Step 2: Build graph
    logger.info("  [2/5] Building traversability graph...")
    graph_builder = GraphBuilder()
    graph_data = graph_builder.build_graph(level_data)

    adjacency = graph_data.get("adjacency", {})
    base_adjacency = graph_data.get("base_adjacency", adjacency)

    if not adjacency:
        raise ValueError("Graph building failed - no adjacency data")

    logger.info(f"    Graph built: {len(adjacency)} nodes")

    # Step 3: Extract goals and prepare index mappings
    logger.info("  [3/5] Extracting goals and waypoints...")
    goals = extract_goal_positions(level_data)

    # Add waypoints to goals list
    if waypoints:
        for i, waypoint_pos in enumerate(waypoints):
            waypoint_id = f"waypoint_{i}"
            goals.append((waypoint_pos, waypoint_id))
        logger.info(f"    Added {len(waypoints)} waypoints to cache")

    # Build goal mappings with aliases
    goal_positions_dict = {}
    for goal_pos, goal_id in goals:
        goal_positions_dict[goal_id] = goal_pos

        # Add generic aliases for backward compatibility
        if goal_id.startswith("exit_switch_") and "switch" not in goal_positions_dict:
            goal_positions_dict["switch"] = goal_pos
        elif goal_id.startswith("exit_door_") and "exit" not in goal_positions_dict:
            goal_positions_dict["exit"] = goal_pos

    # Prepare node and goal lists
    node_positions = list(adjacency.keys())
    goal_ids = list(set(goal_id for _, goal_id in goals))

    # Ensure aliases are in goal_ids
    if "switch" in goal_positions_dict and "switch" not in goal_ids:
        goal_ids.append("switch")
    if "exit" in goal_positions_dict and "exit" not in goal_ids:
        goal_ids.append("exit")

    logger.info(f"    Goals: {len(goal_ids)}, Nodes: {len(node_positions)}")

    # Initialize SharedLevelCache
    shared_cache = SharedLevelCache(
        num_nodes=len(node_positions),
        num_goals=len(goal_ids),
        node_positions=node_positions,
        goal_ids=goal_ids,
    )

    # Store goal positions in path cache view for API compatibility
    path_view = shared_cache.get_path_cache_view()
    path_view._goal_id_to_goal_pos = goal_positions_dict

    # Step 4: Run BFS flood fill for path distances
    logger.info("  [4/5] Computing path distances via BFS flood fill...")

    # Extract spatial lookups for performance
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Get physics cache from graph data
    physics_cache = graph_data.get("node_physics")
    if physics_cache is None:
        raise ValueError("Physics cache not found in graph_data")

    # For each goal, run BFS to compute distances to all reachable nodes
    for goal_pos, goal_id in goals:
        # Also compute for aliases
        goal_ids_to_compute = [goal_id]
        if goal_id.startswith("exit_switch_") and "switch" not in [
            g[1] for g in goals[: goals.index((goal_pos, goal_id))]
        ]:
            goal_ids_to_compute.append("switch")
        elif goal_id.startswith("exit_door_") and "exit" not in [
            g[1] for g in goals[: goals.index((goal_pos, goal_id))]
        ]:
            goal_ids_to_compute.append("exit")

        # Find closest node to goal position
        goal_node = find_closest_node_to_position(
            goal_pos,
            adjacency,
            threshold=50.0,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        if goal_node is None:
            logger.warning(f"    Could not find node near goal {goal_id} at {goal_pos}")
            continue

        # Run BFS from goal node
        physics_distances, _, parents, geometric_distances = bfs_distance_from_start(
            goal_node,
            None,
            adjacency,
            base_adjacency,
            None,
            physics_cache,
            level_data,
            None,  # mine_proximity_cache - will be built separately
            return_parents=True,
            use_geometric_costs=False,
            track_geometric_distances=True,
            mine_sdf=None,  # Will be built separately
        )

        # Store results in shared cache for all goal_id variants
        for gid in goal_ids_to_compute:
            for node_pos, distance in physics_distances.items():
                next_hop = parents.get(node_pos) if parents else None
                geo_dist = (
                    geometric_distances.get(node_pos, float("inf"))
                    if geometric_distances
                    else float("inf")
                )

                shared_cache.set_path_distance(
                    node_pos, gid, distance, geo_dist, next_hop
                )

    logger.info(f"    Path distances computed for {len(goals)} goals")

    # Step 5: Compute mine proximity costs
    logger.info("  [5/5] Computing mine proximity and SDF...")

    # A. Mine proximity costs per node
    if MINE_HAZARD_COST_MULTIPLIER > 1.0:
        mines = level_data.get_entities_by_type(
            EntityType.TOGGLE_MINE
        ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

        deadly_mine_positions = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
                continue
            deadly_mine_positions.append((mine.get("x", 0), mine.get("y", 0)))

        if deadly_mine_positions:
            for node_pos in adjacency.keys():
                # Find closest deadly mine
                min_distance = float("inf")
                for mine_x, mine_y in deadly_mine_positions:
                    dx = node_pos[0] - mine_x
                    dy = node_pos[1] - mine_y
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # Calculate cost multiplier with quadratic falloff
                if min_distance < MINE_HAZARD_RADIUS:
                    proximity_factor = 1.0 - (min_distance / MINE_HAZARD_RADIUS)
                    multiplier = 1.0 + (proximity_factor**2) * (
                        MINE_HAZARD_COST_MULTIPLIER - 1.0
                    )
                    shared_cache.set_mine_proximity_cost(node_pos, multiplier)

            logger.info(
                f"    Mine proximity costs computed for {len(deadly_mine_positions)} mines"
            )

    # B. Mine SDF grid
    from .mine_proximity_cache import SDF_WIDTH, SDF_HEIGHT, SDF_CELL_SIZE

    sdf_grid = np.full((SDF_HEIGHT, SDF_WIDTH), 1.0, dtype=np.float32)
    gradient_grid = np.zeros((SDF_HEIGHT, SDF_WIDTH, 2), dtype=np.float32)

    # Get mines for SDF (respecting MINE_PENALIZE_DEADLY_ONLY for consistency)
    mines = level_data.get_entities_by_type(
        EntityType.TOGGLE_MINE
    ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

    sdf_mine_positions = []
    for mine in mines:
        mine_state = mine.get("state", 0)
        # Use same filter logic as proximity costs for consistency
        if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
            continue  # Skip safe mines if MINE_PENALIZE_DEADLY_ONLY is True
        sdf_mine_positions.append((mine.get("x", 0), mine.get("y", 0)))

    if sdf_mine_positions:
        danger_radius = 20.0  # ninja_radius (10) + mine_radius (4.5) + margin (5.5)

        for row in range(SDF_HEIGHT):
            for col in range(SDF_WIDTH):
                # Cell center in pixel coordinates (12px resolution)
                cell_center_x = (col + 0.5) * SDF_CELL_SIZE
                cell_center_y = (row + 0.5) * SDF_CELL_SIZE

                # Find nearest mine
                min_dist = float("inf")
                nearest_dx, nearest_dy = 0.0, 0.0

                for mine_x, mine_y in sdf_mine_positions:
                    dx = cell_center_x - mine_x
                    dy = cell_center_y - mine_y
                    dist = np.sqrt(dx * dx + dy * dy)

                    if dist < min_dist:
                        min_dist = dist
                        nearest_dx = dx
                        nearest_dy = dy

                # Normalize distance to [-1, 1] range
                if min_dist <= danger_radius:
                    normalized_dist = (min_dist / danger_radius) - 1.0
                else:
                    normalized_dist = min(
                        1.0, (min_dist - danger_radius) / (2 * danger_radius)
                    )

                sdf_grid[row, col] = normalized_dist

                # Compute normalized gradient
                if min_dist > 1e-6:
                    gradient_grid[row, col, 0] = nearest_dx / min_dist
                    gradient_grid[row, col, 1] = nearest_dy / min_dist

        logger.info(f"    Mine SDF grid computed for {len(sdf_mine_positions)} mines")

    # Store SDF data in shared cache
    shared_cache.set_sdf_grid(sdf_grid, gradient_grid)

    logger.info(f"✓ Shared level cache built: {shared_cache.memory_usage_kb:.1f}KB")
    return shared_cache
