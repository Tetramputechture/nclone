"""
Utility to build SharedLevelCache for single-level training.

Precomputes all level-based caches (path distances, mine proximity, SDF)
in the main process before spawning workers. Workers then access the shared
cache via copy-on-write (Linux fork), avoiding redundant computation.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
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

    # CRITICAL: Reset simulator to ensure entities are properly initialized
    # load_map() loads the map data, but entities may not be fully initialized
    # until reset() is called
    nplay.reset()

    logger.info("    Level loaded and reset, entities initialized")

    # Step 1b: Extract level_data from nplay_headless using proper EntityExtractor
    # Build tiles array from tile dictionary
    tile_dic = nplay.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            tiles[inner_y, inner_x] = int(tile_id)

    # Extract entities using the proper EntityExtractor class
    # This ensures entity positions are correctly extracted with xpos/ypos attributes
    from ...gym_environment.entity_extractor import EntityExtractor

    entity_extractor = EntityExtractor(nplay)
    entities = entity_extractor.extract_graph_entities()

    # DIAGNOSTIC: Verify entities were extracted
    entity_types_found = {}
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        entity_types_found[entity_type] = entity_types_found.get(entity_type, 0) + 1

    logger.info(f"    Extracted {len(entities)} entities: {entity_types_found}")

    # STRICT VALIDATION: Check for critical entities
    exit_switches = [e for e in entities if e.get("type") == EntityType.EXIT_SWITCH]
    exit_doors = [e for e in entities if e.get("type") == EntityType.EXIT_DOOR]

    if not exit_switches:
        raise ValueError(
            f"No EXIT_SWITCH entities found in level {level_path}! "
            f"Entity types found: {entity_types_found}. "
            f"Level file may be corrupted or entity extraction failed."
        )

    if not exit_doors:
        raise ValueError(
            f"No EXIT_DOOR entities found in level {level_path}! "
            f"Entity types found: {entity_types_found}. "
            f"Level file may be corrupted or entity extraction failed."
        )

    logger.info(f"    ✓ Found {len(exit_switches)} switches, {len(exit_doors)} exits")

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

    # STRICT VALIDATION: Level must have goals
    if not goals:
        raise ValueError(
            "No goals found in level! "
            "extract_goal_positions returned empty list. "
            "Level must have at least one exit switch and exit door. "
            "Check entity extraction in level_data."
        )

    logger.info(f"    Extracted {len(goals)} base goals: {[g[1] for g in goals]}")

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
            logger.info(f"    Added 'switch' alias for {goal_id} at {goal_pos}")
        elif goal_id.startswith("exit_door_") and "exit" not in goal_positions_dict:
            goal_positions_dict["exit"] = goal_pos
            logger.info(f"    Added 'exit' alias for {goal_id} at {goal_pos}")

    # Prepare node and goal lists
    node_positions = list(adjacency.keys())
    goal_ids = list(set(goal_id for _, goal_id in goals))

    # Ensure aliases are in goal_ids
    if "switch" in goal_positions_dict and "switch" not in goal_ids:
        goal_ids.append("switch")
        logger.info("    Appended 'switch' to goal_ids list")
    if "exit" in goal_positions_dict and "exit" not in goal_ids:
        goal_ids.append("exit")
        logger.info("    Appended 'exit' to goal_ids list")

    logger.info(f"    Final: {len(goal_ids)} goal_ids, {len(node_positions)} nodes")
    logger.info(f"    goal_ids: {goal_ids}")
    logger.info(f"    goal_positions_dict keys: {list(goal_positions_dict.keys())}")

    # Initialize SharedLevelCache with goal position mapping
    # No curriculum_stage for single-cache mode (entities at original positions)
    shared_cache = SharedLevelCache(
        num_nodes=len(node_positions),
        num_goals=len(goal_ids),
        node_positions=node_positions,
        goal_ids=goal_ids,
        goal_pos_mapping=goal_positions_dict,
        curriculum_stage=None,  # Single-cache mode: no curriculum
    )

    # Verify goal positions are stored correctly
    path_view = shared_cache.get_path_cache_view()
    if not path_view._goal_id_to_goal_pos:
        logger.warning(
            f"SharedPathCacheView has empty goal_pos mapping! "
            f"Expected {len(goal_positions_dict)} entries."
        )

    # Step 4: Build mine proximity cache BEFORE computing paths
    # CRITICAL: This MUST be done before BFS to ensure paths use mine avoidance costs
    logger.info("  [4/5] Building mine proximity cache...")
    from .mine_proximity_cache import MineProximityCostCache, MineSignedDistanceField

    mine_proximity_cache = MineProximityCostCache()
    mine_cache_rebuilt = mine_proximity_cache.build_cache(level_data, adjacency)

    mine_cache_size = len(mine_proximity_cache.cache)
    logger.warning(
        f"    Mine proximity cache built: {mine_cache_size} nodes with costs, "
        f"rebuilt={mine_cache_rebuilt}"
    )

    # Build mine SDF for velocity-aware hazard costs
    mine_sdf = MineSignedDistanceField()
    sdf_rebuilt = mine_sdf.build_sdf(level_data)
    logger.info(f"    Mine SDF built: rebuilt={sdf_rebuilt}")

    # Step 5: Run BFS flood fill for path distances
    logger.info("  [5/5] Computing path distances via BFS flood fill...")

    # Extract spatial lookups for performance
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Get physics cache from graph data
    physics_cache = graph_data.get("node_physics")
    if physics_cache is None:
        raise ValueError("Physics cache not found in graph_data")

    # Track which aliases we've already processed to avoid duplicates
    processed_aliases = set()

    # For each goal, run BFS to compute distances to all reachable nodes
    for goal_pos, goal_id in goals:
        # Determine all goal_ids to populate for this BFS run
        # CRITICAL: Populate both the specific goal_id AND generic aliases
        goal_ids_to_compute = [goal_id]

        # Add generic alias if this is the first switch/exit entity
        if goal_id.startswith("exit_switch_") and "switch" not in processed_aliases:
            goal_ids_to_compute.append("switch")
            processed_aliases.add("switch")
        elif goal_id.startswith("exit_door_") and "exit" not in processed_aliases:
            goal_ids_to_compute.append("exit")
            processed_aliases.add("exit")

        logger.debug(
            f"    Computing BFS for goal {goal_id} at {goal_pos}, "
            f"populating cache entries: {goal_ids_to_compute}"
        )

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

        # Run BFS with GEOMETRIC costs only (direction-independent)
        # ARCHITECTURE: Physics costs are directional and cannot be cached via flood-fill
        # They must be computed on-demand via find_shortest_path from actual start position
        geometric_distances, _, parents, _ = bfs_distance_from_start(
            goal_node,
            None,
            adjacency,
            base_adjacency,
            None,
            physics_cache,  # Still needed for horizontal rule validation
            level_data,
            None,  # NO mine_proximity_cache (not used with geometric costs)
            return_parents=True,
            use_geometric_costs=True,  # GEOMETRIC ONLY (direction-independent)
            track_geometric_distances=False,  # Not needed
            mine_sdf=None,  # Not used with geometric costs
        )

        logger.warning(
            f"    BFS for goal '{goal_id}': computed {len(geometric_distances)} GEOMETRIC distances. "
            f"Physics costs will be computed on-demand for correct directionality."
        )

        # Helper function to compute multi-hop direction (8-hop weighted lookahead)
        def compute_multi_hop_direction(
            node_pos: Tuple[int, int],
            parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
            max_hops: int = 8,
        ) -> Optional[Tuple[float, float]]:
            """Compute weighted multi-hop lookahead direction from node toward goal."""
            weights = [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005][:max_hops]
            total_dx = 0.0
            total_dy = 0.0
            current_node = node_pos

            for weight in weights:
                next_node = parents.get(current_node)
                if next_node is None:
                    break
                dx = float(next_node[0] - current_node[0])
                dy = float(next_node[1] - current_node[1])
                total_dx += weight * dx
                total_dy += weight * dy
                current_node = next_node

            magnitude = (total_dx * total_dx + total_dy * total_dy) ** 0.5
            if magnitude < 0.001:
                return None
            return (total_dx / magnitude, total_dy / magnitude)

        # Store results in shared cache for all goal_id variants
        # NOTE: We only cache GEOMETRIC distances (direction-independent)
        # Physics costs are directional and must be computed on-demand
        nodes_stored = 0
        for gid in goal_ids_to_compute:
            for node_pos, geo_dist in geometric_distances.items():
                next_hop = parents.get(node_pos) if parents else None
                # Compute multi-hop direction (8-hop weighted lookahead)
                multi_hop_direction = compute_multi_hop_direction(
                    node_pos, parents, max_hops=8
                )

                # Store geometric distance in BOTH slots for compatibility
                # (legacy code expects physics distance, but we store geometric now)
                shared_cache.set_path_distance(
                    node_pos, gid, geo_dist, geo_dist, next_hop, multi_hop_direction
                )
                nodes_stored += 1

        logger.debug(
            f"    Stored {nodes_stored} GEOMETRIC distance entries for goal_ids {goal_ids_to_compute} "
            f"(goal_node={goal_node}, reachable_nodes={len(geometric_distances)})"
        )

    logger.info(f"    Path distances computed for {len(goals)} goals")

    # Step 6: Store mine proximity costs in shared cache
    # NOTE: Mine cache was already built in Step 4 and used during BFS
    # Now we just need to copy it to shared memory for workers to access
    logger.info("  [6/6] Storing mine proximity costs in shared cache...")

    # A. Transfer mine proximity costs from cache to shared memory
    nodes_with_mine_costs = 0
    for node_pos, multiplier in mine_proximity_cache.cache.items():
        shared_cache.set_mine_proximity_cost(node_pos, multiplier)
        nodes_with_mine_costs += 1

    logger.warning(
        f"    Transferred {nodes_with_mine_costs} mine proximity costs to shared cache "
        f"(used during BFS pathfinding)"
    )

    # B. Mine SDF grid (transfer from already-built mine_sdf)
    # NOTE: mine_sdf was already built in Step 4 and used during BFS
    # Now we just transfer its data to shared memory for workers to access
    from .mine_proximity_cache import SDF_WIDTH, SDF_HEIGHT, SDF_CELL_SIZE

    # Use the SDF that was already built (don't recompute)
    if (
        mine_sdf
        and hasattr(mine_sdf, "distance_grid")
        and mine_sdf.distance_grid is not None
    ):
        sdf_grid = mine_sdf.distance_grid.copy()
        gradient_grid = mine_sdf.gradient_grid.copy()
        logger.info(
            f"    Transferred SDF grid from built mine_sdf (shape={sdf_grid.shape})"
        )
    else:
        # Fallback: Build SDF grid if mine_sdf wasn't available
        logger.warning("    mine_sdf not available, building SDF grid manually...")
        sdf_grid = np.full((SDF_HEIGHT, SDF_WIDTH), 1.0, dtype=np.float32)
        gradient_grid = np.zeros((SDF_HEIGHT, SDF_WIDTH, 2), dtype=np.float32)

        # Get mines for SDF (respecting MINE_PENALIZE_DEADLY_ONLY for consistency)
        mines = level_data.get_all_toggle_mines()

        sdf_mine_positions = []
        for mine in mines:
            mine_state = mine.get("state", 0)
            # Use same filter logic as proximity costs for consistency
            if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
                continue  # Skip safe mines if MINE_PENALIZE_DEADLY_ONLY is True
            sdf_mine_positions.append((mine.get("x", 0), mine.get("y", 0)))

    # Only compute SDF manually if mine_sdf wasn't available (fallback)
    if not (
        mine_sdf
        and hasattr(mine_sdf, "distance_grid")
        and mine_sdf.distance_grid is not None
    ):
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

            logger.info(
                f"    Mine SDF grid computed for {len(sdf_mine_positions)} mines"
            )

    # Store SDF data in shared cache
    shared_cache.set_sdf_grid(sdf_grid, gradient_grid)

    logger.info(f"✓ Shared level cache built: {shared_cache.memory_usage_kb:.1f}KB")
    return shared_cache


def build_multi_stage_shared_cache(
    level_path: str,
    curriculum_config: Dict[str, Any],
    waypoints: Optional[List[Tuple[int, int]]] = None,
) -> Dict[int, SharedLevelCache]:
    """Build SharedLevelCache for each curriculum stage.

    Creates separate caches with entities positioned at each curriculum difficulty stage.
    This enables SharedLevelCache to work with goal curriculum by providing correct
    distances for each stage's entity positions.

    Memory efficiency: 4 stages × 520KB = ~2MB total vs 130MB for 256 per-worker caches

    Args:
        level_path: Path to level file (.json or .txt)
        curriculum_config: Goal curriculum configuration dict with:
            - enabled: bool
            - stage_distance_interval: float (default 150.0)
            - advancement_threshold: float (default 0.50)
            - rolling_window: int (default 100)
        waypoints: Optional list of waypoint positions to include in caches

    Returns:
        Dict mapping stage_index (0 to num_stages-1) to SharedLevelCache
        Each cache has entities positioned for that stage's difficulty
    """
    from ...nplay_headless import NPlayHeadless
    from .graph_builder import GraphBuilder
    from .pathfinding_utils import (
        bfs_distance_from_start,
        find_closest_node_to_position,
        extract_spatial_lookups_from_graph_data,
        find_shortest_path,
    )
    from .level_data_helpers import extract_goal_positions
    from ..level_data import LevelData, extract_start_position_from_map_data
    from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
    from ...constants.physics_constants import (
        EXIT_SWITCH_RADIUS,
        EXIT_DOOR_RADIUS,
        NINJA_RADIUS,
    )
    from .mine_proximity_cache import MineProximityCostCache, MineSignedDistanceField
    import math
    from ...physics import clamp_cell

    logger.info(f"Building multi-stage shared cache for curriculum: {level_path}")
    logger.info(
        f"  Curriculum config: interval={curriculum_config.get('stage_distance_interval', 150.0)}px"
    )

    # Step 1: Load level and extract original entity positions
    logger.info("  [1/7] Loading level...")
    nplay = NPlayHeadless(
        render_mode=None,
        enable_animation=False,
        enable_logging=False,
        enable_debug_overlay=False,
    )
    nplay.load_map(level_path)
    nplay.reset()
    logger.info("    Level loaded and reset, entities initialized")

    # Extract level data
    tile_dic = nplay.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            tiles[inner_y, inner_x] = int(tile_id)

    from ...gym_environment.entity_extractor import EntityExtractor

    entity_extractor = EntityExtractor(nplay)
    entities = entity_extractor.extract_graph_entities()

    start_position = extract_start_position_from_map_data(nplay.sim.map_data)

    # Store original map_data for CurriculumMapCache
    original_map_data = list(nplay.sim.map_data)

    # Store ORIGINAL entity positions
    exit_switches = [e for e in entities if e.get("type") == EntityType.EXIT_SWITCH]
    exit_doors = [e for e in entities if e.get("type") == EntityType.EXIT_DOOR]

    if not exit_switches or not exit_doors:
        raise ValueError(
            f"Level missing critical entities: switches={len(exit_switches)}, doors={len(exit_doors)}"
        )

    original_switch_pos = (exit_switches[0].get("x"), exit_switches[0].get("y"))
    original_exit_pos = (exit_doors[0].get("x"), exit_doors[0].get("y"))

    logger.info(
        f"    Original positions: switch={original_switch_pos}, exit={original_exit_pos}"
    )

    # Step 2: Build graph with ORIGINAL positions to extract optimal paths
    logger.info("  [2/7] Building graph with original positions...")
    level_data = LevelData(
        start_position=start_position,
        tiles=tiles,
        entities=entities,
        switch_states={},
    )

    graph_builder = GraphBuilder()
    graph_data = graph_builder.build_graph(level_data)
    adjacency = graph_data.get("adjacency", {})
    base_adjacency = graph_data.get("base_adjacency", adjacency)

    if not adjacency:
        raise ValueError("Graph building failed")

    logger.info(f"    Graph built: {len(adjacency)} nodes")

    # Step 3: Compute optimal paths using ORIGINAL positions
    logger.info("  [3/7] Computing optimal paths...")
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)
    physics_cache = graph_data.get("node_physics")

    # Build mine cache for pathfinding (same for all stages)
    mine_proximity_cache = MineProximityCostCache()
    mine_proximity_cache.build_cache(level_data, adjacency)

    # Build mine SDF (same for all stages - mines don't move with curriculum)
    mine_sdf_shared = MineSignedDistanceField()
    mine_sdf_shared.build_sdf(level_data)
    logger.info(
        f"    Mine cache: {len(mine_proximity_cache.cache)} nodes, SDF built: {mine_sdf_shared.sdf_grid is not None}"
    )

    # Find nodes for spawn, switch, exit
    spawn_node = find_closest_node_to_position(
        start_position,
        adjacency,
        threshold=50.0,
        entity_radius=0.0,
        ninja_radius=NINJA_RADIUS,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    switch_node = find_closest_node_to_position(
        original_switch_pos,
        adjacency,
        threshold=50.0,
        entity_radius=EXIT_SWITCH_RADIUS,
        ninja_radius=NINJA_RADIUS,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    exit_node = find_closest_node_to_position(
        original_exit_pos,
        adjacency,
        threshold=50.0,
        entity_radius=EXIT_DOOR_RADIUS,
        ninja_radius=NINJA_RADIUS,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    if not all([spawn_node, switch_node, exit_node]):
        raise ValueError(
            f"Could not find nodes: spawn={spawn_node}, switch={switch_node}, exit={exit_node}"
        )

    # Compute paths
    spawn_to_switch_path, _ = find_shortest_path(
        spawn_node,
        switch_node,
        adjacency,
        base_adjacency,
        physics_cache,
        level_data,
        mine_proximity_cache,
    )

    switch_to_exit_path, _ = find_shortest_path(
        switch_node,
        exit_node,
        adjacency,
        base_adjacency,
        physics_cache,
        level_data,
        mine_proximity_cache,
    )

    if not spawn_to_switch_path or not switch_to_exit_path:
        raise ValueError("Could not compute optimal paths")

    # Compute path distances
    def compute_path_distance(path):
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    spawn_to_switch_dist = compute_path_distance(spawn_to_switch_path)
    switch_to_exit_dist = compute_path_distance(switch_to_exit_path)
    combined_dist = spawn_to_switch_dist + switch_to_exit_dist

    logger.info(
        f"    Paths: spawn→switch={spawn_to_switch_dist:.0f}px, switch→exit={switch_to_exit_dist:.0f}px, combined={combined_dist:.0f}px"
    )

    # Step 4: Compute number of stages
    interval = curriculum_config.get("stage_distance_interval", 100.0)
    num_stages = max(2, int(combined_dist / interval) + 1)
    logger.info(
        f"  [4/7] Computing {num_stages} curriculum stages (interval={interval}px)"
    )

    # Helper: Sample position at distance along path
    def sample_position_at_distance(path, target_distance):
        if not path or len(path) < 2:
            if path:
                return (float(path[0][0]) + 24.0, float(path[0][1]) + 24.0)
            return (0.0, 0.0)

        import bisect

        cumulative = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            cumulative.append(cumulative[-1] + math.sqrt(dx * dx + dy * dy))

        total_dist = cumulative[-1]
        target_distance = max(0.0, min(target_distance, total_dist))

        if target_distance <= 0:
            return (float(path[0][0]) + 24.0, float(path[0][1]) + 24.0)
        if target_distance >= total_dist:
            return (float(path[-1][0]) + 24.0, float(path[-1][1]) + 24.0)

        seg_idx = bisect.bisect_right(cumulative, target_distance) - 1
        seg_idx = max(0, min(seg_idx, len(path) - 2))

        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        seg_len = seg_end - seg_start

        t = (target_distance - seg_start) / seg_len if seg_len > 0 else 0.0

        x = path[seg_idx][0] + t * (path[seg_idx + 1][0] - path[seg_idx][0])
        y = path[seg_idx][1] + t * (path[seg_idx + 1][1] - path[seg_idx][1])

        return (x + 24.0, y + 24.0)

    # Step 5: Build cache for each stage
    logger.info("  [5/7] Building caches for each stage...")
    stage_caches = {}
    stage_positions = {}  # Track positions for CurriculumMapCache

    # Get references to actual entity objects in simulator
    sim_switch = (
        nplay.sim.entity_dic.get(3, [])[-1] if nplay.sim.entity_dic.get(3) else None
    )
    sim_exit = (
        sim_switch.parent if sim_switch and hasattr(sim_switch, "parent") else None
    )

    if not sim_switch or not sim_exit:
        raise ValueError("Could not access simulator entities")

    # Store original positions and cells
    original_switch_sim_pos = (sim_switch.xpos, sim_switch.ypos)
    original_exit_sim_pos = (sim_exit.xpos, sim_exit.ypos)
    original_switch_cell = sim_switch.cell
    original_exit_cell = sim_exit.cell

    for stage in range(num_stages):
        logger.info(f"    Building cache for stage {stage}/{num_stages - 1}...")

        # Compute curriculum positions for this stage
        switch_distance = min(stage * interval, spawn_to_switch_dist)
        switch_pos = sample_position_at_distance(spawn_to_switch_path, switch_distance)

        exit_combined_distance = min((stage + 1) * interval, combined_dist)
        if exit_combined_distance <= spawn_to_switch_dist:
            exit_pos = sample_position_at_distance(
                spawn_to_switch_path, exit_combined_distance
            )
        else:
            exit_offset = exit_combined_distance - spawn_to_switch_dist
            exit_pos = sample_position_at_distance(switch_to_exit_path, exit_offset)

        logger.info(
            f"      Stage {stage} positions: switch={switch_pos}, exit={exit_pos}"
        )

        # Store positions for CurriculumMapCache
        stage_positions[stage] = {
            "switch": switch_pos,
            "exit": exit_pos
        }

        # Move entities to this stage's curriculum positions
        sim_switch.xpos = switch_pos[0]
        sim_switch.ypos = switch_pos[1]
        sim_exit.xpos = exit_pos[0]
        sim_exit.ypos = exit_pos[1]

        # Update grid cells
        new_switch_cell = clamp_cell(
            math.floor(switch_pos[0] / 24), math.floor(switch_pos[1] / 24)
        )
        new_exit_cell = clamp_cell(
            math.floor(exit_pos[0] / 24), math.floor(exit_pos[1] / 24)
        )

        if new_switch_cell != sim_switch.cell:
            if (
                sim_switch.cell in nplay.sim.grid_entity
                and sim_switch in nplay.sim.grid_entity[sim_switch.cell]
            ):
                nplay.sim.grid_entity[sim_switch.cell].remove(sim_switch)
            sim_switch.cell = new_switch_cell
            if new_switch_cell not in nplay.sim.grid_entity:
                nplay.sim.grid_entity[new_switch_cell] = []
            nplay.sim.grid_entity[new_switch_cell].append(sim_switch)

        if new_exit_cell != sim_exit.cell:
            # CRITICAL: Remove exit from grid if present, but do NOT add to new cell!
            # Exit door should only be in grid_entity AFTER switch is collected.
            if (
                sim_exit.cell in nplay.sim.grid_entity
                and sim_exit in nplay.sim.grid_entity[sim_exit.cell]
            ):
                nplay.sim.grid_entity[sim_exit.cell].remove(sim_exit)
            # Also remove from new cell if somehow present
            if (
                new_exit_cell in nplay.sim.grid_entity
                and sim_exit in nplay.sim.grid_entity[new_exit_cell]
            ):
                nplay.sim.grid_entity[new_exit_cell].remove(sim_exit)
            # Update cell attribute only (for when switch adds it later)
            sim_exit.cell = new_exit_cell
            # NOTE: Do NOT append to grid_entity - door becomes interactable after switch

        # Extract entities at curriculum positions
        entities_at_stage = entity_extractor.extract_graph_entities()

        # Extract goals at curriculum positions
        level_data_stage = LevelData(
            start_position=start_position,
            tiles=tiles,
            entities=entities_at_stage,
            switch_states={},
        )
        goals_at_stage = extract_goal_positions(level_data_stage)

        if not goals_at_stage:
            raise ValueError(f"No goals found for stage {stage}")

        # Add waypoints
        if waypoints:
            for i, waypoint_pos in enumerate(waypoints):
                goals_at_stage.append((waypoint_pos, f"waypoint_{i}"))

        # Build goal mappings with aliases
        goal_positions_dict = {}
        for goal_pos, goal_id in goals_at_stage:
            goal_positions_dict[goal_id] = goal_pos

            if (
                goal_id.startswith("exit_switch_")
                and "switch" not in goal_positions_dict
            ):
                goal_positions_dict["switch"] = goal_pos
            elif goal_id.startswith("exit_door_") and "exit" not in goal_positions_dict:
                goal_positions_dict["exit"] = goal_pos

        node_positions = list(adjacency.keys())
        goal_ids = list(set(goal_id for _, goal_id in goals_at_stage))

        if "switch" in goal_positions_dict and "switch" not in goal_ids:
            goal_ids.append("switch")
        if "exit" in goal_positions_dict and "exit" not in goal_ids:
            goal_ids.append("exit")

        # Initialize SharedLevelCache for this stage
        stage_cache = SharedLevelCache(
            num_nodes=len(node_positions),
            num_goals=len(goal_ids),
            node_positions=node_positions,
            goal_ids=goal_ids,
            goal_pos_mapping=goal_positions_dict,
            curriculum_stage=stage,  # Track stage for cache identification
        )

        # Run BFS for this stage's goal positions
        processed_aliases = set()
        for goal_pos, goal_id in goals_at_stage:
            goal_ids_to_compute = [goal_id]

            if goal_id.startswith("exit_switch_") and "switch" not in processed_aliases:
                goal_ids_to_compute.append("switch")
                processed_aliases.add("switch")
            elif goal_id.startswith("exit_door_") and "exit" not in processed_aliases:
                goal_ids_to_compute.append("exit")
                processed_aliases.add("exit")

            goal_node = find_closest_node_to_position(
                goal_pos,
                adjacency,
                threshold=50.0,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if goal_node is None:
                continue

            geometric_distances, _, parents, _ = bfs_distance_from_start(
                goal_node,
                None,
                adjacency,
                base_adjacency,
                None,
                physics_cache,
                level_data_stage,
                None,
                return_parents=True,
                use_geometric_costs=True,
                track_geometric_distances=False,
                mine_sdf=None,
            )

            # Compute multi-hop direction helper
            def compute_multi_hop_direction_local(node_pos, parents_dict, max_hops=8):
                weights = [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005][:max_hops]
                total_dx = 0.0
                total_dy = 0.0
                current_node = node_pos

                for weight in weights:
                    next_node = parents_dict.get(current_node)
                    if next_node is None:
                        break
                    dx = float(next_node[0] - current_node[0])
                    dy = float(next_node[1] - current_node[1])
                    total_dx += weight * dx
                    total_dy += weight * dy
                    current_node = next_node

                magnitude = (total_dx * total_dx + total_dy * total_dy) ** 0.5
                if magnitude < 0.001:
                    return None
                return (total_dx / magnitude, total_dy / magnitude)

            # Store in cache
            for gid in goal_ids_to_compute:
                for node_pos, geo_dist in geometric_distances.items():
                    next_hop = parents.get(node_pos) if parents else None
                    multi_hop_direction = compute_multi_hop_direction_local(
                        node_pos, parents, max_hops=8
                    )
                    stage_cache.set_path_distance(
                        node_pos, gid, geo_dist, geo_dist, next_hop, multi_hop_direction
                    )

        # Transfer mine costs (same for all stages)
        for node_pos, multiplier in mine_proximity_cache.cache.items():
            stage_cache.set_mine_proximity_cost(node_pos, multiplier)

        # Transfer SDF (same for all stages - reuse shared SDF)
        if mine_sdf_shared.sdf_grid is not None:
            stage_cache.set_sdf_grid(
                mine_sdf_shared.sdf_grid, mine_sdf_shared.gradient_grid
            )

        stage_caches[stage] = stage_cache
        logger.info(f"      ✓ Stage {stage} cache: {stage_cache.memory_usage_kb:.1f}KB")

    # Restore entities to original positions
    sim_switch.xpos = original_switch_sim_pos[0]
    sim_switch.ypos = original_switch_sim_pos[1]
    sim_exit.xpos = original_exit_sim_pos[0]
    sim_exit.ypos = original_exit_sim_pos[1]
    sim_switch.cell = original_switch_cell
    sim_exit.cell = original_exit_cell

    total_memory = sum(cache.memory_usage_kb for cache in stage_caches.values())
    logger.info(
        f"✓ Multi-stage cache complete: {num_stages} stages, {total_memory:.1f}KB total"
    )
    logger.info(
        f"  Memory per worker saved: {(256 * 520 - total_memory):.0f}KB (130MB → {total_memory / 1024:.1f}MB)"
    )

    # Step 6: Build CurriculumMapCache with pre-modified map_data for each stage
    logger.info("  [6/7] Building CurriculumMapCache...")
    from ...gym_environment.reward_calculation.curriculum_map_cache import (
        CurriculumMapCache,
    )

    curriculum_map_cache = CurriculumMapCache(original_map_data, stage_positions)
    logger.info(
        f"✓ CurriculumMapCache built: {curriculum_map_cache.num_stages} stages"
    )

    return stage_caches, curriculum_map_cache
