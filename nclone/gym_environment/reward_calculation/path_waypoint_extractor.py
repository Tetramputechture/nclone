"""Path-based waypoint extraction from optimal A* paths.

SIMPLIFIED SYSTEM (2026-01-03): Uniform 12px spacing along optimal path.

Extracts waypoints at uniform 12px intervals (sub-node spacing) along the
curriculum-truncated optimal path. This provides coarse rewards that bridge
the small continuous dense rewards of PBRS to the sparse terminal rewards
of exit switch and exit door.

Key features:
- Uniform spacing: One waypoint every 12px along path (matches graph sub-nodes)
- Uniform values: All waypoints have the same reward value (no gradient)
- Exit directions: Computed for velocity alignment in PBRS
- Curriculum-aware: Uses truncated paths based on current curriculum stage
- Phase distinction: Separate pre-switch and post-switch waypoints

Expected density: ~(path_length/12) waypoints per path.
Example: 500px path = ~40 waypoints at uniform intervals.
"""

import logging
import math
from typing import List, Tuple, Dict, Any, NamedTuple, Optional
from collections import OrderedDict

from ...graph.reachability.pathfinding_utils import NODE_WORLD_COORD_OFFSET

logger = logging.getLogger(__name__)


class PathWaypoint(NamedTuple):
    """Waypoint extracted from optimal path.

    Attributes:
        position: World coordinates (x, y) in pixels
        waypoint_type: Type of waypoint for prioritization
        value: Importance score (0.4-1.8) with progress gradient
        phase: "pre_switch" or "post_switch" for phase-aware filtering
        node_index: Index in original path for ordering
        physics_state: "grounded", "aerial", or "walled"
        curvature: Turn angle in degrees (0-180)
        exit_direction: Direction vector (dx, dy) to next waypoint on path (normalized)
    """

    position: Tuple[float, float]
    waypoint_type: str
    value: float
    phase: str
    node_index: int
    physics_state: str
    curvature: float
    exit_direction: Optional[Tuple[float, float]] = None


class PathWaypointExtractor:
    """Extracts dense waypoints from optimal A* paths.

    Provides immediate, deterministic guidance for complex navigation
    by identifying strategic points along the optimal path.
    """

    def __init__(
        self,
        progress_spacing: float = 12.0,
        max_cache_size: int = 100,
    ):
        """Initialize path waypoint extractor.

        Args:
            progress_spacing: Distance between waypoints along path (pixels)
            max_cache_size: Maximum number of levels to cache
        """
        self.progress_spacing = progress_spacing

        # LRU cache: level_id -> List[PathWaypoint]
        self._waypoint_cache: OrderedDict[str, List[PathWaypoint]] = OrderedDict()
        self._max_cache_size = max_cache_size

        # Statistics
        self.total_waypoints_extracted = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"PathWaypointExtractor initialized: "
            f"progress_spacing={progress_spacing}px (simplified 12px uniform system)"
        )

    def extract_waypoints_for_curriculum_stage(
        self,
        spawn_to_switch_path: List[Tuple[int, int]],
        switch_to_exit_path: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        level_id: str,
        curriculum_switch_distance: float,
        curriculum_exit_distance: float,
        use_cache: bool = False,
    ) -> List[PathWaypoint]:
        """Extract waypoints from curriculum-truncated paths.

        IMPORTANT: The paths passed to this function are ALREADY truncated by
        IntermediateGoalManager.get_curriculum_truncated_paths(). Do NOT truncate again.

        Args:
            spawn_to_switch_path: Already curriculum-truncated path from spawn to curriculum switch
            switch_to_exit_path: Already curriculum-truncated path from curriculum switch to curriculum exit
            physics_cache: Pre-computed physics properties per node
            level_id: Level identifier for caching
            curriculum_switch_distance: Distance to curriculum-adjusted switch (pixels, for logging)
            curriculum_exit_distance: Distance to curriculum-adjusted exit (pixels, for logging)
            use_cache: If True, use cached waypoints (NOT USED for curriculum - always recompute)

        Returns:
            List of PathWaypoint objects within curriculum-reachable segment
        """
        # Extract spawn position for filtering (use first node from pre-switch path)
        spawn_node = spawn_to_switch_path[0] if spawn_to_switch_path else (0, 0)
        spawn_world = (
            spawn_node[0] + NODE_WORLD_COORD_OFFSET,
            spawn_node[1] + NODE_WORLD_COORD_OFFSET,
        )

        # Extract waypoints from ALREADY-TRUNCATED curriculum paths
        # NO additional truncation needed - paths are already curriculum-aware
        pre_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=spawn_to_switch_path,
            physics_cache=physics_cache,
            phase="pre_switch",
        )

        post_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=switch_to_exit_path,
            physics_cache=physics_cache,
            phase="post_switch",
        )

        # Filter waypoints too close to spawn (50px minimum)
        MIN_SPAWN_DISTANCE = 50.0
        pre_switch_filtered = []
        filtered_count = 0

        for wp in pre_switch_waypoints:
            dx = wp.position[0] - spawn_world[0]
            dy = wp.position[1] - spawn_world[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist >= MIN_SPAWN_DISTANCE:
                pre_switch_filtered.append(wp)
            else:
                filtered_count += 1

        if filtered_count > 0:
            logger.debug(
                f"Filtered {filtered_count} waypoints within {MIN_SPAWN_DISTANCE}px of spawn"
            )

        # Combine both phases
        all_waypoints = pre_switch_filtered + post_switch_waypoints

        # Update statistics
        self.total_waypoints_extracted += len(all_waypoints)

        logger.info(
            f"Extracted {len(all_waypoints)} curriculum waypoints for {level_id}: "
            f"{len(pre_switch_filtered)} pre-switch, {len(post_switch_waypoints)} post-switch "
            f"(curriculum distances: switch={curriculum_switch_distance:.0f}px, "
            f"exit={curriculum_exit_distance:.0f}px)"
        )

        return all_waypoints

    def _truncate_path_at_distance(
        self, path: List[Tuple[int, int]], max_distance: float
    ) -> List[Tuple[int, int]]:
        """Truncate path to only include nodes within max_distance from start.

        Args:
            path: Full path as list of node positions
            max_distance: Maximum distance along path to include (pixels)

        Returns:
            Truncated path containing only nodes within max_distance
        """
        if not path or len(path) < 2 or max_distance <= 0:
            return path[:1] if path else []  # Return at least start node

        import math

        truncated = [path[0]]  # Always include start
        cumulative_distance = 0.0

        for i in range(1, len(path)):
            # Calculate segment distance
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            segment_dist = math.sqrt(dx * dx + dy * dy)

            # Check if adding this node would exceed max_distance
            if cumulative_distance + segment_dist > max_distance:
                # Interpolate final node at exact max_distance
                if segment_dist > 0:
                    remaining = max_distance - cumulative_distance
                    t = remaining / segment_dist
                    interp_x = int(path[i - 1][0] + t * dx)
                    interp_y = int(path[i - 1][1] + t * dy)
                    truncated.append((interp_x, interp_y))
                break

            truncated.append(path[i])
            cumulative_distance += segment_dist

        return truncated

    def extract_waypoints_from_paths(
        self,
        spawn_to_switch_path: List[Tuple[int, int]],
        switch_to_exit_path: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        level_id: str,
        use_cache: bool = True,
    ) -> List[PathWaypoint]:
        """Extract waypoints from both path segments (spawn→switch, switch→exit).

        Args:
            spawn_to_switch_path: Path nodes from spawn to switch
            switch_to_exit_path: Path nodes from switch to exit
            physics_cache: Pre-computed physics properties per node
            level_id: Level identifier for caching
            use_cache: If True, use cached waypoints if available

        Returns:
            List of PathWaypoint objects for both phases
        """
        # Check cache first
        if use_cache and level_id in self._waypoint_cache:
            self._waypoint_cache.move_to_end(level_id)  # LRU update
            self.cache_hits += 1
            cached_waypoints = self._waypoint_cache[level_id]
            logger.debug(
                f"Using cached waypoints for level {level_id}: "
                f"{len(cached_waypoints)} waypoints"
            )
            return list(cached_waypoints)  # Return copy

        self.cache_misses += 1

        # Extract spawn position (first node in spawn_to_switch_path)
        # Convert to world coordinates for distance filtering
        spawn_node = spawn_to_switch_path[0] if spawn_to_switch_path else (0, 0)
        spawn_world = (
            spawn_node[0] + NODE_WORLD_COORD_OFFSET,
            spawn_node[1] + NODE_WORLD_COORD_OFFSET,
        )

        # Extract waypoints from each path segment
        pre_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=spawn_to_switch_path,
            physics_cache=physics_cache,
            phase="pre_switch",
        )

        post_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=switch_to_exit_path,
            physics_cache=physics_cache,
            phase="post_switch",
        )

        # Filter waypoints too close to spawn (reduced to 20px for early curriculum stages)
        # UPDATED 2026-01-03: Reduced from 50px to 20px to allow waypoints in short curriculum paths
        MIN_SPAWN_DISTANCE = 20.0
        pre_switch_filtered = []
        filtered_count = 0

        # logger.info(
        #     f"[WAYPOINT] Spawn filtering: spawn_world={spawn_world}, "
        #     f"pre_switch_waypoints={len(pre_switch_waypoints)}, "
        #     f"post_switch_waypoints={len(post_switch_waypoints)}"
        # )

        for wp in pre_switch_waypoints:
            dx = wp.position[0] - spawn_world[0]
            dy = wp.position[1] - spawn_world[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist >= MIN_SPAWN_DISTANCE:
                pre_switch_filtered.append(wp)
            else:
                filtered_count += 1
                logger.debug(
                    f"Filtered waypoint too close to spawn: {wp.position} "
                    f"(distance: {dist:.1f}px < {MIN_SPAWN_DISTANCE}px)"
                )

        # if filtered_count > 0:
        #     logger.warning(
        #         f"[WAYPOINT] Filtered {filtered_count} waypoints within {MIN_SPAWN_DISTANCE}px of spawn. "
        #         f"Kept {len(pre_switch_filtered)}/{len(pre_switch_waypoints)} pre-switch waypoints."
        #     )

        # logger.info(
        #     f"[WAYPOINT] Final waypoint counts: "
        #     f"pre_switch={len(pre_switch_filtered)}, post_switch={len(post_switch_waypoints)}"
        # )

        # Combine both phases (using filtered pre-switch waypoints)
        all_waypoints = pre_switch_filtered + post_switch_waypoints

        # Cache the result
        self._waypoint_cache[level_id] = all_waypoints
        self._waypoint_cache.move_to_end(level_id)

        # Evict oldest entry if cache exceeds max size
        if len(self._waypoint_cache) > self._max_cache_size:
            self._waypoint_cache.popitem(last=False)

        self.total_waypoints_extracted += len(all_waypoints)

        logger.info(
            f"Extracted {len(all_waypoints)} path waypoints for level {level_id}: "
            f"{len(pre_switch_waypoints)} pre-switch, {len(post_switch_waypoints)} post-switch"
        )

        return all_waypoints

    def _extract_waypoints_from_path(
        self,
        path_nodes: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        phase: str = "pre_switch",
    ) -> List[PathWaypoint]:
        """Extract waypoints at uniform 12px intervals along path.

        Simplified extraction: places a waypoint every 12px (sub-node spacing) along
        the path with uniform values. This provides coarse rewards bridging PBRS to
        terminal rewards without complex detection logic.

        Args:
            path_nodes: List of (x, y) node positions in path order
            physics_cache: Pre-computed physics properties per node (unused but kept for API compatibility)
            phase: "pre_switch" or "post_switch"

        Returns:
            List of PathWaypoint objects with exit directions
        """
        if len(path_nodes) < 2:
            # logger.warning(
            #     f"[WAYPOINT] {phase}: Path too short ({len(path_nodes)} nodes), skipping waypoint extraction"
            # )
            return []

        # logger.info(
        #     f"[WAYPOINT] {phase}: Extracting from path with {len(path_nodes)} nodes"
        # )

        waypoints = []
        cumulative_distance = 0.0
        last_waypoint_distance = 0.0
        UNIFORM_VALUE = 1.0  # All waypoints have same value

        # # Log first and last nodes for debugging
        # if len(path_nodes) >= 2:
        #     logger.info(
        #         f"[WAYPOINT] {phase}: path start={path_nodes[0]}, end={path_nodes[-1]}"
        #     )

        # First node at cumulative distance 0
        for i in range(len(path_nodes)):
            if i > 0:
                # Calculate edge distance
                dx = path_nodes[i][0] - path_nodes[i - 1][0]
                dy = path_nodes[i][1] - path_nodes[i - 1][1]
                cumulative_distance += math.sqrt(dx * dx + dy * dy)

            # Place waypoint every progress_spacing pixels (default 12px)
            if cumulative_distance - last_waypoint_distance >= self.progress_spacing:
                waypoints.append(
                    PathWaypoint(
                        position=(
                            float(path_nodes[i][0]) + NODE_WORLD_COORD_OFFSET,
                            float(path_nodes[i][1]) + NODE_WORLD_COORD_OFFSET,
                        ),
                        waypoint_type="progress",
                        value=UNIFORM_VALUE,
                        phase=phase,
                        node_index=i,
                        physics_state="unknown",
                        curvature=0.0,
                    )
                )
                last_waypoint_distance = cumulative_distance

        # Compute exit directions for velocity alignment
        waypoints = self._compute_exit_directions(waypoints, path_nodes)

        # # DIAGNOSTIC: Log waypoint extraction with phase information
        # if waypoints:
        #     logger.warning(
        #         f"[WAYPOINT] {phase}: ✓ Extracted {len(waypoints)} waypoints at {self.progress_spacing}px intervals "
        #         f"(total path length: {cumulative_distance:.0f}px from {len(path_nodes)} nodes)"
        #     )
        #     # Verify first waypoint has correct phase
        #     if len(waypoints) > 0:
        #         logger.warning(
        #             f"[WAYPOINT] {phase}: First waypoint phase={waypoints[0].phase}, expected={phase}"
        #         )

        return waypoints

    def _compute_exit_directions(
        self,
        waypoints: List[PathWaypoint],
        path_nodes: List[Tuple[int, int]],
    ) -> List[PathWaypoint]:
        """Compute exit direction for each waypoint pointing toward next waypoint.

        For sequential guidance, each waypoint needs to know which direction the
        agent should continue traveling after collecting it. This is computed by
        finding the direction to the next waypoint in the sequence.

        Args:
            waypoints: List of waypoints (sorted by node_index)
            path_nodes: Original path nodes for fallback direction

        Returns:
            List of waypoints with exit_direction populated
        """
        if not waypoints:
            return waypoints

        updated_waypoints = []

        for i, wp in enumerate(waypoints):
            exit_dir = None

            # Find direction to next waypoint in sequence
            if i < len(waypoints) - 1:
                next_wp = waypoints[i + 1]
                dx = next_wp.position[0] - wp.position[0]
                dy = next_wp.position[1] - wp.position[1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 1.0:
                    # Normalize direction vector
                    exit_dir = (dx / dist, dy / dist)
            else:
                # Last waypoint - use direction along path toward goal
                # Find nodes near this waypoint in the path
                wp_node_idx = wp.node_index
                if wp_node_idx < len(path_nodes) - 1:
                    # Direction from this node to next node in path
                    curr_node = path_nodes[wp_node_idx]
                    next_node = path_nodes[wp_node_idx + 1]

                    # Convert to world coordinates
                    curr_x = curr_node[0] + NODE_WORLD_COORD_OFFSET
                    curr_y = curr_node[1] + NODE_WORLD_COORD_OFFSET
                    next_x = next_node[0] + NODE_WORLD_COORD_OFFSET
                    next_y = next_node[1] + NODE_WORLD_COORD_OFFSET

                    dx = next_x - curr_x
                    dy = next_y - curr_y
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist > 1.0:
                        exit_dir = (dx / dist, dy / dist)

            # Create new waypoint with exit_direction
            updated_waypoints.append(
                PathWaypoint(
                    position=wp.position,
                    waypoint_type=wp.waypoint_type,
                    value=wp.value,
                    phase=wp.phase,
                    node_index=wp.node_index,
                    physics_state=wp.physics_state,
                    curvature=wp.curvature,
                    exit_direction=exit_dir,
                )
            )

        logger.debug(
            f"Computed exit directions for {len(updated_waypoints)} waypoints "
            f"({sum(1 for wp in updated_waypoints if wp.exit_direction is not None)} have valid directions)"
        )

        return updated_waypoints

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction and caching statistics.

        Returns:
            Statistics dictionary
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "total_waypoints_extracted": self.total_waypoints_extracted,
            "cached_levels": len(self._waypoint_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
        }
