"""Intermediate goal curriculum manager for complex level training.

This module implements a curriculum learning approach that accelerates value function
calibration on complex levels by moving switch/exit entities to intermediate positions
along the optimal path.

Design Philosophy:
- Physical entity movement (not virtual goals) maintains observation-reward consistency
- Gradual progression: 25% → 50% → 75% → 100% based on success rates
- Automatic advancement when rolling success rate exceeds threshold
- Preserves Markov property: agent observations match actual entity positions

Integration Points:
- BaseEnvironment: Moves entities after map load, before graph building
- RewardCalculator: No changes needed (reads from entity positions automatically)
- Go-Explore: Stores curriculum stage metadata in checkpoints
- Route Visualization: Renders curriculum stage annotations
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GoalCurriculumState:
    """Tracks current curriculum state for serialization and analysis.

    Attributes:
        unified_stage: Current stage index for sliding window progression (0 to num_stages-1)
        switch_activation_count: Total switch activations across all episodes
        completion_count: Total level completions across all episodes
        episode_count: Total episodes tracked for curriculum advancement
    """

    unified_stage: int = 0
    switch_activation_count: int = 0
    completion_count: int = 0
    episode_count: int = 0


class IntermediateGoalManager:
    """Manages curriculum-based entity repositioning for goal difficulty progression.

    Approach: Moves actual switch/exit entities in sim.entity_dic to intermediate
    positions along physics-optimal paths. All observations automatically update since
    they read from entity positions.

    This maintains observation-reward consistency (Markov property) while providing
    earlier terminal rewards to accelerate value function calibration.

    Workflow:
    1. On level load: Store original positions, extract optimal paths
    2. Apply curriculum: Move entities to current stage positions
    3. Build graph/observations: Automatically reflect curriculum positions
    4. On episode end: Track outcomes, advance stages when threshold reached

    Example progression for complex level:
    - Stage 0 (25%): Switch at 250px from spawn (easy to reach)
    - Stage 1 (50%): Switch at 500px from spawn (moderate)
    - Stage 2 (75%): Switch at 750px from spawn (challenging)
    - Stage 3 (100%): Switch at original 1000px from spawn (full difficulty)
    """

    def __init__(self, config):
        """Initialize curriculum manager with sliding window progression.

        Sliding window approach: Both switch and exit slide forward along the combined
        optimal path (spawn → switch → exit), maintaining fixed interval spacing.
        Each stage advances both entities by stage_distance_interval pixels.

        Args:
            config: GoalCurriculumConfig instance with progression parameters
        """
        self.config = config
        self.state = GoalCurriculumState()

        # Store ORIGINAL entity positions (never modified, used for path extraction)
        self._original_switch_pos: Optional[Tuple[float, float]] = None
        self._original_exit_pos: Optional[Tuple[float, float]] = None

        # Optimal paths extracted ONCE per level using original positions
        # These paths represent the TRUE solution the agent should eventually learn
        self._spawn_to_switch_path: List[Tuple[int, int]] = []
        self._switch_to_exit_path: List[Tuple[int, int]] = []

        # NEW: Combined path for sliding window progression
        self._combined_path: List[Tuple[int, int]] = []
        self._spawn_to_switch_distance: float = 0.0
        self._switch_to_exit_distance: float = 0.0
        self._combined_distance: float = 0.0
        self._num_stages: int = 0

        # Rolling window success tracking for automatic stage advancement
        # Uses deque with maxlen for efficient sliding window
        self._recent_switch_activations: deque = deque(maxlen=config.rolling_window)
        self._recent_completions: deque = deque(maxlen=config.rolling_window)

        # Cache rebuild flag for PBRS integration
        self._cache_needs_rebuild = False

        # Override positions from SharedLevelCache (for multi-stage cache consistency)
        # When set, these override position calculations from paths
        self._curriculum_positions_override: Optional[
            Dict[str, Tuple[float, float]]
        ] = None

        logger.info(
            f"IntermediateGoalManager initialized (sliding window): "
            f"interval={config.stage_distance_interval}px, "
            f"threshold={config.advancement_threshold}, "
            f"window={config.rolling_window}"
        )

    def set_curriculum_position_overrides(
        self, positions: Optional[Dict[str, Tuple[float, float]]]
    ) -> None:
        """Set override positions from SharedLevelCache.

        When using multi-stage SharedLevelCaches, environments MUST use the exact
        positions that were used during cache building to avoid validation failures.
        This method allows injecting those pre-computed positions.

        Args:
            positions: Dict mapping entity type ("switch", "exit") to (x, y) position,
                      or None to clear overrides and use calculated positions
        """
        self._curriculum_positions_override = positions
        if positions:
            logger.debug(
                f"[CURRICULUM] Using position overrides from SharedLevelCache: "
                f"switch={positions.get('switch')}, exit={positions.get('exit')}"
            )
        else:
            logger.debug(
                "[CURRICULUM] Cleared position overrides, will calculate from paths"
            )

    def store_original_positions(
        self, switch_pos: Tuple[float, float], exit_pos: Tuple[float, float]
    ) -> None:
        """Store original entity positions before curriculum modifications.

        Must be called immediately after map load, before build_paths_for_level().

        Args:
            switch_pos: Original exit switch position from level data
            exit_pos: Original exit door position from level data
        """
        self._original_switch_pos = switch_pos
        self._original_exit_pos = exit_pos

        logger.debug(f"Stored original positions: switch={switch_pos}, exit={exit_pos}")

    def build_paths_for_level(
        self,
        graph_data: Dict[str, Any],
        level_data: Any,
        spawn_pos: Tuple[float, float],
    ) -> None:
        """Extract optimal paths using ORIGINAL entity positions.

        Uses find_shortest_path() with physics-aware costs and mine avoidance
        to compute the true optimal solution. Paths are extracted once per level
        and cached for curriculum position sampling.

        Must be called AFTER store_original_positions() and BEFORE apply_to_simulator().

        Args:
            graph_data: Graph data dict with adjacency, physics_cache, etc.
            level_data: LevelData object for mine proximity calculations
            spawn_pos: Ninja spawn position from level data
        """
        if self._original_switch_pos is None or self._original_exit_pos is None:
            logger.warning(
                "Cannot build paths: original positions not stored. "
                "Call store_original_positions() first."
            )
            return

        try:
            from nclone.graph.reachability.pathfinding_utils import (
                find_shortest_path,
                find_ninja_node,
                find_closest_node_to_position,
                extract_spatial_lookups_from_graph_data,
            )
            from nclone.graph.reachability.mine_proximity_cache import (
                MineProximityCostCache,
            )
            from nclone.constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )

            # Extract graph components
            adjacency = graph_data.get("adjacency")
            base_adjacency = graph_data.get("base_adjacency", adjacency)
            physics_cache = graph_data.get("node_physics")

            if not adjacency or not physics_cache:
                logger.warning("Cannot build paths: missing adjacency or physics_cache")
                return

            spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
                graph_data
            )

            # Build mine proximity cache for physics-aware pathfinding
            mine_cache = MineProximityCostCache()
            mine_cache.build_cache(level_data, adjacency)

            # Find spawn node
            spawn_node = find_ninja_node(
                spawn_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )

            if spawn_node is None:
                logger.warning(f"Cannot find spawn node at {spawn_pos}")
                return

            # Find switch node using ORIGINAL position
            switch_node = find_closest_node_to_position(
                self._original_switch_pos,
                adjacency,
                threshold=50.0,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if switch_node is None:
                logger.warning(
                    f"Cannot find switch node at {self._original_switch_pos}"
                )
                return

            # Calculate spawn → switch path
            spawn_to_switch_path, spawn_to_switch_cost = find_shortest_path(
                spawn_node,
                switch_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_cache,
            )

            if spawn_to_switch_path is None:
                logger.warning("No path from spawn to switch (unreachable)")
                return

            self._spawn_to_switch_path = spawn_to_switch_path

            # Find exit node using ORIGINAL position
            exit_node = find_closest_node_to_position(
                self._original_exit_pos,
                adjacency,
                threshold=50.0,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if exit_node is None:
                logger.warning(f"Cannot find exit node at {self._original_exit_pos}")
                return

            # Calculate switch → exit path
            switch_to_exit_path, switch_to_exit_cost = find_shortest_path(
                switch_node,
                exit_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_cache,
            )

            if switch_to_exit_path is None:
                logger.warning("No path from switch to exit (unreachable)")
                return

            self._switch_to_exit_path = switch_to_exit_path

            # NEW: Compute path distances for sliding window progression
            self._spawn_to_switch_distance = self._compute_path_distance(
                self._spawn_to_switch_path
            )
            self._switch_to_exit_distance = self._compute_path_distance(
                self._switch_to_exit_path
            )
            self._combined_distance = (
                self._spawn_to_switch_distance + self._switch_to_exit_distance
            )

            # NEW: Build combined path for unified progression (avoid duplicate switch node)
            self._combined_path = (
                self._spawn_to_switch_path + self._switch_to_exit_path[1:]
                if len(self._switch_to_exit_path) > 0
                else self._spawn_to_switch_path
            )

            # NEW: Compute number of stages based on combined distance
            self._num_stages = self._compute_num_stages()

            # NOTE: Sub-goal extraction deferred to curriculum-aware rebuild
            # Sub-goals will be extracted based on curriculum stage via rebuild_sub_goals_for_stage()
            # This ensures sub-goals are only placed within reachable curriculum distances

            logger.info(
                f"Goal curriculum paths extracted: "
                f"spawn→switch={len(self._spawn_to_switch_path)} nodes "
                f"({self._spawn_to_switch_distance:.1f}px, cost={spawn_to_switch_cost:.2f}), "
                f"switch→exit={len(self._switch_to_exit_path)} nodes "
                f"({self._switch_to_exit_distance:.1f}px, cost={switch_to_exit_cost:.2f}), "
                f"combined={self._combined_distance:.1f}px, "
                f"stages={self._num_stages} "
                f"(sub-goals will be extracted based on curriculum stage)"
            )

        except Exception as e:
            logger.error(f"Failed to build curriculum paths: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _compute_path_distance(self, path: List[Tuple[int, int]]) -> float:
        """Compute total distance along path in pixels.

        Args:
            path: List of node positions from pathfinding

        Returns:
            Total distance in pixels
        """
        if not path or len(path) < 2:
            return 0.0

        import math

        total_distance = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total_distance += math.sqrt(dx * dx + dy * dy)

        return total_distance

    def get_curriculum_truncated_paths(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, float]:
        """Get paths truncated to current curriculum stage distances.

        For curriculum-aware waypoint extraction, only waypoints within the
        currently reachable path segment should be generated. This prevents
        waypoints from appearing beyond curriculum goals.

        Returns:
            Tuple of (truncated_spawn_to_switch, truncated_switch_to_exit,
                     switch_distance, exit_distance)
            Returns empty paths if paths not built yet
        """
        if not self._spawn_to_switch_path or not self._switch_to_exit_path:
            return [], [], 0.0, 0.0

        interval = self.config.stage_distance_interval

        # Calculate curriculum distances
        # Switch at (stage + 1) * interval, clamped to original
        switch_distance = min(
            (self.state.unified_stage + 1) * interval, self._spawn_to_switch_distance
        )

        # Exit at (stage + 2) * interval along combined path
        exit_combined_distance = min(
            (self.state.unified_stage + 2) * interval, self._combined_distance
        )

        # Truncate spawn→switch path
        truncated_spawn_to_switch = self._truncate_path_at_distance(
            self._spawn_to_switch_path, switch_distance
        )

        # Truncate switch→exit path
        # Exit distance relative to switch position
        # FIXED: Compare with curriculum switch_distance, not original _spawn_to_switch_distance
        if exit_combined_distance <= switch_distance:
            # Exit hasn't reached curriculum switch yet (very early curriculum stages)
            truncated_switch_to_exit = []
            exit_distance_from_switch = 0.0
        else:
            # Exit is past curriculum switch - calculate offset from curriculum switch
            exit_distance_from_switch = exit_combined_distance - switch_distance
            truncated_switch_to_exit = self._truncate_path_at_distance(
                self._switch_to_exit_path, exit_distance_from_switch
            )

        # DIAGNOSTIC: Log truncation results
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"[CURRICULUM] get_curriculum_truncated_paths stage={self.state.unified_stage}: "
            f"spawn→switch: {len(self._spawn_to_switch_path)} → {len(truncated_spawn_to_switch)} nodes, "
            f"switch→exit: {len(self._switch_to_exit_path)} → {len(truncated_switch_to_exit)} nodes, "
            f"switch_dist={switch_distance:.0f}px, exit_dist_from_switch={exit_distance_from_switch:.0f}px"
        )

        return (
            truncated_spawn_to_switch,
            truncated_switch_to_exit,
            switch_distance,
            exit_combined_distance,
        )

    def _compute_num_stages(self) -> int:
        """Compute number of curriculum stages based on combined path distance.

        Returns:
            Number of stages (minimum 2)
        """
        if self._combined_distance <= 0:
            return 2

        interval = self.config.stage_distance_interval
        # Number of intervals that fit in combined distance + 1 for final stage
        num_stages = max(2, int(self._combined_distance / interval) + 1)

        return num_stages

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

    def _sample_position_at_distance(
        self, path: List[Tuple[int, int]], target_distance: float
    ) -> Tuple[float, float]:
        """Sample exact position at target_distance along path with interpolation.

        Uses linear interpolation between nodes for precise positioning.
        Supports both grounded and aerial nodes for air trajectory training.

        Args:
            path: List of node positions from pathfinding
            target_distance: Target distance in pixels from path start

        Returns:
            (x, y) position at specified distance along path
        """
        if not path or len(path) < 2:
            if path:
                # Single node - return it with coordinate offset
                return (float(path[0][0]) + 24.0, float(path[0][1]) + 24.0)
            return (0.0, 0.0)

        import math
        import bisect

        # Compute cumulative distances along path
        cumulative_distances = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            segment_dist = math.sqrt(dx * dx + dy * dy)
            cumulative_distances.append(cumulative_distances[-1] + segment_dist)

        total_distance = cumulative_distances[-1]

        # Clamp target to path bounds
        target_distance = max(0.0, min(target_distance, total_distance))

        # Handle edge cases
        if target_distance <= 0:
            return (float(path[0][0]) + 24.0, float(path[0][1]) + 24.0)
        if target_distance >= total_distance:
            return (float(path[-1][0]) + 24.0, float(path[-1][1]) + 24.0)

        # Binary search for segment containing target distance
        segment_idx = bisect.bisect_right(cumulative_distances, target_distance) - 1
        segment_idx = max(0, min(segment_idx, len(path) - 2))

        # Interpolate position within segment
        segment_start_dist = cumulative_distances[segment_idx]
        segment_end_dist = cumulative_distances[segment_idx + 1]
        segment_length = segment_end_dist - segment_start_dist

        if segment_length > 0:
            t = (target_distance - segment_start_dist) / segment_length
        else:
            t = 0.0

        # Linear interpolation between segment nodes
        x = path[segment_idx][0] + t * (path[segment_idx + 1][0] - path[segment_idx][0])
        y = path[segment_idx][1] + t * (path[segment_idx + 1][1] - path[segment_idx][1])

        # Apply coordinate offset (tile data space to world space)
        return (x + 24.0, y + 24.0)

    def is_at_final_stage(self) -> bool:
        """Check if curriculum is at final stage (original entity positions).

        Returns:
            True if unified_stage is at maximum (entities at original positions)
        """
        return self.state.unified_stage >= self._num_stages - 1

    def needs_cache_rebuild(self) -> bool:
        """Check if curriculum stage advanced requiring PBRS cache rebuild.

        Returns True once after stage advancement, then resets flag.
        This allows environment to detect stage changes and invalidate caches.

        Returns:
            True if stage advanced since last check, False otherwise
        """
        if self._cache_needs_rebuild:
            self._cache_needs_rebuild = False
            return True
        return False

    def get_curriculum_switch_position(self) -> Tuple[float, float]:
        """Get curriculum-adjusted switch position for sliding window model.

        Switch position = (unified_stage + 1) * interval, clamped to original switch distance.
        Using (stage + 1) ensures minimum distance of one interval at stage 0.
        This creates a sliding window where switch advances until reaching original position.

        Returns:
            (x, y) position where switch should be placed for current difficulty
        """
        # Use override position from SharedLevelCache if available
        if (
            self._curriculum_positions_override is not None
            and "switch" in self._curriculum_positions_override
        ):
            return self._curriculum_positions_override["switch"]

        if not self._spawn_to_switch_path:
            # Fallback to original if paths not built
            print(
                f"[get_curriculum_switch_position] WARNING: No path built! "
                f"Returning original: {self._original_switch_pos}"
            )
            return self._original_switch_pos or (0.0, 0.0)

        interval = self.config.stage_distance_interval

        # Switch distance along spawn→switch path, clamped to original
        # Use (stage + 1) to ensure minimum distance of one interval at stage 0
        switch_distance = min(
            (self.state.unified_stage + 1) * interval, self._spawn_to_switch_distance
        )

        # # DIAGNOSTIC: Always print position calculation
        # print(
        #     f"[get_curriculum_switch_position] "
        #     f"stage={self.state.unified_stage}, "
        #     f"interval={interval}px, "
        #     f"switch_distance={switch_distance:.1f}px (stage*interval clamped), "
        #     f"spawn_to_switch_distance={self._spawn_to_switch_distance:.1f}px, "
        #     f"original_switch={self._original_switch_pos}, "
        #     f"path_len={len(self._spawn_to_switch_path)} nodes"
        # )

        result = self._sample_position_at_distance(
            self._spawn_to_switch_path, switch_distance
        )

        # # DIAGNOSTIC: Print result and verify it differs from spawn at stage > 0
        # spawn = self._spawn_to_switch_path[0] if self._spawn_to_switch_path else (0, 0)
        # dist_from_spawn = (
        #     (result[0] - spawn[0]) ** 2 + (result[1] - spawn[1]) ** 2
        # ) ** 0.5
        # print(
        #     f"[get_curriculum_switch_position] Result: {result}, "
        #     f"spawn={spawn}, "
        #     f"dist_from_spawn={dist_from_spawn:.1f}px"
        # )

        # # CRITICAL CHECK: At stage > 0, switch should NOT be at spawn
        # if self.state.unified_stage > 0 and dist_from_spawn < 20:
        #     print(
        #         f"[CURRICULUM_BUG] Stage {self.state.unified_stage} but switch only "
        #         f"{dist_from_spawn:.1f}px from spawn! "
        #         f"Expected ~{self.state.unified_stage * interval}px away. "
        #         f"Check path sampling or interval config."
        #     )

        return result

    def get_curriculum_exit_position(self) -> Tuple[float, float]:
        """Get curriculum-adjusted exit position for sliding window model.

        Exit position = (unified_stage + 2) * interval along combined path, clamped to original.
        Using (stage + 2) maintains fixed spacing (one interval) between switch and exit.
        This maintains fixed spacing between switch and exit, sliding forward together.

        Returns:
            (x, y) position where exit door should be placed for current difficulty
        """
        # Use override position from SharedLevelCache if available
        if (
            self._curriculum_positions_override is not None
            and "exit" in self._curriculum_positions_override
        ):
            return self._curriculum_positions_override["exit"]

        if not self._combined_path:
            # Fallback to original if paths not built
            return self._original_exit_pos or (0.0, 0.0)

        interval = self.config.stage_distance_interval

        # Exit position = (stage + 2) * interval along combined path, clamped to total
        # This maintains one-interval spacing from switch position at (stage + 1)
        exit_combined_distance = min(
            (self.state.unified_stage + 2) * interval, self._combined_distance
        )

        # print(
        #     f"[get_curriculum_exit_position] stage={self.state.unified_stage}, "
        #     f"exit_combined_distance={exit_combined_distance:.1f}px, "
        #     f"combined_distance={self._combined_distance:.1f}px, "
        #     f"spawn_to_switch_distance={self._spawn_to_switch_distance:.1f}px"
        # )

        # Sample from combined path at exit distance
        # If exit is in spawn→switch section, use spawn→switch path
        # If exit is in switch→exit section, offset from switch position
        if exit_combined_distance <= self._spawn_to_switch_distance:
            # Exit is still in spawn→switch path section
            # print(
            #     f"[get_curriculum_exit_position] Exit in spawn→switch section (BACKWARDS!)"
            # )
            return self._sample_position_at_distance(
                self._spawn_to_switch_path, exit_combined_distance
            )
        else:
            # Exit is in switch→exit path section
            exit_offset = exit_combined_distance - self._spawn_to_switch_distance
            # print(
            #     f"[get_curriculum_exit_position] Exit in switch→exit section, "
            #     f"offset={exit_offset:.1f}px from switch"
            # )
            return self._sample_position_at_distance(
                self._switch_to_exit_path, exit_offset
            )

    def apply_to_simulator(self, sim) -> None:
        """Move entity positions in sim.entity_dic to curriculum positions.

        CRITICAL: Must be called AFTER map load but BEFORE graph building.
        This ensures all observations reflect curriculum positions.

        Modifies:
        - sim.entity_dic[3]: Exit switch entities
        - sim.entity_dic[4]: Exit door entities (and parent reference)
        - grid_entity: Updates spatial grid for collision detection

        Args:
            sim: Simulator instance with entity_dic
        """
        # frame_at_start = getattr(sim, "frame", -1)
        # print(
        #     f"[apply_to_simulator] CALLED: frame={frame_at_start}, stage={self.state.unified_stage}"
        # )

        try:
            from nclone.physics import clamp_cell
            import math

            # Move exit switch (entity type 3 contains both EntityExit doors and EntityExitSwitch)
            exit_entities = sim.entity_dic.get(3, [])
            # print(
            #     f"[apply_to_simulator] Found {len(exit_entities)} entities in entity_dic[3]: "
            #     f"{[type(e).__name__ for e in exit_entities]}"
            # )

            # CRITICAL FIX: Type-safe entity lookup to prevent door/switch confusion
            # entity_dic[3] contains BOTH EntityExit (doors) and EntityExitSwitch (switches)
            # Using [-1] assumes last entity is switch, but order is not guaranteed after
            # multiple stage advancements with 256+ workers. Must filter by type name.
            switch_entity = None
            for entity in reversed(exit_entities):
                if type(entity).__name__ == "EntityExitSwitch":
                    switch_entity = entity
                    break

            # Log entity state BEFORE modification for diagnostics
            if switch_entity is not None:
                # CRITICAL WARNING: Detect mid-episode corruption
                # If switch.active=False here, it means switch was collected mid-episode
                # and we're about to resurrect it, causing bizarre behavior
                frame_count = getattr(sim, "frame", 0)
                was_collected = not switch_entity.active

                if was_collected and frame_count > 10:
                    # Switch already collected mid-episode - we're resurrecting it!
                    logger.error(
                        f"[CURRICULUM] BUG DETECTED: apply_to_simulator called mid-episode with collected switch! "
                        f"switch.active={switch_entity.active} (False=collected), "
                        f"frame={frame_count}, stage={self.state.unified_stage}. "
                        f"This will RESURRECT the switch mid-episode, causing pathfinding confusion!"
                    )

                # Always log at INFO level for episode-start calls (frame <= 10) to track curriculum
                if frame_count <= 10:
                    logger.info(
                        f"[CURRICULUM] apply_to_simulator at episode start: "
                        f"frame={frame_count}, stage={self.state.unified_stage}, "
                        f"switch.active={switch_entity.active} (should be True for fresh entities), "
                        f"entity_types=[{', '.join(type(e).__name__ for e in exit_entities)}]"
                    )
                else:
                    logger.debug(
                        f"[CURRICULUM] apply_to_simulator mid-episode: "
                        f"frame={frame_count}, stage={self.state.unified_stage}, "
                        f"switch.active={switch_entity.active}"
                    )

            if switch_entity is None:
                # FAIL FAST: No switch found - this should never happen
                entity_types = [type(e).__name__ for e in exit_entities]
                # print(
                #     f"[apply_to_simulator] ERROR: No EntityExitSwitch found in entity_dic[3]! "
                #     f"Found {len(exit_entities)} entities with types: {entity_types}. "
                #     f"frame={getattr(sim, 'frame', -1)}"
                # )
                raise RuntimeError(
                    f"No EntityExitSwitch found in entity_dic[3]! "
                    f"Found {len(exit_entities)} entities with types: {entity_types}. "
                    f"This indicates entity loading failed or entities were corrupted."
                )

            if switch_entity:
                curriculum_pos = self.get_curriculum_switch_position()

                # DIAGNOSTIC: Comprehensive logging for debugging immediate switch activation
                old_pos = (switch_entity.xpos, switch_entity.ypos)
                # ninja = getattr(sim, "ninja", None)
                # ninja_pos = (ninja.xpos, ninja.ypos) if ninja else (0, 0)

                # # Use different markers for stage > 0 to make them stand out
                # stage = self.state.unified_stage
                # marker = "***" if stage > 0 else "==="
                # prefix = f"STAGE_{stage}" if stage > 0 else "CURRICULUM"

                # print(
                #     f"\n{marker * 20}\n"
                #     f"[{prefix}_DEBUG] apply_to_simulator SWITCH REPOSITIONING\n"
                #     f"  Stage: {stage}{' (SWITCH SHOULD BE FAR FROM SPAWN!)' if stage > 0 else ''}\n"
                #     f"  Expected distance from spawn: ~{stage * 100}px\n"
                #     f"  Frame: {getattr(sim, 'frame', -1)}\n"
                #     f"  Switch BEFORE: pos=({old_pos[0]:.1f}, {old_pos[1]:.1f}), active={switch_entity.active}\n"
                #     f"  Switch TARGET: pos=({curriculum_pos[0]:.1f}, {curriculum_pos[1]:.1f})\n"
                #     f"  Ninja pos: ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})\n"
                #     f"  Distance ninja→target: {((ninja_pos[0] - curriculum_pos[0]) ** 2 + (ninja_pos[1] - curriculum_pos[1]) ** 2) ** 0.5:.1f}px\n"
                #     f"{marker * 20}"
                # )

                switch_entity.xpos = curriculum_pos[0]
                switch_entity.ypos = curriculum_pos[1]

                # VERIFICATION: Confirm position was actually set
                actual_pos_after = (switch_entity.xpos, switch_entity.ypos)
                if (
                    abs(actual_pos_after[0] - curriculum_pos[0]) > 0.1
                    or abs(actual_pos_after[1] - curriculum_pos[1]) > 0.1
                ):
                    raise RuntimeError(
                        f"[CURRICULUM_BUG] Switch position NOT set correctly!\n"
                        f"  Expected: {curriculum_pos}\n"
                        f"  Actual: {actual_pos_after}\n"
                        f"  This is a CRITICAL bug - entity.xpos/ypos assignment failed!"
                    )

                # print(
                #     f"[CURRICULUM_DEBUG] Switch AFTER: pos=({switch_entity.xpos:.1f}, {switch_entity.ypos:.1f})"
                # )

                # CRITICAL FIX: Reset switch activation state for new episode
                # When entities aren't freshly reloaded (map_just_loaded=True skips sim.reset()),
                # the switch entity from the previous episode persists with active=False.
                # This causes episodes to start with the switch already "activated", allowing
                # the agent to go straight to the exit door without collecting the switch.
                #
                # DEFENSIVE: Only reset if at episode start (frame <= 10)
                # If called mid-episode (frame > 10), skip reset to avoid resurrection
                frame_count = getattr(sim, "frame", 0)
                if frame_count <= 10:
                    switch_entity.active = True
                else:
                    # Mid-episode call - don't modify switch state
                    logger.warning(
                        f"[CURRICULUM] Skipping switch.active reset at frame {frame_count} "
                        f"(mid-episode, would resurrect collected switch). "
                        f"Current: active={switch_entity.active}"
                    )

                # VALIDATION: Verify switch state was set correctly (only if we actually set it)
                if frame_count <= 10 and not switch_entity.active:
                    raise RuntimeError(
                        f"Failed to reset switch activation state! "
                        f"Entity: {type(switch_entity).__name__}, "
                        f"active={switch_entity.active} (expected True). "
                        f"This indicates entity attribute modification failed."
                    )

                # DEFENSIVE: Clear render caches when entity positions change
                # This ensures visualizations reflect curriculum positions
                if hasattr(sim, "gym_env") and sim.gym_env:
                    if hasattr(sim.gym_env, "nplay_headless"):
                        nplay = sim.gym_env.nplay_headless
                        if hasattr(nplay, "cached_render_surface"):
                            nplay.cached_render_surface = None
                        if hasattr(nplay, "cached_render_buffer"):
                            nplay.cached_render_buffer = None

                # CRITICAL FIX: Update grid placement for collision detection
                # Without this, the switch remains in the old grid cell and won't be found
                # by gather_entities_from_neighbourhood() during collision checks
                old_cell = switch_entity.cell
                new_cell = clamp_cell(
                    math.floor(curriculum_pos[0] / 24),
                    math.floor(curriculum_pos[1] / 24),
                )
                if new_cell != old_cell:
                    # Remove from old cell
                    if (
                        old_cell in sim.grid_entity
                        and switch_entity in sim.grid_entity[old_cell]
                    ):
                        sim.grid_entity[old_cell].remove(switch_entity)
                    # Update entity's cell attribute
                    switch_entity.cell = new_cell
                    # Add to new cell
                    if new_cell not in sim.grid_entity:
                        sim.grid_entity[new_cell] = []
                    sim.grid_entity[new_cell].append(switch_entity)
                    logger.debug(f"Updated switch grid cell: {old_cell} → {new_cell}")

                # Calculate switch distance for logging
                interval = self.config.stage_distance_interval
                switch_distance = min(
                    self.state.unified_stage * interval, self._spawn_to_switch_distance
                )
                logger.info(
                    f"Moved exit switch: {old_pos} → {curriculum_pos} "
                    f"(stage {self.state.unified_stage}, "
                    f"distance={switch_distance:.0f}px)"
                )

                # Log entity state AFTER modification for diagnostics
                logger.debug(
                    f"[CURRICULUM] After switch modification: "
                    f"switch.active={switch_entity.active}, "
                    f"switch_pos=({switch_entity.xpos:.1f}, {switch_entity.ypos:.1f})"
                )

                # VERIFICATION: Confirm exit_switch_activated() returns correct value
                # This verifies the entity state change propagated correctly
                if frame_count <= 10:
                    # Get gym_env reference to call exit_switch_activated()
                    if hasattr(sim, "gym_env") and sim.gym_env:
                        # Access via gym_env.nplay_headless if available
                        nplay = getattr(sim.gym_env, "nplay_headless", None)
                        if nplay:
                            is_activated = nplay.exit_switch_activated()
                            if is_activated:
                                logger.error(
                                    f"[CURRICULUM] VERIFICATION FAILED! "
                                    f"After setting switch.active=True, exit_switch_activated() returns True (activated)! "
                                    f"Entity state: switch.active={switch_entity.active}, "
                                    f"frame={frame_count}, stage={self.state.unified_stage}. "
                                    f"This indicates the entity state modification didn't take effect!"
                                )
                            else:
                                logger.info(
                                    f"[CURRICULUM] ✓ Switch state verified: exit_switch_activated()={is_activated} (correct, not activated)"
                                )

            # Move exit door (entity type 4)
            # Exit door is referenced as .parent from exit switch
            # CRITICAL FIX: Use type-safe switch_entity instead of fragile exit_switches[-1]
            if switch_entity and hasattr(switch_entity, "parent"):
                door_entity = switch_entity.parent
                if door_entity:
                    curriculum_pos = self.get_curriculum_exit_position()

                    # Log entity movement
                    old_pos = (door_entity.xpos, door_entity.ypos)
                    # print(
                    #     f"[apply_to_simulator] Moving exit door: {old_pos} → {curriculum_pos}, "
                    #     f"stage={self.state.unified_stage}"
                    # )
                    door_entity.xpos = curriculum_pos[0]
                    door_entity.ypos = curriculum_pos[1]
                    # print(
                    #     f"[apply_to_simulator] Exit door moved! Verify: "
                    #     f"xpos={door_entity.xpos}, ypos={door_entity.ypos}"
                    # )

                    # CRITICAL: Update door's cell attribute but do NOT add to grid_entity!
                    # The exit door should only be in grid_entity AFTER switch is collected.
                    # EntityExitSwitch.logical_collision() adds the door to grid_entity.
                    old_cell = door_entity.cell
                    new_cell = clamp_cell(
                        math.floor(curriculum_pos[0] / 24),
                        math.floor(curriculum_pos[1] / 24),
                    )
                    if new_cell != old_cell:
                        # Remove from grid if it was mistakenly there
                        if (
                            old_cell in sim.grid_entity
                            and door_entity in sim.grid_entity[old_cell]
                        ):
                            sim.grid_entity[old_cell].remove(door_entity)
                        # Also check new cell and remove if present
                        if (
                            new_cell in sim.grid_entity
                            and door_entity in sim.grid_entity[new_cell]
                        ):
                            sim.grid_entity[new_cell].remove(door_entity)
                        # Update entity's cell attribute (for when switch adds it later)
                        door_entity.cell = new_cell
                        # NOTE: Do NOT append to grid_entity here!
                        # The door only becomes interactable after switch collection
                        logger.debug(
                            f"Updated exit door cell attribute: {old_cell} → {new_cell} "
                            f"(NOT added to grid_entity - awaiting switch activation)"
                        )

                    # Calculate exit distance for logging
                    interval = self.config.stage_distance_interval
                    exit_combined_distance = min(
                        (self.state.unified_stage + 1) * interval,
                        self._combined_distance,
                    )
                    logger.info(
                        f"Moved exit door: {old_pos} → {curriculum_pos} "
                        f"(stage {self.state.unified_stage}, "
                        f"distance={exit_combined_distance:.0f}px)"
                    )

                    # Log entity state AFTER modification for diagnostics
                    logger.debug(
                        f"[CURRICULUM] After door modification: "
                        f"door_pos=({door_entity.xpos:.1f}, {door_entity.ypos:.1f})"
                    )
                else:
                    logger.warning("Exit switch has no parent (exit door) reference")
            else:
                logger.warning("No exit switch found to access exit door")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to apply curriculum entity positions: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # FAIL FAST: Don't continue with corrupt entity state
            # Silent failures lead to agents completing levels by reaching exit
            # without collecting switch, violating game rules
            raise

    def update_from_episode(self, switch_activated: bool, completed: bool) -> None:
        """Update episode tracking (stage advancement handled by global callback).

        Stage advancement is now managed globally by GoalCurriculumCallback, which
        aggregates success rates across all parallel environments and coordinates
        synchronized stage updates. This ensures all environments train at the same
        difficulty level for consistent curriculum progression.

        Called at episode end by BaseEnvironment._build_episode_info().

        Args:
            switch_activated: Whether switch was activated this episode
            completed: Whether level was completed this episode
        """
        self.state.episode_count += 1

        # Track outcomes in rolling windows (for local monitoring only)
        self._recent_switch_activations.append(1 if switch_activated else 0)
        self._recent_completions.append(1 if completed else 0)

        # Update total counts
        if switch_activated:
            self.state.switch_activation_count += 1
        if completed:
            self.state.completion_count += 1

        # NOTE: Stage advancement removed - now handled by GoalCurriculumCallback
        # The callback aggregates success rates across all parallel environments
        # and coordinates synchronized stage advancement via set_curriculum_stage()

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get curriculum state information for episode info dict.

        Note: Stage advancement is now managed globally by GoalCurriculumCallback.
        The unified_stage value is set externally via set_curriculum_stage().

        Returns:
            Dictionary with curriculum metadata for Go-Explore and route visualization
        """
        # Calculate local success rates (for monitoring only, not used for advancement)
        switch_rate = (
            sum(self._recent_switch_activations) / len(self._recent_switch_activations)
            if self._recent_switch_activations
            else 0.0
        )
        completion_rate = (
            sum(self._recent_completions) / len(self._recent_completions)
            if self._recent_completions
            else 0.0
        )

        # Calculate actual distances for current stage
        interval = self.config.stage_distance_interval
        switch_distance = min(
            self.state.unified_stage * interval, self._spawn_to_switch_distance
        )
        exit_distance = min(
            (self.state.unified_stage + 1) * interval, self._combined_distance
        )

        return {
            "enabled": self.config.enabled,
            "unified_stage": self.state.unified_stage,  # Set by global callback
            "num_stages": self._num_stages,
            "stage_distance_interval": interval,
            "switch_distance": switch_distance,
            "exit_distance": exit_distance,
            "curriculum_switch_pos": self.get_curriculum_switch_position(),
            "curriculum_exit_pos": self.get_curriculum_exit_position(),
            "original_switch_pos": self._original_switch_pos,
            "original_exit_pos": self._original_exit_pos,
            "switch_activation_rate": switch_rate,
            "completion_rate": completion_rate,
            "switch_activation_count": self.state.switch_activation_count,
            "completion_count": self.state.completion_count,
            "episode_count": self.state.episode_count,  # Per-environment count
            "stage_control": "global_callback",  # Indicates external stage control
            # Legacy fields for backwards compatibility
            "current_phase": "unified",  # No longer using phases
            "switch_stage": self.state.unified_stage,
            "exit_stage": self.state.unified_stage,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing.

        Returns:
            State dictionary for serialization
        """
        return {
            "unified_stage": self.state.unified_stage,
            "switch_activation_count": self.state.switch_activation_count,
            "completion_count": self.state.completion_count,
            "episode_count": self.state.episode_count,
            "recent_switch_activations": list(self._recent_switch_activations),
            "recent_completions": list(self._recent_completions),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary from checkpointing
        """
        self.state.unified_stage = state.get("unified_stage", 0)
        self.state.switch_activation_count = state.get("switch_activation_count", 0)
        self.state.completion_count = state.get("completion_count", 0)
        self.state.episode_count = state.get("episode_count", 0)

        # Restore rolling windows
        recent_switch = state.get("recent_switch_activations", [])
        recent_complete = state.get("recent_completions", [])

        self._recent_switch_activations = deque(
            recent_switch, maxlen=self.config.rolling_window
        )
        self._recent_completions = deque(
            recent_complete, maxlen=self.config.rolling_window
        )

        logger.info(
            f"Loaded curriculum state: unified_stage={self.state.unified_stage}, "
            f"episodes={self.state.episode_count}"
        )

    def reset_for_new_level(self) -> None:
        """Reset curriculum state when loading a new level.

        Called by curriculum wrapper when advancing to new level.
        Clears paths and tracking but preserves stage progression.
        """
        # Clear level-specific data
        self._original_switch_pos = None
        self._original_exit_pos = None
        self._spawn_to_switch_path = []
        self._switch_to_exit_path = []
        self._combined_path = []
        self._spawn_to_switch_distance = 0.0
        self._switch_to_exit_distance = 0.0
        self._combined_distance = 0.0
        self._num_stages = 0

        # Clear sub-goals for new level
        self._sub_goals_pre_switch = []
        self._sub_goals_post_switch = []

        # Clear rolling windows but keep unified stage
        # (progression can persist across levels if desired)
        self._recent_switch_activations.clear()
        self._recent_completions.clear()

        logger.info(
            f"Goal curriculum reset for new level "
            f"(unified_stage={self.state.unified_stage} preserved)"
        )
