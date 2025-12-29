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

        logger.info(
            f"IntermediateGoalManager initialized (sliding window): "
            f"interval={config.stage_distance_interval}px, "
            f"threshold={config.advancement_threshold}, "
            f"window={config.rolling_window}"
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

            logger.info(
                f"Goal curriculum paths extracted: "
                f"spawn→switch={len(self._spawn_to_switch_path)} nodes "
                f"({self._spawn_to_switch_distance:.1f}px, cost={spawn_to_switch_cost:.2f}), "
                f"switch→exit={len(self._switch_to_exit_path)} nodes "
                f"({self._switch_to_exit_distance:.1f}px, cost={switch_to_exit_cost:.2f}), "
                f"combined={self._combined_distance:.1f}px, "
                f"stages={self._num_stages}"
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

        Switch position = unified_stage * interval, clamped to original switch distance.
        This creates a sliding window where switch advances until reaching original position.

        Returns:
            (x, y) position where switch should be placed for current difficulty
        """
        if not self._spawn_to_switch_path:
            # Fallback to original if paths not built
            return self._original_switch_pos or (0.0, 0.0)

        interval = self.config.stage_distance_interval

        # Switch distance along spawn→switch path, clamped to original
        switch_distance = min(
            self.state.unified_stage * interval, self._spawn_to_switch_distance
        )

        return self._sample_position_at_distance(
            self._spawn_to_switch_path, switch_distance
        )

    def get_curriculum_exit_position(self) -> Tuple[float, float]:
        """Get curriculum-adjusted exit position for sliding window model.

        Exit position = (unified_stage + 1) * interval along combined path, clamped to original.
        This maintains fixed spacing between switch and exit, sliding forward together.

        Returns:
            (x, y) position where exit door should be placed for current difficulty
        """
        if not self._combined_path:
            # Fallback to original if paths not built
            return self._original_exit_pos or (0.0, 0.0)

        interval = self.config.stage_distance_interval

        # Exit position = (stage + 1) * interval along combined path, clamped to total
        exit_combined_distance = min(
            (self.state.unified_stage + 1) * interval, self._combined_distance
        )

        # Sample from combined path at exit distance
        # If exit is in spawn→switch section, use spawn→switch path
        # If exit is in switch→exit section, offset from switch position
        if exit_combined_distance <= self._spawn_to_switch_distance:
            # Exit is still in spawn→switch path section
            return self._sample_position_at_distance(
                self._spawn_to_switch_path, exit_combined_distance
            )
        else:
            # Exit is in switch→exit path section
            exit_offset = exit_combined_distance - self._spawn_to_switch_distance
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
        try:
            from nclone.physics import clamp_cell
            import math

            # Move exit switch (entity type 3)
            exit_switches = sim.entity_dic.get(3, [])
            if exit_switches:
                switch_entity = exit_switches[-1]
                curriculum_pos = self.get_curriculum_switch_position()

                # Log entity movement
                old_pos = (switch_entity.xpos, switch_entity.ypos)
                switch_entity.xpos = curriculum_pos[0]
                switch_entity.ypos = curriculum_pos[1]

                # CRITICAL FIX: Reset switch activation state for new episode
                # When entities aren't freshly reloaded (map_just_loaded=True skips sim.reset()),
                # the switch entity from the previous episode persists with active=False.
                # This causes episodes to start with the switch already "activated", allowing
                # the agent to go straight to the exit door without collecting the switch.
                switch_entity.active = True

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

            # Move exit door (entity type 4)
            # Exit door is referenced as .parent from exit switch
            if exit_switches and hasattr(exit_switches[-1], "parent"):
                door_entity = exit_switches[-1].parent
                if door_entity:
                    curriculum_pos = self.get_curriculum_exit_position()

                    # Log entity movement
                    old_pos = (door_entity.xpos, door_entity.ypos)
                    door_entity.xpos = curriculum_pos[0]
                    door_entity.ypos = curriculum_pos[1]

                    # CRITICAL FIX: Update grid placement for collision detection
                    old_cell = door_entity.cell
                    new_cell = clamp_cell(
                        math.floor(curriculum_pos[0] / 24),
                        math.floor(curriculum_pos[1] / 24),
                    )
                    if new_cell != old_cell:
                        # Remove from old cell
                        if (
                            old_cell in sim.grid_entity
                            and door_entity in sim.grid_entity[old_cell]
                        ):
                            sim.grid_entity[old_cell].remove(door_entity)
                        # Update entity's cell attribute
                        door_entity.cell = new_cell
                        # Add to new cell
                        if new_cell not in sim.grid_entity:
                            sim.grid_entity[new_cell] = []
                        sim.grid_entity[new_cell].append(door_entity)
                        logger.debug(
                            f"Updated exit door grid cell: {old_cell} → {new_cell}"
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
                else:
                    logger.warning("Exit switch has no parent (exit door) reference")
            else:
                logger.warning("No exit switch found to access exit door")

        except Exception as e:
            logger.error(f"Failed to apply curriculum entity positions: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def update_from_episode(self, switch_activated: bool, completed: bool) -> None:
        """Update success tracking and advance stages if thresholds reached.

        Phased progression:
        - Phase 1: Track switch activation rate, advance switch position
        - Phase 2: Track completion rate, advance exit position

        Called at episode end by BaseEnvironment._build_episode_info().

        Args:
            switch_activated: Whether switch was activated this episode
            completed: Whether level was completed this episode
        """
        self.state.episode_count += 1

        # Track outcomes in rolling windows
        self._recent_switch_activations.append(1 if switch_activated else 0)
        self._recent_completions.append(1 if completed else 0)

        # Update total counts
        if switch_activated:
            self.state.switch_activation_count += 1
        if completed:
            self.state.completion_count += 1

        # Check for stage advancement (phased progression)
        # Phase 1: Advances switch based on switch activation rate
        # Phase 2: Advances exit based on completion rate
        self._advance_stage_if_ready("curriculum")

    def _advance_stage_if_ready(self, _unused: str = "curriculum") -> None:
        """Check if ready to advance to next stage based on completion rate.

        Sliding window advancement:
        - Advance unified_stage when completion rate exceeds threshold
        - Both switch and exit advance together maintaining interval spacing
        - No phase transitions - continuous progression

        Advancement criteria:
        - Rolling completion rate > advancement_threshold (default 50%)
        - Not already at final stage
        - Sufficient episodes in rolling window (at least 20)
        """
        # Need sufficient episodes for reliable rate estimate
        if len(self._recent_completions) < 20:
            return

        # Calculate completion rate
        completion_rate = sum(self._recent_completions) / len(self._recent_completions)

        # Check if already at final stage
        if self.state.unified_stage >= self._num_stages - 1:
            # Already at final stage - fully learned!
            return

        # Advance stage if completion rate exceeds threshold
        if completion_rate >= self.config.advancement_threshold:
            old_stage = self.state.unified_stage
            self.state.unified_stage += 1
            self._cache_needs_rebuild = True  # Signal PBRS cache rebuild

            # Calculate distances for logging
            interval = self.config.stage_distance_interval
            new_switch_dist = min(
                self.state.unified_stage * interval, self._spawn_to_switch_distance
            )
            new_exit_dist = min(
                (self.state.unified_stage + 1) * interval, self._combined_distance
            )

            # Clear rolling window for fresh rate at new difficulty
            self._recent_completions.clear()

            logger.info(
                f"🎓 Goal curriculum advanced: Stage {old_stage} → {self.state.unified_stage} | "
                f"Switch: {new_switch_dist:.0f}px, Exit: {new_exit_dist:.0f}px | "
                f"Completion rate: {completion_rate:.1%} | "
                f"Episodes: {self.state.episode_count}"
            )

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get curriculum state information for episode info dict.

        Returns:
            Dictionary with curriculum metadata for Go-Explore and route visualization
        """
        # Calculate current success rates
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
            "unified_stage": self.state.unified_stage,
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
            "episode_count": self.state.episode_count,
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

        # Clear rolling windows but keep unified stage
        # (progression can persist across levels if desired)
        self._recent_switch_activations.clear()
        self._recent_completions.clear()

        logger.info(
            f"Goal curriculum reset for new level "
            f"(unified_stage={self.state.unified_stage} preserved)"
        )
