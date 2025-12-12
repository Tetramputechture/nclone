"""Demo Checkpoint Seeder for Go-Explore Archive.

Seeds the Go-Explore checkpoint archive with cells extracted from expert demonstrations,
using actual cumulative rewards computed by replaying the demos through the reward system.

This provides accurate checkpoint values that match what the agent would experience
during training, ensuring proper UCB selection priority.

Key Benefits:
- Accurate cumulative rewards (not synthetic proxies)
- PBRS-aligned priorities (graph distance incorporated)
- Consistent with Go-Explore's UCB selection mechanism
- Proper frontier identification from expert trajectories

Usage:
    seeder = DemoCheckpointSeeder(replay_dir="/path/to/replays")
    # During training start, when environment is available:
    seeded_count = seeder.seed_archive(go_explore_callback, level_map_data)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import PBRS_GAMMA to ensure consistency with training reward calculation
from nclone.gym_environment.reward_calculation.reward_constants import PBRS_GAMMA

logger = logging.getLogger(__name__)

# Minimum distance from exit door to create checkpoints (prevents trivial episodes)
MIN_EXIT_DISTANCE = 72.0  # pixels


class DemoCheckpointSeeder:
    """Seeds Go-Explore archive with checkpoints from expert demonstrations.

    Replays demonstrations to compute actual cumulative rewards at each frame,
    then extracts checkpoint cells with their true values for archive seeding.
    """

    # Grid size for checkpoint discretization (matches Go-Explore)
    GRID_SIZE = 12

    def __init__(
        self,
        replay_dir: str,
        max_demos_per_level: int = 10,
        min_cumulative_reward: float = 0.0,
        frame_skip: int = 1,
    ):
        """Initialize demo checkpoint seeder.

        Args:
            replay_dir: Directory containing .replay files
            max_demos_per_level: Maximum demos to process per level name
            min_cumulative_reward: Minimum cumulative reward to include checkpoint
            frame_skip: Frame skip used during training (for action subsampling).
                Demos are recorded at 60fps (1 action/frame), but training may use
                frame_skip=4 (1 action/4 frames). We subsample demo actions to match.
        """
        self.replay_dir = Path(replay_dir)
        self.max_demos_per_level = max_demos_per_level
        self.min_cumulative_reward = min_cumulative_reward
        self.frame_skip = max(1, frame_skip)

        # Cache for processed demos (avoid recomputation)
        self._processed_demos: Dict[str, List[Dict[str, Any]]] = {}

    def find_matching_demos(self, level_name: str) -> List[Path]:
        """Find demo replay files matching a level name (EXACT match).

        Parses filename format: YYYYMMDD_HHMMSS_level_name.replay
        and does exact matching on the level_name portion.

        Args:
            level_name: Level name to match (exact match after normalization)

        Returns:
            List of matching replay file paths, sorted by modification time (newest first)
        """
        if not self.replay_dir.exists():
            logger.warning(f"Replay directory not found: {self.replay_dir}")
            return []

        # Normalize level name for matching (unify _ and - separators)
        level_name_lower = level_name.lower().replace("_", "-")

        matching = []
        for replay_file in self.replay_dir.glob("*.replay"):
            # Parse filename: YYYYMMDD_HHMMSS_level_name.replay
            file_stem = replay_file.stem
            parts = file_stem.split("_", 2)  # Split on first 2 underscores

            if len(parts) >= 3:
                # Extract level name (third part after timestamp)
                file_level_name = parts[2].lower().replace("_", "-")
            else:
                # Fallback: entire filename if not in expected format
                file_level_name = file_stem.lower().replace("_", "-")

            # Exact match (after normalization)
            if file_level_name == level_name_lower:
                matching.append(replay_file)
                logger.debug(
                    f"Matched demo: {replay_file.name} for level: {level_name}"
                )

        # Sort by modification time (newest first) and limit
        matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        limited = matching[: self.max_demos_per_level]

        logger.info(
            f"Found {len(limited)} demo(s) for level '{level_name}' "
            f"(from {len(matching)} total matches)"
        )
        return limited

    def _should_filter_checkpoint(
        self,
        ninja_pos: Tuple[float, float],
        exit_door_pos: Tuple[float, float],
        switch_activated: bool,
    ) -> bool:
        """Check if checkpoint should be filtered due to proximity to exit.

        Filters checkpoints that are too close to the exit door after switch
        activation to prevent trivially short episodes.

        Args:
            ninja_pos: (x, y) ninja position
            exit_door_pos: (x, y) exit door position
            switch_activated: Whether switch is activated

        Returns:
            True if checkpoint should be filtered (too close to exit), False otherwise
        """
        # Before switch activation, keep all checkpoints
        if not switch_activated:
            return False

        # After switch activation, filter checkpoints too close to exit
        dx = ninja_pos[0] - exit_door_pos[0]
        dy = ninja_pos[1] - exit_door_pos[1]
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < MIN_EXIT_DISTANCE:
            logger.debug(
                f"Filtered checkpoint at {ninja_pos} "
                f"(distance to exit: {distance:.1f}px < {MIN_EXIT_DISTANCE}px)"
            )
            return True

        return False

    def process_demo_file(
        self,
        replay_path: Path,
        reward_calculator: Any,
        level_data: Any,
        graph_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process a single demo file and extract checkpoints with cumulative rewards.

        Args:
            replay_path: Path to .replay file
            reward_calculator: Initialized MainRewardCalculator instance
            level_data: LevelData for the level
            graph_data: Graph data from graph builder

        Returns:
            List of checkpoint dicts with keys:
            - cell: (x, y) discretized position
            - cumulative_reward: Total reward accumulated to this point
            - action: Action taken at this cell
            - frame_idx: Frame number in trajectory
            - switch_activated: Whether switch was active at this point
        """
        from nclone.replay.gameplay_recorder import CompactReplay

        try:
            # Parse replay file
            with open(replay_path, "rb") as f:
                replay_data = f.read()

            replay = CompactReplay.from_binary(replay_data, episode_id=replay_path.stem)
            inputs = replay.input_sequence
            map_data = replay.map_data

            if not map_data or not inputs:
                logger.warning(f"Incomplete replay data: {replay_path}")
                return []

            # Use ReplayExecutor to simulate the trajectory
            from nclone.replay.replay_executor import (
                ReplayExecutor,
                decode_input_to_controls,
                map_input_to_action,
            )

            executor = ReplayExecutor(enable_rendering=False, frame_skip=1)

            # Load map
            executor.nplay_headless.load_map_from_map_data(list(map_data))

            # Initialize tracking
            checkpoints = []
            cumulative_reward = 0.0
            prev_obs = None

            # Store FULL action sequence from spawn to each position
            # Demos are per-frame control, so we keep all actions for accurate replay
            action_sequence = []

            # Get spawn position for debug logging
            spawn_pos = executor.nplay_headless.ninja_position()

            # DEBUG: Check if verbose logging is enabled via environment variable
            import os
            _verbose_debug = os.environ.get("DEMO_CHECKPOINT_DEBUG", "").lower() in ("1", "true", "yes")

            if _verbose_debug:
                logger.error(
                    f"[EXTRACT_DEBUG] Starting demo extraction: "
                    f"file={replay_path.name}, total_frames={len(inputs)}, "
                    f"spawn_pos=({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f})"
                )

            # Track first 10 positions for debug comparison
            _debug_positions = []

            # Process each frame (demos use per-frame control, always frame_skip=1)
            for frame_idx, input_byte in enumerate(inputs):
                # Decode input
                horizontal, jump = decode_input_to_controls(input_byte)
                action = map_input_to_action(input_byte)

                # Track FULL action sequence (all frames, preserve demo control)
                action_sequence.append(action)

                # Execute simulation step
                executor.nplay_headless.tick(horizontal, jump)

                # Get ninja position (keep as floats for accurate replay validation)
                ninja_x, ninja_y = executor.nplay_headless.ninja_position()
                ninja_pos = (float(ninja_x), float(ninja_y))

                # Track first 10 positions for debug comparison with replay
                if len(action_sequence) <= 10:
                    _debug_positions.append((len(action_sequence), action, ninja_pos))

                # Check switch state
                switch_activated = executor.nplay_headless.exit_switch_activated()

                # Build minimal observation for reward calculation
                obs = {
                    "player_x": ninja_x,
                    "player_y": ninja_y,
                    "switch_activated": switch_activated,
                    "switch_x": executor.nplay_headless.exit_switch_position()[0],
                    "switch_y": executor.nplay_headless.exit_switch_position()[1],
                    "exit_door_x": executor.nplay_headless.exit_door_position()[0],
                    "exit_door_y": executor.nplay_headless.exit_door_position()[1],
                    "player_won": executor.nplay_headless.ninja_has_won(),
                    "player_dead": executor.nplay_headless.sim.ninja.has_died(),
                }

                # Calculate step reward if we have previous observation
                if prev_obs is not None and reward_calculator is not None:
                    try:
                        # Simplified reward calculation for demo processing
                        # We mainly want PBRS component
                        step_reward = self._calculate_demo_step_reward(
                            prev_obs, obs, reward_calculator, level_data, graph_data
                        )
                        cumulative_reward += step_reward
                    except Exception as e:
                        logger.debug(
                            f"Reward calculation failed at frame {frame_idx}: {e}"
                        )
                        # Use distance-based proxy if reward calc fails
                        cumulative_reward += 0.001  # Small increment per frame

                # Discretize position to cell (ensure integer cell coordinates)
                cell = (
                    int(ninja_pos[0] // self.GRID_SIZE),
                    int(ninja_pos[1] // self.GRID_SIZE),
                )

                # Check if checkpoint should be filtered (too close to exit)
                exit_door_pos = (obs["exit_door_x"], obs["exit_door_y"])
                if not self._should_filter_checkpoint(
                    ninja_pos, exit_door_pos, switch_activated
                ):
                    # Store checkpoint with FULL action sequence from spawn to this position
                    # action_sequence is now subsampled at frame_skip intervals
                    checkpoints.append(
                        {
                            "cell": cell,
                            "cumulative_reward": cumulative_reward,
                            "action": action,
                            "frame_idx": frame_idx,
                            "switch_activated": switch_activated,
                            "position": ninja_pos,
                            "action_sequence": action_sequence.copy(),  # Subsampled sequence
                        }
                    )

                prev_obs = obs

                # Stop if agent won or died
                if obs["player_won"] or obs["player_dead"]:
                    break

            executor.close()

            # DEBUG: Log extraction results for comparison with replay
            if _verbose_debug and _debug_positions:
                pos_log = ", ".join(
                    [f"a{i}:{a}->({p[0]:.1f},{p[1]:.1f})" for i, a, p in _debug_positions]
                )
                logger.error(f"[EXTRACT_DEBUG] First 10 positions: {pos_log}")

                if checkpoints:
                    # Log the last checkpoint (most likely to be selected)
                    last_cp = checkpoints[-1]
                    logger.error(
                        f"[EXTRACT_DEBUG] Final checkpoint: "
                        f"cell={last_cp['cell']}, pos={last_cp['position']}, "
                        f"actions={len(last_cp['action_sequence'])}, "
                        f"first_10_actions={list(last_cp['action_sequence'][:10])}"
                    )

            return checkpoints

        except Exception as e:
            logger.error(f"Error processing demo {replay_path}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return []

    def _calculate_demo_step_reward(
        self,
        prev_obs: Dict[str, Any],
        obs: Dict[str, Any],
        reward_calculator: Any,
        level_data: Any,
        graph_data: Dict[str, Any],
    ) -> float:
        """Calculate simplified step reward for demo trajectory.

        Uses PBRS component as the primary reward signal since that's what
        determines checkpoint value for navigation.

        CRITICAL: Uses PBRS_GAMMA (1.0) to match training reward calculation.
        This ensures demo checkpoint cumulative_reward values are directly
        comparable to training checkpoint values for UCB selection.

        Args:
            prev_obs: Previous observation
            obs: Current observation
            reward_calculator: MainRewardCalculator instance
            level_data: Level data
            graph_data: Graph adjacency data

        Returns:
            Step reward (primarily PBRS component)
        """
        # Get PBRS calculator from reward calculator
        pbrs_calc = getattr(reward_calculator, "pbrs_calculator", None)
        if pbrs_calc is None:
            return 0.001  # Fallback

        try:
            # Calculate PBRS potential difference
            current_pos = (obs["player_x"], obs["player_y"])
            prev_pos = (prev_obs["player_x"], prev_obs["player_y"])

            adjacency = graph_data.get("adjacency", {})

            # Get current goal based on switch state
            if obs["switch_activated"]:
                goal_pos = (obs["exit_door_x"], obs["exit_door_y"])
            else:
                goal_pos = (obs["switch_x"], obs["switch_y"])

            # Calculate potentials
            current_potential = pbrs_calc.objective_distance_potential(
                current_pos, goal_pos, adjacency, level_data, graph_data, obs
            )
            prev_potential = pbrs_calc.objective_distance_potential(
                prev_pos, goal_pos, adjacency, level_data, graph_data, prev_obs
            )

            # PBRS reward: γ * Φ(s') - Φ(s)
            # Use PBRS_GAMMA (1.0) to match training reward calculation
            pbrs_reward = PBRS_GAMMA * current_potential - prev_potential

            return pbrs_reward

        except Exception as e:
            logger.debug(f"PBRS calculation failed: {e}")
            return 0.001

    def extract_best_checkpoints(
        self,
        checkpoints: List[Dict[str, Any]],
        max_checkpoints: int = 100,
    ) -> List[Tuple[Tuple[int, int], List[int], float, bool, Tuple[float, float]]]:
        """Extract best checkpoints from processed trajectory.

        Selects checkpoints with highest cumulative rewards while ensuring
        spatial diversity (one checkpoint per cell).

        Args:
            checkpoints: List of checkpoint dicts from process_demo_file
            max_checkpoints: Maximum checkpoints to return

        Returns:
            List of (cell, action_sequence, cumulative_reward, switch_activated, position) tuples
            where action_sequence is the FULL sequence from spawn to reach this cell
            and position is the actual ninja position at this checkpoint
        """
        if not checkpoints:
            return []

        # Group by cell, keeping highest cumulative reward per cell
        cell_to_best: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for cp in checkpoints:
            cell = cp["cell"]
            if (
                cell not in cell_to_best
                or cp["cumulative_reward"] > cell_to_best[cell]["cumulative_reward"]
            ):
                cell_to_best[cell] = cp

        # Sort by cumulative reward (descending) and take top N
        sorted_checkpoints = sorted(
            cell_to_best.values(),
            key=lambda x: x["cumulative_reward"],
            reverse=True,
        )[:max_checkpoints]

        # Convert to tuple format with full action sequences and position
        result = [
            (
                cp["cell"],
                cp.get("action_sequence", [cp["action"]]),  # Full sequence or fallback
                cp["cumulative_reward"],
                cp["switch_activated"],
                cp.get("position", (0.0, 0.0)),  # Position from same checkpoint
            )
            for cp in sorted_checkpoints
        ]

        return result

    def extract_waypoints_from_checkpoints(
        self,
        checkpoints: List[Dict[str, Any]],
        max_waypoints: int = 10,
        cluster_radius: float = 30.0,
    ) -> List[Tuple[Tuple[float, float], float, str, int, str]]:
        """Extract waypoints from demo checkpoints for adaptive waypoint system.

        Converts demo checkpoint positions to waypoint format suitable for
        AdaptiveWaypointSystem. Filters duplicates within cluster_radius and
        assigns values based on cumulative reward.

        Switch-aware extraction: Assigns phase (pre_switch or post_switch) based on
        checkpoint's switch_activated state. This ensures demo waypoints are only
        active in the appropriate phase.

        Args:
            checkpoints: List of checkpoint dicts with 'position', 'cumulative_reward',
                        and 'switch_activated'
            max_waypoints: Maximum waypoints to extract
            cluster_radius: Distance threshold for merging similar waypoints (pixels)

        Returns:
            List of (position, value, type, discovery_count, phase) tuples compatible with
            AdaptiveWaypointSystem format:
            - position: (x, y) in pixels
            - value: importance score (0.5-2.0)
            - type: "demo" for tracking origin
            - discovery_count: 1 (expert-validated)
            - phase: "pre_switch" or "post_switch"
        """
        if not checkpoints:
            return []

        # Split checkpoints by phase to ensure we get waypoints from BOTH phases
        # Without this, high-reward post_switch checkpoints dominate and we get
        # zero pre_switch waypoints, making early-game guidance impossible
        pre_switch_checkpoints = [
            cp for cp in checkpoints if not cp.get("switch_activated", False)
        ]
        post_switch_checkpoints = [
            cp for cp in checkpoints if cp.get("switch_activated", False)
        ]

        logger.info(
            f"Checkpoint phase breakdown: {len(pre_switch_checkpoints)} pre_switch, "
            f"{len(post_switch_checkpoints)} post_switch"
        )

        # Allocate waypoint quota per phase (minimum 2 per phase if available)
        # This ensures both phases get representation
        min_per_phase = 2
        if len(pre_switch_checkpoints) > 0 and len(post_switch_checkpoints) > 0:
            # Both phases have checkpoints - split quota
            pre_switch_quota = max(min_per_phase, max_waypoints // 2)
            post_switch_quota = max(min_per_phase, max_waypoints - pre_switch_quota)
        elif len(pre_switch_checkpoints) > 0:
            # Only pre_switch checkpoints
            pre_switch_quota = max_waypoints
            post_switch_quota = 0
        else:
            # Only post_switch checkpoints
            pre_switch_quota = 0
            post_switch_quota = max_waypoints

        def extract_phase_waypoints(
            phase_checkpoints: List[Dict], quota: int, phase: str
        ) -> List[Tuple]:
            """Extract waypoints from checkpoints of a single phase."""
            if quota == 0 or not phase_checkpoints:
                return []

            # Sort by cumulative reward (descending) to prioritize high-value positions
            sorted_cps = sorted(
                phase_checkpoints,
                key=lambda cp: cp.get("cumulative_reward", 0.0),
                reverse=True,
            )

            phase_waypoints = []
            for cp in sorted_cps:
                position = cp.get("position")
                if position is None:
                    continue

                # Convert to float tuple for consistency
                wp_pos = (float(position[0]), float(position[1]))

                # Check if too close to existing waypoint (clustering)
                is_duplicate = False
                for existing_pos, *_ in phase_waypoints:
                    dx = wp_pos[0] - existing_pos[0]
                    dy = wp_pos[1] - existing_pos[1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance < cluster_radius:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Assign value based on cumulative reward
                cumulative_reward = cp.get("cumulative_reward", 0.0)
                if cumulative_reward > 0:
                    normalized = min(10.0, cumulative_reward) / 10.0
                    value = 0.75 + normalized * 0.75
                else:
                    value = 0.5

                phase_waypoints.append((wp_pos, value, "demo", 1, phase))

                if len(phase_waypoints) >= quota:
                    break

            return phase_waypoints

        # Extract waypoints from each phase
        pre_switch_waypoints = extract_phase_waypoints(
            pre_switch_checkpoints, pre_switch_quota, "pre_switch"
        )
        post_switch_waypoints = extract_phase_waypoints(
            post_switch_checkpoints, post_switch_quota, "post_switch"
        )

        # Combine waypoints from both phases
        waypoints = pre_switch_waypoints + post_switch_waypoints

        logger.info(
            f"Extracted {len(waypoints)} waypoints from {len(checkpoints)} checkpoints "
            f"(pre_switch: {len(pre_switch_waypoints)}, post_switch: {len(post_switch_waypoints)}, "
            f"cluster_radius={cluster_radius}px)"
        )

        return waypoints

    def seed_go_explore_archive(
        self,
        go_explore_callback: Any,
        level_name: str,
        reward_calculator: Optional[Any] = None,
        level_data: Optional[Any] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Seed Go-Explore archive with checkpoints from demos.

        Args:
            go_explore_callback: GoExploreCallback instance with archive
            level_name: Name of level to find demos for
            reward_calculator: Optional reward calculator for cumulative rewards
            level_data: Optional level data for reward calculation
            graph_data: Optional graph data for reward calculation

        Returns:
            Number of checkpoints seeded
        """
        # Find matching demos
        demo_files = self.find_matching_demos(level_name)
        if not demo_files:
            logger.info(f"No demos found matching level: {level_name}")
            return 0

        logger.info(f"Found {len(demo_files)} demos for level: {level_name}")

        all_checkpoints = []

        # Process each demo
        for demo_file in demo_files:
            if (
                reward_calculator is not None
                and level_data is not None
                and graph_data is not None
            ):
                # Full processing with actual cumulative rewards
                checkpoints = self.process_demo_file(
                    demo_file, reward_calculator, level_data, graph_data
                )
            else:
                # Fallback: use simple position-based extraction
                checkpoints = self._extract_positions_from_demo_simple(demo_file)

            all_checkpoints.extend(checkpoints)

        if not all_checkpoints:
            logger.warning("No valid checkpoints extracted from demos")
            return 0

        # Extract best checkpoints
        best_checkpoints = self.extract_best_checkpoints(all_checkpoints)

        # Convert to format expected by go_explore_callback.seed_from_demonstrations
        # Now includes full action_sequence (not just single action)
        # Note: This path uses 3-tuple format (no position), which is fine for backward compatibility
        demo_cells = [
            (cell, action_sequence, cumulative_reward)
            for cell, action_sequence, cumulative_reward, switch_activated, position in best_checkpoints
        ]

        # Seed the archive
        seeded = go_explore_callback.seed_from_demonstrations(demo_cells)

        logger.info(
            f"Seeded {seeded} checkpoints from {len(demo_files)} demos "
            f"(best cumulative_reward: {best_checkpoints[0][2]:.3f} if available)"
            if best_checkpoints
            else f"Seeded {seeded} checkpoints"
        )

        return seeded

    def _extract_positions_from_demo_simple(
        self, replay_path: Path
    ) -> List[Dict[str, Any]]:
        """Simple position extraction without full reward calculation.

        Fallback when reward calculator is not available.
        Uses frame index as a proxy for cumulative reward.

        Args:
            replay_path: Path to replay file

        Returns:
            List of checkpoint dicts
        """
        from nclone.replay.gameplay_recorder import CompactReplay
        from nclone.replay.replay_executor import (
            ReplayExecutor,
            decode_input_to_controls,
            map_input_to_action,
        )

        try:
            # Parse replay file
            with open(replay_path, "rb") as f:
                replay_data = f.read()

            replay = CompactReplay.from_binary(replay_data, episode_id=replay_path.stem)
            inputs = replay.input_sequence
            map_data = replay.map_data

            logger.info(f"Replay path: {replay_path}")
            logger.info(f"Episode ID: {replay.episode_id}")
            logger.info(f"Map data size: {len(map_data)} bytes")
            logger.info(f"Inputs count: {len(inputs)}")

            if not map_data or not inputs:
                return []

            executor = ReplayExecutor(enable_rendering=False, frame_skip=1)
            executor.nplay_headless.load_map_from_map_data(list(map_data))

            # CRITICAL: Initialize entity extractor after map is loaded
            # This is required for the ninja to be properly spawned
            from nclone.gym_environment.entity_extractor import EntityExtractor

            if executor.entity_extractor is None:
                executor.entity_extractor = EntityExtractor(executor.nplay_headless)

            checkpoints = []
            total_frames = len(inputs)
            action_sequence = []  # Track full action sequence from spawn

            # Get initial spawn position to verify ninja is spawned
            # Access ninja object directly to check if it exists
            if not hasattr(executor.nplay_headless, "sim") or not hasattr(
                executor.nplay_headless.sim, "ninja"
            ):
                logger.error(
                    f"Demo extraction failed: sim or ninja not initialized for {replay_path}"
                )
                return []

            spawn_x, spawn_y = executor.nplay_headless.ninja_position()

            # print(
            #     f"[DEMO EXTRACTION SIMPLE] Starting {replay_path.name}: "
            #     f"total_frames={total_frames}, spawn=({spawn_x}, {spawn_y})"
            # )

            # Track previous position to detect if ninja is moving
            prev_pos = (spawn_x, spawn_y)

            # Track FULL action sequence (demos use per-frame control)
            action_sequence = []

            # Process all frames (demos use per-frame control, no subsampling)
            for frame_idx, input_byte in enumerate(inputs):
                horizontal, jump = decode_input_to_controls(input_byte)
                action = map_input_to_action(input_byte)

                # Track FULL action sequence (all frames, preserve demo control)
                action_sequence.append(action)

                # Execute the tick FIRST, then get position
                executor.nplay_headless.tick(horizontal, jump)

                # Get position AFTER tick (keep as floats for accurate replay validation)
                ninja_x, ninja_y = executor.nplay_headless.ninja_position()
                ninja_pos = (float(ninja_x), float(ninja_y))

                # Debug first few positions
                # if len(action_sequence) <= 10:
                #     moved_dist = (
                #         (ninja_x - prev_pos[0]) ** 2 + (ninja_y - prev_pos[1]) ** 2
                #     ) ** 0.5
                #     print(
                #         f"  [DEMO EXTRACTION] Frame {frame_idx}: "
                #         f"action={action}, pos=({ninja_x:.1f}, {ninja_y:.1f}), "
                #         f"moved={moved_dist:.2f}px"
                #     )

                # prev_pos = (ninja_x, ninja_y)
                switch_activated = executor.nplay_headless.exit_switch_activated()

                # Discretize position to cell (ensure integer cell coordinates)
                cell = (
                    int(ninja_pos[0] // self.GRID_SIZE),
                    int(ninja_pos[1] // self.GRID_SIZE),
                )

                # Use progress through trajectory as proxy for cumulative reward
                # Higher frame index = closer to goal = higher "reward"
                progress_reward = frame_idx / max(1, total_frames)

                # Check if checkpoint should be filtered (too close to exit)
                exit_door_pos = executor.nplay_headless.exit_door_position()
                if not self._should_filter_checkpoint(
                    ninja_pos, exit_door_pos, switch_activated
                ):
                    checkpoints.append(
                        {
                            "cell": cell,
                            "cumulative_reward": progress_reward,
                            "action": action,
                            "frame_idx": frame_idx,
                            "switch_activated": switch_activated,
                            "position": ninja_pos,
                            "action_sequence": action_sequence.copy(),  # FULL sequence (demos use frame_skip=1)
                        }
                    )

                if (
                    executor.nplay_headless.ninja_has_won()
                    or executor.nplay_headless.sim.ninja.has_died()
                ):
                    break

            executor.close()
            return checkpoints

        except Exception as e:
            logger.debug(f"Simple extraction failed for {replay_path}: {e}")
            return []


def seed_archive_from_demos(
    go_explore_callback: Any,
    replay_dir: str,
    level_name: str,
    reward_calculator: Optional[Any] = None,
    level_data: Optional[Any] = None,
    graph_data: Optional[Dict[str, Any]] = None,
) -> int:
    """Convenience function to seed Go-Explore archive from demos.

    Args:
        go_explore_callback: GoExploreCallback instance
        replay_dir: Directory containing .replay files
        level_name: Level name to match demos
        reward_calculator: Optional reward calculator
        level_data: Optional level data
        graph_data: Optional graph data

    Returns:
        Number of checkpoints seeded
    """
    seeder = DemoCheckpointSeeder(replay_dir)
    return seeder.seed_go_explore_archive(
        go_explore_callback,
        level_name,
        reward_calculator,
        level_data,
        graph_data,
    )
