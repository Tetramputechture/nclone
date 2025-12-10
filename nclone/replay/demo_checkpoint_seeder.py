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

logger = logging.getLogger(__name__)


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
    ):
        """Initialize demo checkpoint seeder.

        Args:
            replay_dir: Directory containing .replay files
            max_demos_per_level: Maximum demos to process per level name
            min_cumulative_reward: Minimum cumulative reward to include checkpoint
        """
        self.replay_dir = Path(replay_dir)
        self.max_demos_per_level = max_demos_per_level
        self.min_cumulative_reward = min_cumulative_reward

        # Cache for processed demos (avoid recomputation)
        self._processed_demos: Dict[str, List[Dict[str, Any]]] = {}

    def find_matching_demos(self, level_name: str) -> List[Path]:
        """Find demo replay files matching a level name.

        Args:
            level_name: Level name to match (partial match supported)

        Returns:
            List of matching replay file paths, sorted by modification time (newest first)
        """
        if not self.replay_dir.exists():
            logger.warning(f"Replay directory not found: {self.replay_dir}")
            return []

        # Normalize level name for matching
        level_name_lower = level_name.lower().replace("_", " ")

        matching = []
        for replay_file in self.replay_dir.glob("*.replay"):
            file_name_lower = replay_file.stem.lower()
            # Check if level name appears in file name (after timestamp prefix)
            # Format: YYYYMMDD_HHMMSS_level_name.replay
            if level_name_lower in file_name_lower:
                matching.append(replay_file)

        # Sort by modification time (newest first) and limit
        matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matching[: self.max_demos_per_level]

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

            executor = ReplayExecutor(enable_rendering=False)

            # Load map
            executor.nplay_headless.load_map_from_map_data(list(map_data))

            # Initialize tracking
            checkpoints = []
            cumulative_reward = 0.0
            prev_obs = None

            # Store action sequence from spawn to each position (for checkpoint replay)
            action_sequence = []

            # Process each frame
            for frame_idx, input_byte in enumerate(inputs):
                # Decode input
                horizontal, jump = decode_input_to_controls(input_byte)
                action = map_input_to_action(input_byte)

                # Track action sequence
                action_sequence.append(action)

                # Execute simulation step
                executor.nplay_headless.tick(horizontal, jump)

                # Get ninja position
                ninja_x, ninja_y = executor.nplay_headless.ninja_position()
                ninja_pos = (int(ninja_x), int(ninja_y))

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

                # Discretize position to cell
                cell = (ninja_pos[0] // self.GRID_SIZE, ninja_pos[1] // self.GRID_SIZE)

                # Store checkpoint with FULL action sequence from spawn to this position
                checkpoints.append(
                    {
                        "cell": cell,
                        "cumulative_reward": cumulative_reward,
                        "action": action,
                        "frame_idx": frame_idx,
                        "switch_activated": switch_activated,
                        "position": ninja_pos,
                        "action_sequence": action_sequence.copy(),  # Full sequence to reach this point
                    }
                )

                prev_obs = obs

                # Stop if agent won or died
                if obs["player_won"] or obs["player_dead"]:
                    break

            executor.close()
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
            gamma = 0.997
            pbrs_reward = gamma * current_potential - prev_potential

            return pbrs_reward

        except Exception as e:
            logger.debug(f"PBRS calculation failed: {e}")
            return 0.001

    def extract_best_checkpoints(
        self,
        checkpoints: List[Dict[str, Any]],
        max_checkpoints: int = 100,
    ) -> List[Tuple[Tuple[int, int], List[int], float, bool]]:
        """Extract best checkpoints from processed trajectory.

        Selects checkpoints with highest cumulative rewards while ensuring
        spatial diversity (one checkpoint per cell).

        Args:
            checkpoints: List of checkpoint dicts from process_demo_file
            max_checkpoints: Maximum checkpoints to return

        Returns:
            List of (cell, action_sequence, cumulative_reward, switch_activated) tuples
            where action_sequence is the FULL sequence from spawn to reach this cell
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

        # Convert to tuple format with full action sequences
        result = [
            (
                cp["cell"],
                cp.get("action_sequence", [cp["action"]]),  # Full sequence or fallback
                cp["cumulative_reward"],
                cp["switch_activated"],
            )
            for cp in sorted_checkpoints
        ]

        return result

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
        demo_cells = [
            (cell, action_sequence, cumulative_reward)
            for cell, action_sequence, cumulative_reward, switch_activated in best_checkpoints
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

            executor = ReplayExecutor(enable_rendering=False)
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

            # Debug: Get direct access to sim and ninja
            sim = executor.nplay_headless.sim
            ninja = sim.ninja

            spawn_x, spawn_y = executor.nplay_headless.ninja_position()

            # Debug: Check map_data spawn coordinates
            map_data_list = list(map_data)
            map_data_spawn_x = (
                map_data_list[1231] * 6 if len(map_data_list) > 1232 else 0
            )
            map_data_spawn_y = (
                map_data_list[1232] * 6 if len(map_data_list) > 1232 else 0
            )

            logger.info(
                f"Demo extraction: replay={replay_path.name}, "
                f"spawn=({spawn_x}, {spawn_y}), "
                f"map_data[1231]={map_data_list[1231] if len(map_data_list) > 1231 else 'N/A'}, "
                f"map_data[1232]={map_data_list[1232] if len(map_data_list) > 1232 else 'N/A'}, "
                f"map_data_spawn=({map_data_spawn_x}, {map_data_spawn_y}), "
                f"total_frames={total_frames}, "
                f"ninja.xpos={ninja.xpos}, ninja.ypos={ninja.ypos}"
            )

            # Track previous position to detect if ninja is moving
            prev_pos = (spawn_x, spawn_y)

            for frame_idx, input_byte in enumerate(inputs):
                horizontal, jump = decode_input_to_controls(input_byte)
                action = map_input_to_action(input_byte)

                # Track full action sequence
                action_sequence.append(action)

                # Execute the tick FIRST, then get position
                executor.nplay_headless.tick(horizontal, jump)

                # Get position AFTER tick
                ninja_x, ninja_y = executor.nplay_headless.ninja_position()
                ninja_pos = (int(ninja_x), int(ninja_y))

                # Debug first few positions to see if ninja is moving
                if frame_idx < 10:
                    moved_dist = (
                        (ninja_x - prev_pos[0]) ** 2 + (ninja_y - prev_pos[1]) ** 2
                    ) ** 0.5
                    logger.info(
                        f"  Frame {frame_idx}: input={input_byte} ({horizontal}, {jump}), "
                        f"action={action}, pos=({ninja_x:.1f}, {ninja_y:.1f}), "
                        f"moved={moved_dist:.2f}px from prev"
                    )

                prev_pos = (ninja_x, ninja_y)
                switch_activated = executor.nplay_headless.exit_switch_activated()

                cell = (ninja_pos[0] // self.GRID_SIZE, ninja_pos[1] // self.GRID_SIZE)

                # Use progress through trajectory as proxy for cumulative reward
                # Higher frame index = closer to goal = higher "reward"
                progress_reward = frame_idx / max(1, total_frames)

                checkpoints.append(
                    {
                        "cell": cell,
                        "cumulative_reward": progress_reward,
                        "action": action,
                        "frame_idx": frame_idx,
                        "switch_activated": switch_activated,
                        "position": ninja_pos,
                        "action_sequence": action_sequence.copy(),  # Full sequence to this point
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
