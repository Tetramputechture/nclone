"""
Action replayer for deterministic checkpoint restoration.

Replays action sequences to restore game state to any checkpoint.
Since N++ physics is fully deterministic, replaying the same action
sequence from spawn will always produce identical results.

This module provides:
- Fast action replay to reach checkpoint states
- Position validation after replay
- Efficient batch replay for multiple checkpoints
"""

import logging
from typing import Tuple, Optional, List, TYPE_CHECKING

from .state_checkpoint import (
    StateCheckpoint,
    CheckpointValidationResult,
    POSITION_VALIDATION_THRESHOLD,
)

if TYPE_CHECKING:
    from .nsim import Simulator

logger = logging.getLogger(__name__)


# Action to input mapping (matches gym environment action space)
ACTION_TO_INPUTS = [
    (0, 0),  # 0: NOOP
    (-1, 0),  # 1: LEFT
    (1, 0),  # 2: RIGHT
    (0, 1),  # 3: JUMP
    (-1, 1),  # 4: JUMP+LEFT
    (1, 1),  # 5: JUMP+RIGHT
]


class ActionReplayer:
    """
    Replay action sequences deterministically to restore checkpoint states.

    This class leverages N++ physics determinism to restore any checkpoint
    by replaying the action sequence that led to it. This is more robust
    than full state serialization as it uses the game's native physics.

    IMPORTANT: When checkpoints are created with frame_skip > 1, the replay
    must also use the same frame_skip value to reach the correct state.
    Each action in the sequence was executed for frame_skip physics ticks.

    Usage:
        replayer = ActionReplayer(frame_skip=4)  # Match training frame_skip
        success = replayer.replay_to_checkpoint(sim, checkpoint)
        if success:
            # sim is now at checkpoint state
    """

    def __init__(self, validate_position: bool = True, frame_skip: int = 1):
        """
        Initialize the action replayer.

        Args:
            validate_position: Whether to validate position after replay.
                              Disable for performance in production.
            frame_skip: Number of physics ticks per action (must match training).
                       Default is 1 for raw replay; use 4 for typical training.
        """
        self.validate_position = validate_position
        self.frame_skip = max(1, frame_skip)

        # Statistics
        self.replays_attempted = 0
        self.replays_successful = 0
        self.total_actions_replayed = 0
        self.total_physics_frames = 0
        self.validation_failures = 0

    def replay_to_checkpoint(
        self,
        sim: "Simulator",
        checkpoint: StateCheckpoint,
        skip_validation: bool = False,
    ) -> CheckpointValidationResult:
        """
        Reset simulator and replay actions to reach checkpoint state.

        Args:
            sim: Simulator instance to replay on
            checkpoint: Checkpoint containing action sequence to replay
            skip_validation: If True, skip position validation for speed

        Returns:
            CheckpointValidationResult indicating success/failure
        """
        self.replays_attempted += 1

        # Reset simulator to initial spawn state
        sim.reset()

        # Replay all actions in sequence with frame_skip
        # CRITICAL: Each action must execute frame_skip physics ticks to match training
        action_sequence = checkpoint.action_sequence
        for action in action_sequence:
            hor_input, jump_input = self._action_to_inputs(action)
            for _ in range(self.frame_skip):
                sim.tick(hor_input, jump_input)

        self.total_actions_replayed += len(action_sequence)
        self.total_physics_frames += len(action_sequence) * self.frame_skip

        # Validate position if requested
        if self.validate_position and not skip_validation:
            result = self._validate_checkpoint_position(sim, checkpoint)

            if result.success:
                self.replays_successful += 1
            else:
                self.validation_failures += 1
                logger.warning(
                    f"Checkpoint validation failed: expected {checkpoint.ninja_position}, "
                    f"got {result.actual_position}, error={result.position_error:.4f}px"
                )

            return result
        else:
            # Skip validation - assume success
            self.replays_successful += 1
            return CheckpointValidationResult(
                success=True,
                position_error=0.0,
                expected_position=checkpoint.ninja_position,
                actual_position=self._get_ninja_position(sim),
            )

    def replay_partial(
        self,
        sim: "Simulator",
        checkpoint: StateCheckpoint,
        num_actions: int,
    ) -> Tuple[float, float]:
        """
        Replay only a portion of the checkpoint's action sequence.

        Useful for testing or exploring variations from a checkpoint.

        Args:
            sim: Simulator instance
            checkpoint: Checkpoint containing action sequence
            num_actions: Number of actions to replay (from start)

        Returns:
            Final ninja position (x, y)
        """
        sim.reset()

        actions_to_replay = checkpoint.action_sequence[:num_actions]
        for action in actions_to_replay:
            hor_input, jump_input = self._action_to_inputs(action)
            for _ in range(self.frame_skip):
                sim.tick(hor_input, jump_input)

        return self._get_ninja_position(sim)

    def replay_with_variation(
        self,
        sim: "Simulator",
        checkpoint: StateCheckpoint,
        variation_actions: List[int],
        variation_start: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Replay checkpoint actions, then apply variation actions.

        Useful for Go-Explore "go" phase where we explore from checkpoints.

        Args:
            sim: Simulator instance
            checkpoint: Checkpoint to start from
            variation_actions: New actions to try after reaching checkpoint
            variation_start: If provided, start variations at this index
                           (truncates checkpoint actions)

        Returns:
            Final ninja position (x, y)
        """
        sim.reset()

        # Determine how many checkpoint actions to replay
        if variation_start is not None:
            base_actions = checkpoint.action_sequence[:variation_start]
        else:
            base_actions = checkpoint.action_sequence

        # Replay base actions with frame_skip
        for action in base_actions:
            hor_input, jump_input = self._action_to_inputs(action)
            for _ in range(self.frame_skip):
                sim.tick(hor_input, jump_input)

        # Apply variation actions with frame_skip
        for action in variation_actions:
            hor_input, jump_input = self._action_to_inputs(action)
            for _ in range(self.frame_skip):
                sim.tick(hor_input, jump_input)

        return self._get_ninja_position(sim)

    def estimate_replay_time(self, checkpoint: StateCheckpoint) -> float:
        """
        Estimate time to replay checkpoint in seconds.

        Based on empirical measurements of tick() performance.
        Assumes ~0.05ms per tick on average hardware.

        Args:
            checkpoint: Checkpoint to estimate replay time for

        Returns:
            Estimated replay time in seconds
        """
        actions_count = len(checkpoint.action_sequence)
        # Empirical: ~0.05ms per tick, plus ~0.5ms for reset
        return (actions_count * 0.00005) + 0.0005

    def _action_to_inputs(self, action: int) -> Tuple[int, int]:
        """
        Convert action index to (horizontal_input, jump_input) tuple.

        Args:
            action: Action index (0-5)

        Returns:
            (hor_input, jump_input) tuple
        """
        if 0 <= action < len(ACTION_TO_INPUTS):
            return ACTION_TO_INPUTS[action]
        else:
            logger.warning(f"Invalid action {action}, defaulting to NOOP")
            return (0, 0)

    def _get_ninja_position(self, sim: "Simulator") -> Tuple[float, float]:
        """Get current ninja position from simulator."""
        if sim.ninja is not None:
            return (sim.ninja.xpos, sim.ninja.ypos)
        return (0.0, 0.0)

    def _get_ninja_velocity(self, sim: "Simulator") -> Tuple[float, float]:
        """Get current ninja velocity from simulator."""
        if sim.ninja is not None:
            return (sim.ninja.xspeed, sim.ninja.yspeed)
        return (0.0, 0.0)

    def _validate_checkpoint_position(
        self,
        sim: "Simulator",
        checkpoint: StateCheckpoint,
    ) -> CheckpointValidationResult:
        """
        Validate that simulator position matches checkpoint.

        Args:
            sim: Simulator after replay
            checkpoint: Expected checkpoint state

        Returns:
            CheckpointValidationResult with validation details
        """
        actual_pos = self._get_ninja_position(sim)
        expected_pos = checkpoint.ninja_position

        # Calculate position error (Euclidean distance)
        dx = actual_pos[0] - expected_pos[0]
        dy = actual_pos[1] - expected_pos[1]
        error = (dx * dx + dy * dy) ** 0.5

        success = error < POSITION_VALIDATION_THRESHOLD

        return CheckpointValidationResult(
            success=success,
            position_error=error,
            expected_position=expected_pos,
            actual_position=actual_pos,
            error_message=None if success else f"Position mismatch: {error:.6f}px",
        )

    def get_statistics(self) -> dict:
        """Get replay statistics."""
        success_rate = (
            self.replays_successful / self.replays_attempted
            if self.replays_attempted > 0
            else 0.0
        )

        return {
            "replays_attempted": self.replays_attempted,
            "replays_successful": self.replays_successful,
            "total_actions_replayed": self.total_actions_replayed,
            "total_physics_frames": self.total_physics_frames,
            "frame_skip": self.frame_skip,
            "validation_failures": self.validation_failures,
            "success_rate": success_rate,
        }

    def reset_statistics(self):
        """Reset replay statistics."""
        self.replays_attempted = 0
        self.replays_successful = 0
        self.total_actions_replayed = 0
        self.total_physics_frames = 0
        self.validation_failures = 0


class BatchActionReplayer:
    """
    Efficiently replay multiple checkpoints in batch.

    Optimized for cases where we need to evaluate many checkpoints,
    such as selecting the best checkpoint to reset to.
    """

    def __init__(
        self, base_replayer: Optional[ActionReplayer] = None, frame_skip: int = 1
    ):
        """
        Initialize batch replayer.

        Args:
            base_replayer: ActionReplayer to use, creates new one if None
            frame_skip: Number of physics ticks per action (must match training)
        """
        self.replayer = base_replayer or ActionReplayer(
            validate_position=False, frame_skip=frame_skip
        )

    def find_checkpoint_with_min_replay_cost(
        self,
        checkpoints: List[StateCheckpoint],
        max_replay_frames: Optional[int] = None,
    ) -> Optional[StateCheckpoint]:
        """
        Find checkpoint with minimum replay cost (fewest actions).

        Args:
            checkpoints: List of checkpoints to consider
            max_replay_frames: Maximum frames to consider replaying

        Returns:
            Checkpoint with minimum replay cost, or None if empty
        """
        if not checkpoints:
            return None

        valid_checkpoints = checkpoints
        if max_replay_frames is not None:
            valid_checkpoints = [
                cp for cp in checkpoints if cp.frame_count <= max_replay_frames
            ]

        if not valid_checkpoints:
            return None

        return min(valid_checkpoints, key=lambda cp: len(cp.action_sequence))

    def find_checkpoint_closest_to_goal(
        self,
        checkpoints: List[StateCheckpoint],
    ) -> Optional[StateCheckpoint]:
        """
        Find checkpoint closest to goal.

        Args:
            checkpoints: List of checkpoints to consider

        Returns:
            Checkpoint with minimum distance to goal, or None if empty
        """
        if not checkpoints:
            return None

        return min(checkpoints, key=lambda cp: cp.distance_to_goal)

    def find_checkpoint_with_max_novelty(
        self,
        checkpoints: List[StateCheckpoint],
    ) -> Optional[StateCheckpoint]:
        """
        Find checkpoint with highest novelty.

        Args:
            checkpoints: List of checkpoints to consider

        Returns:
            Checkpoint with maximum novelty score, or None if empty
        """
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda cp: cp.novelty)
