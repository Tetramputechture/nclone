"""Physics simulation worker for kinodynamic database construction.

Uses actual NPlayHeadless simulator to explore (position, velocity) states
and discover reachable nodes with perfect physics accuracy.

This module implements the heavy offline computation that makes O(1) runtime
queries possible during training.
"""

import logging
from typing import Dict, Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class KinodynamicStateSimulator:
    """Simulates from (node, velocity) states to find reachable nodes.

    Uses actual N++ physics simulation (NPlayHeadless) for perfect accuracy.
    Explores state space using strategic action sequences.
    """

    def __init__(
        self,
        max_simulation_frames: int = 60,
        action_sequence_strategies: Optional[List[str]] = None,
    ):
        """Initialize state simulator.

        Args:
            max_simulation_frames: Maximum frames to simulate forward
            action_sequence_strategies: List of strategy names to use
        """
        self.max_simulation_frames = max_simulation_frames

        # Define action exploration strategies
        if action_sequence_strategies is None:
            action_sequence_strategies = [
                "hold_single",  # Hold single action for duration
                "jump_and_hold",  # Jump + hold direction
                "momentum_build",  # Build momentum then jump
                "noop",  # Just let physics evolve
            ]

        self.strategies = action_sequence_strategies

    def explore_from_state(
        self,
        src_node: Tuple[int, int],
        initial_velocity: Tuple[float, float],
        nplay_headless: Any,
        spatial_hash: Any,
        adjacency: Dict,
    ) -> Dict[Tuple[int, int], float]:
        """Explore reachable nodes from (src_node, initial_velocity) using simulation.

        Args:
            src_node: Starting node position
            initial_velocity: Starting velocity (vx, vy)
            nplay_headless: NPlayHeadless instance for simulation
            spatial_hash: SpatialHash for finding nodes from positions
            adjacency: Graph adjacency for node lookup

        Returns:
            Dict mapping dst_node -> minimum_cost (in frames)
        """
        from ...graph.reachability.pathfinding_utils import find_ninja_node

        reachable = {}

        # Generate action sequences to try
        action_sequences = self._generate_action_sequences()

        for action_seq in action_sequences:
            # Reset simulator to initial state
            self._set_ninja_state(nplay_headless, src_node, initial_velocity)

            # Simulate action sequence
            for frame_idx, action in enumerate(action_seq):
                # Check if ninja died or won (abort simulation)
                if nplay_headless.ninja_has_died() or nplay_headless.ninja_has_won():
                    break

                # Execute action
                hoz_input, jump_input = self._action_to_inputs(action)
                nplay_headless.tick(hoz_input, jump_input)

                # Find current node
                current_pos = nplay_headless.ninja_position()
                current_node = find_ninja_node(
                    (int(current_pos[0]), int(current_pos[1])),
                    adjacency,
                    spatial_hash=spatial_hash,
                    ninja_radius=10.0,
                )

                if current_node is not None and current_node != src_node:
                    # Record reachable node with cost
                    cost = frame_idx + 1  # Frames taken to reach

                    if current_node not in reachable or cost < reachable[current_node]:
                        reachable[current_node] = cost

        return reachable

    def _generate_action_sequences(self) -> List[List[int]]:
        """Generate strategic action sequences to explore state space.

        Returns diverse sequences that cover:
        - Momentum building (left/right for 10-30 frames)
        - Jumping (immediate, delayed, with direction)
        - Passive evolution (NOOP)
        - Direction changes

        Returns:
            List of action sequences (each sequence is list of action indices)
        """
        sequences = []

        # Strategy 1: Hold single action for duration
        for action in [0, 1, 2, 3, 4, 5]:  # All 6 actions
            for duration in [10, 20, 30, 40, 60]:
                sequences.append([action] * duration)

        # Strategy 2: Build momentum then jump
        for build_dir in [1, 2]:  # LEFT, RIGHT
            for build_frames in [10, 20, 30]:
                for jump_action in [3, 4, 5]:  # JUMP, JUMP+LEFT, JUMP+RIGHT
                    seq = [build_dir] * build_frames + [jump_action] * 20
                    sequences.append(seq)

        # Strategy 3: Jump and hold direction
        for jump_action in [4, 5]:  # JUMP+LEFT, JUMP+RIGHT
            sequences.append([jump_action] * 40)

        # Strategy 4: Alternating (for exploring wall interactions)
        sequences.append([1, 3] * 20)  # LEFT, JUMP, LEFT, JUMP...
        sequences.append([2, 3] * 20)  # RIGHT, JUMP, RIGHT, JUMP...

        # Strategy 5: NOOP (passive physics evolution)
        sequences.append([0] * 60)

        # Strategy 6: Short bursts (for precise control)
        for action in [1, 2, 4, 5]:
            seq = [action] * 5 + [0] * 10  # Burst then NOOP
            sequences.append(seq * 3)  # Repeat pattern

        logger.debug(f"Generated {len(sequences)} action sequences for exploration")
        return sequences

    def _set_ninja_state(
        self,
        nplay_headless: Any,
        position: Tuple[int, int],
        velocity: Tuple[float, float],
    ) -> None:
        """Set ninja to specific position and velocity state.

        Args:
            nplay_headless: NPlayHeadless instance
            position: Ninja position (x, y)
            velocity: Ninja velocity (vx, vy)
        """
        # Reset simulation
        nplay_headless.reset()

        # Set position
        nplay_headless.sim.ninja.xpos = float(position[0])
        nplay_headless.sim.ninja.ypos = float(position[1])

        # Set velocity
        nplay_headless.sim.ninja.xspeed = float(velocity[0])
        nplay_headless.sim.ninja.yspeed = float(velocity[1])

        # Set to appropriate state based on velocity
        if velocity[1] < -0.5:  # Moving upward
            nplay_headless.sim.ninja.state = 3  # Jumping
            nplay_headless.sim.ninja.jump_duration = 0
        elif abs(velocity[0]) > 0.1:  # Moving horizontally
            nplay_headless.sim.ninja.state = 1  # Running
        else:
            nplay_headless.sim.ninja.state = 0  # Immobile

    def _action_to_inputs(self, action: int) -> Tuple[int, int]:
        """Convert action index to (horizontal_input, jump_input).

        Args:
            action: Action index (0-5)

        Returns:
            (horizontal_input, jump_input) where:
                horizontal_input: -1 (left), 0 (none), 1 (right)
                jump_input: 0 (not pressed), 1 (pressed)
        """
        if action == 0:  # NOOP
            return 0, 0
        elif action == 1:  # Left
            return -1, 0
        elif action == 2:  # Right
            return 1, 0
        elif action == 3:  # Jump
            return 0, 1
        elif action == 4:  # Jump + Left
            return -1, 1
        elif action == 5:  # Jump + Right
            return 1, 1
        else:
            return 0, 0
