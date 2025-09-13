"""
Main reachability analyzer orchestrating all reachability components.

This module provides the main ReachabilityAnalyzer class that coordinates
position validation, collision detection, physics movement, and game mechanics
to perform comprehensive reachability analysis.
"""

from collections import deque
from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Optional

from ..common import SUB_CELL_SIZE
from ..trajectory_calculator import TrajectoryCalculator
from .position_validator import PositionValidator
from .collision_checker import CollisionChecker
from .physics_movement import PhysicsMovement
from .game_mechanics import GameMechanics


@dataclass
class ReachabilityState:
    """Represents the state of level reachability analysis."""

    reachable_positions: Set[Tuple[int, int]]  # (sub_row, sub_col) positions
    switch_states: Dict[int, bool]  # entity_id -> activated state
    unlocked_areas: Set[Tuple[int, int]]  # Areas unlocked by switches
    subgoals: List[
        Tuple[int, int, str]
    ]  # (sub_row, sub_col, goal_type) for key objectives


class ReachabilityAnalyzer:
    """
    Analyzes level reachability from player perspective.

    This class coordinates multiple specialized components to determine
    which areas of a level are accessible to the player, considering:
    - Jump and fall physics
    - Switch activation and door unlocking
    - One-way platforms and terrain constraints
    - Subgoal identification for hierarchical planning
    """

    def __init__(
        self, trajectory_calculator: TrajectoryCalculator, debug: bool = False
    ):
        """
        Initialize reachability analyzer with physics calculator.

        Args:
            trajectory_calculator: Physics-based trajectory calculator
            debug: Enable debug output (default: False)
        """
        self.trajectory_calculator = trajectory_calculator
        self.debug = debug

        # Initialize component modules
        self.position_validator = PositionValidator(debug=debug)
        self.collision_checker = CollisionChecker(debug=debug)
        self.physics_movement = PhysicsMovement(self.position_validator, debug=debug)
        self.game_mechanics = GameMechanics(debug=debug)

    def analyze_reachability(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None,
    ) -> ReachabilityState:
        """
        Analyze which areas are reachable from ninja starting position.

        Performs iterative BFS-based reachability analysis, with each iteration
        potentially unlocking new areas via switch activation.

        Args:
            level_data: Level tile and entity data
            ninja_position: Starting position (x, y) in pixels
            initial_switch_states: Initial state of switches (default: all False)

        Returns:
            ReachabilityState with reachable positions and subgoals
        """
        if initial_switch_states is None:
            initial_switch_states = {}

        # Initialize collision detector for this level
        self.position_validator.initialize_for_level(level_data.tiles)

        # Convert ninja position to sub-grid coordinates
        ninja_sub_row, ninja_sub_col = (
            self.position_validator.convert_pixel_to_sub_grid(
                ninja_position[0], ninja_position[1]
            )
        )

        # Initialize reachability state
        state = ReachabilityState(
            reachable_positions=set(),
            switch_states=initial_switch_states.copy(),
            unlocked_areas=set(),
            subgoals=[],
        )

        # Perform iterative reachability analysis
        # Each iteration may unlock new areas via switch activation
        max_iterations = 10  # Prevent infinite loops in complex switch dependencies
        for iteration in range(max_iterations):
            initial_size = len(state.reachable_positions)

            # Analyze reachability from current state
            self._analyze_reachability_iteration(
                level_data, ninja_position, ninja_sub_row, ninja_sub_col, state
            )

            # Early termination: if no new areas were discovered, we're done
            if len(state.reachable_positions) == initial_size:
                if self.debug:
                    print(
                        f"DEBUG: Reachability analysis converged after {iteration + 1} iterations"
                    )
                break

        # Identify subgoals for hierarchical planning
        self.game_mechanics.identify_subgoals(level_data, state)

        return state

    def _analyze_reachability_iteration(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        start_sub_row: int,
        start_sub_col: int,
        state: ReachabilityState,
    ):
        """
        Single iteration of reachability analysis using BFS with physics.

        Args:
            level_data: Level data
            ninja_position: Ninja's actual position in pixels
            start_sub_row: Starting sub-grid row
            start_sub_col: Starting sub-grid column
            state: Current reachability state (modified in-place)
        """
        # BFS queue: (sub_row, sub_col, came_from_direction)
        queue = deque([(start_sub_row, start_sub_col, None)])
        visited_this_iteration = set()

        # If this is the first iteration, mark starting position as reachable
        if not state.reachable_positions:
            # For the ninja's starting position, check if the actual ninja position is traversable
            # even if the sub-cell center is not (due to discretization effects)
            if (
                start_sub_row,
                start_sub_col,
            ) == self.position_validator.convert_pixel_to_sub_grid(
                ninja_position[0], ninja_position[1]
            ):
                # Check if the actual ninja position is traversable
                if self.position_validator.is_position_traversable_with_radius(
                    ninja_position[0], ninja_position[1], level_data.tiles, 10.0
                ):
                    state.reachable_positions.add((start_sub_row, start_sub_col))
                    if self.debug:
                        print(
                            f"DEBUG: Added ninja starting position ({start_sub_row}, {start_sub_col}) "
                            f"based on actual ninja position ({ninja_position[0]}, {ninja_position[1]})"
                        )
                elif self.debug:
                    print(
                        f"DEBUG: Ninja starting position ({ninja_position[0]}, {ninja_position[1]}) "
                        f"is not traversable"
                    )
            else:
                state.reachable_positions.add((start_sub_row, start_sub_col))

        while queue:
            sub_row, sub_col, came_from = queue.popleft()

            # Skip if already processed this iteration
            if (sub_row, sub_col) in visited_this_iteration:
                continue
            visited_this_iteration.add((sub_row, sub_col))

            # Skip if position is out of bounds
            if not self.position_validator.is_valid_sub_grid_position(sub_row, sub_col):
                continue

            # Skip if position is not traversable (solid tile)
            # For the ninja's starting position, use the actual ninja position instead of sub-cell center
            ninja_pos_override = None
            if (sub_row, sub_col) == self.position_validator.convert_pixel_to_sub_grid(
                ninja_position[0], ninja_position[1]
            ):
                ninja_pos_override = ninja_position

            if not self.position_validator.is_position_traversable(
                level_data, sub_row, sub_col, ninja_pos_override
            ):
                continue

            # Mark as reachable
            state.reachable_positions.add((sub_row, sub_col))

            # Check for switch activation at this position
            self.game_mechanics.check_switch_activation(
                level_data, sub_row, sub_col, state
            )

            # Explore neighboring positions using physics-based movement
            neighbors = self.physics_movement.get_physics_based_neighbors(
                level_data, sub_row, sub_col, came_from, state
            )

            if self.debug and (
                sub_row,
                sub_col,
            ) == self.position_validator.convert_pixel_to_sub_grid(
                ninja_position[0], ninja_position[1]
            ):
                print(
                    f"DEBUG: Exploring {len(neighbors)} neighbors from ninja position ({sub_row}, {sub_col})"
                )
                for neighbor_row, neighbor_col, movement_type in neighbors:
                    print(
                        f"  Neighbor ({neighbor_row}, {neighbor_col}) via {movement_type}"
                    )

            for neighbor_row, neighbor_col, movement_type in neighbors:
                if (neighbor_row, neighbor_col) not in visited_this_iteration:
                    queue.append((neighbor_row, neighbor_col, movement_type))
