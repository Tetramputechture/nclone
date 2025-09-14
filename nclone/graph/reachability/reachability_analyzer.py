"""
Main reachability analyzer orchestrating all reachability components.

This module provides the main ReachabilityAnalyzer class that coordinates
position validation, collision detection, physics movement, and game mechanics
to perform comprehensive reachability analysis.
"""

from collections import deque
from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Optional, Any

from ..trajectory_calculator import TrajectoryCalculator
from .position_validator import PositionValidator
from .collision_checker import CollisionChecker
from .physics_movement import PhysicsMovement
from .game_mechanics import GameMechanics
from .reachability_cache import ReachabilityCache
from .hazard_integration import ReachabilityHazardExtension
from ..hazard_system import HazardClassificationSystem
from .frontier_detector import FrontierDetector
from .rl_integration import RLIntegrationAPI


from .reachability_state import ReachabilityState


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
        self, 
        trajectory_calculator: TrajectoryCalculator, 
        debug: bool = False,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 300.0
    ):
        """
        Initialize reachability analyzer with physics calculator.

        Args:
            trajectory_calculator: Physics-based trajectory calculator
            debug: Enable debug output (default: False)
            enable_caching: Enable intelligent caching system (default: True)
            cache_size: Maximum cache size (default: 1000)
            cache_ttl: Cache time-to-live in seconds (default: 300)
        """
        self.trajectory_calculator = trajectory_calculator
        self.debug = debug
        self.enable_caching = enable_caching

        # Initialize component modules
        self.position_validator = PositionValidator(debug=debug)
        self.collision_checker = CollisionChecker(debug=debug)
        self.physics_movement = PhysicsMovement(self.position_validator, debug=debug)
        self.game_mechanics = GameMechanics(debug=debug)
        # Initialize hazard system integration
        self.hazard_system = HazardClassificationSystem()
        self.hazard_extension = ReachabilityHazardExtension(self.hazard_system, debug=debug)
        self.frontier_detector = FrontierDetector(debug=debug)
        self.rl_api = RLIntegrationAPI(self, debug=debug)
        
        # Initialize caching system
        if enable_caching:
            self.cache = ReachabilityCache(max_size=cache_size, ttl_seconds=cache_ttl)
        else:
            self.cache = None

    def analyze_reachability(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None,
    ) -> ReachabilityState:
        """
        Analyze which areas are reachable from ninja starting position.

        Performs iterative BFS-based reachability analysis, with each iteration
        potentially unlocking new areas via switch activation. Uses intelligent
        caching for improved performance during RL training.

        Args:
            level_data: Level tile and entity data
            ninja_position: Starting position (x, y) in pixels
            initial_switch_states: Initial state of switches (default: all False)

        Returns:
            ReachabilityState with reachable positions and subgoals
        """
        if initial_switch_states is None:
            initial_switch_states = {}
        
        # Check cache first
        if self.cache is not None:
            cached_result = self.cache.get(
                ninja_position, 
                initial_switch_states, 
                getattr(level_data, 'level_id', 'unknown')
            )
            if cached_result is not None:
                if self.debug:
                    print("DEBUG: Using cached reachability result")
                return cached_result

        # Initialize collision detector and entity handler for this level
        self.position_validator.initialize_for_level(level_data.tiles)
        self.hazard_extension.initialize_for_reachability(level_data)
        
        # Connect hazard extension to physics movement
        self.physics_movement.hazard_extension = self.hazard_extension

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
        global_visited = set()  # Track positions across all iterations to prevent re-exploration
        
        for iteration in range(max_iterations):
            initial_size = len(state.reachable_positions)

            # Analyze reachability from current state
            self._analyze_reachability_iteration(
                level_data, ninja_position, ninja_sub_row, ninja_sub_col, state, global_visited
            )

            # Early termination: if no new areas were discovered, we're done
            if len(state.reachable_positions) == initial_size:
                if self.debug:
                    print(
                        f"DEBUG: Reachability analysis converged after {iteration + 1} iterations"
                    )
                break

        # Identify subgoals for hierarchical planning
        self.game_mechanics.identify_subgoals(
            level_data, state, self.hazard_extension, self.position_validator
        )
        
        # Detect exploration frontiers for curiosity-driven RL
        frontiers = self.frontier_detector.detect_frontiers(
            level_data, state, self.hazard_extension, self.position_validator
        )
        
        # Store frontiers in reachability state
        if not hasattr(state, 'frontiers'):
            state.frontiers = []
        state.frontiers = frontiers

        # Cache the result
        if self.cache is not None:
            self.cache.put(
                ninja_position,
                initial_switch_states,
                getattr(level_data, 'level_id', 'unknown'),
                state
            )

        return state

    def _analyze_reachability_iteration(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        start_sub_row: int,
        start_sub_col: int,
        state: ReachabilityState,
        global_visited: set,
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
        # but DON'T add to global_visited yet - let the BFS loop process it
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
                    # DON'T add to global_visited here - let BFS loop process it
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
                # DON'T add to global_visited here - let BFS loop process it

        # Add safety limit to prevent runaway exploration
        max_positions = 50000  # Reasonable limit for large levels
        positions_processed = 0
        
        while queue and positions_processed < max_positions:
            sub_row, sub_col, came_from = queue.popleft()

            # Skip if already processed globally (across all iterations)
            if (sub_row, sub_col) in global_visited:
                continue
            global_visited.add((sub_row, sub_col))
            visited_this_iteration.add((sub_row, sub_col))
            positions_processed += 1

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

            # Check if position is safe from entities
            pixel_x, pixel_y = self.position_validator.convert_sub_grid_to_pixel(sub_row, sub_col)
            if ninja_pos_override:
                pixel_x, pixel_y = ninja_pos_override
            
            if not self.hazard_extension.is_position_safe_for_reachability((pixel_x, pixel_y)):
                if self.debug:
                    print(f"DEBUG: Position ({sub_row}, {sub_col}) is unsafe due to entities")
                continue

            # Mark as reachable
            state.reachable_positions.add((sub_row, sub_col))

            # Check for switch activation at this position
            old_switch_states = state.switch_states.copy()
            self.game_mechanics.check_switch_activation(
                level_data, sub_row, sub_col, state
            )
            
            # Update entity handler if switch states changed
            if state.switch_states != old_switch_states:
                self.hazard_extension.update_switch_states(state.switch_states)

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
        
        # Warn if we hit the safety limit
        if positions_processed >= max_positions:
            if self.debug:
                print(f"WARNING: Reachability analysis hit safety limit of {max_positions} positions")
                print(f"This may indicate a performance issue or very large level")
    
    def get_cache_hit_rate(self) -> float:
        """
        Get the cache hit rate for performance monitoring.
        
        Returns:
            Cache hit rate as a float between 0.0 and 1.0
        """
        if self.cache is None:
            return 0.0
        return self.cache.get_hit_rate()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance statistics
        """
        if self.cache is None:
            return {'caching_disabled': True}
        return self.cache.get_stats()
    
    def invalidate_cache_for_level(self, level_id: str):
        """
        Invalidate all cached results for a specific level.
        
        Args:
            level_id: Level identifier to invalidate
        """
        if self.cache is not None:
            self.cache.invalidate_level(level_id)
    
    def invalidate_cache_for_switches(self, changed_switches: Set[int]):
        """
        Invalidate cached results that depend on changed switches.
        
        Args:
            changed_switches: Set of switch IDs that have changed state
        """
        if self.cache is not None:
            self.cache.invalidate_switch_dependent(changed_switches)
    
    def clear_cache(self):
        """Clear all cached results."""
        if self.cache is not None:
            self.cache.clear()
    
    def warm_cache(self, scenarios: list):
        """
        Pre-populate cache with common scenarios.
        
        Args:
            scenarios: List of (ninja_pos, switch_states, level_id, result) tuples
        """
        if self.cache is not None:
            self.cache.warm_cache(scenarios)
    
    # RL Integration Methods
    
    def get_rl_state(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None
    ):
        """
        Get RL-optimized state representation.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            initial_switch_states: Initial switch states
            
        Returns:
            RL-optimized state representation
        """
        return self.rl_api.get_rl_state(level_data, ninja_position, initial_switch_states)
    
    def calculate_curiosity_reward(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        target_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None
    ) -> float:
        """
        Calculate curiosity reward for moving to a target position.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            target_position: Target position to evaluate
            initial_switch_states: Initial switch states
            
        Returns:
            Curiosity reward value (0.0 to 1.0)
        """
        rl_state = self.get_rl_state(level_data, ninja_position, initial_switch_states)
        return self.rl_api.calculate_curiosity_reward(rl_state, target_position)
    
    def get_hierarchical_subgoals(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None,
        max_subgoals: int = 5
    ):
        """
        Get prioritized subgoals for hierarchical RL.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            initial_switch_states: Initial switch states
            max_subgoals: Maximum number of subgoals to return
            
        Returns:
            List of prioritized subgoals
        """
        rl_state = self.get_rl_state(level_data, ninja_position, initial_switch_states)
        return self.rl_api.get_hierarchical_subgoals(rl_state, max_subgoals)
    
    def get_exploration_targets(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None,
        max_targets: int = 3
    ):
        """
        Get high-value exploration targets.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            initial_switch_states: Initial switch states
            max_targets: Maximum number of targets to return
            
        Returns:
            List of exploration target positions
        """
        rl_state = self.get_rl_state(level_data, ninja_position, initial_switch_states)
        return self.rl_api.get_exploration_targets(rl_state, max_targets)
    
    def get_reachability_features(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        query_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None
    ) -> Dict[str, float]:
        """
        Extract RL features for a query position.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            query_position: Position to extract features for
            initial_switch_states: Initial switch states
            
        Returns:
            Dictionary of RL features
        """
        rl_state = self.get_rl_state(level_data, ninja_position, initial_switch_states)
        return self.rl_api.get_reachability_features(rl_state, query_position)
