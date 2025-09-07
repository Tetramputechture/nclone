"""
Player-centric reachability analysis for graph optimization.

This module implements reachability analysis to determine which areas of a level
are accessible to the player from their starting position, considering game physics
and mechanics like switches, doors, and one-way platforms.
"""

import numpy as np
from collections import deque
from typing import Set, Tuple, List, Dict, Optional
from dataclasses import dataclass

from .common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
from .trajectory_calculator import TrajectoryCalculator
from ..constants.physics_constants import (
    MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, GRAVITY_FALL, JUMP_INITIAL_VELOCITY,
    TILE_PIXEL_SIZE
)
from ..constants.entity_types import EntityType


@dataclass
class ReachabilityState:
    """Represents the state of level reachability analysis."""
    reachable_positions: Set[Tuple[int, int]]  # (sub_row, sub_col) positions
    switch_states: Dict[int, bool]  # entity_id -> activated state
    unlocked_areas: Set[Tuple[int, int]]  # Areas unlocked by switches
    subgoals: List[Tuple[int, int, str]]  # (sub_row, sub_col, goal_type) for key objectives


class ReachabilityAnalyzer:
    """
    Analyzes level reachability from player perspective.
    
    This class implements physics-based reachability analysis to determine
    which areas of a level are accessible to the player, considering:
    - Jump and fall physics
    - Switch activation and door unlocking
    - One-way platforms and terrain constraints
    - Subgoal identification for hierarchical planning
    """
    
    def __init__(self, trajectory_calculator: TrajectoryCalculator, debug: bool = False):
        """
        Initialize reachability analyzer with physics calculator.
        
        Args:
            trajectory_calculator: Physics-based trajectory calculator
            debug: Enable debug output (default: False)
        """
        self.trajectory_calculator = trajectory_calculator
        self.debug = debug
        
    def analyze_reachability(
        self, 
        level_data, 
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None
    ) -> ReachabilityState:
        """
        Analyze which areas are reachable from ninja starting position.
        
        Args:
            level_data: Level tile and entity data
            ninja_position: Starting position (x, y) in pixels
            initial_switch_states: Initial state of switches (default: all False)
            
        Returns:
            ReachabilityState with reachable positions and subgoals
        """
        if initial_switch_states is None:
            initial_switch_states = {}
            
        # Convert ninja position to sub-grid coordinates
        ninja_sub_row = int(ninja_position[1] // SUB_CELL_SIZE)
        ninja_sub_col = int(ninja_position[0] // SUB_CELL_SIZE)
        
        # Initialize reachability state
        state = ReachabilityState(
            reachable_positions=set(),
            switch_states=initial_switch_states.copy(),
            unlocked_areas=set(),
            subgoals=[]
        )
        
        # Perform iterative reachability analysis
        # Each iteration may unlock new areas via switch activation
        max_iterations = 10  # Prevent infinite loops in complex switch dependencies
        for iteration in range(max_iterations):
            initial_size = len(state.reachable_positions)
            
            # Analyze reachability from current state
            self._analyze_reachability_iteration(
                level_data, ninja_sub_row, ninja_sub_col, state
            )
            
            # Early termination: if no new areas were discovered, we're done
            if len(state.reachable_positions) == initial_size:
                if self.debug:
                    print(f"DEBUG: Reachability analysis converged after {iteration + 1} iterations")
                break
                
        # Identify subgoals for hierarchical planning
        self._identify_subgoals(level_data, state)
        
        return state
    
    def _analyze_reachability_iteration(
        self, 
        level_data, 
        start_sub_row: int, 
        start_sub_col: int, 
        state: ReachabilityState
    ):
        """
        Single iteration of reachability analysis using BFS with physics.
        
        Args:
            level_data: Level data
            start_sub_row: Starting sub-grid row
            start_sub_col: Starting sub-grid column  
            state: Current reachability state (modified in-place)
        """
        # BFS queue: (sub_row, sub_col, came_from_direction)
        queue = deque([(start_sub_row, start_sub_col, None)])
        visited_this_iteration = set()
        
        # If this is the first iteration, mark starting position as reachable
        if not state.reachable_positions:
            state.reachable_positions.add((start_sub_row, start_sub_col))
            
        while queue:
            sub_row, sub_col, came_from = queue.popleft()
            
            # Skip if already processed this iteration
            if (sub_row, sub_col) in visited_this_iteration:
                continue
            visited_this_iteration.add((sub_row, sub_col))
            
            # Skip if position is out of bounds
            if not self._is_valid_position(sub_row, sub_col):
                continue
                
            # Skip if position is not traversable (solid tile)
            if not self._is_traversable_position(level_data, sub_row, sub_col, state):
                continue
                
            # Mark as reachable
            state.reachable_positions.add((sub_row, sub_col))
            
            # Check for switch activation at this position
            self._check_switch_activation(level_data, sub_row, sub_col, state)
            
            # Explore neighboring positions using physics-based movement
            neighbors = self._get_physics_based_neighbors(
                level_data, sub_row, sub_col, came_from, state
            )
            
            for neighbor_row, neighbor_col, movement_type in neighbors:
                if (neighbor_row, neighbor_col) not in visited_this_iteration:
                    queue.append((neighbor_row, neighbor_col, movement_type))
    
    def _is_valid_position(self, sub_row: int, sub_col: int) -> bool:
        """Check if sub-grid position is within level bounds."""
        return (0 <= sub_row < SUB_GRID_HEIGHT and 
                0 <= sub_col < SUB_GRID_WIDTH)
    
    def _is_traversable_position(
        self, 
        level_data, 
        sub_row: int, 
        sub_col: int, 
        state: ReachabilityState
    ) -> bool:
        """
        Check if position is traversable considering current game state.
        
        Args:
            level_data: Level data
            sub_row: Sub-grid row
            sub_col: Sub-grid column
            state: Current reachability state
            
        Returns:
            True if position can be occupied by player
        """
        # Convert to pixel coordinates
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Check if position is in solid tile
        tile_x = int(pixel_x // TILE_PIXEL_SIZE)
        tile_y = int(pixel_y // TILE_PIXEL_SIZE)
        
        if (0 <= tile_y < len(level_data.tiles) and 
            0 <= tile_x < len(level_data.tiles[0])):
            tile_value = level_data.tiles[tile_y][tile_x]
            
            # Solid tiles are not traversable
            if tile_value == 1:  # Solid tile
                return False
                
            # Check for locked doors that are still locked
            if self._is_position_blocked_by_door(level_data, pixel_x, pixel_y, state):
                return False
                
        return True
    
    def _is_position_blocked_by_door(
        self, 
        level_data, 
        pixel_x: float, 
        pixel_y: float, 
        state: ReachabilityState
    ) -> bool:
        """Check if position is blocked by a locked door."""
        for entity in level_data.entities:
            if entity.get('type') == EntityType.LOCKED_DOOR:
                # Check if this door is still locked
                entity_id = id(entity)  # Use object id as unique identifier
                if not state.switch_states.get(entity_id, False):
                    # Door is locked, check if position overlaps with door
                    door_x = entity.get('door_x', entity.get('x', 0))
                    door_y = entity.get('door_y', entity.get('y', 0))
                    
                    # Simple overlap check (could be refined)
                    half_tile = TILE_PIXEL_SIZE // 2
                    if (abs(pixel_x - door_x) < half_tile and
                        abs(pixel_y - door_y) < half_tile):
                        return True
        return False
    
    def _check_switch_activation(
        self, 
        level_data, 
        sub_row: int, 
        sub_col: int, 
        state: ReachabilityState
    ):
        """Check if player can activate any switches at this position."""
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        for entity in level_data.entities:
            entity_type = entity.get('type')
            if entity_type in [EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR, EntityType.EXIT_SWITCH]:
                # Check if player is close enough to activate switch
                switch_x = entity.get('x', 0)
                switch_y = entity.get('y', 0)
                
                # Activation range (within ~1 tile)
                if (abs(pixel_x - switch_x) < TILE_PIXEL_SIZE and 
                    abs(pixel_y - switch_y) < TILE_PIXEL_SIZE):
                    entity_id = id(entity)
                    if not state.switch_states.get(entity_id, False):
                        # Activate the switch
                        state.switch_states[entity_id] = True
                        if self.debug:
                            print(f"DEBUG: Activated switch at ({switch_x}, {switch_y}) - type {entity_type}")
    
    def _get_physics_based_neighbors(
        self, 
        level_data, 
        sub_row: int, 
        sub_col: int, 
        came_from: Optional[str],
        state: ReachabilityState
    ) -> List[Tuple[int, int, str]]:
        """
        Get neighboring positions reachable via physics-based movement.
        
        Args:
            level_data: Level data
            sub_row: Current sub-grid row
            sub_col: Current sub-grid column
            came_from: Direction we came from (to avoid backtracking)
            state: Current reachability state
            
        Returns:
            List of (neighbor_row, neighbor_col, movement_type) tuples
        """
        neighbors = []
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Walking neighbors (adjacent sub-cells)
        walk_directions = [
            (-1, 0, 'walk_up'),    # Up
            (1, 0, 'walk_down'),   # Down  
            (0, -1, 'walk_left'),  # Left
            (0, 1, 'walk_right'),  # Right
        ]
        
        for dr, dc, movement_type in walk_directions:
            if came_from == movement_type:
                continue  # Don't immediately backtrack
                
            new_row = sub_row + dr
            new_col = sub_col + dc
            
            if self._is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col, movement_type))
        
        # Jump neighbors (physics-based)
        jump_neighbors = self._get_jump_neighbors(level_data, pixel_x, pixel_y, state)
        neighbors.extend(jump_neighbors)
        
        # Fall neighbors (physics-based)
        fall_neighbors = self._get_fall_neighbors(level_data, pixel_x, pixel_y, state)
        neighbors.extend(fall_neighbors)
        
        return neighbors
    
    def _get_jump_neighbors(
        self, 
        level_data, 
        pixel_x: float, 
        pixel_y: float, 
        state: ReachabilityState
    ) -> List[Tuple[int, int, str]]:
        """Get positions reachable via jumping."""
        neighbors = []
        
        # Sample jump targets within physics range
        max_jump_pixels = min(MAX_JUMP_DISTANCE, 150)  # Reasonable limit to avoid excessive computation
        
        # Sample jump directions and distances (8×4=32 samples total)
        for angle in np.linspace(15, 165, 8):  # Jump angles from 15° to 165° (upward arcs)
            angle_rad = np.radians(angle)
            
            for distance in np.linspace(TILE_PIXEL_SIZE, max_jump_pixels, 4):  # Jump distances from min to max
                target_x = pixel_x + distance * np.cos(angle_rad)
                target_y = pixel_y - distance * np.sin(angle_rad)  # Y decreases upward
                
                # Convert to sub-grid coordinates
                target_sub_row = int(target_y // SUB_CELL_SIZE)
                target_sub_col = int(target_x // SUB_CELL_SIZE)
                
                if self._is_valid_position(target_sub_row, target_sub_col):
                    # Validate jump trajectory (simplified)
                    if self._is_jump_trajectory_valid(
                        level_data, pixel_x, pixel_y, target_x, target_y, state
                    ):
                        neighbors.append((target_sub_row, target_sub_col, 'jump'))
        
        return neighbors
    
    def _get_fall_neighbors(
        self, 
        level_data, 
        pixel_x: float, 
        pixel_y: float, 
        state: ReachabilityState
    ) -> List[Tuple[int, int, str]]:
        """Get positions reachable via falling."""
        neighbors = []
        
        # Sample fall targets below current position
        max_fall_pixels = min(MAX_FALL_DISTANCE, 200)  # Reasonable limit
        
        # Horizontal drift while falling (in multiples of tile size)
        horizontal_offsets = [-2 * TILE_PIXEL_SIZE, -TILE_PIXEL_SIZE, 0, TILE_PIXEL_SIZE, 2 * TILE_PIXEL_SIZE]
        for horizontal_offset in horizontal_offsets:
            for fall_distance in np.linspace(TILE_PIXEL_SIZE, max_fall_pixels, 6):
                target_x = pixel_x + horizontal_offset
                target_y = pixel_y + fall_distance
                
                # Convert to sub-grid coordinates
                target_sub_row = int(target_y // SUB_CELL_SIZE)
                target_sub_col = int(target_x // SUB_CELL_SIZE)
                
                if self._is_valid_position(target_sub_row, target_sub_col):
                    # Validate fall trajectory (simplified)
                    if self._is_fall_trajectory_valid(
                        level_data, pixel_x, pixel_y, target_x, target_y, state
                    ):
                        neighbors.append((target_sub_row, target_sub_col, 'fall'))
        
        return neighbors
    
    def _is_jump_trajectory_valid(
        self, 
        level_data, 
        start_x: float, 
        start_y: float, 
        end_x: float, 
        end_y: float,
        state: ReachabilityState
    ) -> bool:
        """Validate if jump trajectory is clear of obstacles."""
        # Simplified trajectory validation
        # Sample points along parabolic trajectory
        num_samples = 8
        for i in range(1, num_samples):
            t = i / num_samples
            
            # Parabolic interpolation (simplified)
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y) - 0.5 * GRAVITY_FALL * t * t * 10  # Simplified gravity
            
            # Check if sample point is in solid tile
            tile_x = int(sample_x // TILE_PIXEL_SIZE)
            tile_y = int(sample_y // TILE_PIXEL_SIZE)
            
            if (0 <= tile_y < len(level_data.tiles) and 
                0 <= tile_x < len(level_data.tiles[0])):
                if level_data.tiles[tile_y][tile_x] == 1:  # Solid tile
                    return False
        
        return True
    
    def _is_fall_trajectory_valid(
        self, 
        level_data, 
        start_x: float, 
        start_y: float, 
        end_x: float, 
        end_y: float,
        state: ReachabilityState
    ) -> bool:
        """Validate if fall trajectory is clear of obstacles."""
        # Simplified fall validation - check vertical path
        num_samples = 6
        for i in range(1, num_samples):
            t = i / num_samples
            
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)
            
            # Check if sample point is in solid tile
            tile_x = int(sample_x // TILE_PIXEL_SIZE)
            tile_y = int(sample_y // TILE_PIXEL_SIZE)
            
            if (0 <= tile_y < len(level_data.tiles) and 
                0 <= tile_x < len(level_data.tiles[0])):
                if level_data.tiles[tile_y][tile_x] == 1:  # Solid tile
                    return False
        
        return True
    
    def _identify_subgoals(self, level_data, state: ReachabilityState):
        """
        Identify key subgoals for hierarchical pathfinding.
        
        Subgoals are entities that unlock new areas or represent key objectives:
        - Switches that unlock doors
        - Exit switches
        - Exit doors
        """
        state.subgoals.clear()
        
        for entity in level_data.entities:
            entity_type = entity.get('type')
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            
            # Convert to sub-grid coordinates
            sub_row = int(entity_y // SUB_CELL_SIZE)
            sub_col = int(entity_x // SUB_CELL_SIZE)
            
            # Check if entity is in reachable area
            if (sub_row, sub_col) in state.reachable_positions:
                if entity_type == EntityType.LOCKED_DOOR:
                    state.subgoals.append((sub_row, sub_col, 'locked_door_switch'))
                elif entity_type == EntityType.TRAP_DOOR:
                    state.subgoals.append((sub_row, sub_col, 'trap_door_switch'))
                elif entity_type == EntityType.EXIT_SWITCH:
                    state.subgoals.append((sub_row, sub_col, 'exit_switch'))
                elif entity_type == EntityType.EXIT_DOOR:
                    state.subgoals.append((sub_row, sub_col, 'exit'))
        
        # Sort subgoals by priority (switches before exits)
        priority_order = {
            'locked_door_switch': 1,
            'trap_door_switch': 2, 
            'exit_switch': 3,
            'exit': 4
        }
        state.subgoals.sort(key=lambda x: priority_order.get(x[2], 999))
        
        if self.debug:
            print(f"DEBUG: Identified {len(state.subgoals)} subgoals: {[g[2] for g in state.subgoals]}")