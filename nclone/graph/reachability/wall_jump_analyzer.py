"""
Wall jump analysis for advanced reachability calculations.

This module provides wall jump trajectory analysis to identify positions
that are only reachable through wall jumping mechanics.
"""

import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from ...constants.physics_constants import (
    MAX_VER_SPEED, TERMINAL_VELOCITY, GRAVITY_FALL, TILE_PIXEL_SIZE,
    NINJA_RADIUS, MAX_JUMP_DISTANCE
)


@dataclass
class WallJumpTrajectory:
    """Represents a wall jump trajectory."""
    
    start_position: Tuple[float, float]
    wall_position: Tuple[float, float]
    end_position: Tuple[float, float]
    wall_contact_time: float
    total_time: float
    feasible: bool


class WallJumpAnalyzer:
    """
    Analyzes wall jump possibilities for reachability analysis.
    
    Features:
    - Wall detection and classification
    - Wall jump trajectory calculation
    - Multi-wall jump sequence analysis
    - Wall jump reachability validation
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize wall jump analyzer.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.wall_positions: Set[Tuple[int, int]] = set()
        self.wall_jump_cache: dict = {}
        
    def initialize_for_level(self, tiles):
        """
        Initialize wall jump analyzer for a specific level.
        
        Args:
            tiles: 2D array of tile data
        """
        self.wall_positions.clear()
        self.wall_jump_cache.clear()
        
        # Identify wall positions (solid tiles adjacent to empty space)
        height, width = tiles.shape
        
        for row in range(height):
            for col in range(width):
                if tiles[row, col] != 0:  # Solid tile
                    # Check if it's adjacent to empty space (potential wall)
                    if self._is_wall_tile(tiles, row, col):
                        self.wall_positions.add((row, col))
        
        if self.debug:
            print(f"DEBUG: Identified {len(self.wall_positions)} wall positions")
    
    def _is_wall_tile(self, tiles, row: int, col: int) -> bool:
        """
        Check if a tile is a wall (solid tile adjacent to empty space).
        
        Args:
            tiles: 2D tile array
            row: Tile row
            col: Tile column
            
        Returns:
            True if tile is a wall
        """
        height, width = tiles.shape
        
        # Check adjacent positions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row, adj_col = row + dr, col + dc
            
            if (0 <= adj_row < height and 0 <= adj_col < width and
                tiles[adj_row, adj_col] == 0):  # Adjacent empty space
                return True
        
        return False
    
    def find_wall_jump_neighbors(
        self, 
        start_row: int, 
        start_col: int, 
        tiles,
        position_validator
    ) -> List[Tuple[int, int, str]]:
        """
        Find positions reachable via wall jumping from the current position.
        
        Args:
            start_row: Starting row position
            start_col: Starting column position
            tiles: 2D tile array
            position_validator: Position validator for traversability checks
            
        Returns:
            List of (row, col, movement_type) tuples for wall jump destinations
        """
        neighbors = []
        start_pixel_x, start_pixel_y = position_validator.convert_sub_grid_to_pixel(
            start_row, start_col
        )
        
        # Find nearby walls within wall jump range
        nearby_walls = self._find_nearby_walls(start_row, start_col, max_distance=5)
        
        for wall_row, wall_col in nearby_walls:
            wall_pixel_x, wall_pixel_y = position_validator.convert_sub_grid_to_pixel(
                wall_row, wall_col
            )
            
            # Calculate possible wall jump trajectories
            trajectories = self._calculate_wall_jump_trajectories(
                (start_pixel_x, start_pixel_y),
                (wall_pixel_x, wall_pixel_y),
                tiles,
                position_validator
            )
            
            for trajectory in trajectories:
                if trajectory.feasible:
                    end_row, end_col = position_validator.convert_pixel_to_sub_grid(
                        trajectory.end_position[0], trajectory.end_position[1]
                    )
                    
                    # Validate end position
                    end_pixel_x, end_pixel_y = position_validator.convert_sub_grid_to_pixel(end_row, end_col)
                    if (position_validator.is_valid_sub_grid_position(end_row, end_col) and
                        position_validator.is_position_traversable_with_radius(
                            end_pixel_x, end_pixel_y, tiles, NINJA_RADIUS
                        )):
                        neighbors.append((end_row, end_col, "wall_jump"))
                        
                        if self.debug:
                            print(f"DEBUG: Wall jump from ({start_row}, {start_col}) "
                                  f"to ({end_row}, {end_col}) via wall at "
                                  f"({wall_row}, {wall_col})")
        
        return neighbors
    
    def _find_nearby_walls(
        self, 
        center_row: int, 
        center_col: int, 
        max_distance: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Find wall positions within a certain distance.
        
        Args:
            center_row: Center row position
            center_col: Center column position
            max_distance: Maximum distance in grid cells
            
        Returns:
            List of nearby wall positions
        """
        nearby_walls = []
        
        for wall_row, wall_col in self.wall_positions:
            distance = max(abs(wall_row - center_row), abs(wall_col - center_col))
            if distance <= max_distance:
                nearby_walls.append((wall_row, wall_col))
        
        return nearby_walls
    
    def _calculate_wall_jump_trajectories(
        self,
        start_pos: Tuple[float, float],
        wall_pos: Tuple[float, float],
        tiles,
        position_validator
    ) -> List[WallJumpTrajectory]:
        """
        Calculate possible wall jump trajectories from start to wall.
        
        Args:
            start_pos: Starting position (x, y)
            wall_pos: Wall position (x, y)
            tiles: 2D tile array
            position_validator: Position validator
            
        Returns:
            List of possible wall jump trajectories
        """
        trajectories = []
        start_x, start_y = start_pos
        wall_x, wall_y = wall_pos
        
        # Calculate initial trajectory to reach the wall
        dx = wall_x - start_x
        dy = wall_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > MAX_JUMP_DISTANCE:
            return trajectories  # Wall too far for initial jump
        
        # Calculate time to reach wall
        # Simplified physics: assume constant horizontal velocity, gravity affects vertical
        jump_speed = MAX_VER_SPEED  # Use available constant
        
        if jump_speed == 0 or distance == 0:
            return trajectories  # Avoid division by zero
            
        horizontal_velocity = dx / (distance / jump_speed)
        initial_vertical_velocity = jump_speed
        
        # Time to reach wall (simplified)
        time_to_wall = distance / abs(horizontal_velocity) if horizontal_velocity != 0 else 0
        
        if time_to_wall <= 0:
            return trajectories
        
        # Calculate wall contact position
        initial_vertical_velocity = MAX_VER_SPEED  # Use available constant
        wall_contact_y = start_y + initial_vertical_velocity * time_to_wall - 0.5 * GRAVITY_FALL * time_to_wall * time_to_wall
        
        # Check if wall contact is valid
        if not self._is_valid_wall_contact((wall_x, wall_contact_y), tiles, position_validator):
            return trajectories
        
        # Calculate wall jump trajectories (multiple angles)
        wall_jump_angles = [30, 45, 60, 75]  # Degrees from horizontal
        
        for angle_deg in wall_jump_angles:
            angle_rad = math.radians(angle_deg)
            
            # Wall jump velocity components
            wall_jump_speed = MAX_VER_SPEED * 0.8  # Slightly reduced for wall jump
            
            # Determine wall jump direction (away from wall)
            wall_normal_x = 1 if dx < 0 else -1  # Jump away from wall
            
            jump_vx = wall_jump_speed * math.cos(angle_rad) * wall_normal_x
            jump_vy = wall_jump_speed * math.sin(angle_rad)
            
            # Calculate landing position
            flight_time = 2 * jump_vy / GRAVITY_FALL  # Time to reach peak and fall back
            
            end_x = wall_x + jump_vx * flight_time
            end_y = wall_contact_y + jump_vy * flight_time - 0.5 * GRAVITY_FALL * flight_time * flight_time
            
            # Validate trajectory
            trajectory = WallJumpTrajectory(
                start_position=start_pos,
                wall_position=(wall_x, wall_contact_y),
                end_position=(end_x, end_y),
                wall_contact_time=time_to_wall,
                total_time=time_to_wall + flight_time,
                feasible=self._validate_wall_jump_trajectory(
                    start_pos, (wall_x, wall_contact_y), (end_x, end_y),
                    tiles, position_validator
                )
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _is_valid_wall_contact(
        self, 
        contact_pos: Tuple[float, float], 
        tiles, 
        position_validator
    ) -> bool:
        """
        Check if wall contact position is valid.
        
        Args:
            contact_pos: Wall contact position (x, y)
            tiles: 2D tile array
            position_validator: Position validator
            
        Returns:
            True if wall contact is valid
        """
        x, y = contact_pos
        
        # Convert to grid coordinates
        row = int(y // TILE_PIXEL_SIZE)
        col = int(x // TILE_PIXEL_SIZE)
        
        # Check if position is within bounds
        if not position_validator.is_valid_sub_grid_position(row, col):
            return False
        
        # Check if there's a solid tile for wall contact
        height, width = tiles.shape
        if 0 <= row < height and 0 <= col < width:
            return tiles[row, col] != 0
        
        return False
    
    def _validate_wall_jump_trajectory(
        self,
        start_pos: Tuple[float, float],
        wall_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        tiles,
        position_validator
    ) -> bool:
        """
        Validate that a wall jump trajectory is feasible.
        
        Args:
            start_pos: Starting position
            wall_pos: Wall contact position
            end_pos: Landing position
            tiles: 2D tile array
            position_validator: Position validator
            
        Returns:
            True if trajectory is feasible
        """
        # Check if end position is valid and traversable
        end_row, end_col = position_validator.convert_pixel_to_sub_grid(
            end_pos[0], end_pos[1]
        )
        
        if not position_validator.is_valid_sub_grid_position(end_row, end_col):
            return False
        
        if not position_validator.is_position_traversable_with_radius(
            end_pos[0], end_pos[1], tiles, NINJA_RADIUS
        ):
            return False
        
        # Sample trajectory path for collisions
        return self._check_trajectory_path_clear(
            start_pos, wall_pos, end_pos, tiles, position_validator
        )
    
    def _check_trajectory_path_clear(
        self,
        start_pos: Tuple[float, float],
        wall_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        tiles,
        position_validator,
        samples: int = 20
    ) -> bool:
        """
        Check if trajectory path is clear of obstacles.
        
        Args:
            start_pos: Starting position
            wall_pos: Wall contact position
            end_pos: Landing position
            tiles: 2D tile array
            position_validator: Position validator
            samples: Number of samples along path
            
        Returns:
            True if path is clear
        """
        # Check path from start to wall
        for i in range(samples + 1):
            t = i / samples
            sample_x = start_pos[0] + t * (wall_pos[0] - start_pos[0])
            sample_y = start_pos[1] + t * (wall_pos[1] - start_pos[1])
            
            sample_row, sample_col = position_validator.convert_pixel_to_sub_grid(
                sample_x, sample_y
            )
            
            if not position_validator.is_position_traversable_with_radius(
                sample_x, sample_y, tiles, NINJA_RADIUS
            ):
                return False
        
        # Check path from wall to end
        for i in range(samples + 1):
            t = i / samples
            sample_x = wall_pos[0] + t * (end_pos[0] - wall_pos[0])
            sample_y = wall_pos[1] + t * (end_pos[1] - wall_pos[1])
            
            sample_row, sample_col = position_validator.convert_pixel_to_sub_grid(
                sample_x, sample_y
            )
            
            if not position_validator.is_position_traversable_with_radius(
                sample_x, sample_y, tiles, NINJA_RADIUS
            ):
                return False
        
        return True
    
    def get_wall_jump_statistics(self) -> dict:
        """
        Get statistics about wall jump analysis.
        
        Returns:
            Dictionary with wall jump statistics
        """
        return {
            'wall_positions': len(self.wall_positions),
            'cache_size': len(self.wall_jump_cache),
            'wall_density': len(self.wall_positions) / max(1, len(self.wall_positions) + 100)  # Rough estimate
        }
    
    def clear_cache(self):
        """Clear wall jump calculation cache."""
        self.wall_jump_cache.clear()