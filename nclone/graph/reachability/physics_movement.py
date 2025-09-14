"""
Physics-based movement calculations for reachability analysis.

This module handles all physics-related movement calculations including:
- Jump trajectory validation and neighbor finding
- Fall trajectory validation and neighbor finding
- Walking neighbor calculations
- Physics-based neighbor enumeration
"""

import numpy as np
from typing import List, Tuple, Optional

from .position_validator import PositionValidator
from .wall_jump_analyzer import WallJumpAnalyzer
from ...constants.physics_constants import (
    MAX_JUMP_DISTANCE,
    EXTENDED_JUMP_DISTANCE,
    MAX_FALL_DISTANCE,
    GRAVITY_FALL,
    TILE_PIXEL_SIZE,
)


class PhysicsMovement:
    """Handles physics-based movement calculations for reachability analysis."""

    def __init__(self, position_validator: PositionValidator, debug: bool = False):
        """
        Initialize physics movement calculator.

        Args:
            position_validator: Position validator for checking traversability
            debug: Enable debug output
        """
        self.position_validator = position_validator
        self.debug = debug
        self.hazard_extension = None  # Will be set by ReachabilityAnalyzer
        self.wall_jump_analyzer = WallJumpAnalyzer(debug=debug)

    def get_physics_based_neighbors(
        self,
        level_data,
        sub_row: int,
        sub_col: int,
        came_from: Optional[str],
        reachability_state,
    ) -> List[Tuple[int, int, str]]:
        """
        Get neighboring positions reachable via physics-based movement.

        Calculates all possible neighbors reachable through walking, jumping,
        and falling from the current position.

        Args:
            level_data: Level data containing tiles and entities
            sub_row: Current sub-grid row
            sub_col: Current sub-grid column
            came_from: Direction we came from (to avoid backtracking)
            reachability_state: Current reachability state

        Returns:
            List of (neighbor_row, neighbor_col, movement_type) tuples
        """
        neighbors = []
        # Use position validator's coordinate conversion for consistency
        pixel_x, pixel_y = self.position_validator.convert_sub_grid_to_pixel(
            sub_row, sub_col
        )

        if self.debug:
            print(
                f"DEBUG: get_physics_based_neighbors for ({sub_row}, {sub_col}) "
                f"at pixel ({pixel_x}, {pixel_y})"
            )

        # Walking neighbors (adjacent sub-cells)
        walk_neighbors = self._get_walking_neighbors(sub_row, sub_col, came_from)
        neighbors.extend(walk_neighbors)

        # Jump neighbors (physics-based)
        jump_neighbors = self._get_jump_neighbors(
            level_data, pixel_x, pixel_y, reachability_state
        )
        neighbors.extend(jump_neighbors)

        # Fall neighbors (physics-based)
        fall_neighbors = self._get_fall_neighbors(
            level_data, pixel_x, pixel_y, reachability_state
        )
        neighbors.extend(fall_neighbors)

        # Wall jump neighbors (advanced physics)
        wall_jump_neighbors = self._get_wall_jump_neighbors(
            level_data, sub_row, sub_col
        )
        neighbors.extend(wall_jump_neighbors)

        return neighbors

    def _get_walking_neighbors(
        self, sub_row: int, sub_col: int, came_from: Optional[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Get neighbors reachable by walking (adjacent sub-cells).

        Args:
            sub_row: Current sub-grid row
            sub_col: Current sub-grid column
            came_from: Direction we came from (to avoid backtracking)

        Returns:
            List of walking neighbors
        """
        neighbors = []
        walk_directions = [
            (-1, 0, "walk_up"),  # Up
            (1, 0, "walk_down"),  # Down
            (0, -1, "walk_left"),  # Left
            (0, 1, "walk_right"),  # Right
        ]

        for dr, dc, movement_type in walk_directions:
            if came_from == movement_type:
                continue  # Don't immediately backtrack

            new_row = sub_row + dr
            new_col = sub_col + dc

            if self.position_validator.is_valid_sub_grid_position(new_row, new_col):
                neighbors.append((new_row, new_col, movement_type))
                if self.debug:
                    print(
                        f"  Added walking neighbor ({new_row}, {new_col}) via {movement_type}"
                    )
            elif self.debug:
                print(
                    f"  Rejected walking neighbor ({new_row}, {new_col}) via {movement_type} - out of bounds"
                )

        return neighbors

    def _get_jump_neighbors(
        self, level_data, pixel_x: float, pixel_y: float, reachability_state
    ) -> List[Tuple[int, int, str]]:
        """
        Get positions reachable via jumping using physics simulation.

        Args:
            level_data: Level data
            pixel_x: Current x position in pixels
            pixel_y: Current y position in pixels
            reachability_state: Current reachability state

        Returns:
            List of jump neighbors
        """
        neighbors = []

        # Detect if we need extended jump distance for gap crossing
        max_jump_pixels = self._get_appropriate_jump_distance(pixel_x, pixel_y, level_data)

        # Use adaptive sampling based on max jump distance to avoid performance issues
        if max_jump_pixels <= 200:
            # Standard sampling for shorter jumps
            num_angles = 8
            num_distances = 4
        else:
            # Much more aggressive reduction for longer jumps to maintain performance
            num_angles = 4  # Focus on key angles: 30°, 60°, 90°, 120°
            num_distances = 2  # Just short and max distance

        # Sample jump directions and distances
        for angle in np.linspace(
            15, 165, num_angles
        ):  # Jump angles from 15° to 165° (upward arcs)
            angle_rad = np.radians(angle)

            for distance in np.linspace(
                TILE_PIXEL_SIZE, max_jump_pixels, num_distances
            ):  # Jump distances
                target_x = pixel_x + distance * np.cos(angle_rad)
                target_y = pixel_y - distance * np.sin(angle_rad)  # Y decreases upward

                # Convert to sub-grid coordinates
                target_sub_row, target_sub_col = (
                    self.position_validator.convert_pixel_to_sub_grid(
                        target_x, target_y
                    )
                )

                if self.position_validator.is_valid_sub_grid_position(
                    target_sub_row, target_sub_col
                ):
                    # Calculate jump distance for physics validation
                    jump_distance = ((target_x - pixel_x)**2 + (target_y - pixel_y)**2)**0.5
                    
                    # For very long jumps (>200px), apply stricter physics validation
                    if jump_distance > 200:  # ~8.3 tiles - beyond reasonable jump range
                        # Import trajectory calculator for long jump validation
                        from ..trajectory_calculator import TrajectoryCalculator
                        trajectory_calc = TrajectoryCalculator()
                        
                        # Check if jump is physically possible
                        start_pos = (pixel_x, pixel_y)
                        end_pos = (target_x, target_y)
                        trajectory_result = trajectory_calc.calculate_jump_trajectory(start_pos, end_pos)
                        
                        # If basic jump fails, try with momentum
                        if not trajectory_result.feasible or trajectory_result.success_probability <= 0.1:
                            from ...constants.physics_constants import MAX_HOR_SPEED, JUMP_FLOOR_Y
                            initial_velocity = (MAX_HOR_SPEED, JUMP_FLOOR_Y)
                            trajectory_result = trajectory_calc.calculate_momentum_trajectory(
                                start_pos, end_pos, initial_velocity
                            )
                        
                        # Only allow if trajectory calculator confirms feasibility
                        if not (trajectory_result.feasible and trajectory_result.success_probability > 0.1):
                            continue  # Skip this jump - not physically possible
                    
                    # Validate jump trajectory (simplified for shorter jumps, strict for long jumps)
                    if self._is_jump_trajectory_valid(
                        level_data,
                        pixel_x,
                        pixel_y,
                        target_x,
                        target_y,
                        reachability_state,
                    ):
                        neighbors.append((target_sub_row, target_sub_col, "jump"))

        return neighbors

    def _get_appropriate_jump_distance(self, pixel_x: float, pixel_y: float, level_data) -> float:
        """
        Determine appropriate jump distance based on gap detection.
        
        Uses standard jump distance normally, but extended distance when a large gap is detected.
        
        Args:
            pixel_x: Current X position in pixels
            pixel_y: Current Y position in pixels  
            level_data: Level data containing tiles
            
        Returns:
            Appropriate jump distance in pixels
        """
        # Check for large horizontal gaps to the right
        gap_detected = self._detect_horizontal_gap(pixel_x, pixel_y, level_data)
        
        if gap_detected:
            return EXTENDED_JUMP_DISTANCE
        else:
            return MAX_JUMP_DISTANCE
    
    def _detect_horizontal_gap(self, pixel_x: float, pixel_y: float, level_data) -> bool:
        """
        Detect if there's a large horizontal gap that might require extended jumping.
        
        Args:
            pixel_x: Current X position in pixels
            pixel_y: Current Y position in pixels
            level_data: Level data containing tiles
            
        Returns:
            True if a large gap is detected
        """
        # Convert to tile coordinates
        current_tile_x = int(pixel_x // TILE_PIXEL_SIZE)
        current_tile_y = int(pixel_y // TILE_PIXEL_SIZE)
        
        # Account for padding offset
        data_tile_x = current_tile_x - 1
        data_tile_y = current_tile_y - 1
        
        # Check if we're at the edge of a platform
        if (0 <= data_tile_y < len(level_data.tiles) and 
            0 <= data_tile_x < len(level_data.tiles[0])):
            
            # Look for a gap pattern: solid ground, then empty space, then solid ground again
            gap_start = None
            gap_end = None
            
            # Scan horizontally to the right for up to 15 tiles (360 pixels)
            for offset in range(1, 16):
                check_x = data_tile_x + offset
                if check_x >= len(level_data.tiles[0]):
                    break
                    
                tile_value = level_data.tiles[data_tile_y][check_x]
                
                if gap_start is None and tile_value == 0:  # Found start of gap
                    gap_start = offset
                elif gap_start is not None and tile_value == 1:  # Found end of gap
                    gap_end = offset
                    break
            
            # If we found a gap that's larger than standard jump distance
            if gap_start is not None and gap_end is not None:
                gap_size_pixels = (gap_end - gap_start) * TILE_PIXEL_SIZE
                if gap_size_pixels > MAX_JUMP_DISTANCE:
                    return True
        
        return False

    def _get_fall_neighbors(
        self, level_data, pixel_x: float, pixel_y: float, reachability_state
    ) -> List[Tuple[int, int, str]]:
        """
        Get positions reachable via falling with horizontal drift.

        Args:
            level_data: Level data
            pixel_x: Current x position in pixels
            pixel_y: Current y position in pixels
            reachability_state: Current reachability state

        Returns:
            List of fall neighbors
        """
        neighbors = []

        # Sample fall targets below current position
        max_fall_pixels = min(MAX_FALL_DISTANCE, 200)  # Reasonable limit

        # Horizontal drift while falling (in multiples of tile size)
        horizontal_offsets = [
            -2 * TILE_PIXEL_SIZE,
            -TILE_PIXEL_SIZE,
            0,
            TILE_PIXEL_SIZE,
            2 * TILE_PIXEL_SIZE,
        ]

        for horizontal_offset in horizontal_offsets:
            for fall_distance in np.linspace(TILE_PIXEL_SIZE, max_fall_pixels, 6):
                target_x = pixel_x + horizontal_offset
                target_y = pixel_y + fall_distance

                # Convert to sub-grid coordinates
                target_sub_row, target_sub_col = (
                    self.position_validator.convert_pixel_to_sub_grid(
                        target_x, target_y
                    )
                )

                if self.position_validator.is_valid_sub_grid_position(
                    target_sub_row, target_sub_col
                ):
                    # Validate fall trajectory (simplified)
                    if self._is_fall_trajectory_valid(
                        level_data,
                        pixel_x,
                        pixel_y,
                        target_x,
                        target_y,
                        reachability_state,
                    ):
                        neighbors.append((target_sub_row, target_sub_col, "fall"))

        return neighbors

    def _is_jump_trajectory_valid(
        self,
        level_data,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        reachability_state,
    ) -> bool:
        """
        Validate if jump trajectory is clear of obstacles.

        Uses simplified parabolic trajectory sampling to check for collisions
        with solid tiles along the jump path.

        Args:
            level_data: Level data
            start_x: Jump start x coordinate
            start_y: Jump start y coordinate
            end_x: Jump end x coordinate
            end_y: Jump end y coordinate
            reachability_state: Current reachability state

        Returns:
            True if trajectory is clear
        """
        # Simplified trajectory validation
        # Sample points along parabolic trajectory
        num_samples = 8
        for i in range(1, num_samples):
            t = i / num_samples

            # Parabolic interpolation (simplified)
            sample_x = start_x + t * (end_x - start_x)
            sample_y = (
                start_y + t * (end_y - start_y) - 0.5 * GRAVITY_FALL * t * t * 10
            )  # Simplified gravity

            # Check if sample point is in solid tile
            tile_x = int(sample_x // TILE_PIXEL_SIZE)
            tile_y = int(sample_y // TILE_PIXEL_SIZE)

            if 0 <= tile_y < len(level_data.tiles) and 0 <= tile_x < len(
                level_data.tiles[0]
            ):
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
        reachability_state,
    ) -> bool:
        """
        Validate if fall trajectory is clear of obstacles.

        Uses simplified linear trajectory sampling to check for collisions
        with solid tiles along the fall path.

        Args:
            level_data: Level data
            start_x: Fall start x coordinate
            start_y: Fall start y coordinate
            end_x: Fall end x coordinate
            end_y: Fall end y coordinate
            reachability_state: Current reachability state

        Returns:
            True if trajectory is clear
        """
        # Simplified fall validation - check vertical path
        num_samples = 6
        for i in range(1, num_samples):
            t = i / num_samples

            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)

            # Check if sample point is in solid tile
            tile_x = int(sample_x // TILE_PIXEL_SIZE)
            tile_y = int(sample_y // TILE_PIXEL_SIZE)

            if 0 <= tile_y < len(level_data.tiles) and 0 <= tile_x < len(
                level_data.tiles[0]
            ):
                if level_data.tiles[tile_y][tile_x] == 1:  # Solid tile
                    return False

        return True
    
    def _get_wall_jump_neighbors(
        self, 
        level_data, 
        sub_row: int, 
        sub_col: int
    ) -> List[Tuple[int, int, str]]:
        """
        Get neighbors reachable via wall jumping.
        
        Args:
            level_data: Level data
            sub_row: Current sub-grid row
            sub_col: Current sub-grid column
            
        Returns:
            List of (neighbor_row, neighbor_col, movement_type) tuples
        """
        # Initialize wall jump analyzer for this level if not done
        if not hasattr(self.wall_jump_analyzer, '_initialized') or not self.wall_jump_analyzer._initialized:
            self.wall_jump_analyzer.initialize_for_level(level_data.tiles)
            self.wall_jump_analyzer._initialized = True
        
        # Find wall jump neighbors
        wall_jump_neighbors = self.wall_jump_analyzer.find_wall_jump_neighbors(
            sub_row, sub_col, level_data.tiles, self.position_validator
        )
        
        # Filter out neighbors that are unsafe due to entities
        safe_neighbors = []
        for neighbor_row, neighbor_col, movement_type in wall_jump_neighbors:
            if self.hazard_extension is not None:
                pixel_x, pixel_y = self.position_validator.convert_sub_grid_to_pixel(
                    neighbor_row, neighbor_col
                )
                if not self.hazard_extension.is_position_safe_for_reachability((pixel_x, pixel_y)):
                    continue
            
            safe_neighbors.append((neighbor_row, neighbor_col, movement_type))
        
        return safe_neighbors
