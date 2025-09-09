"""
Physics validation for N++ pathfinding movements.

This module validates that proposed movements respect the ninja's
physical capabilities and constraints.
"""

import math
from typing import Tuple
from .movement_types import MovementType
from ..constants.physics_constants import (
    MAX_HOR_SPEED, JUMP_INITIAL_VELOCITY, GRAVITY_FALL, GRAVITY_JUMP,
    MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE
)

class PhysicsValidator:
    """
    Validates that movements respect N++ physics constraints.
    
    This class ensures that all proposed movements are physically
    achievable by the ninja given the game's physics system.
    """
    
    def validate_movement(self, start_pos: Tuple[float, float], 
                         end_pos: Tuple[float, float], 
                         movement_type: MovementType) -> bool:
        """
        Validate that a movement is physically possible.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            movement_type: Type of movement to validate
            
        Returns:
            True if movement is physically valid, False otherwise
        """
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if movement_type == MovementType.WALK:
            return self._validate_walk(dx, dy, distance)
        elif movement_type == MovementType.JUMP:
            return self._validate_jump(dx, dy, distance)
        elif movement_type == MovementType.FALL:
            return self._validate_fall(dx, dy, distance)
        elif movement_type == MovementType.WALL_SLIDE:
            return self._validate_wall_slide(dx, dy, distance)
        elif movement_type in [MovementType.WALL_JUMP, MovementType.LAUNCH_PAD, 
                               MovementType.BOUNCE_BLOCK, MovementType.BOUNCE_CHAIN]:
            return True  # Special movements are always valid for now
        
        return False
    
    def _validate_walk(self, dx: float, dy: float, distance: float) -> bool:
        """Validate walking movement."""
        
        # Walk movements should be mostly horizontal
        if abs(dy) > 6:  # More than 6 pixels vertical is not walkable
            return False
        
        # Check if horizontal distance is reasonable
        # For walking, we allow very long distances since the ninja can walk indefinitely
        max_walk_distance = 2000  # Allow walks up to 2000px (full map width)
        return distance <= max_walk_distance
    
    def _validate_jump(self, dx: float, dy: float, distance: float) -> bool:
        """Validate jumping movement."""
        
        # Check total distance
        if distance > MAX_JUMP_DISTANCE:
            return False
        
        # Check vertical component for upward jumps
        if dy < 0:  # Jumping up
            max_jump_height = abs(JUMP_INITIAL_VELOCITY)**2 / (2 * GRAVITY_JUMP)
            if abs(dy) > max_jump_height:
                return False
        
        # Check horizontal velocity requirements
        if abs(dx) > 0:
            # Estimate jump duration
            if dy < 0:  # Jumping up
                time_to_peak = abs(JUMP_INITIAL_VELOCITY) / GRAVITY_JUMP
                jump_duration = time_to_peak * 2
            else:  # Jumping down or horizontal
                jump_duration = 45  # Default frames
            
            required_horizontal_velocity = abs(dx) / jump_duration
            if required_horizontal_velocity > MAX_HOR_SPEED:
                return False
        
        return True
    
    def _validate_fall(self, dx: float, dy: float, distance: float) -> bool:
        """Validate falling movement."""
        
        # Check total distance
        if distance > MAX_FALL_DISTANCE:
            return False
        
        # Falls should generally be downward (though upward falls can occur)
        # No strict validation needed as falls are gravity-driven
        return True
    
    def _validate_wall_slide(self, dx: float, dy: float, distance: float) -> bool:
        """Validate wall sliding movement."""
        
        # Wall slides should be mostly vertical
        if abs(dx) > abs(dy):
            return False
        
        # Check reasonable distance
        max_slide_distance = 200  # Reasonable wall slide distance
        return distance <= max_slide_distance