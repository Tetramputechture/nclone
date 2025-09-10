"""
Movement types for N++ pathfinding system.

This module defines the movement types that the ninja can perform,
based on the physics-aware analysis of the game mechanics.
"""

from enum import IntEnum

class MovementType(IntEnum):
    """
    Movement types for N++ ninja navigation.
    
    These correspond to the actual movement capabilities of the ninja
    in the N++ physics system.
    """
    WALK = 0        # Ground-based horizontal movement
    JUMP = 1        # Airborne movement with initial velocity
    FALL = 2        # Gravity-driven descent
    WALL_SLIDE = 3  # Sliding along vertical surfaces
    WALL_JUMP = 4   # Wall-assisted jump
    LAUNCH_PAD = 5  # Launch pad boost
    BOUNCE_BLOCK = 6  # Bounce block interaction
    BOUNCE_CHAIN = 7  # Chained bounce block sequence

# Movement type colors for visualization
MOVEMENT_COLORS = {
    MovementType.WALK: (0.0, 1.0, 0.0),      # Green
    MovementType.JUMP: (0.0, 0.5, 1.0),      # Blue  
    MovementType.FALL: (1.0, 0.0, 0.0),      # Red
    MovementType.WALL_SLIDE: (1.0, 0.5, 0.0), # Orange
    MovementType.WALL_JUMP: (0.0, 1.0, 1.0), # Cyan
    MovementType.LAUNCH_PAD: (1.0, 0.0, 1.0), # Magenta
    MovementType.BOUNCE_BLOCK: (1.0, 1.0, 0.0), # Yellow
    MovementType.BOUNCE_CHAIN: (0.5, 0.5, 0.5), # Gray
}