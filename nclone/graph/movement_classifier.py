"""
Movement classification for N++ physics simulation.

This module defines movement types used in the graph construction and
segment consolidation systems.
"""

from enum import Enum


class MovementType(Enum):
    """
    Classification of different movement types in N++ physics.
    
    These types are used to categorize different kinds of movement
    segments in the graph construction process.
    """
    WALK = "walk"
    JUMP = "jump"
    FALL = "fall"
    COMBO = "combo"