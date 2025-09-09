"""
Consolidated N++ Physics-Aware Pathfinding System

This module provides the authoritative pathfinding implementation for N++,
incorporating accurate physics simulation and movement classification.
"""

from .core_pathfinder import CorePathfinder
from .movement_types import MovementType
from .physics_validator import PhysicsValidator

__all__ = ['CorePathfinder', 'MovementType', 'PhysicsValidator']