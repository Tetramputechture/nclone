"""
Reachability analysis package for hierarchical graph optimization.

This package provides modular components for analyzing level reachability:
- Position validation and traversability checking
- Collision detection with tiles and entities
- Physics-based movement calculations
- Game mechanics (switches, doors, subgoals)
- Main reachability analysis orchestration

The components work together to determine which areas of a level are
accessible to the player from their starting position.
"""

from .reachability_analyzer import ReachabilityAnalyzer
from .reachability_state import ReachabilityState
from .position_validator import PositionValidator
from .collision_checker import CollisionChecker
from .physics_movement import PhysicsMovement
from .game_mechanics import GameMechanics

__all__ = [
    "ReachabilityAnalyzer",
    "ReachabilityState",
    "PositionValidator",
    "CollisionChecker",
    "PhysicsMovement",
    "GameMechanics",
]
