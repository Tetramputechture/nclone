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

from .tiered_system import TieredReachabilitySystem
from .reachability_types import (
    ReachabilityApproximation,
    ReachabilityResult,
    PerformanceTarget,
)
from .position_validator import PositionValidator

__all__ = [
    "TieredReachabilitySystem",
    "ReachabilityApproximation",
    "ReachabilityResult",
    "PerformanceTarget",
    "PositionValidator",
]
