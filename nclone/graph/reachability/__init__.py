"""
Reachability analysis package for hierarchical graph optimization.

The components work together to determine which areas of a level are
accessible to the player from their starting position.
"""

from .tiered_system import TieredReachabilitySystem
from .reachability_types import (
    ReachabilityApproximation,
    ReachabilityResult,
    PerformanceTarget,
)

__all__ = [
    "TieredReachabilitySystem",
    "ReachabilityApproximation",
    "ReachabilityResult",
    "PerformanceTarget",
]
