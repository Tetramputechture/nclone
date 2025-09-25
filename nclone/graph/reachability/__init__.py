"""
Reachability analysis package for hierarchical graph optimization.

The components work together to determine which areas of a level are
accessible to the player from their starting position.
"""

from .reachability_system import ReachabilitySystem
from .reachability_types import (
    ReachabilityApproximation,
)

__all__ = [
    "ReachabilitySystem",
    "ReachabilityApproximation",
]
