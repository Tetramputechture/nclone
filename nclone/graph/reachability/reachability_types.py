"""
Shared data types for reachability analysis systems.

This module contains common data structures used across different
reachability analysis implementations to avoid circular imports.
"""

from typing import Set, Tuple
from dataclasses import dataclass


@dataclass
class ReachabilityResult:
    """
    Result of a graph-based reachability analysis.
    """

    reachable_positions: Set[Tuple[int, int]]

    def is_position_reachable(self, position: Tuple[int, int]) -> bool:
        """Check if a specific position is reachable."""
        return position in self.reachable_positions
