"""
Reachability state data structures.

This module contains the core data structures used by the reachability system
to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Set, Tuple, List, Dict


@dataclass
class ReachabilityState:
    """Represents the state of level reachability analysis."""

    reachable_positions: Set[Tuple[int, int]] = field(default_factory=set)  # (sub_row, sub_col) positions
    switch_states: Dict[int, bool] = field(default_factory=dict)  # entity_id -> activated state
    unlocked_areas: Set[Tuple[int, int]] = field(default_factory=set)  # Areas unlocked by switches
    subgoals: List[Tuple[int, int, str]] = field(default_factory=list)  # (sub_row, sub_col, goal_type) for key objectives
    
    def copy(self) -> 'ReachabilityState':
        """Create a deep copy of this reachability state."""
        return ReachabilityState(
            reachable_positions=self.reachable_positions.copy(),
            switch_states=self.switch_states.copy(),
            unlocked_areas=self.unlocked_areas.copy(),
            subgoals=self.subgoals.copy()
        )
    
    def __hash__(self) -> int:
        """Make ReachabilityState hashable for caching."""
        # Convert sets and lists to tuples for hashing
        reachable_tuple = tuple(sorted(self.reachable_positions))
        switch_tuple = tuple(sorted(self.switch_states.items()))
        unlocked_tuple = tuple(sorted(self.unlocked_areas))
        subgoals_tuple = tuple(self.subgoals)
        
        return hash((reachable_tuple, switch_tuple, unlocked_tuple, subgoals_tuple))