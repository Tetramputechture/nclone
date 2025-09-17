"""
Shared types for subgoal planning system.

This module contains data classes and types used by both the subgoal planner
and reachability analysis components, preventing circular imports.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Subgoal:
    """Represents a single subgoal in hierarchical planning."""
    goal_type: str  # 'locked_door_switch', 'trap_door_switch', 'exit_switch', 'exit'
    position: Tuple[int, int]  # (sub_row, sub_col)
    node_idx: Optional[int] = None  # Graph node index
    priority: int = 0  # Lower numbers = higher priority
    dependencies: List[str] = field(default_factory=list)  # List of goal_types this depends on
    unlocks: List[str] = field(default_factory=list)  # List of goal_types this unlocks


@dataclass
class SubgoalPlan:
    """Complete plan with ordered subgoals and execution strategy."""
    subgoals: List[Subgoal]
    execution_order: List[int]  # Indices into subgoals list
    total_estimated_cost: float
    
    def get_next_subgoal(self) -> Optional[Subgoal]:
        """Get the next subgoal to execute."""
        if self.execution_order:
            next_idx = self.execution_order[0]
            return self.subgoals[next_idx] if 0 <= next_idx < len(self.subgoals) else None
        return None
    
    def mark_completed(self, subgoal_idx: int):
        """Mark a subgoal as completed and remove from execution order."""
        if subgoal_idx in self.execution_order:
            self.execution_order.remove(subgoal_idx)


@dataclass
class CompletionStrategyInfo:
    """Information about level completion strategy for RL agents."""
    is_completable: bool
    confidence: float
    primary_subgoals: List[str]  # Ordered list of primary objectives
    required_switches: List[Tuple[int, int]]  # Switch positions that must be activated
    blocking_doors: List[Tuple[int, int]]  # Door positions that block progress
    switch_dependencies: Dict[Tuple[int, int], List[Tuple[int, int]]]  # switch_pos -> [required_switches]
    completion_sequence: List[str]  # Step-by-step completion plan
    alternative_paths: List[List[str]]  # Alternative completion strategies
    estimated_difficulty: float  # 0.0 (easy) to 1.0 (very hard)
    
    def get_next_subgoal(self) -> Optional[str]:
        """Get the next subgoal in the completion sequence."""
        return self.completion_sequence[0] if self.completion_sequence else None
    
    def get_reachable_subgoals(self, reachable_positions: set) -> List[str]:
        """Filter subgoals to only those currently reachable."""
        reachable_subgoals = []
        for subgoal in self.primary_subgoals:
            if self._is_subgoal_reachable(subgoal, reachable_positions):
                reachable_subgoals.append(subgoal)
        return reachable_subgoals
    
    def _is_subgoal_reachable(self, subgoal: str, reachable_positions: set) -> bool:
        """Check if a specific subgoal is reachable given current reachable positions."""
        # Extract position from subgoal string and check if it's in reachable set
        # This is a simplified implementation - in practice would need more sophisticated parsing
        return True  # Placeholder - would implement proper position extraction