"""
Shared data types for reachability analysis systems.

This module contains common data structures used across different
reachability analysis implementations to avoid circular imports.
"""

from typing import Set, Tuple, Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class ReachabilityApproximation:
    """
    Result of a reachability approximation analysis.
    
    This represents a fast, approximate analysis of what positions
    the ninja can reach from a given starting position.
    """
    reachable_positions: Set[Tuple[int, int]]
    confidence: float  # 0.0 to 1.0, how confident we are in this approximation
    computation_time_ms: float
    method: str  # Description of the method used
    tier_used: int = 1  # Which tier was used for this analysis
    
    def is_position_reachable(self, position: Tuple[int, int]) -> bool:
        """Check if a specific position is reachable."""
        return position in self.reachable_positions
    
    def get_reachable_count(self) -> int:
        """Get the number of reachable positions."""
        return len(self.reachable_positions)
    
    def is_level_completable(self) -> bool:
        """
        Heuristic to determine if the level is completable.
        
        This is a connectivity-based heuristic that assumes if we can reach
        a reasonable number of positions, the level is likely completable.
        This works well for OpenCV flood fill which does pure connectivity analysis.
        """
        # Simple heuristic: if we can reach more than 3 positions, 
        # the level is probably completable (very permissive for connectivity analysis)
        return len(self.reachable_positions) > 3


@dataclass 
class ReachabilityResult:
    """
    Result of a comprehensive reachability analysis.
    
    This represents a more detailed analysis that may include
    additional information like subgoals, switch dependencies, etc.
    """
    reachable_positions: Set[Tuple[int, int]]
    confidence: float  # 0.0 to 1.0
    computation_time_ms: float
    method: str
    tier_used: int = 2
    
    # Additional analysis results
    switch_dependencies: Dict[str, bool] = None
    subgoals: Set[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.switch_dependencies is None:
            self.switch_dependencies = {}
        if self.subgoals is None:
            self.subgoals = set()
    
    def is_position_reachable(self, position: Tuple[int, int]) -> bool:
        """Check if a specific position is reachable."""
        return position in self.reachable_positions
    
    def get_reachable_count(self) -> int:
        """Get the number of reachable positions."""
        return len(self.reachable_positions)
    
    def is_level_completable(self) -> bool:
        """
        Heuristic to determine if the level is completable.
        
        This is a connectivity-based heuristic that assumes if we can reach
        a reasonable number of positions, the level is likely completable.
        This works well for OpenCV flood fill which does pure connectivity analysis.
        """
        # Simple heuristic: if we can reach more than 3 positions, 
        # the level is probably completable (very permissive for connectivity analysis)
        return len(self.reachable_positions) > 3


@dataclass
class CompletionStrategyInfo:
    """
    Enhanced completion strategy information for RL integration.
    
    This provides strategic information beyond simple reachability analysis,
    including switch dependencies and completion sequences for Deep RL agents.
    """
    primary_subgoals: List[str]  # Ordered list of primary objectives
    required_switches: List[Tuple[int, int]]  # Switch positions that must be activated
    blocking_doors: List[Tuple[int, int]]  # Door positions that block progress
    completion_sequence: List[str]  # Step-by-step completion plan
    estimated_difficulty: float  # 0.0 (easy) to 1.0 (very hard)
    alternative_paths: List[List[str]]  # Alternative completion strategies
    
    def get_next_subgoal(self) -> Optional[str]:
        """Get the next subgoal in the completion sequence."""
        return self.completion_sequence[0] if self.completion_sequence else None


class PerformanceTarget(Enum):
    """Performance targets for adaptive tier selection."""
    ULTRA_FAST = "ultra_fast"  # <1ms, lowest accuracy
    FAST = "fast"              # <5ms, medium accuracy  
    BALANCED = "balanced"      # <20ms, good accuracy
    ACCURATE = "accurate"      # <100ms, high accuracy
    PRECISE = "precise"        # No time limit, maximum accuracy