"""
Planning module for hierarchical subgoal generation and level completion strategies.

This module provides reachability-based planning components that integrate with
the nclone physics simulation and reachability analysis systems.
"""

from .subgoals import (
    Subgoal,
    EntityInteractionSubgoal,
    NavigationSubgoal,  # Backward compatibility alias
    SwitchActivationSubgoal,  # Backward compatibility alias
    CompletionStep,
    CompletionStrategy,
    SubgoalPlan,
)

from .completion_planner import LevelCompletionPlanner
from .analyzers import PathAnalyzer, DependencyAnalyzer
from .prioritizer import SubgoalPrioritizer

__all__ = [
    "Subgoal",
    "EntityInteractionSubgoal",
    "NavigationSubgoal",  # Backward compatibility alias
    "SwitchActivationSubgoal",  # Backward compatibility alias
    "CompletionStep",
    "CompletionStrategy",
    "SubgoalPlan",
    "LevelCompletionPlanner",
    "PathAnalyzer",
    "DependencyAnalyzer",
    "SubgoalPrioritizer",
]
