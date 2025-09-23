"""
Hierarchical subgoal framework for reachability-guided planning.

This module provides the core subgoal classes that integrate with nclone's
reachability analysis systems for strategic level completion planning.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from nclone.constants.entity_types import EntityType


@dataclass
class Subgoal(ABC):
    """
    Base class for hierarchical subgoals in the Options framework.

    This abstract base class defines the interface for all hierarchical subgoals
    used in the reachability-guided HRL system. Implementation follows the Options
    framework from Sutton et al. (1999) adapted for NPP level completion.

    Subgoals represent temporally extended actions with clear termination conditions
    and progress measurement capabilities for reward shaping integration.

    References:
    - Options framework: Sutton et al. (1999) "Between MDPs and semi-MDPs"
    - Hierarchical planning: Bacon et al. (2017) "The Option-Critic Architecture"
    - NPP-specific implementation: Custom design for level completion objectives
    """

    priority: float  # Strategic priority for subgoal selection (0.0-1.0)
    estimated_time: float  # Estimated completion time in seconds
    success_probability: float  # Likelihood of successful completion (0.0-1.0)
    
    # Enhanced fields from graph/subgoal_types.py for better integration
    goal_type: str = ""  # 'locked_door_switch', 'trap_door_switch', 'exit_switch', 'exit'
    position: Optional[Tuple[int, int]] = None  # (sub_row, sub_col) for graph integration
    node_idx: Optional[int] = None  # Graph node index for pathfinding
    dependencies: List[str] = field(default_factory=list)  # List of goal_types this depends on
    unlocks: List[str] = field(default_factory=list)  # List of goal_types this unlocks

    @abstractmethod
    def get_target_position(self) -> Tuple[float, float]:
        """Get the target position for this subgoal."""
        pass

    @abstractmethod
    def is_completed(
        self, ninja_pos: Tuple[float, float], level_data, switch_states: Dict
    ) -> bool:
        """Check if this subgoal has been completed."""
        pass

    @abstractmethod
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        """Get reward shaping signal for progress toward this subgoal."""
        pass


@dataclass
class NavigationSubgoal(Subgoal):
    """Navigate to a specific position."""

    target_position: Tuple[float, float] = (0.0, 0.0)
    target_type: str = "navigation"  # 'exit_door', 'exit_switch', 'door_switch', etc.
    distance: float = 0.0

    def get_target_position(self) -> Tuple[float, float]:
        return self.target_position

    def is_completed(
        self, ninja_pos: Tuple[float, float], level_data, switch_states: Dict
    ) -> bool:
        # Check if ninja is within interaction range of target
        distance = math.sqrt(
            (ninja_pos[0] - self.target_position[0]) ** 2
            + (ninja_pos[1] - self.target_position[1]) ** 2
        )
        return distance <= 24.0  # One tile distance

    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        # Reward for getting closer to target
        distance = math.sqrt(
            (ninja_pos[0] - self.target_position[0]) ** 2
            + (ninja_pos[1] - self.target_position[1]) ** 2
        )
        max_distance = 500.0  # Normalize by reasonable max distance
        return (max_distance - distance) / max_distance


@dataclass
class SwitchActivationSubgoal(Subgoal):
    """Activate a specific switch."""

    switch_id: str = ""
    switch_position: Tuple[float, float] = (0.0, 0.0)
    switch_type: str = "switch"
    reachability_score: float = 0.0

    def get_target_position(self) -> Tuple[float, float]:
        return self.switch_position

    def is_completed(
        self, ninja_pos: Tuple[float, float], level_data, switch_states: Dict
    ) -> bool:
        # Use authoritative simulation data first, fall back to passed states
        return self._is_switch_activated_authoritative(
            self.switch_id, level_data, switch_states
        )

    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        # Reward for getting closer to switch
        distance = math.sqrt(
            (ninja_pos[0] - self.switch_position[0]) ** 2
            + (ninja_pos[1] - self.switch_position[1]) ** 2
        )
        max_distance = 500.0
        proximity_reward = (max_distance - distance) / max_distance

        # Bonus for high reachability score
        reachability_bonus = self.reachability_score * 0.5

        return proximity_reward + reachability_bonus

    def _is_switch_activated_authoritative(
        self, switch_id: str, level_data, switch_states: Dict
    ) -> bool:
        """
        Check switch activation using authoritative simulation data first.
        Falls back to passed switch_states if simulation data unavailable.

        Uses actual NppEnvironment data structures from nclone.
        """
        # Method 1: Check level_data.entities for switch with matching entity_id
        if hasattr(level_data, "entities") and level_data.entities:
            for entity in level_data.entities:
                if (
                    entity.get("entity_id") == switch_id
                    and entity.get("type") == EntityType.EXIT_SWITCH
                ):
                    # For exit switches, activated means active=False (inverted logic in nclone)
                    return not entity.get("active", True)

        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, "switch_activated"):
            # This is the direct boolean from NppEnvironment observation
            return level_data.switch_activated

        # Method 3: Fall back to passed switch_states (legacy compatibility)
        return switch_states.get(switch_id, False)


@dataclass
class CompletionStep:
    """Single step in a level completion strategy."""

    action_type: str  # 'navigate_and_activate', 'navigate_to_exit', etc.
    target_position: Tuple[float, float]
    target_id: str
    description: str
    priority: float


@dataclass
class CompletionStrategy:
    """Strategic plan for level completion."""

    steps: List[CompletionStep]
    description: str
    confidence: float

    def get_next_subgoal(self) -> Optional["Subgoal"]:
        """Get the next subgoal from the completion strategy."""
        if not self.steps:
            return None

        # Convert first step to appropriate subgoal
        step = self.steps[0]

        if step.action_type == "navigate_and_activate":
            return SwitchActivationSubgoal(
                switch_id=step.target_id,
                switch_position=step.target_position,
                switch_type="exit_switch",
                reachability_score=0.8,
                priority=step.priority,
                estimated_time=5.0,
                success_probability=0.9,
            )
        elif step.action_type == "navigate_to_exit":
            return NavigationSubgoal(
                target_position=step.target_position,
                target_type="exit_door",
                distance=0.0,
                priority=step.priority,
                estimated_time=3.0,
                success_probability=0.95,
            )

        return None


@dataclass
class SubgoalPlan:
    """
    Complete plan with ordered subgoals and execution strategy.
    
    Unified class combining features from both planning and graph approaches
    for comprehensive subgoal planning with execution tracking.
    """
    subgoals: List[Subgoal]
    execution_order: List[int]  # Indices into subgoals list
    total_estimated_cost: float
    description: str = ""
    confidence: float = 0.0
    
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
    
    def get_progress(self) -> float:
        """Get completion progress as a percentage (0.0 to 1.0)."""
        if not self.subgoals:
            return 1.0
        completed = len(self.subgoals) - len(self.execution_order)
        return completed / len(self.subgoals)
