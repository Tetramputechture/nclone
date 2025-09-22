"""
Subgoal prioritizer for strategic planning.

This module provides the SubgoalPrioritizer that ranks subgoals based on
strategic value and neural reachability analysis.
"""

import math
from typing import List, Tuple

import torch

from .subgoals import Subgoal, NavigationSubgoal, SwitchActivationSubgoal, CollectionSubgoal


class SubgoalPrioritizer:
    """
    Prioritizes hierarchical subgoals based on strategic value and completion likelihood.
    
    This prioritizer uses neural reachability features and strategic analysis to rank
    subgoals for optimal level completion. Implementation focuses on switch-based
    progression following the NPP level completion heuristic.
    
    Prioritization Strategy:
    - Exit-related subgoals receive highest priority
    - Switch activation subgoals prioritized by strategic value
    - Collection subgoals weighted by value and accessibility
    - Navigation subgoals ranked by distance and strategic importance
    
    References:
    - Strategic planning: Custom NPP level completion analysis
    - Hierarchical RL: Bacon et al. (2017) "The Option-Critic Architecture"
    - Reachability integration: Neural feature-based prioritization
    """
    
    def __init__(self):
        self.strategic_weights = {
            'exit_door': 1.0,
            'exit_switch': 0.9,
            'door_switch': 0.7,
            'collectible': 0.3,
            'exploration': 0.1
        }
    
    def prioritize(self, subgoals: List[Subgoal], ninja_pos: Tuple[float, float], 
                  level_data, reachability_features: torch.Tensor) -> List[Subgoal]:
        """
        Prioritize subgoals based on strategic value and neural reachability analysis.
        
        Uses neural network features (graph transformer + CNN + MLP) for strategic
        assessment rather than expensive physics calculations.
        """
        if not subgoals:
            return []
        
        # Calculate priority scores for each subgoal
        scored_subgoals = []
        for subgoal in subgoals:
            priority_score = self._calculate_priority_score(
                subgoal, ninja_pos, level_data, reachability_features
            )
            scored_subgoals.append((subgoal, priority_score))
        
        # Sort by priority score (highest first)
        scored_subgoals.sort(key=lambda x: x[1], reverse=True)
        
        # Update subgoal priorities and return sorted list
        prioritized_subgoals = []
        for subgoal, score in scored_subgoals:
            subgoal.priority = score
            prioritized_subgoals.append(subgoal)
        
        return prioritized_subgoals
    
    def _calculate_priority_score(self, subgoal: Subgoal, ninja_pos: Tuple[float, float],
                                 level_data, reachability_features: torch.Tensor) -> float:
        """Calculate priority score for a subgoal using neural features."""
        base_score = subgoal.priority
        
        # Strategic value based on subgoal type
        if isinstance(subgoal, NavigationSubgoal):
            strategic_weight = self.strategic_weights.get(subgoal.target_type, 0.5)
        elif isinstance(subgoal, SwitchActivationSubgoal):
            strategic_weight = self.strategic_weights.get(subgoal.switch_type, 0.7)
        elif isinstance(subgoal, CollectionSubgoal):
            strategic_weight = self.strategic_weights.get('collectible', 0.3)
        else:
            strategic_weight = 0.5
        
        # Distance penalty (closer is better)
        target_pos = subgoal.get_target_position()
        distance = math.sqrt((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)
        distance_factor = max(0.1, 1.0 - distance / 500.0)  # Normalize by max reasonable distance
        
        # Reachability bonus from neural features
        reachability_bonus = self._get_reachability_bonus(subgoal, reachability_features)
        
        # Success probability factor
        success_factor = subgoal.success_probability
        
        # Combine factors
        priority_score = (base_score * strategic_weight * distance_factor * 
                         success_factor + reachability_bonus)
        
        return min(1.0, priority_score)
    
    def _get_reachability_bonus(self, subgoal: Subgoal, reachability_features: torch.Tensor) -> float:
        """Get reachability bonus from neural features."""
        if isinstance(subgoal, SwitchActivationSubgoal):
            return subgoal.reachability_score * 0.2
        elif len(reachability_features) > 0:
            # Use average neural feature strength as reachability indicator
            return torch.mean(torch.abs(reachability_features)).item() * 0.1
        return 0.0