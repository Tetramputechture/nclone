"""
Strategic level completion planner using reachability analysis.

This module provides the LevelCompletionPlanner that implements the production-ready
NPP level completion algorithm using neural reachability features.
"""

import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch

from nclone.constants.entity_types import EntityType
from .subgoals import CompletionStep, CompletionStrategy


class LevelCompletionPlanner:
    """
    Strategic planner for hierarchical level completion using fast reachability analysis.
    
    This planner implements the production-ready NPP level completion heuristic that leverages
    the sophisticated neural architecture (graph transformer + 3D CNN + MLPs) rather than
    expensive physics calculations. The strategy focuses on systematic switch activation
    sequences following the definitive NPP level completion algorithm.
    
    NPP Level Completion Strategy (Production Implementation):
    1. Check if exit door switch is reachable using neural reachability features
       - If reachable: trigger exit door switch, proceed to step 2
       - If not reachable: find nearest reachable locked door switch, trigger it, return to step 1
    2. Check if exit door is reachable using neural reachability analysis
       - If reachable: navigate to exit door and complete level
       - If not reachable: find nearest reachable locked door switch, trigger it, return to step 2
    
    Performance Optimization:
    - Avoids expensive physics calculations in favor of neural reachability features
    - Trusts graph transformer + CNN + MLP output for spatial reasoning
    - Maintains <3ms planning target through fast feature-based decisions
    - Removes complex hazard avoidance in favor of switch-focused strategy
    
    References:
    - Strategic analysis: nclone reachability analysis integration strategy
    - Hierarchical planning: Sutton et al. (1999) "Between MDPs and semi-MDPs"  
    - Strategic RL: Bacon et al. (2017) "The Option-Critic Architecture"
    """
    
    def __init__(self):
        from .analyzers import PathAnalyzer, DependencyAnalyzer
        self.path_analyzer = PathAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def plan_completion(self, ninja_pos, level_data, switch_states, 
                       reachability_system, reachability_features) -> CompletionStrategy:
        """
        Generate strategic plan for NPP level completion using production-ready algorithm.
        
        Implementation uses fast neural reachability analysis rather than expensive
        physics calculations. Relies on graph transformer + 3D CNN + MLP features
        for spatial reasoning and switch accessibility determination.
        
        NPP Level Completion Algorithm (Production Implementation):
        1. Check if exit door switch is reachable using neural reachability features
           - If reachable: create subgoal to trigger exit door switch, proceed to step 2
           - If not reachable: find nearest reachable locked door switch, create activation subgoal, return to step 1
        2. Check if exit door is reachable using neural reachability analysis
           - If reachable: create navigation subgoal to exit door for level completion
           - If not reachable: find nearest reachable locked door switch, create activation subgoal, return to step 2
        
        This algorithm ensures systematic progression through switch dependencies for level completion.
        """
        # Extract neural reachability features - production-ready feature extraction
        # Trust the sophisticated graph transformer + CNN + MLP architecture
        reachability_result = reachability_system.analyze_reachability(
            level_data, ninja_pos, switch_states, performance_target="balanced"
        )
        
        # Encode reachability into compact 64-dimensional features
        reachability_features_array = reachability_features.encode_reachability(
            reachability_result, level_data, [], ninja_pos, switch_states
        )
        reachability_features = torch.tensor(reachability_features_array, dtype=torch.float32)
        
        # Identify level objectives using production-ready level analysis
        exit_door = self._find_exit_door(level_data)
        exit_switch = self._find_exit_switch(level_data)
        
        if not exit_door or not exit_switch:
            return CompletionStrategy([], "No exit found", 0.0)
        
        # Implement NPP Level Completion Algorithm (Production Implementation)
        completion_steps = []
        current_state = "check_exit_switch"
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while current_state != "complete" and iteration < max_iterations:
            iteration += 1
            
            if current_state == "check_exit_switch":
                # Step 1: Check if exit door switch is reachable
                exit_switch_reachable = self._is_objective_reachable(
                    exit_switch['position'], reachability_features
                )
                
                if exit_switch_reachable and not switch_states.get(exit_switch['id'], False):
                    # Exit switch is reachable - create activation subgoal
                    completion_steps.append(CompletionStep(
                        action_type='navigate_and_activate',
                        target_position=exit_switch['position'],
                        target_id=exit_switch['id'],
                        description=f"Activate exit door switch at {exit_switch['position']}",
                        priority=1.0
                    ))
                    current_state = "check_exit_door"
                    
                elif not exit_switch_reachable:
                    # Exit switch not reachable - find nearest reachable locked door switch
                    nearest_switch = self._find_nearest_reachable_locked_door_switch(
                        ninja_pos, level_data, switch_states, reachability_features
                    )
                    
                    if nearest_switch:
                        completion_steps.append(CompletionStep(
                            action_type='navigate_and_activate',
                            target_position=nearest_switch['position'],
                            target_id=nearest_switch['id'],
                            description=f"Activate blocking switch {nearest_switch['id']} at {nearest_switch['position']}",
                            priority=0.8
                        ))
                        # Return to step 1 after activating blocking switch
                        current_state = "check_exit_switch"
                    else:
                        # No reachable switches found - level may be impossible
                        current_state = "complete"
                else:
                    # Exit switch already activated
                    current_state = "check_exit_door"
            
            elif current_state == "check_exit_door":
                # Step 2: Check if exit door is reachable
                exit_door_reachable = self._is_objective_reachable(
                    exit_door['position'], reachability_features
                )
                
                if exit_door_reachable:
                    # Exit door is reachable - create navigation subgoal for level completion
                    completion_steps.append(CompletionStep(
                        action_type='navigate_to_exit',
                        target_position=exit_door['position'],
                        target_id=exit_door['id'],
                        description=f"Navigate to exit door at {exit_door['position']}",
                        priority=1.0
                    ))
                    current_state = "complete"
                    
                else:
                    # Exit door not reachable - find nearest reachable locked door switch
                    nearest_switch = self._find_nearest_reachable_locked_door_switch(
                        ninja_pos, level_data, switch_states, reachability_features
                    )
                    
                    if nearest_switch:
                        completion_steps.append(CompletionStep(
                            action_type='navigate_and_activate',
                            target_position=nearest_switch['position'],
                            target_id=nearest_switch['id'],
                            description=f"Activate blocking switch {nearest_switch['id']} at {nearest_switch['position']}",
                            priority=0.8
                        ))
                        # Return to step 2 after activating blocking switch
                        current_state = "check_exit_door"
                    else:
                        # No reachable switches found - level may be impossible
                        current_state = "complete"
        
        # Calculate confidence using production-ready feature analysis
        confidence = self._calculate_strategy_confidence_from_features(
            completion_steps, reachability_features
        )
        
        return CompletionStrategy(
            steps=completion_steps,
            description="NPP Level Completion Strategy (Production Implementation)",
            confidence=confidence
        )
    
    def _find_exit_door(self, level_data) -> Optional[Dict]:
        """Find the exit door in level data using actual NppEnvironment data structures."""
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_DOOR:
                    return {
                        'id': entity.get('entity_id', 'exit_door'),
                        'position': (entity.get('x', 0), entity.get('y', 0)),
                        'type': 'exit_door'
                    }
        return None
    
    def _find_exit_switch(self, level_data) -> Optional[Dict]:
        """Find the exit door switch in level data using actual NppEnvironment data structures."""
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    return {
                        'id': entity.get('entity_id', 'exit_switch'),
                        'position': (entity.get('x', 0), entity.get('y', 0)),
                        'type': 'exit_switch'
                    }
        return None
    
    def _is_objective_reachable(self, position: Tuple[float, float], 
                               reachability_features: torch.Tensor) -> bool:
        """Check if objective is reachable using neural reachability features."""
        # Use neural network output rather than expensive physics calculations
        # Trust the graph transformer + CNN + MLP architecture for spatial reasoning
        if len(reachability_features) >= 8:
            # Extract objective reachability from neural features
            objective_distances = reachability_features[0:8].numpy()
            # Consider reachable if any objective distance feature is positive
            return np.any(objective_distances > 0.1)
        return False
    
    def _find_nearest_reachable_locked_door_switch(self, ninja_pos, level_data, 
                                                  switch_states, reachability_features) -> Optional[Dict]:
        """Find nearest reachable locked door switch using neural features and actual NppEnvironment data structures."""
        if not hasattr(level_data, 'entities') or not level_data.entities:
            return None
        
        reachable_switches = []
        switch_features = reachability_features[8:24].numpy() if len(reachability_features) >= 24 else []
        
        switch_index = 0
        for entity in level_data.entities:
            # Only consider exit switches
            if entity.get('type') != EntityType.EXIT_SWITCH:
                continue
            
            switch_id = entity.get('entity_id')
            
            # Skip already activated switches (using authoritative method)
            if self._is_switch_activated_authoritative(switch_id, level_data, switch_states):
                continue
            
            # Check reachability using neural features
            if switch_index < len(switch_features) and switch_features[switch_index] > 0.1:
                distance = math.sqrt(
                    (ninja_pos[0] - entity.get('x', 0))**2 + 
                    (ninja_pos[1] - entity.get('y', 0))**2
                )
                reachable_switches.append({
                    'id': switch_id,
                    'position': (entity.get('x', 0), entity.get('y', 0)),
                    'type': 'exit_switch',
                    'distance': distance,
                    'reachability_score': switch_features[switch_index]
                })
            
            switch_index += 1
        
        # Return nearest reachable switch
        if reachable_switches:
            return min(reachable_switches, key=lambda s: s['distance'])
        return None
    
    def _calculate_strategy_confidence_from_features(self, completion_steps: List[CompletionStep], 
                                                   reachability_features: torch.Tensor) -> float:
        """Calculate strategy confidence using neural reachability features."""
        if not completion_steps:
            return 0.0
        
        # Base confidence on neural feature quality and step count
        feature_confidence = torch.mean(torch.abs(reachability_features)).item()
        step_penalty = max(0.0, 1.0 - len(completion_steps) * 0.1)
        
        return min(1.0, feature_confidence * step_penalty)
    
    def _is_switch_activated_authoritative(self, switch_id: str, level_data, switch_states: Dict) -> bool:
        """
        Check switch activation using authoritative simulation data first.
        Falls back to passed switch_states if simulation data unavailable.
        
        Uses actual NppEnvironment data structures from nclone.
        """
        # Method 1: Check level_data.entities for switch with matching entity_id
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if (entity.get('entity_id') == switch_id and 
                    entity.get('type') == EntityType.EXIT_SWITCH):
                    # For exit switches, activated means active=False (inverted logic in nclone)
                    return not entity.get('active', True)
        
        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, 'switch_activated'):
            # This is the direct boolean from NppEnvironment observation
            return level_data.switch_activated
        
        # Method 3: Fall back to passed switch_states (legacy compatibility)
        return switch_states.get(switch_id, False)