"""
Hierarchical RL functionality mixin for N++ environment.

This module contains hierarchical RL functionality including completion planner
integration, subtask management, and reward shaping for strategic gameplay.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from ...planning.completion_planner import LevelCompletionPlanner


class Subtask(Enum):
    """Enumeration of available subtasks for hierarchical control."""
    NAVIGATE_TO_EXIT_SWITCH = 0
    NAVIGATE_TO_LOCKED_DOOR_SWITCH = 1
    NAVIGATE_TO_EXIT_DOOR = 2
    AVOID_MINE = 3


class HierarchicalMixin:
    """
    Mixin class providing hierarchical RL functionality for N++ environment.
    
    This mixin handles:
    - Completion planner integration for strategic subtask selection
    - Subtask state management and transitions
    - Subtask-specific reward shaping
    - Performance tracking and logging
    """
    
    def _init_hierarchical_system(
        self,
        enable_hierarchical: bool = True,
        completion_planner: Optional[LevelCompletionPlanner] = None,
        enable_subtask_rewards: bool = True,
        subtask_reward_scale: float = 0.1,
        max_subtask_steps: int = 1000,
        debug: bool = False
    ):
        """Initialize the hierarchical system components."""
        self.enable_hierarchical = enable_hierarchical
        self.enable_subtask_rewards = enable_subtask_rewards
        self.subtask_reward_scale = subtask_reward_scale
        self.max_subtask_steps = max_subtask_steps
        self.debug = debug
        
        # Initialize completion planner
        self.completion_planner = completion_planner or LevelCompletionPlanner()
        
        # Hierarchical state tracking
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = 0
        self.subtask_step_count = 0
        
        # Subtask transition history for logging
        self.subtask_history = []
        self.last_switch_states = {}
        self.last_ninja_pos = None
        
        # Performance tracking
        self.hierarchical_times = []
        self.max_time_samples = 100
        
        if self.debug:
            logging.info("Hierarchical system initialized")
    
    def _get_current_subtask(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Subtask:
        """
        Use completion planner to determine current subtask.
        
        Args:
            obs: Environment observation containing multimodal data
            info: Environment info containing game state
            
        Returns:
            Current subtask based on completion planner analysis
        """
        if not self.enable_hierarchical:
            return self.current_subtask
            
        start_time = time.time()
        
        try:
            # Extract game state information
            ninja_pos = self._extract_ninja_position(obs, info)
            level_data = self._extract_level_data(obs, info)
            switch_states = self._extract_switch_states(obs, info)
            reachability_features = self._extract_reachability_features(obs)
            
            # Check if we should switch subtasks based on completion planner
            if self._should_switch_subtask(obs, info):
                new_subtask = self._determine_next_subtask(
                    ninja_pos, level_data, switch_states, reachability_features
                )
                
                if new_subtask != self.current_subtask:
                    self._transition_to_subtask(new_subtask)
            
        except Exception as e:
            if self.debug:
                logging.warning(f"Hierarchical subtask selection failed: {e}")
        
        # Track performance
        elapsed_time = time.time() - start_time
        self.hierarchical_times.append(elapsed_time)
        if len(self.hierarchical_times) > self.max_time_samples:
            self.hierarchical_times.pop(0)
        
        return self.current_subtask
    
    def _should_switch_subtask(self, obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
        """
        Determine if subtask should change based on completion planner.
        
        Args:
            obs: Environment observation
            info: Environment info
            
        Returns:
            True if subtask should switch, False otherwise
        """
        # Check for forced transition due to step limit
        if self.subtask_step_count >= self.max_subtask_steps:
            return True
            
        # Check for completion of current subtask
        switch_states = self._extract_switch_states(obs, info)
        ninja_pos = self._extract_ninja_position(obs, info)
        
        # Detect switch state changes (subtask completion)
        if self.last_switch_states != switch_states:
            return True
            
        # Check for specific subtask completion conditions
        if self.current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Check if exit switch was activated
            exit_switch_id = self._find_exit_switch_id(obs, info)
            if exit_switch_id and switch_states.get(exit_switch_id, False):
                return True
                
        elif self.current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Check if any locked door switch was activated
            for switch_id, activated in switch_states.items():
                if activated and not self.last_switch_states.get(switch_id, False):
                    return True
                    
        elif self.current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Check if ninja reached exit (level completion)
            if info.get('level_complete', False):
                return True
                
        # Check for significant position change (potential mine avoidance completion)
        if self.current_subtask == Subtask.AVOID_MINE:
            if self.last_ninja_pos and ninja_pos:
                distance_moved = np.linalg.norm(
                    np.array(ninja_pos) - np.array(self.last_ninja_pos)
                )
                if distance_moved > 5.0:  # Moved significant distance
                    return True
        
        return False
    
    def _determine_next_subtask(
        self, 
        ninja_pos: Tuple[int, int], 
        level_data: Dict[str, Any], 
        switch_states: Dict[str, bool],
        reachability_features: np.ndarray
    ) -> Subtask:
        """
        Determine the next subtask using completion planner logic.
        
        Args:
            ninja_pos: Current ninja position
            level_data: Level layout data
            switch_states: Current switch activation states
            reachability_features: 8D reachability features
            
        Returns:
            Next subtask to execute
        """
        try:
            # Use completion planner to get strategic plan
            reachability_system = self._create_reachability_system(reachability_features)
            
            completion_strategy = self.completion_planner.plan_completion(
                ninja_pos, level_data, switch_states, reachability_system
            )
            
            if completion_strategy.steps:
                # Map completion step to subtask
                first_step = completion_strategy.steps[0]
                return self._map_completion_step_to_subtask(first_step)
                
        except Exception as e:
            if self.debug:
                logging.warning(f"Completion planner failed: {e}")
            
        # Fallback logic based on reachability features and game state
        return self._fallback_subtask_selection(switch_states, reachability_features)
    
    def _map_completion_step_to_subtask(self, completion_step) -> Subtask:
        """Map completion planner step to subtask enum."""
        action_type = completion_step.action_type
        
        if action_type == "navigate_and_activate":
            # Determine if it's exit switch or locked door switch
            if "exit" in completion_step.description.lower():
                return Subtask.NAVIGATE_TO_EXIT_SWITCH
            else:
                return Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
        elif action_type == "navigate_to_exit":
            return Subtask.NAVIGATE_TO_EXIT_DOOR
        else:
            # Default to exit switch navigation
            return Subtask.NAVIGATE_TO_EXIT_SWITCH
    
    def _fallback_subtask_selection(
        self, 
        switch_states: Dict[str, bool], 
        reachability_features: np.ndarray
    ) -> Subtask:
        """Fallback subtask selection when completion planner fails."""
        # Simple heuristic: if exit switch not activated, go for it
        # Otherwise, go for exit door
        exit_switch_activated = any(switch_states.values())
        
        if not exit_switch_activated:
            return Subtask.NAVIGATE_TO_EXIT_SWITCH
        else:
            return Subtask.NAVIGATE_TO_EXIT_DOOR
    
    def _transition_to_subtask(self, new_subtask: Subtask):
        """Transition to a new subtask with logging."""
        old_subtask = self.current_subtask
        self.current_subtask = new_subtask
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0
        
        # Log transition
        transition = {
            'timestamp': time.time(),
            'from_subtask': old_subtask.name,
            'to_subtask': new_subtask.name,
            'step_count': self.subtask_step_count
        }
        self.subtask_history.append(transition)
        
        if self.debug:
            logging.info(f"Subtask transition: {old_subtask.name} -> {new_subtask.name}")
    
    def _calculate_subtask_reward(
        self, 
        current_subtask: Subtask, 
        obs: Dict[str, Any], 
        info: Dict[str, Any], 
        terminated: bool
    ) -> float:
        """
        Calculate subtask-specific reward shaping.
        
        Args:
            current_subtask: Current subtask enum
            obs: Observation dictionary
            info: Environment info
            terminated: Whether episode terminated
            
        Returns:
            Subtask-specific reward
        """
        if not self.enable_subtask_rewards:
            return 0.0
            
        reward = 0.0
        
        # Reward based on subtask progress
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Reward getting closer to exit switch
            if 'switch_distance' in info:
                # Negative distance as reward (closer = higher reward)
                reward += -info['switch_distance'] * 0.01
                
        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Reward getting closer to locked door switches
            if 'locked_door_distance' in info:
                reward += -info['locked_door_distance'] * 0.01
                
        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Reward getting closer to exit door
            if 'exit_distance' in info:
                reward += -info['exit_distance'] * 0.01
                
        elif current_subtask == Subtask.AVOID_MINE:
            # Reward staying away from mines
            if 'mine_distance' in info:
                reward += info['mine_distance'] * 0.005  # Positive distance reward
        
        # Bonus for subtask completion
        if 'subtask_transition' in info:
            reward += 0.5  # Bonus for successful subtask transition
        
        # Penalty for taking too long on a subtask
        if self.subtask_step_count > 500:  # 500 steps without progress
            reward -= 0.1
        
        return reward
    
    def _update_hierarchical_state(self, obs: Dict[str, Any], info: Dict[str, Any]):
        """Update hierarchical state after environment step."""
        if not self.enable_hierarchical:
            return
            
        self.subtask_step_count += 1
        self.last_switch_states = self._extract_switch_states(obs, info)
        self.last_ninja_pos = self._extract_ninja_position(obs, info)
    
    def _reset_hierarchical_state(self):
        """Reset hierarchical state for new episode."""
        if not self.enable_hierarchical:
            return
            
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0
        self.last_switch_states = {}
        self.last_ninja_pos = None
    
    def _get_subtask_features(self) -> np.ndarray:
        """
        Get current subtask as one-hot encoded features.
        
        Returns:
            4-dimensional one-hot vector representing current subtask
        """
        if not self.enable_hierarchical:
            return np.zeros(4, dtype=np.float32)
            
        features = np.zeros(4, dtype=np.float32)
        features[self.current_subtask.value] = 1.0
        return features
    
    def _get_hierarchical_info(self) -> Dict[str, Any]:
        """Get hierarchical information for environment info."""
        if not self.enable_hierarchical:
            return {}
            
        return {
            'current_subtask': self.current_subtask.name,
            'subtask_features': self._get_subtask_features(),
            'subtask_step_count': self.subtask_step_count,
            'subtask_duration': time.time() - self.subtask_start_time,
            'total_transitions': len(self.subtask_history),
            'recent_transitions': self.subtask_history[-5:] if self.subtask_history else []
        }
    
    def _get_hierarchical_performance_stats(self) -> Dict[str, float]:
        """Get hierarchical performance statistics."""
        if not self.hierarchical_times:
            return {}
            
        return {
            'avg_hierarchical_time': np.mean(self.hierarchical_times),
            'max_hierarchical_time': np.max(self.hierarchical_times),
            'min_hierarchical_time': np.min(self.hierarchical_times),
            'hierarchical_time_std': np.std(self.hierarchical_times),
        }
    
    # Helper methods for extracting information from observations
    def _extract_ninja_position(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, int]:
        """Extract ninja position from observation/info."""
        # Try to get from info first
        if 'ninja_pos' in info:
            return tuple(info['ninja_pos'])
        
        # Fallback: try to extract from observation
        # This would need to be implemented based on actual observation structure
        return (0, 0)  # Placeholder
    
    def _extract_level_data(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract level data from observation/info."""
        # This would extract level layout information
        return info.get('level_data', {})
    
    def _extract_switch_states(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, bool]:
        """Extract switch states from observation/info."""
        return info.get('switch_states', {})
    
    def _extract_reachability_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract 8D reachability features from observation."""
        # Try to get from observation if available
        if 'reachability_features' in obs:
            return obs['reachability_features']
        
        # Fallback: return zeros as placeholder
        return np.zeros(8, dtype=np.float32)
    
    def _find_exit_switch_id(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Optional[str]:
        """Find the exit switch ID from level data."""
        level_data = self._extract_level_data(obs, info)
        # This would find the exit switch in the level data
        return None  # Placeholder
    
    def _create_reachability_system(self, reachability_features: np.ndarray):
        """Create a mock reachability system for the completion planner."""
        # This would create a reachability system compatible with the planner
        # For now, return a simple mock
        class MockReachabilitySystem:
            def analyze_reachability(self, level_data, ninja_pos, switch_states):
                return {'reachable_positions': set(), 'features': reachability_features}
        
        return MockReachabilitySystem()