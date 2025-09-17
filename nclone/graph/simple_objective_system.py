"""
Simplified objective system for Phase 1 RL training.

This module provides a simple, reactive approach to level completion:
1. Check if exit switch is reachable → if yes, that's the goal
2. If not, find nearest reachable locked door switch → that's the goal
3. Check if exit door is reachable → if yes, that's the goal  
4. If not, find nearest reachable switch → that's the goal

This approach is optimized for RL learning with clear, unambiguous objectives.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math

from .reachability.tiered_system import TieredReachabilitySystem
from .reachability.reachability_types import PerformanceTarget


class ObjectiveType(Enum):
    """Types of objectives for the RL agent."""
    REACH_EXIT_SWITCH = "reach_exit_switch"
    REACH_DOOR_SWITCH = "reach_door_switch" 
    REACH_EXIT_DOOR = "reach_exit_door"
    REACH_SWITCH = "reach_switch"
    LEVEL_COMPLETE = "level_complete"
    NO_OBJECTIVE = "no_objective"


@dataclass
class SimpleObjective:
    """
    A simple objective for the RL agent.
    
    Attributes:
        objective_type: Type of objective
        position: Target position (x, y) in game coordinates
        distance: Distance from ninja to objective
        priority: Priority score (higher = more important)
        entity_id: ID of the target entity (switch/door)
        description: Human-readable description
    """
    objective_type: ObjectiveType
    position: Tuple[float, float]
    distance: float
    priority: float = 1.0
    entity_id: Optional[str] = None
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.objective_type.value} at {self.position}"


class SimplifiedCompletionStrategy:
    """
    Simplified completion strategy for Phase 1 RL training.
    
    This system provides clear, simple objectives that RL agents can learn to follow.
    It uses the tiered reachability system for fast objective validation.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the simplified completion strategy.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.tiered_system = TieredReachabilitySystem(debug=debug)
        self.current_objective: Optional[SimpleObjective] = None
        self.last_switch_states: Dict[str, bool] = {}
        
    def get_next_objective(
        self,
        ninja_position: Tuple[float, float],
        level_data: Any,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None
    ) -> Optional[SimpleObjective]:
        """
        Get the next simple objective for the RL agent.
        
        This method implements the simplified strategy:
        1. Check if exit switch is reachable → if yes, go there
        2. If not, find nearest reachable locked door switch → go there
        3. Check if exit door is reachable → if yes, go there
        4. If not, find nearest reachable switch → go there
        
        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level tile data
            entities: List of game entities
            switch_states: Current switch states (optional)
            
        Returns:
            SimpleObjective or None if no valid objective found
        """
        if switch_states is None:
            switch_states = {}
            
        # Check if we need to recompute (switch states changed)
        if switch_states != self.last_switch_states:
            self.last_switch_states = switch_states.copy()
            self.current_objective = None  # Force recomputation
            
        # Step 1: Check if exit switch is reachable
        exit_switch = self._find_exit_switch(entities)
        if exit_switch and self._is_reachable(ninja_position, exit_switch['position'], level_data, switch_states):
            objective = SimpleObjective(
                objective_type=ObjectiveType.REACH_EXIT_SWITCH,
                position=exit_switch['position'],
                distance=self._calculate_distance(ninja_position, exit_switch['position']),
                priority=1.0,
                entity_id=exit_switch.get('id'),
                description=f"Reach exit switch at {exit_switch['position']}"
            )
            self.current_objective = objective
            if self.debug:
                print(f"DEBUG: Exit switch reachable at {exit_switch['position']}")
            return objective
            
        # Step 2: If exit switch not reachable, find nearest reachable locked door switch
        if exit_switch:  # Exit switch exists but not reachable
            door_switch = self._find_nearest_reachable_door_switch(ninja_position, entities, level_data, switch_states)
            if door_switch:
                objective = SimpleObjective(
                    objective_type=ObjectiveType.REACH_DOOR_SWITCH,
                    position=door_switch['position'],
                    distance=self._calculate_distance(ninja_position, door_switch['position']),
                    priority=0.9,
                    entity_id=door_switch.get('id'),
                    description=f"Reach door switch at {door_switch['position']}"
                )
                self.current_objective = objective
                if self.debug:
                    print(f"DEBUG: Door switch reachable at {door_switch['position']}")
                return objective
                
        # Step 3: Check if exit door is reachable
        exit_door = self._find_exit_door(entities)
        if exit_door and self._is_reachable(ninja_position, exit_door['position'], level_data, switch_states):
            objective = SimpleObjective(
                objective_type=ObjectiveType.REACH_EXIT_DOOR,
                position=exit_door['position'],
                distance=self._calculate_distance(ninja_position, exit_door['position']),
                priority=1.0,
                entity_id=exit_door.get('id'),
                description=f"Reach exit door at {exit_door['position']}"
            )
            self.current_objective = objective
            if self.debug:
                print(f"DEBUG: Exit door reachable at {exit_door['position']}")
            return objective
            
        # Step 4: Find any reachable switch
        any_switch = self._find_nearest_reachable_switch(ninja_position, entities, level_data, switch_states)
        if any_switch:
            objective = SimpleObjective(
                objective_type=ObjectiveType.REACH_SWITCH,
                position=any_switch['position'],
                distance=self._calculate_distance(ninja_position, any_switch['position']),
                priority=0.5,
                entity_id=any_switch.get('id'),
                description=f"Reach switch at {any_switch['position']}"
            )
            self.current_objective = objective
            if self.debug:
                print(f"DEBUG: Any switch reachable at {any_switch['position']}")
            return objective
            
        # No valid objective found
        if self.debug:
            print("DEBUG: No reachable objectives found")
        return None
        
    def _is_reachable(
        self,
        ninja_position: Tuple[float, float],
        target_position: Tuple[float, float],
        level_data: Any,
        switch_states: Dict[str, bool]
    ) -> bool:
        """
        Check if target position is reachable from ninja position.
        
        Uses the tiered reachability system with FAST performance target
        for good balance of speed and accuracy.
        """
        try:
            result = self.tiered_system.analyze_reachability(
                level_data=level_data,
                ninja_position=ninja_position,
                switch_states=switch_states,
                performance_target=PerformanceTarget.FAST
            )
            
            # Convert target position to tile coordinates for comparison
            target_tile = (int(target_position[0] // 24), int(target_position[1] // 24))
            target_pixel = (target_tile[0] * 24 + 12, target_tile[1] * 24 + 12)  # Center of tile
            
            # Check if target pixel is in reachable positions
            is_reachable = target_pixel in result.reachable_positions
            
            if self.debug:
                print(f"DEBUG: Checking reachability from {ninja_position} to {target_position}")
                print(f"DEBUG: Target tile: {target_tile}, Target pixel: {target_pixel}")
                print(f"DEBUG: Reachable positions count: {len(result.reachable_positions)}")
                print(f"DEBUG: Is reachable: {is_reachable}")
            
            return is_reachable
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Reachability check failed: {e}")
            return False
            
    def _find_exit_switch(self, entities: List[Any]) -> Optional[Dict[str, Any]]:
        """Find the exit switch entity."""
        for entity in entities:
            if hasattr(entity, 'entity_type') and entity.entity_type == 'exit_switch':
                return {
                    'position': (entity.x, entity.y),
                    'id': getattr(entity, 'id', None),
                    'entity': entity
                }
        return None
        
    def _find_exit_door(self, entities: List[Any]) -> Optional[Dict[str, Any]]:
        """Find the exit door entity."""
        for entity in entities:
            if hasattr(entity, 'entity_type') and entity.entity_type == 'exit_door':
                return {
                    'position': (entity.x, entity.y),
                    'id': getattr(entity, 'id', None),
                    'entity': entity
                }
        return None
        
    def _find_nearest_reachable_door_switch(
        self,
        ninja_position: Tuple[float, float],
        entities: List[Any],
        level_data: Any,
        switch_states: Dict[str, bool]
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest reachable locked door switch."""
        door_switches = []
        
        for entity in entities:
            if hasattr(entity, 'entity_type') and 'door_switch' in entity.entity_type:
                # Skip if already activated
                entity_id = getattr(entity, 'id', None)
                if entity_id and switch_states.get(entity_id, False):
                    continue
                    
                position = (entity.x, entity.y)
                if self._is_reachable(ninja_position, position, level_data, switch_states):
                    distance = self._calculate_distance(ninja_position, position)
                    door_switches.append({
                        'position': position,
                        'distance': distance,
                        'id': entity_id,
                        'entity': entity
                    })
                    
        if door_switches:
            # Return nearest door switch
            door_switches.sort(key=lambda x: x['distance'])
            return door_switches[0]
            
        return None
        
    def _find_nearest_reachable_switch(
        self,
        ninja_position: Tuple[float, float],
        entities: List[Any],
        level_data: Any,
        switch_states: Dict[str, bool]
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest reachable switch of any type."""
        switches = []
        
        for entity in entities:
            if hasattr(entity, 'entity_type') and 'switch' in entity.entity_type:
                # Skip if already activated
                entity_id = getattr(entity, 'id', None)
                if entity_id and switch_states.get(entity_id, False):
                    continue
                    
                position = (entity.x, entity.y)
                if self._is_reachable(ninja_position, position, level_data, switch_states):
                    distance = self._calculate_distance(ninja_position, position)
                    switches.append({
                        'position': position,
                        'distance': distance,
                        'id': entity_id,
                        'entity': entity
                    })
                    
        if switches:
            # Return nearest switch
            switches.sort(key=lambda x: x['distance'])
            return switches[0]
            
        return None
        
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
    def get_current_objective(self) -> Optional[SimpleObjective]:
        """Get the current objective without recomputing."""
        return self.current_objective
        
    def clear_objective(self):
        """Clear the current objective (e.g., when objective is reached)."""
        self.current_objective = None
        
    def is_objective_reached(self, ninja_position: Tuple[float, float], threshold: float = 24.0) -> bool:
        """
        Check if the current objective has been reached.
        
        Args:
            ninja_position: Current ninja position
            threshold: Distance threshold for "reached" (default: 24 pixels = 1 tile)
            
        Returns:
            True if objective is reached
        """
        if not self.current_objective:
            return False
            
        distance = self._calculate_distance(ninja_position, self.current_objective.position)
        return distance <= threshold
        
    def get_objective_for_rl_features(self, ninja_position: Tuple[float, float]) -> Dict[str, float]:
        """
        Get objective information formatted for RL feature encoding.
        
        This method provides the objective data in a format suitable for
        TASK_003's compact feature encoding.
        
        Returns:
            Dictionary with objective features for RL integration
        """
        if not self.current_objective:
            return {
                'has_objective': 0.0,
                'objective_distance': 1.0,  # Max distance (unreachable)
                'objective_type_exit_switch': 0.0,
                'objective_type_door_switch': 0.0,
                'objective_type_exit_door': 0.0,
                'objective_type_switch': 0.0,
                'objective_priority': 0.0
            }
            
        # Normalize distance (0-1 range)
        max_distance = 1000.0  # Reasonable max for normalization
        normalized_distance = min(self.current_objective.distance / max_distance, 1.0)
        
        # One-hot encoding for objective type
        type_features = {
            'objective_type_exit_switch': 1.0 if self.current_objective.objective_type == ObjectiveType.REACH_EXIT_SWITCH else 0.0,
            'objective_type_door_switch': 1.0 if self.current_objective.objective_type == ObjectiveType.REACH_DOOR_SWITCH else 0.0,
            'objective_type_exit_door': 1.0 if self.current_objective.objective_type == ObjectiveType.REACH_EXIT_DOOR else 0.0,
            'objective_type_switch': 1.0 if self.current_objective.objective_type == ObjectiveType.REACH_SWITCH else 0.0,
        }
        
        return {
            'has_objective': 1.0,
            'objective_distance': normalized_distance,
            'objective_priority': self.current_objective.priority,
            **type_features
        }