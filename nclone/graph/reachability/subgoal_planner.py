"""
Subgoal-based planning system for complex puzzle levels.

This system handles levels where the exit is not immediately reachable due to
locked doors, and requires a sequence of switch activations to open doors
in the correct order.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from ...constants.entity_types import EntityType
from ...constants.physics_constants import TILE_PIXEL_SIZE


@dataclass
class Subgoal:
    """Represents a subgoal in the planning process."""
    goal_type: str  # 'reach_exit', 'activate_switch', 'open_door'
    target_id: str  # entity_id of the target
    target_position: Tuple[float, float]  # (x, y) position
    dependencies: List['Subgoal']  # subgoals that must be completed first
    completed: bool = False


class SubgoalPlanner:
    """Plans sequences of actions to complete complex puzzle levels."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.entities = []
        self.switch_positions = {}  # switch_id -> (x, y)
        self.door_positions = {}    # door_id -> (x, y)
        self.switch_door_map = {}   # switch_id -> door_id
        self.door_switch_map = {}   # door_id -> switch_id
        
    def initialize(self, entities: List[Dict[str, Any]]):
        """Initialize the planner with level entities."""
        self.entities = entities
        self._parse_entities()
        self._build_relationships()
        
    def _parse_entities(self):
        """Parse entities to extract switches and doors."""
        self.switch_positions.clear()
        self.door_positions.clear()
        
        for entity in self.entities:
            entity_type = entity.get('type')
            entity_id = entity.get('entity_id', entity.get('id', 0))
            x, y = entity.get('x', 0), entity.get('y', 0)
            
            # Track switches
            if entity_type == EntityType.EXIT_SWITCH:
                self.switch_positions[entity_id] = (x, y)
            elif entity_type == EntityType.LOCKED_DOOR and not entity.get('is_door_part', False):
                self.switch_positions[entity_id] = (x, y)
            elif entity_type == EntityType.TRAP_DOOR and not entity.get('is_door_part', False):
                self.switch_positions[entity_id] = (x, y)
                
            # Track doors (door parts only)
            elif entity_type == EntityType.LOCKED_DOOR and entity.get('is_door_part', False):
                self.door_positions[entity_id] = (x, y)
            elif entity_type == EntityType.TRAP_DOOR and entity.get('is_door_part', False):
                self.door_positions[entity_id] = (x, y)
            elif entity_type == EntityType.EXIT_DOOR:
                self.door_positions[entity_id] = (x, y)
                
        if self.debug:
            print(f"DEBUG: SubgoalPlanner found {len(self.switch_positions)} switches, {len(self.door_positions)} doors")
            
    def _build_relationships(self):
        """Build switch-door relationships."""
        self.switch_door_map.clear()
        self.door_switch_map.clear()
        
        for entity in self.entities:
            entity_type = entity.get('type')
            entity_id = entity.get('entity_id', entity.get('id', 0))
            
            # For exit doors, check switch_entity_id
            if entity_type == EntityType.EXIT_DOOR:
                switch_id = entity.get('switch_entity_id')
                if switch_id:
                    self.switch_door_map[switch_id] = entity_id
                    self.door_switch_map[entity_id] = switch_id
                    
            # For locked/trap doors, the switch and door share the same entity_id
            elif entity_type in [EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR]:
                # The switch controls the door with the same entity_id
                if entity_id in self.switch_positions and entity_id in self.door_positions:
                    self.switch_door_map[entity_id] = entity_id
                    self.door_switch_map[entity_id] = entity_id
                    
        if self.debug:
            print(f"DEBUG: Switch-door relationships: {self.switch_door_map}")
            
    def plan_subgoals(self, reachable_positions: Set[Tuple[int, int]], 
                     exit_positions: List[Tuple[float, float]]) -> List[Subgoal]:
        """
        Plan the sequence of subgoals needed to complete the level.
        
        Args:
            reachable_positions: Set of (tile_x, tile_y) positions currently reachable
            exit_positions: List of (x, y) exit positions to reach
            
        Returns:
            List of subgoals in execution order
        """
        if self.debug:
            print(f"DEBUG: Planning subgoals for {len(exit_positions)} exits")
            
        # Try to reach any exit directly
        for exit_x, exit_y in exit_positions:
            exit_tile_x = int(exit_x // TILE_PIXEL_SIZE)
            exit_tile_y = int(exit_y // TILE_PIXEL_SIZE)
            
            if self.debug:
                print(f"DEBUG: Checking exit at ({exit_x}, {exit_y}) -> tile ({exit_tile_x}, {exit_tile_y})")
                print(f"DEBUG: Exit tile reachable: {(exit_tile_x, exit_tile_y) in reachable_positions}")
            
            if (exit_tile_x, exit_tile_y) in reachable_positions:
                if self.debug:
                    print(f"DEBUG: Exit at ({exit_x}, {exit_y}) is directly reachable")
                return []  # No subgoals needed
                
        # Exit not directly reachable - need to plan subgoals
        subgoals = []
        
        # Find which doors are blocking access to exits
        blocking_doors = self._find_blocking_doors(reachable_positions, exit_positions)
        
        if self.debug:
            print(f"DEBUG: Found {len(blocking_doors)} doors potentially blocking exits")
            
        # Create subgoals to open blocking doors
        for door_id in blocking_doors:
            door_subgoals = self._plan_door_opening(door_id, reachable_positions)
            subgoals.extend(door_subgoals)
            
        # Add final subgoal to reach exit
        if exit_positions:
            exit_x, exit_y = exit_positions[0]  # Use first exit
            subgoals.append(Subgoal(
                goal_type='reach_exit',
                target_id='exit',
                target_position=(exit_x, exit_y),
                dependencies=[]
            ))
            
        if self.debug:
            print(f"DEBUG: Generated {len(subgoals)} total subgoals")
            
        return subgoals
        
    def _find_blocking_doors(self, reachable_positions: Set[Tuple[int, int]], 
                           exit_positions: List[Tuple[float, float]]) -> List[str]:
        """Find doors that are likely blocking access to exits."""
        blocking_doors = []
        
        # Simple heuristic: find doors that are close to exits but not in reachable areas
        for exit_x, exit_y in exit_positions:
            exit_tile_x = int(exit_x // TILE_PIXEL_SIZE)
            exit_tile_y = int(exit_y // TILE_PIXEL_SIZE)
            
            # Check doors near this exit
            for door_id, (door_x, door_y) in self.door_positions.items():
                door_tile_x = int(door_x // TILE_PIXEL_SIZE)
                door_tile_y = int(door_y // TILE_PIXEL_SIZE)
                
                # If door is close to exit and exit area is not reachable
                distance = abs(door_tile_x - exit_tile_x) + abs(door_tile_y - exit_tile_y)
                if distance <= 5 and (exit_tile_x, exit_tile_y) not in reachable_positions:
                    if door_id not in blocking_doors:
                        blocking_doors.append(door_id)
                        if self.debug:
                            print(f"DEBUG: Door {door_id} at ({door_x}, {door_y}) may be blocking exit at ({exit_x}, {exit_y})")
                            
        return blocking_doors
        
    def _plan_door_opening(self, door_id: str, reachable_positions: Set[Tuple[int, int]]) -> List[Subgoal]:
        """Plan the subgoals needed to open a specific door."""
        subgoals = []
        
        # Find the switch that controls this door
        switch_id = self.door_switch_map.get(door_id)
        if not switch_id:
            if self.debug:
                print(f"DEBUG: No switch found for door {door_id}")
            return subgoals
            
        switch_x, switch_y = self.switch_positions.get(switch_id, (0, 0))
        switch_tile_x = int(switch_x // TILE_PIXEL_SIZE)
        switch_tile_y = int(switch_y // TILE_PIXEL_SIZE)
        
        if self.debug:
            print(f"DEBUG: Door {door_id} controlled by switch {switch_id} at ({switch_x}, {switch_y})")
            
        # Check if switch is directly reachable
        if (switch_tile_x, switch_tile_y) in reachable_positions:
            # Switch is reachable - just need to activate it
            subgoals.append(Subgoal(
                goal_type='activate_switch',
                target_id=switch_id,
                target_position=(switch_x, switch_y),
                dependencies=[]
            ))
        else:
            # Switch is not reachable - need to open doors to reach it
            if self.debug:
                print(f"DEBUG: Switch {switch_id} is not reachable, need to find path to it")
                
            # Recursively find doors blocking access to this switch
            switch_blocking_doors = self._find_doors_blocking_position(
                (switch_x, switch_y), reachable_positions
            )
            
            # Create subgoals for opening those doors first
            for blocking_door_id in switch_blocking_doors:
                door_subgoals = self._plan_door_opening(blocking_door_id, reachable_positions)
                subgoals.extend(door_subgoals)
                
            # Then add subgoal to activate the switch
            subgoals.append(Subgoal(
                goal_type='activate_switch',
                target_id=switch_id,
                target_position=(switch_x, switch_y),
                dependencies=[]
            ))
            
        return subgoals
        
    def _find_doors_blocking_position(self, target_position: Tuple[float, float], 
                                    reachable_positions: Set[Tuple[int, int]]) -> List[str]:
        """Find doors that are blocking access to a target position."""
        target_x, target_y = target_position
        target_tile_x = int(target_x // TILE_PIXEL_SIZE)
        target_tile_y = int(target_y // TILE_PIXEL_SIZE)
        
        blocking_doors = []
        
        # Simple heuristic: find doors between reachable area and target
        for door_id, (door_x, door_y) in self.door_positions.items():
            door_tile_x = int(door_x // TILE_PIXEL_SIZE)
            door_tile_y = int(door_y // TILE_PIXEL_SIZE)
            
            # Check if door is between reachable area and target
            # This is a simplified heuristic - in practice would need pathfinding
            min_distance_to_reachable = float('inf')
            for reachable_x, reachable_y in reachable_positions:
                distance = abs(door_tile_x - reachable_x) + abs(door_tile_y - reachable_y)
                min_distance_to_reachable = min(min_distance_to_reachable, distance)
                
            distance_to_target = abs(door_tile_x - target_tile_x) + abs(door_tile_y - target_tile_y)
            
            # If door is close to both reachable area and target, it might be blocking
            if min_distance_to_reachable <= 3 and distance_to_target <= 3:
                blocking_doors.append(door_id)
                if self.debug:
                    print(f"DEBUG: Door {door_id} at ({door_x}, {door_y}) may be blocking path to ({target_x}, {target_y})")
                    
        return blocking_doors
        
    def validate_plan(self, subgoals: List[Subgoal]) -> bool:
        """Validate that the subgoal plan is achievable."""
        if not subgoals:
            return True  # No subgoals needed
            
        # Check for circular dependencies
        if self._has_circular_dependencies(subgoals):
            if self.debug:
                print("DEBUG: Plan has circular dependencies")
            return False
            
        # Check that all required switches exist
        for subgoal in subgoals:
            if subgoal.goal_type == 'activate_switch':
                if subgoal.target_id not in self.switch_positions:
                    if self.debug:
                        print(f"DEBUG: Switch {subgoal.target_id} not found")
                    return False
                    
        return True
        
    def _has_circular_dependencies(self, subgoals: List[Subgoal]) -> bool:
        """Check if the subgoal plan has circular dependencies."""
        # Simple cycle detection - in practice would need more sophisticated algorithm
        visited = set()
        
        def visit(subgoal: Subgoal, path: Set[str]) -> bool:
            if subgoal.target_id in path:
                return True  # Circular dependency found
                
            if subgoal.target_id in visited:
                return False
                
            visited.add(subgoal.target_id)
            path.add(subgoal.target_id)
            
            for dep in subgoal.dependencies:
                if visit(dep, path):
                    return True
                    
            path.remove(subgoal.target_id)
            return False
            
        for subgoal in subgoals:
            if visit(subgoal, set()):
                return True
                
        return False