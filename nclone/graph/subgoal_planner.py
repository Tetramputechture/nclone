"""
Simplified subgoal planning for Phase 1 RL training.

This module provides a simplified, reactive approach to level completion
that is optimized for RL learning. It uses the SimplifiedCompletionStrategy
to provide clear, unambiguous objectives.

For backward compatibility, it maintains the same API as the original
hierarchical planner while using the simplified approach internally.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

from .common import SUB_CELL_SIZE
from .reachability.reachability_types import ReachabilityApproximation, ReachabilityResult
from .navigation import PathfindingEngine
from .reachability.opencv_flood_fill import OpenCVFloodFill
from .subgoal_types import Subgoal, SubgoalPlan, CompletionStrategyInfo
from .simple_objective_system import SimplifiedCompletionStrategy, SimpleObjective, ObjectiveType


class SubgoalPlanner:
    """
    Simplified planner for Phase 1 RL training.
    
    This class provides a simplified, reactive approach to level completion
    that is optimized for RL learning. It maintains backward compatibility
    with the original hierarchical API while using simplified logic internally.
    
    The simplified approach:
    1. Check if exit switch is reachable → if yes, that's the goal
    2. If not, find nearest reachable locked door switch → that's the goal
    3. Check if exit door is reachable → if yes, that's the goal
    4. If not, find nearest reachable switch → that's the goal
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize simplified subgoal planner.
        
        Args:
            debug: Enable debug output (default: False)
        """
        self.navigation_engine = PathfindingEngine(debug=debug)
        self.debug = debug
        self.simplified_strategy = SimplifiedCompletionStrategy(debug=debug)
        
    def create_subgoal_plan(
        self, 
        reachability_state,  # Union[ReachabilityApproximation, ReachabilityResult]
        graph_data,
        ninja_node_idx: int,
        target_goal_type: str = 'exit'
    ) -> Optional[SubgoalPlan]:
        """
        Create hierarchical plan to reach target goal.
        
        Args:
            reachability_state: Current reachability analysis results
            graph_data: Graph data with node positions
            ninja_node_idx: Starting node index for ninja
            target_goal_type: Final objective type ('exit', 'exit_switch', etc.)
            
        Returns:
            SubgoalPlan with ordered sequence of subgoals, or None if impossible
        """
        # Convert reachability subgoals to Subgoal objects
        subgoals = self._create_subgoal_objects(reachability_state, graph_data)
        
        if not subgoals:
            if self.debug:
                print("DEBUG: No subgoals found in reachability analysis")
            return None
            
        # Analyze dependencies between subgoals
        self._analyze_subgoal_dependencies(subgoals)
        
        # Find target subgoal
        target_subgoal = None
        for subgoal in subgoals:
            if subgoal.goal_type == target_goal_type:
                target_subgoal = subgoal
                break
                
        if not target_subgoal:
            if self.debug:
                print(f"DEBUG: Target goal type '{target_goal_type}' not found in subgoals")
            return None
            
        # Create execution plan using topological sort
        execution_order = self._create_execution_order(subgoals, target_subgoal)
        
        if not execution_order:
            if self.debug:
                print("DEBUG: Could not create valid execution order (circular dependencies?)")
            return None
            
        # Estimate total cost
        total_cost = self._estimate_plan_cost(subgoals, execution_order, ninja_node_idx)
        
        plan = SubgoalPlan(
            subgoals=subgoals,
            execution_order=execution_order,
            total_estimated_cost=total_cost
        )
        
        if self.debug:
            print(f"DEBUG: Created subgoal plan with {len(execution_order)} steps, estimated cost: {total_cost:.1f}")
        return plan
    
    def _create_subgoal_objects(
        self, 
        reachability_state,  # Union[ReachabilityApproximation, ReachabilityResult]
        graph_data
    ) -> List[Subgoal]:
        """Convert reachability subgoals to Subgoal objects with node indices."""
        subgoals = []
        
        # Handle both ReachabilityApproximation and ReachabilityResult
        subgoals_data = getattr(reachability_state, 'subgoals', set())
        if not subgoals_data:
            return []
            
        for sub_row, sub_col, goal_type in subgoals_data:
            # Find closest graph node to this subgoal position
            pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            # Find closest node by checking all valid nodes
            closest_node_idx = None
            min_distance = float('inf')
            
            for node_idx in range(graph_data.num_nodes):
                if graph_data.node_mask[node_idx] == 0:
                    continue
                    
                node_pos = self._get_node_position(graph_data, node_idx)
                distance = np.sqrt((node_pos[0] - pixel_x) ** 2 + (node_pos[1] - pixel_y) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_node_idx = node_idx
            
            # Set priority based on goal type
            priority_map = {
                'locked_door_switch': 1,
                'trap_door_switch': 2,
                'exit_switch': 3,
                'exit': 4
            }
            
            if closest_node_idx is not None:
                subgoal = Subgoal(
                    goal_type=goal_type,
                    position=(sub_row, sub_col),
                    node_idx=closest_node_idx,
                    priority=priority_map.get(goal_type, 999)
                )
                subgoals.append(subgoal)
            
        return subgoals
    
    def _analyze_subgoal_dependencies(self, subgoals: List[Subgoal]):
        """Analyze and set dependencies between subgoals."""
        # Basic dependency rules for N++ levels
        for subgoal in subgoals:
            if subgoal.goal_type == 'exit':
                # Exit typically requires exit switch to be activated first
                subgoal.dependencies = ['exit_switch']
                
            elif subgoal.goal_type == 'exit_switch':
                # Exit switch may require doors to be unlocked first
                door_switches = [s.goal_type for s in subgoals 
                               if s.goal_type in ['locked_door_switch', 'trap_door_switch']]
                if door_switches:
                    subgoal.dependencies = door_switches
                    
            elif subgoal.goal_type in ['locked_door_switch', 'trap_door_switch']:
                # Door switches are typically independent or depend on other doors
                # For now, assume they can be done in any order
                pass
                
        # Set unlocks relationships (inverse of dependencies)
        for subgoal in subgoals:
            for dep_type in subgoal.dependencies:
                for other_subgoal in subgoals:
                    if other_subgoal.goal_type == dep_type:
                        other_subgoal.unlocks.append(subgoal.goal_type)
    
    def _create_execution_order(
        self, 
        subgoals: List[Subgoal], 
        target_subgoal: Subgoal
    ) -> List[int]:
        """
        Create execution order using topological sort from target backwards.
        
        Args:
            subgoals: List of all subgoals
            target_subgoal: Final target to reach
            
        Returns:
            List of indices into subgoals list, in execution order
        """
        # Build dependency graph for efficient lookup
        subgoal_map = {s.goal_type: i for i, s in enumerate(subgoals)}
        
        # Find all subgoals needed to reach target using backwards BFS
        # This ensures we only include necessary subgoals in the execution order
        needed_subgoals = set()
        queue = deque([target_subgoal.goal_type])
        needed_subgoals.add(target_subgoal.goal_type)
        
        while queue:
            current_type = queue.popleft()
            current_idx = subgoal_map.get(current_type)
            
            if current_idx is not None:
                current_subgoal = subgoals[current_idx]
                for dep_type in current_subgoal.dependencies:
                    if dep_type not in needed_subgoals and dep_type in subgoal_map:
                        needed_subgoals.add(dep_type)
                        queue.append(dep_type)
        
        # Topological sort of needed subgoals
        execution_order = []
        remaining = {subgoal_map[goal_type] for goal_type in needed_subgoals}
        
        while remaining:
            # Find subgoals with no unmet dependencies
            ready = []
            for idx in remaining:
                subgoal = subgoals[idx]
                deps_met = all(
                    dep_type not in needed_subgoals or 
                    subgoal_map.get(dep_type) in execution_order
                    for dep_type in subgoal.dependencies
                )
                if deps_met:
                    ready.append(idx)
            
            if not ready:
                # Circular dependency or missing dependency
                if self.debug:
                    print("DEBUG: Circular dependency detected in subgoal planning")
                return []
                
            # Sort ready subgoals by priority
            ready.sort(key=lambda idx: subgoals[idx].priority)
            
            # Add highest priority ready subgoal
            next_idx = ready[0]
            execution_order.append(next_idx)
            remaining.remove(next_idx)
        
        return execution_order
    
    def _estimate_plan_cost(
        self, 
        subgoals: List[Subgoal], 
        execution_order: List[int], 
        start_node_idx: int
    ) -> float:
        """Estimate total cost of executing the subgoal plan."""
        total_cost = 0.0
        current_node = start_node_idx
        
        for subgoal_idx in execution_order:
            subgoal = subgoals[subgoal_idx]
            
            if subgoal.node_idx is not None:
                # Find path from current position to this subgoal
                path = self.navigation_engine.find_shortest_path(
                    current_node, subgoal.node_idx
                )
                
                if path:
                    # Estimate cost based on path length (simplified)
                    path_cost = len(path) * 10.0  # Rough cost per step
                    total_cost += path_cost
                    current_node = subgoal.node_idx
                else:
                    # If no path found, add penalty cost
                    total_cost += 1000.0  # High penalty for unreachable subgoals
        
        return total_cost
    
    def execute_subgoal_plan(
        self, 
        plan: SubgoalPlan, 
        start_node_idx: int
    ) -> List[List[int]]:
        """
        Execute subgoal plan and return sequence of paths.
        
        Args:
            plan: SubgoalPlan to execute
            start_node_idx: Starting node index
            
        Returns:
            List of paths, where each path is a list of node indices
        """
        paths = []
        current_node = start_node_idx
        
        for subgoal_idx in plan.execution_order:
            subgoal = plan.subgoals[subgoal_idx]
            
            if subgoal.node_idx is not None:
                path = self.navigation_engine.find_shortest_path(
                    current_node, subgoal.node_idx
                )
                
                if path:
                    paths.append(path)
                    current_node = subgoal.node_idx
                    if self.debug:
                        print(f"DEBUG: Path to {subgoal.goal_type}: {len(path)} nodes")
                else:
                    if self.debug:
                        print(f"DEBUG: No path found to {subgoal.goal_type}")
                    break
        
        return paths
    
    def _get_node_position(self, graph_data, node_idx: int) -> Tuple[float, float]:
        """Extract world position from node index and features."""
        if node_idx >= graph_data.num_nodes or graph_data.node_mask[node_idx] == 0:
            return (0.0, 0.0)

        from .common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
        from ..constants.physics_constants import (
            TILE_PIXEL_SIZE,
        )

        # Calculate sub-grid nodes count
        sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT

        if node_idx < sub_grid_nodes_count:
            # Sub-grid node: extract position from features (already in correct coordinates)
            node_features = graph_data.node_features[node_idx]
            if len(node_features) >= 2:
                x = float(node_features[0])
                y = float(node_features[1])
                return (float(x), float(y))
            else:
                # Fallback: calculate position from index
                sub_row = node_idx // SUB_GRID_WIDTH
                sub_col = node_idx % SUB_GRID_WIDTH
                # Center in sub-cell, add 1-tile offset for simulator border
                x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                return (float(x), float(y))
        else:
            # Entity node: extract position from features
            node_features = graph_data.node_features[node_idx]
            # Feature layout: position(2) + tile_type + 4 + entity_type + state_features
            # Position coordinates are stored at indices 0 and 1 (already in pixel coordinates)
            if len(node_features) >= 2:
                x = float(node_features[0])
                y = float(node_features[1])
                return (float(x), float(y))
            else:
                return (0.0, 0.0)
    
    def create_hierarchical_completion_plan(
        self,
        ninja_position: Tuple[float, float],
        level_data,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None,
        reachability_analyzer: Optional[OpenCVFloodFill] = None
    ) -> Optional[SubgoalPlan]:
        """
        Create simplified completion plan optimized for RL training.
        
        This method now uses the simplified strategy:
        1. Check if exit switch is reachable → if yes, that's the goal
        2. If not, find nearest reachable locked door switch → that's the goal
        3. Check if exit door is reachable → if yes, that's the goal
        4. If not, find nearest reachable switch → that's the goal
        
        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level tile data
            entities: List of entities in the level
            switch_states: Current state of switches (activated/not activated)
            reachability_analyzer: OpenCV flood fill analyzer (unused in simplified version)
            
        Returns:
            SubgoalPlan with single clear objective, or None if no objective found
        """
        if switch_states is None:
            switch_states = {}
        
        # Use simplified strategy to get next objective
        objective = self.simplified_strategy.get_next_objective(
            ninja_position, level_data, entities, switch_states
        )
        
        if objective is None:
            if self.debug:
                print("DEBUG: No reachable objectives found")
            return None
        
        # Convert SimpleObjective to Subgoal for backward compatibility
        subgoal = self._convert_objective_to_subgoal(objective)
        
        # Create simple plan with single objective
        plan = SubgoalPlan(
            subgoals=[subgoal],
            execution_order=[0],  # Single objective at index 0
            total_estimated_cost=objective.distance
        )
        
        if self.debug:
            print(f"DEBUG: Created simplified plan: {objective.description}")
        
        return plan
    
    def _convert_objective_to_subgoal(self, objective: SimpleObjective) -> Subgoal:
        """
        Convert SimpleObjective to Subgoal for backward compatibility.
        
        Args:
            objective: SimpleObjective from simplified strategy
            
        Returns:
            Subgoal object compatible with existing API
        """
        # Map objective types to goal types
        goal_type_map = {
            ObjectiveType.REACH_EXIT_SWITCH: 'exit_switch',
            ObjectiveType.REACH_DOOR_SWITCH: 'locked_door_switch',
            ObjectiveType.REACH_EXIT_DOOR: 'exit',
            ObjectiveType.REACH_SWITCH: 'switch',
        }
        
        goal_type = goal_type_map.get(objective.objective_type, 'unknown')
        
        # Convert position from pixel coordinates to sub-cell coordinates
        sub_row = int(objective.position[1] // SUB_CELL_SIZE)
        sub_col = int(objective.position[0] // SUB_CELL_SIZE)
        
        return Subgoal(
            goal_type=goal_type,
            position=(sub_row, sub_col),
            priority=int(objective.priority * 10),  # Convert to integer priority
            node_idx=None  # Not needed for simplified approach
        )
    
    def get_current_objective(self) -> Optional[SimpleObjective]:
        """
        Get the current objective from the simplified strategy.
        
        This method provides direct access to the simplified objective
        for RL integration and feature encoding.
        
        Returns:
            Current SimpleObjective or None
        """
        return self.simplified_strategy.get_current_objective()
    
    def get_objective_for_rl_features(self, ninja_position: Tuple[float, float]) -> Dict[str, float]:
        """
        Get objective information formatted for RL feature encoding.
        
        This method provides objective data in a format suitable for
        TASK_003's compact feature encoding.
        
        Args:
            ninja_position: Current ninja position
            
        Returns:
            Dictionary with objective features for RL integration
        """
        return self.simplified_strategy.get_objective_for_rl_features(ninja_position)
    
    def is_objective_reached(self, ninja_position: Tuple[float, float], threshold: float = 24.0) -> bool:
        """
        Check if the current objective has been reached.
        
        Args:
            ninja_position: Current ninja position
            threshold: Distance threshold for "reached" (default: 24 pixels = 1 tile)
            
        Returns:
            True if objective is reached
        """
        return self.simplified_strategy.is_objective_reached(ninja_position, threshold)
    
    def clear_objective(self):
        """Clear the current objective (e.g., when objective is reached)."""
        self.simplified_strategy.clear_objective()
    
    def _extract_entity_relationships(self, entities: List[Any]) -> Dict[str, Any]:
        """
        Extract entity positions and switch-door relationships.
        
        Args:
            entities: List of entity objects
            
        Returns:
            Dictionary with entity positions and relationships
        """
        entity_info = {
            'exit_switches': [],      # List of (x, y) positions
            'exit_doors': [],         # List of (x, y) positions
            'locked_doors': [],       # List of {'position': (x, y), 'switch': (x, y)}
            'door_switches': [],      # List of (x, y) positions
            'gold': [],               # List of (x, y) positions
            'hazards': []             # List of (x, y) positions
        }
        
        for entity in entities:
            entity_type = getattr(entity, 'type', None)
            x_pos = getattr(entity, 'xpos', getattr(entity, 'x', 0))
            y_pos = getattr(entity, 'ypos', getattr(entity, 'y', 0))
            position = (float(x_pos), float(y_pos))
            
            if entity_type == 4:  # Exit switch
                entity_info['exit_switches'].append(position)
            elif entity_type == 3:  # Exit door
                entity_info['exit_doors'].append(position)
            elif entity_type == 6:  # Locked door
                # Extract switch position for this door
                sw_x = getattr(entity, 'sw_xcoord', 0)
                sw_y = getattr(entity, 'sw_ycoord', 0)
                switch_pos = (float(sw_x), float(sw_y))
                
                entity_info['locked_doors'].append({
                    'position': position,
                    'switch': switch_pos
                })
                entity_info['door_switches'].append(switch_pos)
            elif entity_type == 2:  # Gold
                entity_info['gold'].append(position)
            elif entity_type in [1, 14, 20, 25, 26]:  # Various hazards
                entity_info['hazards'].append(position)
                
        return entity_info
    
    def _has_required_entities(self, entity_info: Dict[str, Any]) -> bool:
        """Check if level has minimum required entities for completion."""
        return (len(entity_info['exit_switches']) > 0 and 
                len(entity_info['exit_doors']) > 0)
    
    def _recursive_completion_analysis(
        self,
        ninja_position: Tuple[float, float],
        entity_info: Dict[str, Any],
        reachability_result,
        switch_states: Dict[str, bool],
        level_data,
        entities: List[Any],
        reachability_analyzer: OpenCVFloodFill
    ) -> List[Subgoal]:
        """
        Implement recursive completion algorithm from HIERARCHICAL_SUBGOAL_PLANNING.md.
        
        Returns list of Subgoal objects in optimal completion order.
        """
        subgoals = []
        
        # Get primary objectives
        exit_switch_pos = entity_info['exit_switches'][0] if entity_info['exit_switches'] else None
        exit_door_pos = entity_info['exit_doors'][0] if entity_info['exit_doors'] else None
        
        if not exit_switch_pos or not exit_door_pos:
            return []
        
        # Step 1: Check exit switch reachability
        if self._is_position_reachable(exit_switch_pos, reachability_result):
            # Exit switch is reachable
            if switch_states.get('exit_switch', False):
                # Switch already activated, check door reachability
                if self._is_position_reachable(exit_door_pos, reachability_result):
                    # Direct path to completion
                    subgoals.append(Subgoal(
                        goal_type='exit',
                        position=self._world_to_sub_coords(exit_door_pos),
                        priority=1
                    ))
                else:
                    # Need to unlock path to exit door
                    door_unlock_subgoals = self._find_door_unlock_subgoals(
                        exit_door_pos, entity_info, reachability_result, switch_states,
                        level_data, entities, reachability_analyzer
                    )
                    subgoals.extend(door_unlock_subgoals)
                    subgoals.append(Subgoal(
                        goal_type='exit',
                        position=self._world_to_sub_coords(exit_door_pos),
                        priority=len(door_unlock_subgoals) + 1
                    ))
            else:
                # Need to activate exit switch first
                subgoals.extend([
                    Subgoal(
                        goal_type='exit_switch',
                        position=self._world_to_sub_coords(exit_switch_pos),
                        priority=1
                    ),
                    Subgoal(
                        goal_type='exit',
                        position=self._world_to_sub_coords(exit_door_pos),
                        priority=2
                    )
                ])
        else:
            # Exit switch not reachable - find blocking doors
            blocking_door_subgoals = self._find_blocking_door_subgoals(
                ninja_position, exit_switch_pos, entity_info, reachability_result,
                switch_states, level_data, entities, reachability_analyzer
            )
            
            subgoals.extend(blocking_door_subgoals)
            
            # Add final exit sequence
            subgoals.extend([
                Subgoal(
                    goal_type='exit_switch',
                    position=self._world_to_sub_coords(exit_switch_pos),
                    priority=len(blocking_door_subgoals) + 1
                ),
                Subgoal(
                    goal_type='exit',
                    position=self._world_to_sub_coords(exit_door_pos),
                    priority=len(blocking_door_subgoals) + 2
                )
            ])
        
        return subgoals
    
    def _is_position_reachable(self, position: Tuple[float, float], reachability_result) -> bool:
        """Check if a specific position is reachable."""
        # Convert to pixel position for comparison with reachability result
        return position in reachability_result.reachable_positions
    
    def _find_door_unlock_subgoals(
        self,
        target_position: Tuple[float, float],
        entity_info: Dict[str, Any],
        reachability_result,
        switch_states: Dict[str, bool],
        level_data,
        entities: List[Any],
        reachability_analyzer: OpenCVFloodFill
    ) -> List[Subgoal]:
        """Find subgoals needed to unlock doors blocking path to target."""
        unlock_subgoals = []
        
        # For each locked door, check if it blocks path to target
        for door_info in entity_info['locked_doors']:
            door_pos = door_info['position']
            switch_pos = door_info['switch']
            
            # Simple heuristic: if switch is reachable and door might help, add it
            if self._is_position_reachable(switch_pos, reachability_result):
                unlock_subgoals.append(Subgoal(
                    goal_type='locked_door_switch',
                    position=self._world_to_sub_coords(switch_pos),
                    priority=len(unlock_subgoals) + 1
                ))
        
        return unlock_subgoals
    
    def _find_blocking_door_subgoals(
        self,
        start_position: Tuple[float, float],
        target_position: Tuple[float, float],
        entity_info: Dict[str, Any],
        reachability_result,
        switch_states: Dict[str, bool],
        level_data,
        entities: List[Any],
        reachability_analyzer: OpenCVFloodFill
    ) -> List[Subgoal]:
        """Find subgoals for doors that block path to target."""
        blocking_subgoals = []
        
        # Find switches that need to be activated to reach target
        for door_info in entity_info['locked_doors']:
            switch_pos = door_info['switch']
            
            # If switch is reachable, it might help unlock path to target
            if self._is_position_reachable(switch_pos, reachability_result):
                blocking_subgoals.append(Subgoal(
                    goal_type='locked_door_switch',
                    position=self._world_to_sub_coords(switch_pos),
                    priority=len(blocking_subgoals) + 1
                ))
        
        return blocking_subgoals
    
    def _world_to_sub_coords(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to sub-grid coordinates."""
        # Simple conversion - in practice would use proper coordinate transformation
        x, y = world_pos
        sub_x = int(x // SUB_CELL_SIZE)
        sub_y = int(y // SUB_CELL_SIZE)
        return (sub_x, sub_y)