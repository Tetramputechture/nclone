"""
Hierarchical subgoal planning for complex multi-step objectives.

This module implements a hierarchical navigation system that breaks down
complex objectives into manageable subgoals, considering game mechanics
like switch activation, door unlocking, and sequential dependencies.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from .common import SUB_CELL_SIZE
from .reachability import ReachabilityState
from .navigation import PathfindingEngine


@dataclass
class Subgoal:
    """Represents a single subgoal in hierarchical planning."""
    goal_type: str  # 'locked_door_switch', 'trap_door_switch', 'exit_switch', 'exit'
    position: Tuple[int, int]  # (sub_row, sub_col)
    node_idx: Optional[int] = None  # Graph node index
    priority: int = 0  # Lower numbers = higher priority
    dependencies: List[str] = None  # List of goal_types this depends on
    unlocks: List[str] = None  # List of goal_types this unlocks
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.unlocks is None:
            self.unlocks = []


@dataclass
class SubgoalPlan:
    """Complete hierarchical plan with ordered subgoals."""
    subgoals: List[Subgoal]
    execution_order: List[int]  # Indices into subgoals list
    total_estimated_cost: float = 0.0


class SubgoalPlanner:
    """
    Hierarchical planner for multi-step objectives.
    
    This class analyzes level structure and creates optimal plans for
    reaching complex objectives that require multiple steps, such as:
    - Activating switches to unlock doors
    - Sequential switch activation
    - Navigating to final objectives (exits)
    """
    
    def __init__(self, navigation_engine: PathfindingEngine, debug: bool = False):
        """
        Initialize subgoal planner with navigation engine.
        
        Args:
            navigation_engine: Engine for finding paths between nodes
            debug: Enable debug output (default: False)
        """
        self.navigation_engine = navigation_engine
        self.debug = debug
        
    def create_subgoal_plan(
        self, 
        reachability_state: ReachabilityState,
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
        reachability_state: ReachabilityState, 
        graph_data
    ) -> List[Subgoal]:
        """Convert reachability subgoals to Subgoal objects with node indices."""
        subgoals = []
        
        for sub_row, sub_col, goal_type in reachability_state.subgoals:
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