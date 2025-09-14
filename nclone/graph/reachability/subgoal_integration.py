"""
Integration layer between reachability analysis and existing subgoal planner.

This module extends the existing SubgoalPlanner with reachability-specific
enhancements without duplicating the core planning logic.
"""

from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..subgoal_planner import SubgoalPlanner, Subgoal, SubgoalPlan
    from .reachability_state import ReachabilityState
else:
    # Import at runtime to avoid circular imports
    SubgoalPlanner = None
    Subgoal = None
    SubgoalPlan = None
    ReachabilityState = None


class ReachabilitySubgoalIntegration:
    """
    Extends SubgoalPlanner with reachability-specific functionality.
    
    This class wraps the existing SubgoalPlanner and adds methods specifically
    needed for reachability analysis without duplicating existing functionality.
    """
    
    def __init__(self, subgoal_planner, debug: bool = False):
        """
        Initialize reachability subgoal integration.
        
        Args:
            subgoal_planner: Existing SubgoalPlanner instance
            debug: Enable debug output
        """
        self.subgoal_planner = subgoal_planner
        self.debug = debug
        
    def enhance_subgoals_with_reachability(
        self, 
        level_data, 
        reachability_state,
        hazard_extension=None,
        position_validator=None
    ):
        """
        Enhance existing subgoal identification with reachability-specific analysis.
        
        Args:
            level_data: Level data containing tiles and entities
            reachability_state: Current reachability state
            hazard_extension: Optional hazard extension for hazard information
            position_validator: Optional position validator for coordinate conversion
            
        Returns:
            List of enhanced subgoals
        """
        # Start with existing subgoal planning
        existing_subgoals = []
        if hasattr(reachability_state, 'subgoals') and reachability_state.subgoals:
            # Import Subgoal class at runtime to avoid circular imports
            from ..subgoal_planner import Subgoal
            
            # Convert existing subgoals to Subgoal objects
            for sub_row, sub_col, goal_type in reachability_state.subgoals:
                subgoal = Subgoal(
                    goal_type=goal_type,
                    position=(sub_row, sub_col),
                    priority=self._calculate_subgoal_priority(goal_type)
                )
                existing_subgoals.append(subgoal)
        
        # Add reachability-specific enhancements
        enhanced_subgoals = existing_subgoals.copy()
        
        # Add strategic waypoints for complex navigation
        strategic_waypoints = self._identify_strategic_waypoints(
            level_data, reachability_state, position_validator
        )
        enhanced_subgoals.extend(strategic_waypoints)
        
        # Add exploration frontiers as subgoals
        if hasattr(reachability_state, 'frontiers'):
            frontier_subgoals = self._convert_frontiers_to_subgoals(
                reachability_state.frontiers
            )
            enhanced_subgoals.extend(frontier_subgoals)
        
        if self.debug:
            print(f"DEBUG: Enhanced {len(existing_subgoals)} existing subgoals to {len(enhanced_subgoals)} total")
            
        return enhanced_subgoals
    
    def _identify_strategic_waypoints(
        self, 
        level_data, 
        reachability_state, 
        position_validator=None
    ):
        """
        Identify strategic waypoints for complex navigation.
        
        This is a simplified version that focuses on key navigation points
        without duplicating the complex hazard analysis from enhanced_subgoals.py.
        """
        waypoints = []
        
        if not hasattr(reachability_state, 'reachable_positions'):
            return waypoints
            
        reachable_positions = reachability_state.reachable_positions
        height, width = level_data.tiles.shape
        
        # Import Subgoal class at runtime
        from ..subgoal_planner import Subgoal
        
        # Find positions that are at intersections or have multiple paths
        for row, col in reachable_positions:
            if self._is_strategic_position(row, col, level_data, reachable_positions):
                waypoint = Subgoal(
                    goal_type='strategic_waypoint',
                    position=(row, col),
                    priority=50  # Medium priority
                )
                waypoints.append(waypoint)
                
        return waypoints
    
    def _is_strategic_position(
        self, 
        row: int, 
        col: int, 
        level_data, 
        reachable_positions
    ) -> bool:
        """
        Check if a position is strategically important for navigation.
        
        A position is strategic if it:
        - Has multiple reachable neighbors (intersection)
        - Is near level boundaries
        - Is adjacent to unreachable areas (potential exploration point)
        """
        # Count reachable neighbors
        reachable_neighbors = 0
        unreachable_neighbors = 0
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                neighbor_row, neighbor_col = row + dr, col + dc
                
                # Check bounds
                if (0 <= neighbor_row < level_data.tiles.shape[0] and 
                    0 <= neighbor_col < level_data.tiles.shape[1]):
                    
                    if (neighbor_row, neighbor_col) in reachable_positions:
                        reachable_neighbors += 1
                    else:
                        unreachable_neighbors += 1
        
        # Strategic if it's an intersection (3+ neighbors) or exploration point
        return reachable_neighbors >= 3 or (reachable_neighbors >= 2 and unreachable_neighbors >= 2)
    
    def _convert_frontiers_to_subgoals(self, frontiers) -> List[Subgoal]:
        """
        Convert exploration frontiers to subgoals.
        
        Args:
            frontiers: List of Frontier objects
            
        Returns:
            List of subgoals representing exploration targets
        """
        subgoals = []
        
        for frontier in frontiers:
            # Convert frontier to exploration subgoal
            subgoal = Subgoal(
                goal_type='exploration_frontier',
                position=frontier.position,
                priority=self._calculate_frontier_priority(frontier)
            )
            subgoals.append(subgoal)
            
        return subgoals
    
    def _calculate_subgoal_priority(self, goal_type: str) -> int:
        """
        Calculate priority for different subgoal types.
        
        Lower numbers = higher priority.
        """
        priority_map = {
            'exit': 1,  # Highest priority
            'exit_switch': 10,
            'locked_door_switch': 20,
            'trap_door_switch': 30,
            'strategic_waypoint': 50,
            'exploration_frontier': 60,  # Lowest priority
        }
        
        return priority_map.get(goal_type, 50)
    
    def _calculate_frontier_priority(self, frontier) -> int:
        """
        Calculate priority for exploration frontiers.
        
        Frontiers with more potential for discovery get higher priority.
        """
        base_priority = 60
        
        # Adjust based on frontier type if available
        if hasattr(frontier, 'frontier_type'):
            if frontier.frontier_type == 'skill_challenge':
                return base_priority - 10  # Higher priority for skill challenges
            elif frontier.frontier_type == 'exploration':
                return base_priority + 10  # Lower priority for pure exploration
                
        return base_priority
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for subgoal integration.
        
        Returns:
            Dictionary with debug information
        """
        return {
            "subgoal_planner_available": self.subgoal_planner is not None,
            "integration_active": True
        }