"""
Reachability-specific extensions to the existing hazard system.

This module extends the existing hazard_system.py with reachability-specific
functionality without duplicating the core hazard detection logic.
"""

from typing import Dict, Any, List, Tuple, Set, Optional
from ..hazard_system import HazardClassificationSystem, HazardInfo
from ...constants.entity_types import EntityType


class ReachabilityHazardExtension:
    """
    Extends HazardSystem with reachability-specific functionality.
    
    This class wraps the existing HazardSystem and adds methods specifically
    needed for reachability analysis without duplicating existing functionality.
    """
    
    def __init__(self, hazard_system: HazardClassificationSystem, debug: bool = False):
        """
        Initialize reachability hazard extension.
        
        Args:
            hazard_system: Existing HazardSystem instance
            debug: Enable debug output
        """
        self.hazard_system = hazard_system
        self.debug = debug
        self.switch_states: Dict[int, bool] = {}
        
    def initialize_for_reachability(self, entities: List[Dict[str, Any]], tiles):
        """
        Initialize hazard system for reachability analysis.
        
        Args:
            entities: List of entity dictionaries from level data
            tiles: Tile array for collision detection
        """
        # Use existing hazard system initialization
        self.hazard_system.set_tile_data(tiles)
        self.hazard_system.build_static_hazard_cache(entities)
        
        # Initialize switch states
        self.switch_states.clear()
        for entity in entities:
            if entity.get('type') in [EntityType.DOOR_SWITCH, EntityType.EXIT_SWITCH]:
                entity_id = entity.get('id', 0)
                self.switch_states[entity_id] = entity.get('switch_state', False)
    
    def is_position_safe_for_reachability(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is safe for reachability analysis.
        
        This extends the hazard system's safety checks with reachability-specific logic.
        
        Args:
            position: (x, y) position to check
            
        Returns:
            True if position is safe for reachability analysis
        """
        x, y = position
        
        # Get dynamic hazards in range
        dynamic_hazards = self.hazard_system.get_dynamic_hazards_in_range(
            [{"x": x, "y": y}], x, y, radius=200.0
        )
        
        # Check if position intersects with any hazards
        for hazard in dynamic_hazards:
            if self._position_intersects_hazard(position, hazard):
                return False
                
        return True
    
    def can_traverse_between_positions_safely(
        self, 
        start_pos: Tuple[float, float], 
        end_pos: Tuple[float, float]
    ) -> bool:
        """
        Check if movement between two positions is safe from hazards.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            
        Returns:
            True if path is safe from hazards
        """
        # Get hazards that might affect this path
        center_x = (start_pos[0] + end_pos[0]) / 2
        center_y = (start_pos[1] + end_pos[1]) / 2
        
        dynamic_hazards = self.hazard_system.get_dynamic_hazards_in_range(
            [{"x": center_x, "y": center_y}], center_x, center_y, radius=300.0
        )
        
        # Check path intersection with each hazard
        for hazard in dynamic_hazards:
            if self.hazard_system.check_path_hazard_intersection(
                start_pos[0], start_pos[1], end_pos[0], end_pos[1], hazard
            ):
                return False
                
        return True
    
    def get_bounce_trajectory_safe(
        self, 
        bounce_block_pos: Tuple[float, float], 
        ninja_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate safe bounce trajectory from a bounce block.
        
        Args:
            bounce_block_pos: Position of bounce block
            ninja_pos: Current ninja position
            
        Returns:
            Target position after bounce, or None if unsafe
        """
        # Use existing hazard system's bounce block analysis
        # This is a simplified implementation - in production, this would
        # integrate with the physics system for accurate trajectory calculation
        
        dx = ninja_pos[0] - bounce_block_pos[0]
        dy = ninja_pos[1] - bounce_block_pos[1]
        
        # Simple bounce calculation (could be enhanced with physics integration)
        bounce_distance = 100.0  # Simplified constant
        if abs(dx) > abs(dy):
            # Horizontal bounce
            target_x = bounce_block_pos[0] + (bounce_distance if dx > 0 else -bounce_distance)
            target_y = bounce_block_pos[1]
        else:
            # Vertical bounce
            target_x = bounce_block_pos[0]
            target_y = bounce_block_pos[1] + (bounce_distance if dy > 0 else -bounce_distance)
            
        target_pos = (target_x, target_y)
        
        # Check if target position is safe
        if self.is_position_safe_for_reachability(target_pos):
            return target_pos
        else:
            return None
    
    def update_switch_states(self, switch_states: Dict[int, bool]):
        """
        Update switch states for dynamic hazard calculation.
        
        Args:
            switch_states: Dictionary mapping switch IDs to their states
        """
        self.switch_states.update(switch_states)
        
        # Note: The existing hazard system handles switch-dependent hazards
        # through its dynamic hazard detection, so we don't need to rebuild
        # the entire cache here
    
    def get_entity_influenced_positions(self) -> Set[Tuple[int, int]]:
        """
        Get grid positions that are influenced by entities.
        
        Returns:
            Set of (row, col) positions that have entity influence
        """
        influenced_positions = set()
        
        # This would integrate with the hazard system's static hazard cache
        # For now, return empty set as this is primarily used for visualization
        # In a full implementation, this would extract positions from the
        # hazard system's internal data structures
        
        return influenced_positions
    
    def _position_intersects_hazard(
        self, 
        position: Tuple[float, float], 
        hazard: HazardInfo
    ) -> bool:
        """
        Check if a position intersects with a specific hazard.
        
        Args:
            position: Position to check
            hazard: Hazard information
            
        Returns:
            True if position intersects hazard
        """
        x, y = position
        
        # Use the hazard's blocking area to check intersection
        for blocked_x, blocked_y, radius in hazard.blocking_area:
            distance_sq = (x - blocked_x) ** 2 + (y - blocked_y) ** 2
            if distance_sq <= radius ** 2:
                return True
                
        return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for reachability hazard analysis.
        
        Returns:
            Dictionary with debug information
        """
        return {
            "switch_states": self.switch_states,
            "hazard_system_stats": {
                "static_hazards": len(getattr(self.hazard_system, '_static_hazards', {})),
                "dynamic_hazards": len(getattr(self.hazard_system, '_dynamic_hazards', {})),
            }
        }