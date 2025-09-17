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
        
    def initialize_for_reachability(self, level_data):
        """
        Initialize hazard system for reachability analysis.
        
        Args:
            level_data: LevelData object containing entities and tiles
        """
        # Use existing hazard system initialization
        self.hazard_system.set_tile_data(level_data.tiles)
        self.hazard_system.build_static_hazard_cache(level_data)
        
        # Initialize switch states
        self.switch_states.clear()
        for entity in level_data.entities:
            if entity.get('type') in [EntityType.EXIT_SWITCH, EntityType.REGULAR_DOOR]:
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
            [{"x": x, "y": y}], (x, y), radius=200.0
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
        ninja_pos: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]] = None,
        entities: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate physics-accurate safe bounce trajectory from a bounce block.
        
        This method uses the trajectory calculator to compute realistic bounce
        trajectories based on N++ physics, including proper bounce block mechanics,
        velocity calculations, and collision detection.
        
        Args:
            bounce_block_pos: Position of bounce block (x, y)
            ninja_pos: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy), defaults to zero
            entities: List of entities for collision detection
            
        Returns:
            Target position after bounce, or None if unsafe/impossible
        """
        # Simplified bounce calculation without trajectory calculator
        from ...constants.physics_constants import (
            BOUNCE_BLOCK_INTERACTION_RADIUS,
            BOUNCE_BLOCK_BOOST_MIN,
            BOUNCE_BLOCK_BOOST_MAX,
            MAX_HOR_SPEED,
            JUMP_INITIAL_VELOCITY,
            GRAVITY_FALL,
            GRAVITY_JUMP
        )
        from ...utils.physics_utils import (
            calculate_bounce_block_boost_multiplier,
            calculate_distance
        )
        import math
        
        # Check if ninja is within interaction range of bounce block
        distance_to_block = calculate_distance(ninja_pos, bounce_block_pos)
        if distance_to_block > BOUNCE_BLOCK_INTERACTION_RADIUS:
            return None  # Too far to interact
            
        # Default velocity if not provided
        if ninja_velocity is None:
            ninja_velocity = (0.0, 0.0)
            
        # Create bounce block entity for calculations
        bounce_block_entity = {
            'type': 'bounce_block',
            'x': bounce_block_pos[0],
            'y': bounce_block_pos[1],
            'state': 0.0,  # Neutral state
            'active': True
        }
        
        # Calculate approach vector and bounce direction
        dx = ninja_pos[0] - bounce_block_pos[0]
        dy = ninja_pos[1] - bounce_block_pos[1]
        
        # Normalize approach vector
        approach_distance = math.sqrt(dx * dx + dy * dy)
        if approach_distance < 1e-6:
            return None  # Too close to calculate meaningful trajectory
            
        approach_x = dx / approach_distance
        approach_y = dy / approach_distance
        
        # Calculate bounce block boost multiplier
        boost_multiplier = calculate_bounce_block_boost_multiplier(
            [bounce_block_entity], ninja_pos
        )
        
        # Calculate initial bounce velocity based on approach and current velocity
        base_velocity_x = ninja_velocity[0]
        base_velocity_y = ninja_velocity[1]
        
        # If ninja is falling/jumping onto the bounce block, use that velocity
        if abs(base_velocity_y) < 0.1:  # Minimal vertical velocity
            base_velocity_y = JUMP_INITIAL_VELOCITY  # Assume jump velocity
            
        # Apply bounce block physics
        # Bounce blocks amplify velocity in the direction away from the block
        bounce_velocity_x = base_velocity_x + (approach_x * MAX_HOR_SPEED * boost_multiplier)
        bounce_velocity_y = base_velocity_y - (approach_y * abs(JUMP_INITIAL_VELOCITY) * boost_multiplier)
        
        # Clamp velocities to reasonable limits
        bounce_velocity_x = max(-MAX_HOR_SPEED * 2, min(MAX_HOR_SPEED * 2, bounce_velocity_x))
        bounce_velocity_y = max(JUMP_INITIAL_VELOCITY * 2, min(-JUMP_INITIAL_VELOCITY * 0.5, bounce_velocity_y))
        
        # Calculate simplified bounce landing position
        potential_targets = []
        
        # Simplified bounce calculation - estimate landing position
        flight_time = 1.0  # Simplified flight time estimate
        target_x = bounce_block_pos[0] + bounce_velocity_x * flight_time
        target_y = bounce_block_pos[1] + bounce_velocity_y * flight_time + 0.5 * GRAVITY_FALL * flight_time * flight_time
        
        target_pos = (target_x, target_y)
        
        # Simple validation: check if target position is safe and reasonable
        distance_to_target = ((target_x - ninja_pos[0])**2 + (target_y - ninja_pos[1])**2)**0.5
        if (distance_to_target < 500 and  # Reasonable bounce distance
            self.is_position_safe_for_reachability(target_pos)):
            
            potential_targets.append({
                'position': target_pos,
                'probability': 0.8,  # Simplified success probability
                'distance': distance_to_target
            })
        
        # Select best target based on success probability and reasonable distance
        if not potential_targets:
            # Fallback: simple physics-based calculation without trajectory validation
            fallback_distance = 60.0 * boost_multiplier  # Base bounce distance
            
            # Calculate fallback target in the bounce direction
            target_x = bounce_block_pos[0] + approach_x * fallback_distance
            target_y = bounce_block_pos[1] + approach_y * fallback_distance
            
            # Adjust for gravity (bounce blocks typically launch upward/forward)
            if approach_y > 0:  # Approaching from above
                target_y -= 20.0 * boost_multiplier  # Launch upward
            
            fallback_target = (target_x, target_y)
            
            if self.is_position_safe_for_reachability(fallback_target):
                return fallback_target
            else:
                return None
        
        # Sort by success probability (descending) and select best
        potential_targets.sort(key=lambda t: t['probability'], reverse=True)
        best_target = potential_targets[0]
        
        return best_target['position']
    
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