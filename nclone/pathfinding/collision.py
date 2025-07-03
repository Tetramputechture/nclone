"""
Unified collision detection system for pathfinding that leverages existing physics.py.
This module provides a simplified interface to the existing collision system.
"""

import math
from typing import Tuple, List, Optional, Dict, Any

# Import existing physics functions instead of reimplementing them
from ..physics import (
    sweep_circle_vs_tiles,
    get_single_closest_point,
    gather_entities_from_neighbourhood,
    penetration_square_vs_point,
    overlap_circle_vs_circle,
    overlap_circle_vs_segment,
    is_empty_row,
    is_empty_column,
    raycast_vs_player,
    get_raycast_distance
)
from ..ninja import NINJA_RADIUS


class CollisionChecker:
    """Unified collision checker that uses the existing simulation infrastructure."""
    
    def __init__(self, sim):
        """Initialize with a reference to the simulator."""
        self.sim = sim
        
    def check_collision(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                       radius: float = None) -> bool:
        """Check for collision along a movement path using the existing physics system."""
        if radius is None:
            radius = NINJA_RADIUS
            
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Use the existing sweep_circle_vs_tiles function from physics.py
        time = sweep_circle_vs_tiles(self.sim, pos1[0], pos1[1], dx, dy, radius)
        return time < 1.0  # Collision if we can't move the full distance
    
    def point_in_wall(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside solid geometry using existing collision system."""
        # Use the existing closest point detection to check if point is inside geometry
        result, closest_point = get_single_closest_point(self.sim, point[0], point[1], 0.1)
        return result != 0  # Non-zero result means point is inside or very close to geometry
    
    def get_surface_normal_at_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Get the surface normal at a given point using existing collision system."""
        result, closest_point = get_single_closest_point(
            self.sim, point[0], point[1], NINJA_RADIUS)
        
        if result != 0 and closest_point:
            # Calculate normal from point to closest surface point
            dx = point[0] - closest_point[0]
            dy = point[1] - closest_point[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                return (dx/length, dy/length)
        
        return None
    
    def check_line_of_sight(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Check if there's a clear line of sight between two points using existing raycast."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return True
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Use existing raycast function
        raycast_distance = get_raycast_distance(self.sim, pos1[0], pos1[1], dx, dy)
        return raycast_distance is None or raycast_distance >= distance
    
    def check_grid_path_clear(self, start_pos: Tuple[float, float], 
                             end_pos: Tuple[float, float], direction: Tuple[int, int]) -> bool:
        """Check if a grid-aligned path is clear using existing grid functions."""
        start_cell_x = int(start_pos[0] // 12)
        start_cell_y = int(start_pos[1] // 12)
        end_cell_x = int(end_pos[0] // 12)
        end_cell_y = int(end_pos[1] // 12)
        
        if direction[0] != 0:  # Horizontal movement
            return is_empty_column(self.sim, start_cell_x, 
                                 min(start_cell_y, end_cell_y), 
                                 max(start_cell_y, end_cell_y), 
                                 direction[0])
        elif direction[1] != 0:  # Vertical movement
            return is_empty_row(self.sim, 
                              min(start_cell_x, end_cell_x),
                              max(start_cell_x, end_cell_x), 
                              start_cell_y, direction[1])
        
        return True
    
    def entity_blocks_path(self, entity: Any, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float], radius: float = None) -> bool:
        """Check if an entity blocks a movement path using existing collision functions."""
        if not entity.active:
            return False
            
        if radius is None:
            radius = NINJA_RADIUS
            
        # Check if entity has collision radius
        entity_radius = getattr(entity, 'RADIUS', getattr(entity, 'SEMI_SIDE', 0))
        if entity_radius == 0:
            return False
            
        # Use existing circle-segment overlap function
        return overlap_circle_vs_segment(
            entity.xpos, entity.ypos, entity_radius + radius,
            pos1[0], pos1[1], pos2[0], pos2[1])
    
    def get_nearby_entities(self, pos: Tuple[float, float]) -> List[Any]:
        """Get nearby entities using the existing physics system."""
        return gather_entities_from_neighbourhood(self.sim, pos[0], pos[1])
    
    def check_entity_collision(self, pos: Tuple[float, float], 
                              radius: float = None) -> bool:
        """Check if a position collides with any nearby entities."""
        if radius is None:
            radius = NINJA_RADIUS
            
        nearby_entities = self.get_nearby_entities(pos)
        
        for entity in nearby_entities:
            if not entity.active or not entity.is_physical_collidable:
                continue
                
            entity_radius = getattr(entity, 'RADIUS', getattr(entity, 'SEMI_SIDE', 0))
            if entity_radius == 0:
                continue
                
            # Use existing circle overlap function
            if overlap_circle_vs_circle(pos[0], pos[1], radius,
                                      entity.xpos, entity.ypos, entity_radius):
                return True
                
        return False
    
    def get_depenetration_from_entity(self, entity: Any, pos: Tuple[float, float], 
                                    radius: float = None) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get depenetration vector from an entity using existing physics functions."""
        if not entity.active or not entity.is_physical_collidable:
            return None
            
        if radius is None:
            radius = NINJA_RADIUS
            
        # Use existing depenetration function for square entities
        if hasattr(entity, 'SEMI_SIDE'):
            return penetration_square_vs_point(
                entity.xpos, entity.ypos, pos[0], pos[1], 
                entity.SEMI_SIDE + radius)
        
        return None 