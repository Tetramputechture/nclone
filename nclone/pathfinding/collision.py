"""
Unified collision detection system for both simulation and pathfinding.
This module consolidates collision checking functionality to eliminate duplication.
"""

import math
from typing import Tuple, List, Optional, Dict, Any

from ..physics import (
    sweep_circle_vs_tiles,
    get_single_closest_point,
)
from ..ninja import NINJA_RADIUS
from ..nsim import TILE_PIXEL_SIZE


class CollisionChecker:
    """Unified collision checker that works with the existing simulation infrastructure."""
    
    def __init__(self, sim):
        """Initialize with a reference to the simulator."""
        self.sim = sim
        self.tile_size = TILE_PIXEL_SIZE
        
    def check_collision(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                       radius: float = None) -> bool:
        """Check for collision along a movement path using the existing physics system."""
        if radius is None:
            radius = NINJA_RADIUS
            
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Use the existing sweep_circle_vs_tiles function
        time = sweep_circle_vs_tiles(self.sim, pos1[0], pos1[1], dx, dy, radius)
        return time < 1.0  # Collision if we can't move the full distance
    
    def point_in_wall(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside solid geometry using existing collision system."""
        x, y = point
        grid_x = int(x // self.tile_size)
        grid_y = int(y // self.tile_size)
        
        # Check bounds
        if (grid_x < 0 or grid_x >= self.sim.current_map_width_tiles or 
            grid_y < 0 or grid_y >= self.sim.current_map_height_tiles):
            return True  # Outside map bounds
        
        # Use existing tile data
        if hasattr(self.sim, 'tile_dic'):
            tile_type = self.sim.tile_dic.get((grid_x, grid_y), 0)
            return tile_type != 0  # Non-empty tiles are walls
        
        return False
    
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
    
    def entity_blocks_path(self, entity: Any, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float], radius: float = None) -> bool:
        """Check if an entity blocks a movement path."""
        if not entity.active:
            return False
            
        if radius is None:
            radius = NINJA_RADIUS
            
        # Check if entity has collision radius
        entity_radius = getattr(entity, 'RADIUS', 0)
        if entity_radius == 0:
            return False
            
        # Simple line-circle intersection test
        return self._line_intersects_circle(
            pos1, pos2, (entity.xpos, entity.ypos), entity_radius + radius)
    
    def _line_intersects_circle(self, line_start: Tuple[float, float], 
                               line_end: Tuple[float, float],
                               circle_center: Tuple[float, float], 
                               radius: float) -> bool:
        """Check if a line segment intersects with a circle."""
        x1, y1 = line_start
        x2, y2 = line_end
        cx, cy = circle_center
        
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from line start to circle center
        fx = cx - x1
        fy = cy - y1
        
        # Project circle center onto line
        line_length_sq = dx*dx + dy*dy
        if line_length_sq == 0:
            # Degenerate line, check point-circle distance
            dist_sq = fx*fx + fy*fy
            return dist_sq <= radius*radius
        
        t = max(0, min(1, (fx*dx + fy*dy) / line_length_sq))
        
        # Closest point on line segment
        closest_x = x1 + t*dx
        closest_y = y1 + t*dy
        
        # Distance from circle center to closest point
        dist_x = cx - closest_x
        dist_y = cy - closest_y
        dist_sq = dist_x*dist_x + dist_y*dist_y
        
        return dist_sq <= radius*radius 