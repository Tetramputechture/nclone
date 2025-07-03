import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any

from .collision import CollisionChecker
from .entity_wrapper import EntityManager, EntityWrapper
from ..ninja import NINJA_RADIUS
from ..nsim import TILE_PIXEL_SIZE

# Legacy compatibility - use EntityWrapper instead
def create_enemy_from_entity_data(entity_data: Dict[str, Any]) -> EntityWrapper:
    """
    Legacy function for creating enemy objects from entity data.
    Now returns an EntityWrapper for compatibility.
    
    This function is deprecated - use EntityManager.get_pathfinding_entities() instead.
    """
    # This would need an actual entity object, which we don't have from raw data
    # In practice, pathfinding should use EntityManager with the simulation
    raise NotImplementedError(
        "Use EntityManager with a simulation instance instead of raw entity data"
    )


class PathfindingUtils:
    """Unified pathfinding utilities that work with the existing simulation."""
    
    def __init__(self, sim):
        """Initialize with a reference to the simulator."""
        self.sim = sim
        self.collision_checker = CollisionChecker(sim)
        self.entity_manager = EntityManager(sim)
    
    def check_path_collision(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                           radius: float = None) -> bool:
        """Check if a path between two points has any collisions."""
        return self.collision_checker.check_collision(pos1, pos2, radius)
    
    def point_in_wall(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside solid geometry."""
        return self.collision_checker.point_in_wall(point)
    
    def get_surface_normal(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Get surface normal at a point."""
        return self.collision_checker.get_surface_normal_at_point(point)
    
    def get_nearby_enemies(self, pos: Tuple[float, float], radius: float = 240.0) -> List[EntityWrapper]:
        """Get enemies near a position that could affect pathfinding."""
        return self.entity_manager.get_pathfinding_entities(pos, radius)
    
    def update_entities(self):
        """Update entity states from the simulation."""
        self.entity_manager.update_all()
    
    def path_blocked_by_entities(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                                radius: float = None) -> bool:
        """Check if a path is blocked by any entities."""
        if radius is None:
            radius = NINJA_RADIUS
            
        # Get entities along the path
        path_center = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        path_length = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        search_radius = path_length / 2 + radius + 50  # Add buffer
        
        nearby_entities = self.entity_manager.get_pathfinding_entities(path_center, search_radius)
        
        for entity in nearby_entities:
            if self.collision_checker.entity_blocks_path(entity.entity, pos1, pos2, radius):
                return True
                
        return False
