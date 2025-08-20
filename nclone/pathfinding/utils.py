import math
from typing import Tuple, List, Optional, Dict, Any

from .collision import CollisionChecker
from ..ninja import NINJA_RADIUS


class EntityWrapper:
    """Wrapper for existing entities to provide pathfinding-specific interface."""
    
    def __init__(self, entity):
        self.entity = entity
        self.id = id(entity)  # Use object id as unique identifier
        self.xpos = entity.xpos
        self.ypos = entity.ypos
        self.active = entity.active
        self.radius = getattr(entity, 'RADIUS', getattr(entity, 'SEMI_SIDE', 0))
        self.entity_type = entity.type
        
        # Get movement properties from entity
        self.xspeed = getattr(entity, 'xspeed', 0)
        self.yspeed = getattr(entity, 'yspeed', 0)
        
        # Determine entity behavior type for pathfinding
        self.behavior_type = self._determine_behavior_type()
        
        # Get entity-specific properties
        self.properties = self._extract_properties()
    
    def _determine_behavior_type(self) -> str:
        """Determine pathfinding behavior type from entity type."""
        # Map entity types to pathfinding behavior types
        type_map = {
            9: 'thwump',          # Thwump
            16: 'shove_thwump',   # Shove Thwump  
            12: 'death_ball',     # Death Ball
            13: 'drone',          # Drone (various types)
            # Add more entity type mappings as needed
        }
        return type_map.get(self.entity_type, 'static')
    
    def _extract_properties(self) -> Dict[str, Any]:
        """Extract relevant properties for pathfinding prediction."""
        props = {}
        
        if self.behavior_type == 'thwump':
            # Extract thwump-specific properties
            props['rest_frames'] = getattr(self.entity, 'rest_frames', 120)
            props['smash_duration'] = getattr(self.entity, 'smash_duration', 15)
            props['wait_extended'] = getattr(self.entity, 'wait_extended', 60)
            props['return_duration'] = getattr(self.entity, 'return_duration', 30)
            props['smash_distance'] = getattr(self.entity, 'smash_distance', 96)
            props['direction'] = getattr(self.entity, 'direction', 1)
            
        elif self.behavior_type == 'drone':
            # Extract drone-specific properties
            props['patrol_speed'] = getattr(self.entity, 'patrol_speed', 2.0)
            props['patrol_nodes'] = getattr(self.entity, 'patrol_nodes', [])
            props['patrol_is_loop'] = getattr(self.entity, 'patrol_is_loop', True)
            
        elif self.behavior_type == 'death_ball':
            # Death balls use their current velocity and acceleration
            props['acceleration'] = getattr(self.entity, 'acceleration', 0.1)
            props['max_speed'] = getattr(self.entity, 'max_speed', 3.0)
            
        return props
    
    def update(self):
        """Update wrapper with current entity state."""
        self.xpos = self.entity.xpos
        self.ypos = self.entity.ypos
        self.active = self.entity.active
        self.xspeed = getattr(self.entity, 'xspeed', 0)
        self.yspeed = getattr(self.entity, 'yspeed', 0)


class EntityManager:
    """Manages entities for pathfinding using the existing entity system."""
    
    def __init__(self, sim):
        self.sim = sim
        self.entity_wrappers: Dict[int, EntityWrapper] = {}
        self.last_update_frame = -1
    
    def update_all(self):
        """Update all entity wrappers from the simulation."""
        if self.sim.frame == self.last_update_frame:
            return  # Already updated this frame
            
        self.last_update_frame = self.sim.frame
        
        # Update existing wrappers
        for wrapper in self.entity_wrappers.values():
            if wrapper.entity.active:
                wrapper.update()
        
        # Remove inactive entities
        inactive_ids = [wrapper_id for wrapper_id, wrapper in self.entity_wrappers.items() 
                       if not wrapper.entity.active]
        for wrapper_id in inactive_ids:
            del self.entity_wrappers[wrapper_id]
    
    def get_pathfinding_entities(self, center_pos: Tuple[float, float] = None, 
                               search_radius: float = None) -> List[EntityWrapper]:
        """Get entities relevant for pathfinding near a position."""
        self.update_all()
        
        if center_pos is None:
            # Return all entities if no position specified
            return list(self.entity_wrappers.values())
        
        relevant_entities = []
        
        # Get entities from simulation near the position
        if search_radius and search_radius < 100:  # Use neighborhood search for small radius
            from ..physics import gather_entities_from_neighbourhood
            nearby_entities = gather_entities_from_neighbourhood(self.sim, center_pos[0], center_pos[1])
        else:
            # For larger radius or no radius, check all entities
            nearby_entities = []
            for entity_list in self.sim.entity_dic.values():
                nearby_entities.extend([e for e in entity_list if e.active])
        
        for entity in nearby_entities:
            entity_id = id(entity)
            
            # Create wrapper if it doesn't exist
            if entity_id not in self.entity_wrappers:
                # Only wrap entities that are relevant for pathfinding
                if self._is_pathfinding_relevant(entity):
                    self.entity_wrappers[entity_id] = EntityWrapper(entity)
            
            if entity_id in self.entity_wrappers:
                wrapper = self.entity_wrappers[entity_id]
                
                # Check distance if search radius specified
                if search_radius is not None:
                    dx = wrapper.xpos - center_pos[0]
                    dy = wrapper.ypos - center_pos[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= search_radius:
                        relevant_entities.append(wrapper)
                else:
                    relevant_entities.append(wrapper)
        
        return relevant_entities
    
    def _is_pathfinding_relevant(self, entity) -> bool:
        """Check if an entity is relevant for pathfinding."""
        # Exclude certain entity types that don't affect pathfinding
        irrelevant_types = {
            0,   # Gold
            1,   # Exit switch
            2,   # Exit door
            5,   # Player spawn
            # Add other non-blocking entity types
        }
        
        if entity.type in irrelevant_types:
            return False
        
        # Include entities that are physically collidable or move
        return (getattr(entity, 'is_physical_collidable', False) or 
                getattr(entity, 'is_movable', False) or
                getattr(entity, 'is_thinkable', False))


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
    
    def get_nearby_entities(self, pos: Tuple[float, float], radius: float = 240.0) -> List[EntityWrapper]:
        """Get entities near a position that could affect pathfinding."""
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
        
        for entity_wrapper in nearby_entities:
            if self.collision_checker.entity_blocks_path(entity_wrapper.entity, pos1, pos2, radius):
                return True
                
        return False
    
    def check_line_of_sight(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Check if there's a clear line of sight between two points."""
        return self.collision_checker.check_line_of_sight(pos1, pos2)
    
    def get_entities_for_prediction(self) -> List[EntityWrapper]:
        """Get all entities that need movement prediction for dynamic pathfinding."""
        all_entities = self.entity_manager.get_pathfinding_entities()
        return [e for e in all_entities if e.behavior_type != 'static']
