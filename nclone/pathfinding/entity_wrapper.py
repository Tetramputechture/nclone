"""
Entity wrapper for pathfinding that works with the existing entity system.
This eliminates duplication by wrapping existing entities rather than recreating them.
"""

import math
from typing import Tuple, Dict, Any, Optional

from ..ninja import NINJA_RADIUS


class EntityWrapper:
    """Wrapper around existing entities to provide pathfinding-specific functionality."""
    
    # Entity type constants (from existing entities)
    TOGGLE_MINE = 1
    GOLD = 2
    EXIT_DOOR = 3
    EXIT_SWITCH = 4
    REGULAR_DOOR = 5
    LOCKED_DOOR = 6
    TRAP_DOOR = 8
    LAUNCH_PAD = 10
    ONE_WAY = 11
    DRONE_CLOCKWISE = 14
    BOUNCE_BLOCK = 17
    THWUMP = 20
    TOGGLE_MINE_TOGGLED = 21
    DEATH_BALL = 25
    DRONE_COUNTER_CLOCKWISE = 26
    SHWUMP = 28
    
    def __init__(self, entity):
        """Initialize wrapper with an existing entity."""
        self.entity = entity
        self.type = entity.type
        self.xpos = entity.xpos
        self.ypos = entity.ypos
        self.active = entity.active
        
        # Cache entity properties for pathfinding
        self._cache_properties()
    
    def _cache_properties(self):
        """Cache entity properties for efficient pathfinding queries."""
        # Get radius from entity
        self.radius = getattr(self.entity, 'RADIUS', 0)
        
        # Determine if entity is deadly
        self.is_deadly = self._determine_if_deadly()
        
        # Get movement properties
        self.is_movable = getattr(self.entity, 'is_movable', False)
        self.is_thinkable = getattr(self.entity, 'is_thinkable', False)
        
        # Get speeds for prediction
        self.xspeed = getattr(self.entity, 'xspeed', 0.0)
        self.yspeed = getattr(self.entity, 'yspeed', 0.0)
    
    def _determine_if_deadly(self) -> bool:
        """Determine if this entity can kill the ninja."""
        deadly_types = {
            self.TOGGLE_MINE_TOGGLED,  # Toggled mines
            self.DRONE_CLOCKWISE,
            self.DRONE_COUNTER_CLOCKWISE, 
            self.DEATH_BALL,
            self.THWUMP,
            self.SHWUMP
        }
        
        # Special case for toggle mines - only deadly when toggled
        if self.type == self.TOGGLE_MINE:
            return getattr(self.entity, 'state', 1) == 0  # State 0 = toggled = deadly
            
        return self.type in deadly_types
    
    def predict_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict entity position after given number of frames."""
        if not self.active or not self.is_movable:
            return (self.xpos, self.ypos)
        
        # Use entity-specific prediction if available
        if hasattr(self.entity, 'predict_position'):
            return self.entity.predict_position(frames_ahead)
        
        # Simple linear prediction for basic entities
        predicted_x = self.xpos + self.xspeed * frames_ahead
        predicted_y = self.ypos + self.yspeed * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def get_danger_radius(self, time_frame: int = 60) -> float:
        """Get the radius of danger around this entity for pathfinding."""
        if not self.is_deadly:
            return 0.0
            
        base_radius = self.radius
        
        # Entity-specific danger calculations
        if self.type == self.THWUMP:
            # Thwumps can move quickly
            forward_speed = getattr(self.entity, 'FORWARD_SPEED', 20.0/7.0)
            return base_radius + forward_speed * time_frame
            
        elif self.type in [self.DRONE_CLOCKWISE, self.DRONE_COUNTER_CLOCKWISE]:
            # Drones patrol at moderate speed
            speed = getattr(self.entity, 'speed', 1.0)
            return base_radius + speed * time_frame * 0.5
            
        elif self.type == self.DEATH_BALL:
            # Death balls accelerate toward ninja
            max_speed = getattr(self.entity, 'MAX_SPEED', 0.85)
            return base_radius + max_speed * time_frame
            
        else:
            return base_radius
    
    def affects_pathfinding(self) -> bool:
        """Check if this entity should be considered in pathfinding."""
        if not self.active:
            return False
            
        # Deadly entities always affect pathfinding
        if self.is_deadly:
            return True
            
        # Physical obstacles affect pathfinding
        if getattr(self.entity, 'is_physical_collidable', False):
            return True
            
        # Doors and switches can block paths
        if self.type in [self.REGULAR_DOOR, self.LOCKED_DOOR, self.TRAP_DOOR]:
            return getattr(self.entity, 'closed', False)
            
        return False
    
    def update_from_entity(self):
        """Update wrapper state from the underlying entity."""
        self.xpos = self.entity.xpos
        self.ypos = self.entity.ypos
        self.active = self.entity.active
        self.xspeed = getattr(self.entity, 'xspeed', 0.0)
        self.yspeed = getattr(self.entity, 'yspeed', 0.0)
        
        # Update deadly status for toggle mines
        if self.type == self.TOGGLE_MINE:
            self.is_deadly = getattr(self.entity, 'state', 1) == 0


class EntityManager:
    """Manages entity wrappers for pathfinding."""
    
    def __init__(self, sim):
        """Initialize with a reference to the simulator."""
        self.sim = sim
        self.entity_wrappers: Dict[int, EntityWrapper] = {}
        self._build_wrappers()
    
    def _build_wrappers(self):
        """Build wrappers for all entities in the simulation."""
        self.entity_wrappers.clear()
        
        if not hasattr(self.sim, 'entity_dic'):
            return
            
        entity_id = 0
        for entity_list in self.sim.entity_dic.values():
            for entity in entity_list:
                wrapper = EntityWrapper(entity)
                self.entity_wrappers[entity_id] = wrapper
                entity_id += 1
    
    def get_pathfinding_entities(self, pos: Tuple[float, float], 
                                radius: float = 240.0) -> list[EntityWrapper]:
        """Get entities near a position that affect pathfinding."""
        nearby_entities = []
        
        for wrapper in self.entity_wrappers.values():
            if not wrapper.affects_pathfinding():
                continue
                
            # Check distance
            dx = wrapper.xpos - pos[0]
            dy = wrapper.ypos - pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= radius:
                nearby_entities.append(wrapper)
        
        return nearby_entities
    
    def update_all(self):
        """Update all wrappers from their underlying entities."""
        for wrapper in self.entity_wrappers.values():
            wrapper.update_from_entity()
    
    def get_deadly_entities(self) -> list[EntityWrapper]:
        """Get all deadly entities that could kill the ninja."""
        return [wrapper for wrapper in self.entity_wrappers.values() 
                if wrapper.is_deadly and wrapper.active] 