
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS
from ..utils.physics_utils import (
    BounceBlockState, 
    calculate_compression_amount,
    calculate_stored_energy,
    determine_bounce_block_state,
    calculate_clearance_directions
)
from ..utils.collision_utils import (
    find_entities_in_radius
)
from ..constants.physics_constants import *


class EntityBounceBlock(Entity):
    """Bounce Block Entity (Type 17) with Enhanced Data Tracking
    
    This class extends the base bounce block entity to provide comprehensive
    data tracking for RL integration without modifying the original physics behavior.
    
    Data Tracking Features:
        - State tracking (NEUTRAL, COMPRESSING, COMPRESSED, EXTENDING)
        - Compression amount and direction monitoring
        - Energy storage calculations
        - Clearance detection in all directions
        - Multi-block interaction tracking
        - Boost potential calculations
    
    Note: This implementation only adds data collection capabilities.
    The original physics behavior remains unchanged.
    """
    ENTITY_TYPE = 17
    SEMI_SIDE = 9
    STIFFNESS = 0.02222222222222222
    DAMPENING = 0.98
    STRENGTH = 0.2
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_physical_collidable = True
        self.is_logical_collidable = True
        self.is_movable = True
        self.xspeed, self.yspeed = 0, 0
        self.xorigin, self.yorigin = self.xpos, self.ypos
        
        # Data tracking attributes (do not affect physics behavior)
        self.bounce_state = BounceBlockState.NEUTRAL
        self.compression_amount = 0.0  # 0.0 to 1.0 normalized compression
        self.previous_compression = 0.0  # Previous frame compression for state detection
        self.stored_energy = 0.0  # Calculated stored energy
        self.clearance_directions = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
        self.nearby_bounce_blocks = []  # List of nearby bounce blocks
        self.ninja_contact = False  # Current ninja contact status
        
        # Cache update counters
        self._clearance_update_counter = 0
        self._nearby_blocks_update_counter = 0

    def move(self):
        """Original move method with added data tracking (no behavior changes)."""
        # Original physics behavior
        self.xspeed *= self.DAMPENING
        self.yspeed *= self.DAMPENING
        self.xpos += self.xspeed
        self.ypos += self.yspeed
        
        xforce = self.STIFFNESS * (self.xorigin - self.xpos)
        yforce = self.STIFFNESS * (self.yorigin - self.ypos)
        
        self.xpos += xforce
        self.ypos += yforce
        self.xspeed += xforce
        self.yspeed += yforce
        
        self.grid_move()
        
        # Data tracking only (does not affect physics)
        self._update_data_tracking()

    def physical_collision(self):
        """Original physical collision with data tracking (no behavior changes)."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            
            # Original physics behavior
            self.xpos -= depen_x * depen_len * (1 - self.STRENGTH)
            self.ypos -= depen_y * depen_len * (1 - self.STRENGTH)
            self.xspeed -= depen_x * depen_len * (1 - self.STRENGTH)
            self.yspeed -= depen_y * depen_len * (1 - self.STRENGTH)
            
            # Data tracking only (does not affect physics)
            self.ninja_contact = True
            
            return (depen_x, depen_y), (depen_len * self.STRENGTH, depen[1][1])

    def logical_collision(self):
        """Original logical collision (no changes)."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen:
            return depen[0][0]
    
    def _update_data_tracking(self):
        """Update all data tracking attributes (does not affect physics behavior)."""
        # Update compression amount based on displacement from origin
        current_pos = (self.xpos, self.ypos)
        original_pos = (self.xorigin, self.yorigin)
        self.previous_compression = self.compression_amount
        self.compression_amount = calculate_compression_amount(current_pos, original_pos)
        
        # Update stored energy calculation
        self.stored_energy = calculate_stored_energy(self.compression_amount)
        
        # Update bounce state based on compression changes and ninja contact
        self.bounce_state = determine_bounce_block_state(
            self.compression_amount, 
            self.previous_compression, 
            self.ninja_contact
        )
        
        # Update nearby blocks cache periodically
        self._nearby_blocks_update_counter += 1
        if self._nearby_blocks_update_counter >= 10:  # Every 10 frames
            self._update_nearby_blocks_cache()
            self._nearby_blocks_update_counter = 0
        
        # Update clearance cache periodically
        self._clearance_update_counter += 1
        if self._clearance_update_counter >= 30:  # Every 30 frames
            self._update_clearance_cache()
            self._clearance_update_counter = 0
        
        # Reset ninja contact for next frame
        self.ninja_contact = False
    
    def _update_nearby_blocks_cache(self):
        """Update cache of nearby bounce blocks."""
        # Get all entities from entity_dic and convert to dictionary format
        all_entities = []
        for entity_list in self.sim.entity_dic.values():
            for entity in entity_list:
                if hasattr(entity, 'ENTITY_TYPE'):
                    entity_dict = {
                        'type': entity.ENTITY_TYPE,
                        'x': entity.xpos,
                        'y': entity.ypos,
                        'entity_ref': entity  # Keep reference to original entity
                    }
                    all_entities.append(entity_dict)
        
        found_entity_dicts = find_entities_in_radius(
            (self.xpos, self.ypos),
            BOUNCE_BLOCK_CHAIN_DISTANCE,
            all_entities,
            ENTITY_TYPE_BOUNCE_BLOCK
        )
        
        # Extract original entity references
        self.nearby_bounce_blocks = [entity_dict['entity_ref'] for entity_dict in found_entity_dicts]
        # Remove self from the list
        self.nearby_bounce_blocks = [block for block in self.nearby_bounce_blocks 
                                   if block != self]
    
    def _update_clearance_cache(self):
        """Update clearance directions cache."""
        # Get all active entities from entity_dic and convert to dictionary format
        level_entities = []
        for entity_list in self.sim.entity_dic.values():
            for entity in entity_list:
                if hasattr(entity, 'ENTITY_TYPE') and entity.active:
                    entity_dict = {
                        'type': entity.ENTITY_TYPE,
                        'x': entity.xpos,
                        'y': entity.ypos
                    }
                    level_entities.append(entity_dict)
        self.clearance_directions = calculate_clearance_directions(
            (self.xpos, self.ypos), 
            level_entities
        )
    
    def get_bounce_block_data(self) -> dict:
        """Get comprehensive bounce block data for RL system."""
        return {
            'position': (self.xpos, self.ypos),
            'original_position': (self.xorigin, self.yorigin),
            'velocity': (self.xspeed, self.yspeed),
            'bounce_state': self.bounce_state,
            'compression_amount': self.compression_amount,
            'stored_energy': self.stored_energy,
            'clearance_directions': self.clearance_directions.copy(),
            'nearby_blocks_count': len(self.nearby_bounce_blocks),
            'ninja_contact': self.ninja_contact
        }
