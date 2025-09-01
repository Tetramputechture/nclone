
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
    """Bounce Block Entity (Type 17)

    A dynamic physics-based block that acts as a spring-mass system, providing momentum-preserving
    interactions with the ninja. Bounce blocks are crucial for advanced movement techniques and
    can be used for creative routing strategies.

    Physical Properties:
        - Size: 9*9 pixel square (18*18 total)
        - Collision Type: Square with spring physics
        - Max Per Level: 512 instances
        - Spring Constants:
            * Stiffness: 0.0222 (determines spring force)
            * Dampening: 0.98 (velocity decay per frame)
            * Strength: 0.2 (force distribution ratio to ninja)

    Behavior:
        - Spring Physics:
            * Maintains an origin point and current position
            * Applies spring force based on displacement from origin
            * Force proportional to distance (Hooke's Law)
            * Velocity decays through dampening
        - Collision Response:
            * 80% of collision force applied to block
            * 20% of collision force applied to ninja
            * Supports both physical and logical (wall) collisions
            * Additional 0.1 pixel buffer for wall interactions
        - Block Boosting (Advanced Technique):
            * Rapid successive jump key presses while block compresses back to origin
            * Creates large velocity boosts in movement direction
            * Can be performed on top (upward) or side (wall-jump style)
            * Timing: Press/release jump frame-by-frame during compression phase
            * Natural physics interaction - no special mechanics required
        - Multiple Block Interactions:
            * Each bounce block applies independent opposing force to ninja
            * Forces accumulate additively when multiple blocks contact ninja
            * Creates compound dampening effects and complex momentum transfer
            * Natural result of physics system with overlapping collision zones

    AI Strategy Notes:
        - Use for momentum preservation in complex jumps
        - Can create dynamic platforms for height gain
        - Chain interactions for extended movement sequences
        - Consider block's current state (compressed/extended) for timing
        - Block Boosting Applications:
            * Time rapid jump inputs during compression phase for speed boosts
            * Use wall-side boosting for complex routing and height gain
            * Combine with directional inputs for precise trajectory control
        - Multiple Block Strategy:
            * Account for additive dampening when navigating block clusters
            * Use overlapping blocks for fine-tuned momentum control
            * Consider compound forces when planning multi-block sequences

    Technical Implementation:
        - Updates position and velocity each frame
        - Applies spring forces relative to origin point
        - Handles both physical and logical collision types
        - Supports position logging for debugging/replay
    """
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
        
        # Data tracking attributes
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
        """Move the bounce block."""
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
        
        # Data tracking
        self._update_data_tracking()

    def physical_collision(self):
        """Physical collision with data tracking."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            
            self.xpos -= depen_x * depen_len * (1 - self.STRENGTH)
            self.ypos -= depen_y * depen_len * (1 - self.STRENGTH)
            self.xspeed -= depen_x * depen_len * (1 - self.STRENGTH)
            self.yspeed -= depen_y * depen_len * (1 - self.STRENGTH)
            
            # Data tracking
            self.ninja_contact = True
            
            return (depen_x, depen_y), (depen_len * self.STRENGTH, depen[1][1])

    def logical_collision(self):
        """Logical collision."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen:
            return depen[0][0]
    
    def _update_data_tracking(self):
        """Update all data tracking attributes."""
        # Update compression amount based on displacement from origin
        current_pos = (self.xpos, self.ypos)
        original_pos = (self.xorigin, self.yorigin)
        self.previous_compression = self.compression_amount
        self.compression_amount = calculate_compression_amount(current_pos, original_pos)
        
        # Update stored energy
        self.stored_energy = calculate_stored_energy(self.compression_amount)
        
        # Update bounce state
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
                entity_dict = {
                    'type': entity.type,
                    'x': entity.xpos,
                    'y': entity.ypos,
                    'entity_ref': entity  # Keep reference to original entity
                }
                all_entities.append(entity_dict)
        
        found_entity_dicts = find_entities_in_radius(
            (self.xpos, self.ypos),
            BOUNCE_BLOCK_CHAIN_DISTANCE,
            all_entities,
            self.type
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
                if entity.active:
                    entity_dict = {
                        'type': entity.type,
                        'x': entity.xpos,
                        'y': entity.ypos
                    }
                    level_entities.append(entity_dict)
        self.clearance_directions = calculate_clearance_directions(
            (self.xpos, self.ypos), 
            level_entities
        )
    
    def get_bounce_block_data(self) -> dict:
        """Get bounce block data for RL system."""
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
