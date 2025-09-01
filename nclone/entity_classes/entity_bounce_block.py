
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS
from enum import IntEnum
import math


class BounceBlockState(IntEnum):
    """States of bounce block compression/extension cycle."""
    NEUTRAL = 0      # At rest position
    COMPRESSING = 1  # Being compressed by ninja
    COMPRESSED = 2   # Fully compressed, storing energy
    EXTENDING = 3    # Extending back to neutral, can provide boost


class EntityBounceBlock(Entity):
    """Enhanced Bounce Block Entity (Type 17)

    A dynamic physics-based block that acts as a spring-mass system with advanced compression
    mechanics and boost capabilities. Bounce blocks enable complex movement strategies through
    state-dependent interactions and repeated boost mechanics.

    Physical Properties:
        - Size: 9*9 pixel square (18*18 total)
        - Collision Type: Square with spring physics
        - Max Per Level: 512 instances
        - Spring Constants:
            * Stiffness: 0.0222 (determines spring force)
            * Dampening: 0.98 (velocity decay per frame)
            * Strength: 0.2 (force distribution ratio to ninja)
        - Compression Properties:
            * Max Compression: 6 pixels in any direction
            * Compression Resistance: Increases with multiple blocks
            * Boost Multiplier: 1.5x to 3.0x based on compression

    Enhanced Behavior:
        - State-Based Physics:
            * NEUTRAL: Standard spring behavior
            * COMPRESSING: Stores energy, increases resistance
            * COMPRESSED: Maximum energy storage, ready for boost
            * EXTENDING: Provides boost force to ninja
        - Boost Mechanics:
            * Repeated jumping during extension provides cumulative boost
            * Boost direction opposite to available clearance
            * Can chain with other bounce blocks for extended sequences
        - Multi-Block Interactions:
            * Force accumulation when multiple blocks are compressed
            * Reduced individual compression when stacked
            * Can create barriers or platforms depending on density

    AI Strategy Notes:
        - Use compression state for timing-based strategies
        - Chain multiple blocks for extended movement sequences
        - Exploit repeated boost mechanics for maximum height/distance
        - Consider clearance directions for optimal boost angles
        - Use density patterns to create traversable vs impassable areas

    Technical Implementation:
        - Enhanced state machine with compression tracking
        - Force accumulation system for multi-block interactions
        - Boost calculation based on compression energy and clearance
        - Clearance detection for boost direction optimization
    """
    ENTITY_TYPE = 17
    SEMI_SIDE = 9
    STIFFNESS = 0.02222222222222222
    DAMPENING = 0.98
    STRENGTH = 0.2
    MAX_COUNT_PER_LEVEL = 512
    
    # Enhanced physics constants
    MAX_COMPRESSION = 6.0  # Maximum compression distance in pixels
    MIN_BOOST_MULTIPLIER = 1.5  # Minimum boost when lightly compressed
    MAX_BOOST_MULTIPLIER = 3.0  # Maximum boost when fully compressed
    COMPRESSION_RESISTANCE_BASE = 1.0  # Base resistance to compression
    COMPRESSION_RESISTANCE_MULTI = 0.3  # Additional resistance per nearby block
    BOOST_DECAY_RATE = 0.95  # How quickly repeated boosts decay
    CLEARANCE_CHECK_DISTANCE = 24.0  # Distance to check for clearance (1 tile)

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_physical_collidable = True
        self.is_logical_collidable = True
        self.is_movable = True
        self.xspeed, self.yspeed = 0, 0
        self.xorigin, self.yorigin = self.xpos, self.ypos
        
        # Enhanced state tracking
        self.bounce_state = BounceBlockState.NEUTRAL
        self.compression_amount = 0.0  # 0.0 to 1.0 (fraction of MAX_COMPRESSION)
        self.compression_direction = (0.0, 0.0)  # Direction of compression
        self.stored_energy = 0.0  # Energy stored from compression
        self.boost_count = 0  # Number of consecutive boosts this cycle
        self.last_ninja_contact = False  # Whether ninja was in contact last frame
        self.compression_timer = 0  # Frames since compression started
        self.nearby_blocks = []  # Cache of nearby bounce blocks
        self.clearance_cache = {}  # Cache of clearance in each direction

    def move(self):
        """Enhanced movement with state-based physics and compression mechanics."""
        # Update nearby blocks cache periodically
        if self.sim.frame_count % 10 == 0:  # Update every 10 frames
            self._update_nearby_blocks()
        
        # Check ninja contact
        ninja_contact = self._check_ninja_contact()
        
        # Update state based on ninja interaction
        self._update_bounce_state(ninja_contact)
        
        # Apply dampening
        self.xspeed *= self.DAMPENING
        self.yspeed *= self.DAMPENING
        
        # Apply position updates
        self.xpos += self.xspeed
        self.ypos += self.yspeed
        
        # Calculate spring forces with state-dependent modifications
        base_xforce = self.STIFFNESS * (self.xorigin - self.xpos)
        base_yforce = self.STIFFNESS * (self.yorigin - self.ypos)
        
        # Modify forces based on state and compression
        xforce, yforce = self._calculate_state_modified_forces(base_xforce, base_yforce)
        
        # Apply forces
        self.xpos += xforce
        self.ypos += yforce
        self.xspeed += xforce
        self.yspeed += yforce
        
        # Update compression tracking
        self._update_compression_tracking()
        
        # Update clearance cache if needed
        if self.sim.frame_count % 30 == 0:  # Update every 30 frames
            self._update_clearance_cache()
        
        self.grid_move()

    def physical_collision(self):
        """Enhanced collision with boost mechanics and state-dependent behavior."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            
            # Calculate compression resistance based on nearby blocks
            resistance = self._calculate_compression_resistance()
            
            # Apply forces with resistance
            block_force = (1 - self.STRENGTH) * resistance
            ninja_force = self.STRENGTH / resistance
            
            self.xpos -= depen_x * depen_len * block_force
            self.ypos -= depen_y * depen_len * block_force
            self.xspeed -= depen_x * depen_len * block_force
            self.yspeed -= depen_y * depen_len * block_force
            
            # Update compression state
            self._update_compression_from_collision(depen_x, depen_y, depen_len)
            
            # Apply boost if in extending state
            boost_force = self._calculate_boost_force(depen_x, depen_y)
            
            return (depen_x, depen_y), (depen_len * ninja_force + boost_force, depen[1][1])

    def logical_collision(self):
        """Enhanced logical collision with wall jump support."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + NINJA_RADIUS + 0.1)
        if depen:
            # Bounce blocks support wall jumping from all sides
            return depen[0][0]
    
    def _check_ninja_contact(self) -> bool:
        """Check if ninja is currently in contact with this bounce block."""
        ninja = self.sim.ninja
        distance = math.sqrt((self.xpos - ninja.xpos)**2 + (self.ypos - ninja.ypos)**2)
        return distance <= (self.SEMI_SIDE + NINJA_RADIUS + 1.0)
    
    def _update_bounce_state(self, ninja_contact: bool):
        """Update bounce block state based on ninja interaction."""
        if ninja_contact and not self.last_ninja_contact:
            # Ninja just made contact
            if self.bounce_state == BounceBlockState.NEUTRAL:
                self.bounce_state = BounceBlockState.COMPRESSING
                self.compression_timer = 0
                self.boost_count = 0
        elif not ninja_contact and self.last_ninja_contact:
            # Ninja just lost contact
            if self.bounce_state in [BounceBlockState.COMPRESSING, BounceBlockState.COMPRESSED]:
                self.bounce_state = BounceBlockState.EXTENDING
                self.stored_energy = self.compression_amount * self.MAX_BOOST_MULTIPLIER
        
        # Update compression timer
        if self.bounce_state == BounceBlockState.COMPRESSING:
            self.compression_timer += 1
            if self.compression_timer > 10:  # After 10 frames of compression
                self.bounce_state = BounceBlockState.COMPRESSED
        elif self.bounce_state == BounceBlockState.EXTENDING:
            # Check if back to neutral position
            distance_from_origin = math.sqrt((self.xpos - self.xorigin)**2 + (self.ypos - self.yorigin)**2)
            if distance_from_origin < 1.0:  # Close to origin
                self.bounce_state = BounceBlockState.NEUTRAL
                self.compression_amount = 0.0
                self.stored_energy = 0.0
                self.boost_count = 0
        
        self.last_ninja_contact = ninja_contact
    
    def _calculate_state_modified_forces(self, base_xforce: float, base_yforce: float) -> tuple[float, float]:
        """Calculate forces modified by current state."""
        if self.bounce_state == BounceBlockState.COMPRESSING:
            # Reduce spring force during compression to allow more displacement
            return base_xforce * 0.7, base_yforce * 0.7
        elif self.bounce_state == BounceBlockState.EXTENDING:
            # Increase spring force during extension for boost effect
            return base_xforce * 1.3, base_yforce * 1.3
        else:
            return base_xforce, base_yforce
    
    def _update_compression_tracking(self):
        """Update compression amount and direction."""
        displacement_x = self.xpos - self.xorigin
        displacement_y = self.ypos - self.yorigin
        displacement_magnitude = math.sqrt(displacement_x**2 + displacement_y**2)
        
        if displacement_magnitude > 0:
            self.compression_amount = min(displacement_magnitude / self.MAX_COMPRESSION, 1.0)
            self.compression_direction = (displacement_x / displacement_magnitude, 
                                        displacement_y / displacement_magnitude)
        else:
            self.compression_amount = 0.0
            self.compression_direction = (0.0, 0.0)
    
    def _update_nearby_blocks(self):
        """Update cache of nearby bounce blocks."""
        self.nearby_blocks = []
        for entity in self.sim.entities:
            if (entity != self and 
                hasattr(entity, 'ENTITY_TYPE') and 
                entity.ENTITY_TYPE == 17 and 
                entity.active):
                distance = math.sqrt((self.xpos - entity.xpos)**2 + (self.ypos - entity.ypos)**2)
                if distance < 50.0:  # Within 50 pixels
                    self.nearby_blocks.append(entity)
    
    def _calculate_compression_resistance(self) -> float:
        """Calculate resistance to compression based on nearby blocks."""
        base_resistance = self.COMPRESSION_RESISTANCE_BASE
        nearby_count = len(self.nearby_blocks)
        return base_resistance + (nearby_count * self.COMPRESSION_RESISTANCE_MULTI)
    
    def _update_compression_from_collision(self, depen_x: float, depen_y: float, depen_len: float):
        """Update compression state from collision data."""
        if self.bounce_state == BounceBlockState.COMPRESSING:
            # Store compression direction for boost calculation
            if depen_len > 0:
                self.compression_direction = (-depen_x, -depen_y)  # Opposite to penetration
    
    def _calculate_boost_force(self, depen_x: float, depen_y: float) -> float:
        """Calculate boost force when in extending state."""
        if self.bounce_state != BounceBlockState.EXTENDING or self.stored_energy <= 0:
            return 0.0
        
        # Check if ninja is jumping (would indicate repeated boost attempt)
        ninja = self.sim.ninja
        ninja_jumping = hasattr(ninja, 'movement_state') and ninja.movement_state == 3
        
        if ninja_jumping:
            # Calculate boost based on stored energy and clearance
            clearance_boost = self._get_optimal_boost_direction()
            boost_multiplier = self.stored_energy * (self.BOOST_DECAY_RATE ** self.boost_count)
            self.boost_count += 1
            
            # Reduce stored energy
            self.stored_energy *= 0.8
            
            return boost_multiplier * clearance_boost
        
        return 0.0
    
    def _get_optimal_boost_direction(self) -> float:
        """Get boost multiplier based on optimal clearance direction."""
        if not self.clearance_cache:
            return 1.0
        
        # Find direction with most clearance
        max_clearance = 0.0
        for direction, clearance in self.clearance_cache.items():
            max_clearance = max(max_clearance, clearance)
        
        # Boost is stronger when there's more clearance
        return min(max_clearance / self.CLEARANCE_CHECK_DISTANCE, 2.0)
    
    def _update_clearance_cache(self):
        """Update cache of clearance in each direction."""
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-0.707, -0.707),
            'up_right': (0.707, -0.707),
            'down_left': (-0.707, 0.707),
            'down_right': (0.707, 0.707)
        }
        
        self.clearance_cache = {}
        for direction_name, (dx, dy) in directions.items():
            clearance = self._check_clearance_in_direction(dx, dy)
            self.clearance_cache[direction_name] = clearance
    
    def _check_clearance_in_direction(self, dx: float, dy: float) -> float:
        """Check how much clearance exists in a given direction."""
        # Simple implementation - check for solid tiles
        check_distance = 0.0
        step_size = 6.0  # Check every 6 pixels (quarter tile)
        
        while check_distance < self.CLEARANCE_CHECK_DISTANCE:
            check_x = self.xpos + dx * check_distance
            check_y = self.ypos + dy * check_distance
            
            # Convert to tile coordinates
            tile_x = int(check_x // 24)  # TILE_PIXEL_SIZE = 24
            tile_y = int(check_y // 24)
            
            # Check if tile is solid (simplified check)
            if (tile_x < 0 or tile_x >= 44 or tile_y < 0 or tile_y >= 25):
                break  # Hit boundary
            
            # Check for solid tiles (would need access to level data)
            # For now, assume open space
            check_distance += step_size
        
        return check_distance
    
    def get_boost_potential(self, ninja_position: tuple[float, float], ninja_velocity: tuple[float, float]) -> dict:
        """Get boost potential information for AI planning."""
        return {
            'state': self.bounce_state.value,
            'compression_amount': self.compression_amount,
            'stored_energy': self.stored_energy,
            'boost_count': self.boost_count,
            'optimal_direction': self._get_optimal_boost_direction_vector(),
            'clearance_map': self.clearance_cache.copy(),
            'nearby_blocks': len(self.nearby_blocks),
            'can_boost': self.bounce_state == BounceBlockState.EXTENDING and self.stored_energy > 0
        }
    
    def _get_optimal_boost_direction_vector(self) -> tuple[float, float]:
        """Get the optimal boost direction as a vector."""
        if not self.clearance_cache:
            return (0.0, -1.0)  # Default upward
        
        # Find direction with maximum clearance
        max_clearance = 0.0
        best_direction = (0.0, -1.0)
        
        direction_vectors = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-0.707, -0.707),
            'up_right': (0.707, -0.707),
            'down_left': (-0.707, 0.707),
            'down_right': (0.707, 0.707)
        }
        
        for direction_name, clearance in self.clearance_cache.items():
            if clearance > max_clearance and direction_name in direction_vectors:
                max_clearance = clearance
                best_direction = direction_vectors[direction_name]
        
        return best_direction
