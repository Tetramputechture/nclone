"""
Enhanced entity handling for reachability analysis.

This module provides comprehensive entity integration for hazard detection,
collision handling, and entity-specific movement constraints in reachability analysis.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import math

from ...constants.entity_types import EntityType
from ...constants.physics_constants import (
    NINJA_RADIUS, TILE_PIXEL_SIZE, MAX_VER_SPEED, TERMINAL_VELOCITY
)


@dataclass
class EntityState:
    """Represents the current state of an entity."""
    
    entity_id: int
    entity_type: EntityType
    position: Tuple[float, float]
    active: bool = True
    switch_state: bool = False
    movement_data: Optional[Dict[str, Any]] = None


class EntityHandler:
    """
    Handles entity-specific logic for reachability analysis.
    
    Features:
    - Hazard detection and avoidance
    - Entity collision checking
    - Switch and door state management
    - Bounce block trajectory calculations
    - One-way platform constraints
    - Drone and thwump movement prediction
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize entity handler.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.entity_states: Dict[int, EntityState] = {}
        self.hazard_zones: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.one_way_platforms: List[Dict[str, Any]] = []
        self.bounce_blocks: List[Dict[str, Any]] = []
        
    def initialize_for_level(self, entities: List[Dict[str, Any]]):
        """
        Initialize entity handler for a specific level.
        
        Args:
            entities: List of entity dictionaries from level data
        """
        self.entity_states.clear()
        self.hazard_zones.clear()
        self.one_way_platforms.clear()
        self.bounce_blocks.clear()
        
        for entity in entities:
            entity_id = entity.get('id', 0)
            entity_type = entity.get('type')
            position = (entity.get('x', 0), entity.get('y', 0))
            
            # Create entity state
            state = EntityState(
                entity_id=entity_id,
                entity_type=entity_type,
                position=position,
                active=True,
                switch_state=entity.get('switch_state', False),
                movement_data=entity.get('movement_data', {})
            )
            
            self.entity_states[entity_id] = state
            
            # Process entity-specific initialization
            self._process_entity_for_reachability(entity, state)
    
    def _process_entity_for_reachability(self, entity: Dict[str, Any], state: EntityState):
        """
        Process entity for reachability-specific data structures.
        
        Args:
            entity: Entity dictionary
            state: Entity state object
        """
        entity_type = state.entity_type
        x, y = state.position
        
        if entity_type == EntityType.TOGGLE_MINE:
            # Toggle mines create hazard zones when active
            if state.active:
                # Mine explosion radius (approximate)
                explosion_radius = 30.0
                self.hazard_zones.append((x, y, explosion_radius))
        
        elif entity_type == EntityType.DRONE_ZAP:
            # Drones create moving hazard zones
            # For reachability, we consider their patrol area as hazardous
            patrol_radius = entity.get('patrol_radius', 50.0)
            self.hazard_zones.append((x, y, patrol_radius))
        
        elif entity_type == EntityType.THWUMP:
            # Thwumps create hazard zones in their movement path
            # For simplicity, consider a rectangular hazard zone
            thwump_width = 40.0
            thwump_height = 100.0
            # Add as circular hazard for now
            self.hazard_zones.append((x, y, max(thwump_width, thwump_height) / 2))
        
        elif entity_type == EntityType.BOUNCE_BLOCK:
            # Bounce blocks affect movement trajectories
            self.bounce_blocks.append({
                'id': state.entity_id,
                'position': (x, y),
                'size': entity.get('size', (24, 24)),
                'bounce_factor': entity.get('bounce_factor', 1.5)
            })
        
        elif entity_type == EntityType.ONE_WAY_PLATFORM:
            # One-way platforms restrict movement direction
            self.one_way_platforms.append({
                'id': state.entity_id,
                'position': (x, y),
                'size': entity.get('size', (48, 12)),
                'direction': entity.get('direction', 'up')  # up, down, left, right
            })
    
    def is_position_safe(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is safe from hazards.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if position is safe from all hazards
        """
        x, y = position
        
        # Check against all hazard zones
        for hazard_x, hazard_y, hazard_radius in self.hazard_zones:
            distance = math.sqrt((x - hazard_x) ** 2 + (y - hazard_y) ** 2)
            if distance < hazard_radius + NINJA_RADIUS:
                if self.debug:
                    print(f"DEBUG: Position ({x}, {y}) is unsafe - hazard at ({hazard_x}, {hazard_y})")
                return False
        
        return True
    
    def can_traverse_between_positions(
        self, 
        start_pos: Tuple[float, float], 
        end_pos: Tuple[float, float],
        movement_type: str = "walk"
    ) -> bool:
        """
        Check if ninja can safely traverse between two positions.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            movement_type: Type of movement (walk, jump, fall)
            
        Returns:
            True if traversal is safe and possible
        """
        # Check if both positions are safe
        if not self.is_position_safe(start_pos) or not self.is_position_safe(end_pos):
            return False
        
        # Check one-way platform constraints
        if not self._check_one_way_platform_constraints(start_pos, end_pos, movement_type):
            return False
        
        # Sample points along the path for hazard checking
        return self._check_path_safety(start_pos, end_pos)
    
    def _check_one_way_platform_constraints(
        self, 
        start_pos: Tuple[float, float], 
        end_pos: Tuple[float, float],
        movement_type: str
    ) -> bool:
        """
        Check one-way platform movement constraints.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            movement_type: Type of movement
            
        Returns:
            True if movement respects one-way platform constraints
        """
        for platform in self.one_way_platforms:
            platform_x, platform_y = platform['position']
            platform_w, platform_h = platform['size']
            direction = platform['direction']
            
            # Check if either position is on the platform
            start_on_platform = self._is_position_on_platform(start_pos, platform)
            end_on_platform = self._is_position_on_platform(end_pos, platform)
            
            if start_on_platform or end_on_platform:
                # Apply direction constraints
                if direction == 'up' and movement_type == 'fall':
                    # Can't fall through upward one-way platform
                    if start_pos[1] < platform_y and end_pos[1] > platform_y:
                        return False
                elif direction == 'down' and movement_type == 'jump':
                    # Can't jump through downward one-way platform
                    if start_pos[1] > platform_y and end_pos[1] < platform_y:
                        return False
        
        return True
    
    def _is_position_on_platform(
        self, 
        position: Tuple[float, float], 
        platform: Dict[str, Any]
    ) -> bool:
        """
        Check if position is on a platform.
        
        Args:
            position: Position to check
            platform: Platform data
            
        Returns:
            True if position is on the platform
        """
        x, y = position
        platform_x, platform_y = platform['position']
        platform_w, platform_h = platform['size']
        
        return (platform_x - platform_w/2 <= x <= platform_x + platform_w/2 and
                platform_y - platform_h/2 <= y <= platform_y + platform_h/2)
    
    def _check_path_safety(
        self, 
        start_pos: Tuple[float, float], 
        end_pos: Tuple[float, float],
        samples: int = 10
    ) -> bool:
        """
        Check if path between positions is safe from hazards.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            samples: Number of points to sample along path
            
        Returns:
            True if entire path is safe
        """
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        for i in range(samples + 1):
            t = i / samples
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)
            
            if not self.is_position_safe((sample_x, sample_y)):
                return False
        
        return True
    
    def get_bounce_trajectory(
        self, 
        position: Tuple[float, float], 
        velocity: Tuple[float, float]
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculate bounce trajectory if ninja hits a bounce block.
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            Tuple of (new_position, new_velocity) or None if no bounce
        """
        x, y = position
        vx, vy = velocity
        
        for bounce_block in self.bounce_blocks:
            block_x, block_y = bounce_block['position']
            block_w, block_h = bounce_block['size']
            bounce_factor = bounce_block['bounce_factor']
            
            # Check collision with bounce block
            if (block_x - block_w/2 - NINJA_RADIUS <= x <= block_x + block_w/2 + NINJA_RADIUS and
                block_y - block_h/2 - NINJA_RADIUS <= y <= block_y + block_h/2 + NINJA_RADIUS):
                
                # Calculate bounce direction based on collision side
                dx = x - block_x
                dy = y - block_y
                
                if abs(dx) > abs(dy):
                    # Horizontal bounce
                    new_vx = -vx * bounce_factor if dx * vx < 0 else vx
                    new_vy = vy
                else:
                    # Vertical bounce
                    new_vx = vx
                    new_vy = -vy * bounce_factor if dy * vy < 0 else vy
                
                # Calculate new position after bounce
                new_x = x + new_vx * 0.1  # Small time step
                new_y = y + new_vy * 0.1
                
                return ((new_x, new_y), (new_vx, new_vy))
        
        return None
    
    def update_switch_states(self, switch_states: Dict[int, bool]):
        """
        Update entity states based on switch activations.
        
        Args:
            switch_states: Dictionary of switch ID to state
        """
        for entity_id, state in self.entity_states.items():
            if state.entity_type in [EntityType.REGULAR_DOOR, EntityType.LOCKED_DOOR]:
                # Doors are controlled by switches
                controlling_switch = state.movement_data.get('controlling_switch')
                if controlling_switch is not None and controlling_switch in switch_states:
                    state.active = not switch_states[controlling_switch]  # Door open when switch active
            
            elif state.entity_type == EntityType.TOGGLE_MINE:
                # Toggle mines can be activated/deactivated
                if entity_id in switch_states:
                    state.active = switch_states[entity_id]
        
        # Rebuild hazard zones after state changes
        self._rebuild_hazard_zones()
    
    def _rebuild_hazard_zones(self):
        """Rebuild hazard zones based on current entity states."""
        self.hazard_zones.clear()
        
        for state in self.entity_states.values():
            if not state.active:
                continue
                
            x, y = state.position
            
            if state.entity_type == EntityType.TOGGLE_MINE:
                explosion_radius = 30.0
                self.hazard_zones.append((x, y, explosion_radius))
            
            elif state.entity_type == EntityType.DRONE_ZAP:
                patrol_radius = state.movement_data.get('patrol_radius', 50.0)
                self.hazard_zones.append((x, y, patrol_radius))
            
            elif state.entity_type == EntityType.THWUMP:
                thwump_size = max(40.0, 100.0) / 2
                self.hazard_zones.append((x, y, thwump_size))
    
    def get_entity_influenced_positions(self) -> Set[Tuple[int, int]]:
        """
        Get all grid positions that are influenced by entities.
        
        Returns:
            Set of (row, col) positions that are affected by entities
        """
        influenced_positions = set()
        
        # Add positions around hazard zones
        for hazard_x, hazard_y, hazard_radius in self.hazard_zones:
            # Convert to grid coordinates
            center_row = int(hazard_y // TILE_PIXEL_SIZE)
            center_col = int(hazard_x // TILE_PIXEL_SIZE)
            
            # Add surrounding positions based on hazard radius
            grid_radius = int(hazard_radius // TILE_PIXEL_SIZE) + 1
            
            for dr in range(-grid_radius, grid_radius + 1):
                for dc in range(-grid_radius, grid_radius + 1):
                    influenced_positions.add((center_row + dr, center_col + dc))
        
        # Add positions around bounce blocks and one-way platforms
        for bounce_block in self.bounce_blocks:
            x, y = bounce_block['position']
            row = int(y // TILE_PIXEL_SIZE)
            col = int(x // TILE_PIXEL_SIZE)
            influenced_positions.add((row, col))
        
        for platform in self.one_way_platforms:
            x, y = platform['position']
            row = int(y // TILE_PIXEL_SIZE)
            col = int(x // TILE_PIXEL_SIZE)
            influenced_positions.add((row, col))
        
        return influenced_positions
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about entity states.
        
        Returns:
            Dictionary with entity debug information
        """
        return {
            'entity_count': len(self.entity_states),
            'hazard_zones': len(self.hazard_zones),
            'bounce_blocks': len(self.bounce_blocks),
            'one_way_platforms': len(self.one_way_platforms),
            'active_entities': sum(1 for state in self.entity_states.values() if state.active),
            'entity_types': [state.entity_type.name for state in self.entity_states.values()]
        }