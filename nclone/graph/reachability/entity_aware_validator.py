"""
Entity-aware position validation for advanced reachability analysis.

This module extends the basic position validation with entity awareness including:
- Switch and door state handling
- Dynamic hazard detection (drones, thwumps, etc.)
- Platform interaction logic
- Precise sub-pixel pathfinding for misaligned tiles
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import math

from .position_validator import PositionValidator
from .hazard_integration import ReachabilityHazardExtension
from ..hazard_system import HazardClassificationSystem
from ...constants.entity_types import EntityType
from ...constants.physics_constants import TILE_PIXEL_SIZE
from ..subgoal_planner import SubgoalPlanner


class EntityAwareValidator(PositionValidator):
    """Enhanced position validator with entity awareness."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize entity-aware validator.
        
        Args:
            debug: Enable debug output
        """
        super().__init__(debug)
        self.hazard_system = None
        self.hazard_extension = None
        self.switch_states: Dict[int, bool] = {}
        self.door_states: Dict[int, bool] = {}
        self.entity_positions: Dict[str, List[Dict[str, Any]]] = {}
        self.ignore_doors_mode: bool = False  # Global flag to ignore door states
        
    def initialize_for_level(self, level_data):
        """
        Initialize validator for the current level with entity awareness.
        
        Args:
            level_data: Level data containing tiles and entities
        """
        # Initialize base collision detection
        super().initialize_for_level(level_data.tiles)
        
        # Initialize hazard system
        self.hazard_system = HazardClassificationSystem()
        self.hazard_extension = ReachabilityHazardExtension(self.hazard_system, debug=self.debug)
        self.hazard_extension.initialize_for_reachability(level_data)
        
        # Initialize subgoal planner
        self.subgoal_planner = SubgoalPlanner(debug=self.debug)
        self.subgoal_planner.initialize(level_data.entities)
        
        # Parse entities and their states
        self._parse_entities(level_data.entities)
        
        # Build switch-door relationships
        self._build_switch_door_relationships(level_data.entities)
        
        if self.debug:
            print(f"DEBUG: EntityAwareValidator initialized with {len(self.switch_states)} switches, "
                  f"{len(self.door_states)} doors, {len(self.entity_positions)} entity types")
    
    def _parse_entities(self, entities: List[Dict[str, Any]]):
        """Parse entities and extract relevant state information."""
        self.switch_states.clear()
        self.door_states.clear()
        self.entity_positions.clear()
        
        for entity in entities:
            entity_type = entity.get('type')
            entity_id = entity.get('entity_id', entity.get('id', 0))
            
            # Track switch states
            if entity_type == EntityType.EXIT_SWITCH:
                self.switch_states[entity_id] = entity.get('switch_state', False)
            elif entity_type == EntityType.LOCKED_DOOR and not entity.get('is_door_part', False):
                # Locked door switches (non-door parts) can be activated
                self.switch_states[entity_id] = False  # Start inactive
                
            # Track door states (doors can be opened by switches)
            elif entity_type in [EntityType.REGULAR_DOOR, EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR]:
                # Only track actual door parts, not switch parts
                if entity_type == EntityType.LOCKED_DOOR and entity.get('is_door_part', False):
                    # Locked door parts start closed and are opened by their switch
                    self.door_states[entity_id] = entity.get('closed', True) == False  # Open if not closed
                elif entity_type == EntityType.TRAP_DOOR and entity.get('is_door_part', False):
                    # Trap door parts start open and can be closed by their switch
                    self.door_states[entity_id] = entity.get('closed', False) == False  # Open if not closed
                elif entity_type == EntityType.REGULAR_DOOR:
                    # Regular doors
                    switch_id = entity.get('switch_id', entity_id)
                    self.door_states[entity_id] = self.switch_states.get(switch_id, False)
                
            # Group entities by type for spatial queries
            type_name = str(entity_type) if entity_type else 'unknown'
            if type_name not in self.entity_positions:
                self.entity_positions[type_name] = []
            self.entity_positions[type_name].append(entity)
    
    def _build_switch_door_relationships(self, entities: List[Dict[str, Any]]):
        """Build relationships between switches and doors."""
        # Map switch entity IDs to their positions for reachability checking
        self.switch_positions = {}
        self.door_switch_map = {}  # door_id -> switch_id
        
        for entity in entities:
            entity_type = entity.get('type')
            entity_id = entity.get('entity_id', entity.get('id', 0))
            
            if self.debug:
                print(f"DEBUG: Processing entity {entity_id}, type={entity_type}, is_door_part={entity.get('is_door_part', False)}")
            
            if entity_type == EntityType.EXIT_SWITCH:
                self.switch_positions[entity_id] = (entity.get('x', 0), entity.get('y', 0))
                if self.debug:
                    print(f"DEBUG: Added EXIT_SWITCH {entity_id} at ({entity.get('x', 0)}, {entity.get('y', 0)})")
            elif entity_type == EntityType.LOCKED_DOOR and not entity.get('is_door_part', False):
                # Locked door switches (non-door parts) can be activated
                self.switch_positions[entity_id] = (entity.get('x', 0), entity.get('y', 0))
                if self.debug:
                    print(f"DEBUG: Added LOCKED_DOOR switch {entity_id} at ({entity.get('x', 0)}, {entity.get('y', 0)})")
            elif entity_type == EntityType.TRAP_DOOR and not entity.get('is_door_part', False):
                # Trap door switches (non-door parts) can be activated
                self.switch_positions[entity_id] = (entity.get('x', 0), entity.get('y', 0))
                if self.debug:
                    print(f"DEBUG: Added TRAP_DOOR switch {entity_id} at ({entity.get('x', 0)}, {entity.get('y', 0)})")
    
    def is_position_traversable_with_entities(
        self,
        x: float,
        y: float,
        tiles: np.ndarray,
        radius: float,
        switch_states: Optional[Dict[int, bool]] = None,
        ignore_doors: bool = False
    ) -> bool:
        """
        Check if position is traversable considering both tiles and entities.
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels  
            tiles: Tile array
            radius: Ninja radius
            switch_states: Current switch states (if different from default)
            ignore_doors: If True, ignore door states for initial exploration
            
        Returns:
            True if position is traversable
        """
        # First check basic tile traversability
        if not self.is_position_traversable_with_radius(x, y, tiles, radius):
            return False
            
        # Check entity-based obstacles (unless ignoring doors)
        should_ignore = ignore_doors or self.ignore_doors_mode
        if self.debug and should_ignore:
            print(f"DEBUG: Ignoring doors at ({x}, {y}) - ignore_doors={ignore_doors}, ignore_doors_mode={self.ignore_doors_mode}")
        
        if not should_ignore and not self._is_position_clear_of_entities(x, y, radius, switch_states):
            return False
            
        # Check dynamic hazards (drones, etc.)
        if self.hazard_extension and not self.hazard_extension.is_position_safe_for_reachability((x, y)):
            return False
            
        return True
    
    def _is_position_clear_of_entities(
        self,
        x: float,
        y: float,
        radius: float,
        switch_states: Optional[Dict[int, bool]] = None
    ) -> bool:
        """Check if position is clear of entity-based obstacles."""
        if switch_states is None:
            switch_states = self.switch_states
            
        # Check doors - closed doors block movement
        for door_id, default_state in self.door_states.items():
            # Use provided switch state or default
            switch_id = door_id  # Assume door ID matches switch ID for simplicity
            is_open = switch_states.get(switch_id, default_state)
            
            if not is_open:
                # Find door entity and check collision
                door_entity = self._find_entity_by_id(door_id, EntityType.REGULAR_DOOR)
                if door_entity and self._position_intersects_entity(x, y, radius, door_entity):
                    if self.debug:
                        print(f"DEBUG: Position ({x}, {y}) blocked by closed door {door_id}")
                    return False
        
        # Check other blocking entities (could be expanded)
        # For now, focus on doors as the main dynamic obstacle
        
        return True
    
    def _find_entity_by_id(self, entity_id: int, entity_type: EntityType) -> Optional[Dict[str, Any]]:
        """Find entity by ID and type."""
        type_name = str(entity_type)
        entities = self.entity_positions.get(type_name, [])
        
        for entity in entities:
            if entity.get('id') == entity_id:
                return entity
        return None
    
    def _position_intersects_entity(
        self,
        x: float,
        y: float,
        radius: float,
        entity: Dict[str, Any]
    ) -> bool:
        """Check if position intersects with an entity."""
        entity_x = entity.get('x', 0)
        entity_y = entity.get('y', 0)
        entity_width = entity.get('width', TILE_PIXEL_SIZE)
        entity_height = entity.get('height', TILE_PIXEL_SIZE)
        
        # Simple rectangular collision detection
        # Check if ninja circle intersects with entity rectangle
        closest_x = max(entity_x, min(x, entity_x + entity_width))
        closest_y = max(entity_y, min(y, entity_y + entity_height))
        
        distance_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
        return distance_sq <= radius ** 2
    
    def can_traverse_path_with_entities(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        switch_states: Optional[Dict[int, bool]] = None
    ) -> bool:
        """
        Check if a path between two positions is traversable considering entities.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            switch_states: Current switch states
            
        Returns:
            True if path is traversable
        """
        # Sample points along the path
        num_samples = max(5, int(math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2) / 10))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            sample_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            if not self._is_position_clear_of_entities(sample_x, sample_y, 10.0, switch_states):
                return False
                
        # Check hazard-based path safety
        if self.hazard_extension:
            return self.hazard_extension.can_traverse_between_positions_safely(start_pos, end_pos)
            
        return True
    
    def find_reachable_switch_states(
        self,
        ninja_pos: Tuple[float, float],
        current_reachable_positions: Set[Tuple[int, int]]
    ) -> Dict[int, bool]:
        """
        Find which switches can be activated from current reachable positions.
        
        Args:
            ninja_pos: Current ninja position
            current_reachable_positions: Set of currently reachable (sub_row, sub_col) positions
            
        Returns:
            Dictionary of switch states that can be achieved
        """
        achievable_states = self.switch_states.copy()
        
        # Check each switch to see if it's reachable
        switch_entities = self.entity_positions.get(str(EntityType.EXIT_SWITCH), [])
        
        for switch_entity in switch_entities:
            switch_id = switch_entity.get('id', 0)
            switch_x = switch_entity.get('x', 0)
            switch_y = switch_entity.get('y', 0)
            
            # Convert switch position to sub-grid coordinates
            switch_sub_row = int(switch_y // 6)  # 6px subcell size
            switch_sub_col = int(switch_x // 6)
            
            # Check if switch position is reachable
            if (switch_sub_row, switch_sub_col) in current_reachable_positions:
                # Switch can be activated
                achievable_states[switch_id] = True
                if self.debug:
                    print(f"DEBUG: Switch {switch_id} at ({switch_x}, {switch_y}) is reachable")
        
        return achievable_states
    
    def get_precision_traversable_positions(
        self,
        center_x: float,
        center_y: float,
        search_radius: float,
        tiles: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Find precisely traversable positions around a center point for misaligned paths.
        
        This method uses sub-pixel sampling to find traversable positions even when
        the standard grid-based approach fails due to tile misalignment.
        
        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate  
            search_radius: Radius to search within
            tiles: Tile array
            
        Returns:
            List of precisely traversable (x, y) positions
        """
        traversable_positions = []
        
        # Sample at sub-pixel resolution
        sample_step = 3.0  # 3-pixel steps for precision
        samples_per_axis = int(search_radius * 2 / sample_step) + 1
        
        for i in range(samples_per_axis):
            for j in range(samples_per_axis):
                offset_x = (i - samples_per_axis // 2) * sample_step
                offset_y = (j - samples_per_axis // 2) * sample_step
                
                test_x = center_x + offset_x
                test_y = center_y + offset_y
                
                # Skip if outside search radius
                if offset_x**2 + offset_y**2 > search_radius**2:
                    continue
                
                # Test traversability with high precision
                if self.is_position_traversable_with_entities(test_x, test_y, tiles, 10.0):
                    traversable_positions.append((test_x, test_y))
        
        return traversable_positions
    
    def update_switch_states(self, new_switch_states: Dict[int, bool]):
        """Update switch states and recalculate dependent door states."""
        self.switch_states.update(new_switch_states)
        
        # Update door states based on new switch states
        # For locked doors, the switch and door have the same entity_id but different is_door_part values
        for door_id in self.door_states:
            # For locked doors, the switch has the same entity_id as the door
            if door_id in self.switch_states:
                self.door_states[door_id] = self.switch_states[door_id]
        
        # Update hazard extension
        if self.hazard_extension:
            self.hazard_extension.update_switch_states(new_switch_states)
    
    def find_reachable_switches(self, reachable_positions: Set[Tuple[int, int]]) -> Dict[str, bool]:
        """Find switches that are reachable from current positions."""
        newly_reachable_switches = {}
        
        if self.debug:
            print(f"DEBUG: Checking {len(self.switch_positions)} switches against {len(reachable_positions)} reachable positions")
        
        for switch_id, (switch_x, switch_y) in self.switch_positions.items():
            # Convert switch position to tile coordinates
            switch_tile_x = int(switch_x // TILE_PIXEL_SIZE)
            switch_tile_y = int(switch_y // TILE_PIXEL_SIZE)
            
            if self.debug:
                print(f"DEBUG: Switch {switch_id} at pixel ({switch_x}, {switch_y}) -> tile ({switch_tile_x}, {switch_tile_y})")
            
            # Check if switch tile is reachable
            if (switch_tile_x, switch_tile_y) in reachable_positions:
                newly_reachable_switches[switch_id] = True
                if self.debug:
                    print(f"DEBUG: Switch {switch_id} at ({switch_x}, {switch_y}) is reachable")
            else:
                if self.debug:
                    print(f"DEBUG: Switch {switch_id} at tile ({switch_tile_x}, {switch_tile_y}) is NOT reachable")
        
        return newly_reachable_switches
    
    def set_ignore_doors_mode(self, ignore: bool):
        """Enable or disable ignore doors mode for initial exploration."""
        self.ignore_doors_mode = ignore
        if self.debug:
            print(f"DEBUG: Set ignore_doors_mode = {ignore}")
        
        # Clear any cached results that might depend on door states
        if hasattr(self, 'tile_traversability_cache'):
            self.tile_traversability_cache.clear()
        if hasattr(self, 'region_traversability_cache'):
            self.region_traversability_cache.clear()
    
    def plan_subgoals(self, reachable_positions: Set[Tuple[int, int]], 
                     exit_positions: List[Tuple[float, float]]) -> List[str]:
        """
        Plan subgoals needed to complete the level.
        
        Args:
            reachable_positions: Set of (tile_x, tile_y) positions currently reachable
            exit_positions: List of (x, y) exit positions to reach
            
        Returns:
            List of subgoal descriptions
        """
        if not hasattr(self, 'subgoal_planner'):
            return []
            
        subgoals = self.subgoal_planner.plan_subgoals(reachable_positions, exit_positions)
        
        # Convert subgoals to string descriptions
        subgoal_descriptions = []
        for subgoal in subgoals:
            if subgoal.goal_type == 'activate_switch':
                subgoal_descriptions.append(f"activate_switch_{subgoal.target_id}")
            elif subgoal.goal_type == 'open_door':
                subgoal_descriptions.append(f"open_door_{subgoal.target_id}")
            elif subgoal.goal_type == 'reach_exit':
                subgoal_descriptions.append("reach_exit")
                
        if self.debug and subgoal_descriptions:
            print(f"DEBUG: Generated subgoals: {subgoal_descriptions}")
            
        return subgoal_descriptions

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about entity states."""
        info = {
            "switch_states": self.switch_states,
            "door_states": self.door_states,
            "entity_counts": {k: len(v) for k, v in self.entity_positions.items()}
        }
        
        if self.hazard_extension:
            info["hazard_info"] = self.hazard_extension.get_debug_info()
            
        return info