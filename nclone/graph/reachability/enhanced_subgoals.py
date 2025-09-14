"""
Enhanced subgoal identification for hierarchical RL.

This module provides comprehensive subgoal identification including strategic
waypoints, hazard navigation points, and hierarchical planning support.
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

from ...constants.entity_types import EntityType
from ...constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS


class SubgoalType(Enum):
    """Types of subgoals for hierarchical planning."""
    
    # Critical path subgoals
    EXIT_SWITCH = "exit_switch"
    EXIT_DOOR = "exit_door"
    
    # Door and switch subgoals
    DOOR_SWITCH = "door_switch"
    LOCKED_DOOR = "locked_door"
    REGULAR_DOOR = "regular_door"
    
    # Collection subgoals
    GOLD = "gold"
    KEY = "key"
    
    # Hazard navigation subgoals
    HAZARD_WAYPOINT = "hazard_waypoint"
    SAFE_ZONE = "safe_zone"
    
    # Strategic waypoints
    JUNCTION = "junction"
    BOTTLENECK = "bottleneck"
    EXPLORATION_FRONTIER = "exploration_frontier"
    
    # Advanced movement subgoals
    WALL_JUMP_POINT = "wall_jump_point"
    BOUNCE_BLOCK_TARGET = "bounce_block_target"


@dataclass
class Subgoal:
    """Represents a strategic subgoal for hierarchical planning."""
    
    position: Tuple[int, int]  # Sub-grid coordinates
    subgoal_type: SubgoalType
    priority: int
    entity_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[int]] = None  # Entity IDs this subgoal depends on
    unlocks: Optional[List[int]] = None  # Entity IDs this subgoal unlocks


class EnhancedSubgoalIdentifier:
    """
    Enhanced subgoal identification for hierarchical RL.
    
    Features:
    - Strategic subgoal identification
    - Dependency analysis
    - Priority assignment
    - Hazard navigation waypoints
    - Exploration frontier detection
    - Junction and bottleneck identification
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize enhanced subgoal identifier.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.subgoals: List[Subgoal] = []
        self.entity_dependencies: Dict[int, List[int]] = {}
        
    def identify_subgoals(
        self, 
        level_data, 
        reachability_state, 
        entity_handler=None,
        position_validator=None
    ) -> List[Subgoal]:
        """
        Identify comprehensive subgoals for hierarchical planning.
        
        Args:
            level_data: Level data containing tiles and entities
            reachability_state: Current reachability state
            entity_handler: Entity handler for hazard information
            position_validator: Position validator for coordinate conversion
            
        Returns:
            List of identified subgoals
        """
        self.subgoals.clear()
        self.entity_dependencies.clear()
        
        # Identify entity-based subgoals
        self._identify_entity_subgoals(level_data, reachability_state, position_validator)
        
        # Identify strategic waypoints
        self._identify_strategic_waypoints(level_data, reachability_state, position_validator)
        
        # Identify hazard navigation points
        if entity_handler:
            self._identify_hazard_waypoints(level_data, reachability_state, entity_handler, position_validator)
        
        # Identify exploration frontiers
        self._identify_exploration_frontiers(level_data, reachability_state, position_validator)
        
        # Analyze dependencies and assign priorities
        self._analyze_dependencies()
        self._assign_priorities()
        
        # Sort by priority
        self.subgoals.sort(key=lambda s: s.priority)
        
        if self.debug:
            print(f"DEBUG: Identified {len(self.subgoals)} enhanced subgoals")
            for subgoal in self.subgoals[:5]:  # Show top 5
                print(f"  {subgoal.subgoal_type.value} at {subgoal.position} (priority: {subgoal.priority})")
        
        return self.subgoals.copy()
    
    def _identify_entity_subgoals(self, level_data, reachability_state, position_validator):
        """Identify subgoals based on entities."""
        for entity in level_data.entities:
            entity_type = entity.get('type')
            entity_id = entity.get('id', 0)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            
            if position_validator:
                sub_row, sub_col = position_validator.convert_pixel_to_sub_grid(entity_x, entity_y)
            else:
                sub_row = int(entity_y // TILE_PIXEL_SIZE)
                sub_col = int(entity_x // TILE_PIXEL_SIZE)
            
            # Only consider reachable positions
            if (sub_row, sub_col) not in reachability_state.reachable_positions:
                continue
            
            subgoal = self._create_entity_subgoal(entity, sub_row, sub_col)
            if subgoal:
                self.subgoals.append(subgoal)
    
    def _create_entity_subgoal(self, entity: Dict[str, Any], sub_row: int, sub_col: int) -> Optional[Subgoal]:
        """Create subgoal from entity data."""
        entity_type = entity.get('type')
        entity_id = entity.get('id', 0)
        
        subgoal_mapping = {
            EntityType.EXIT_SWITCH: (SubgoalType.EXIT_SWITCH, 1),
            EntityType.EXIT_DOOR: (SubgoalType.EXIT_DOOR, 2),
            EntityType.REGULAR_DOOR: (SubgoalType.REGULAR_DOOR, 5),
            EntityType.LOCKED_DOOR: (SubgoalType.LOCKED_DOOR, 4),
            EntityType.GOLD: (SubgoalType.GOLD, 6),
            EntityType.BOUNCE_BLOCK: (SubgoalType.BOUNCE_BLOCK_TARGET, 7),
        }
        
        if entity_type in subgoal_mapping:
            subgoal_type, base_priority = subgoal_mapping[entity_type]
            
            metadata = {
                'entity_data': entity,
                'requires_activation': entity_type in [EntityType.EXIT_SWITCH],
                'blocks_progress': entity_type in [EntityType.REGULAR_DOOR, EntityType.LOCKED_DOOR]
            }
            
            return Subgoal(
                position=(sub_row, sub_col),
                subgoal_type=subgoal_type,
                priority=base_priority,
                entity_id=entity_id,
                metadata=metadata
            )
        
        return None
    
    def _identify_strategic_waypoints(self, level_data, reachability_state, position_validator):
        """Identify strategic waypoints like junctions and bottlenecks."""
        reachable_positions = reachability_state.reachable_positions
        
        # Find junctions (positions with 3+ reachable neighbors)
        for row, col in reachable_positions:
            neighbor_count = self._count_reachable_neighbors(row, col, reachable_positions)
            
            if neighbor_count >= 3:
                # This is a junction
                subgoal = Subgoal(
                    position=(row, col),
                    subgoal_type=SubgoalType.JUNCTION,
                    priority=8,
                    metadata={'neighbor_count': neighbor_count}
                )
                self.subgoals.append(subgoal)
            
            elif neighbor_count == 1:
                # This might be a bottleneck or dead end
                # Check if it's strategically important
                if self._is_strategic_bottleneck(row, col, reachable_positions):
                    subgoal = Subgoal(
                        position=(row, col),
                        subgoal_type=SubgoalType.BOTTLENECK,
                        priority=9,
                        metadata={'bottleneck_type': 'strategic'}
                    )
                    self.subgoals.append(subgoal)
    
    def _identify_hazard_waypoints(self, level_data, reachability_state, entity_handler, position_validator):
        """Identify waypoints for navigating around hazards."""
        hazard_zones = entity_handler.hazard_zones
        reachable_positions = reachability_state.reachable_positions
        
        for hazard_x, hazard_y, hazard_radius in hazard_zones:
            # Find safe positions around hazards
            hazard_row = int(hazard_y // TILE_PIXEL_SIZE)
            hazard_col = int(hazard_x // TILE_PIXEL_SIZE)
            
            # Look for safe waypoints around the hazard
            search_radius = int(hazard_radius // TILE_PIXEL_SIZE) + 2
            
            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    waypoint_row = hazard_row + dr
                    waypoint_col = hazard_col + dc
                    
                    if (waypoint_row, waypoint_col) in reachable_positions:
                        # Check if this position is safe but close to hazard
                        pixel_x, pixel_y = position_validator.convert_sub_grid_to_pixel(
                            waypoint_row, waypoint_col
                        )
                        
                        if entity_handler.is_position_safe((pixel_x, pixel_y)):
                            distance_to_hazard = math.sqrt(
                                (pixel_x - hazard_x) ** 2 + (pixel_y - hazard_y) ** 2
                            )
                            
                            # If it's close to hazard but safe, it's a good waypoint
                            if hazard_radius < distance_to_hazard < hazard_radius * 1.5:
                                subgoal = Subgoal(
                                    position=(waypoint_row, waypoint_col),
                                    subgoal_type=SubgoalType.HAZARD_WAYPOINT,
                                    priority=10,
                                    metadata={
                                        'hazard_position': (hazard_x, hazard_y),
                                        'distance_to_hazard': distance_to_hazard
                                    }
                                )
                                self.subgoals.append(subgoal)
    
    def _identify_exploration_frontiers(self, level_data, reachability_state, position_validator):
        """Identify positions at the boundary of reachable areas."""
        reachable_positions = reachability_state.reachable_positions
        height, width = level_data.tiles.shape
        
        for row, col in reachable_positions:
            # Check if this position is at the frontier
            is_frontier = False
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adj_row, adj_col = row + dr, col + dc
                
                if (0 <= adj_row < height and 0 <= adj_col < width and
                    (adj_row, adj_col) not in reachable_positions and
                    level_data.tiles[adj_row, adj_col] == 0):  # Empty but unreachable
                    is_frontier = True
                    break
            
            if is_frontier:
                subgoal = Subgoal(
                    position=(row, col),
                    subgoal_type=SubgoalType.EXPLORATION_FRONTIER,
                    priority=12,
                    metadata={'frontier_type': 'exploration'}
                )
                self.subgoals.append(subgoal)
    
    def _count_reachable_neighbors(self, row: int, col: int, reachable_positions: Set[Tuple[int, int]]) -> int:
        """Count reachable neighbors of a position."""
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (row + dr, col + dc) in reachable_positions:
                count += 1
        return count
    
    def _is_strategic_bottleneck(self, row: int, col: int, reachable_positions: Set[Tuple[int, int]]) -> bool:
        """Check if a position is a strategic bottleneck."""
        # A bottleneck is strategic if removing it would disconnect reachable areas
        # For simplicity, we'll consider positions with exactly one neighbor as potential bottlenecks
        neighbor_count = self._count_reachable_neighbors(row, col, reachable_positions)
        return neighbor_count == 1
    
    def _analyze_dependencies(self):
        """Analyze dependencies between subgoals."""
        # Build dependency graph based on entity relationships
        for subgoal in self.subgoals:
            if subgoal.entity_id is not None:
                entity_type = subgoal.metadata.get('entity_data', {}).get('type')
                
                if entity_type == EntityType.EXIT_DOOR:
                    # Exit door depends on exit switch
                    exit_switches = [s for s in self.subgoals 
                                   if s.subgoal_type == SubgoalType.EXIT_SWITCH]
                    if exit_switches:
                        subgoal.dependencies = [s.entity_id for s in exit_switches]
                
                elif entity_type == EntityType.REGULAR_DOOR:
                    # Regular door might depend on switches (simplified)
                    door_switches = [s for s in self.subgoals 
                                   if s.subgoal_type == SubgoalType.DOOR_SWITCH]
                    if door_switches:
                        subgoal.dependencies = [s.entity_id for s in door_switches]
    
    def _assign_priorities(self):
        """Assign final priorities based on dependencies and strategic value."""
        # Lower numbers = higher priority
        priority_adjustments = {
            SubgoalType.EXIT_SWITCH: -2,  # Highest priority
            SubgoalType.DOOR_SWITCH: -1,
            SubgoalType.HAZARD_WAYPOINT: 1,
            SubgoalType.JUNCTION: 2,
            SubgoalType.EXPLORATION_FRONTIER: 3
        }
        
        for subgoal in self.subgoals:
            adjustment = priority_adjustments.get(subgoal.subgoal_type, 0)
            subgoal.priority += adjustment
            
            # Boost priority if subgoal has no dependencies
            if not subgoal.dependencies:
                subgoal.priority -= 1
    
    def get_subgoals_by_type(self, subgoal_type: SubgoalType) -> List[Subgoal]:
        """Get all subgoals of a specific type."""
        return [s for s in self.subgoals if s.subgoal_type == subgoal_type]
    
    def get_critical_path_subgoals(self) -> List[Subgoal]:
        """Get subgoals that are on the critical path to level completion."""
        critical_types = {SubgoalType.EXIT_SWITCH, SubgoalType.EXIT_DOOR}
        return [s for s in self.subgoals if s.subgoal_type in critical_types]
    
    def get_subgoals_in_radius(self, center_pos: Tuple[int, int], radius: int) -> List[Subgoal]:
        """Get subgoals within a certain radius of a position."""
        center_row, center_col = center_pos
        nearby_subgoals = []
        
        for subgoal in self.subgoals:
            sub_row, sub_col = subgoal.position
            distance = max(abs(sub_row - center_row), abs(sub_col - center_col))
            
            if distance <= radius:
                nearby_subgoals.append(subgoal)
        
        return nearby_subgoals