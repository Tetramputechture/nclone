"""
Hazard-aware traversability system for graph construction.

This module implements comprehensive hazard detection and classification
for static and dynamic hazards that can block player movement, including
toggle mines, thwumps, shove thwumps, one-way platforms, and drones.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import IntEnum

from ..constants.entity_types import EntityType
from ..constants.physics_constants import NINJA_RADIUS, TILE_PIXEL_SIZE


class HazardType(IntEnum):
    """Types of hazards that can block movement."""
    STATIC_BLOCKING = 0      # Permanent blocks (active toggle mines)
    DIRECTIONAL_BLOCKING = 1 # Direction-specific blocks (one-way platforms)
    DYNAMIC_THREAT = 2       # Moving threats (drones)
    ACTIVATION_TRIGGER = 3   # Triggered hazards (thwumps, shove thwumps)


class HazardState(IntEnum):
    """States of dynamic hazards."""
    INACTIVE = 0    # Safe, not threatening
    ACTIVE = 1      # Currently dangerous
    CHARGING = 2    # Building up to dangerous state
    RETREATING = 3  # Moving away from dangerous state


@dataclass
class HazardInfo:
    """Information about a hazard entity."""
    entity_id: int
    hazard_type: HazardType
    entity_type: int
    position: Tuple[float, float]
    state: HazardState
    blocked_directions: Set[int]  # Bitmask of blocked directions (0-7)
    danger_radius: float
    activation_range: float
    # Static hazard properties
    blocked_cells: Set[Tuple[int, int]]  # Sub-grid cells permanently blocked
    # Dynamic hazard properties
    velocity: Tuple[float, float]
    predicted_positions: List[Tuple[float, float]]
    # Directional hazard properties
    orientation: int  # For one-way platforms (0-7)
    # Activation hazard properties
    charge_direction: Tuple[float, float]  # For thwumps
    core_position: Tuple[float, float]     # For shove thwumps
    launch_trajectories: List[Tuple[float, float]]  # Possible launch paths


@dataclass
class EdgeHazardMeta:
    """Hazard metadata for graph edges."""
    last_update_frame: int
    hazard_risk: float  # 0.0 = safe, 1.0 = blocked
    affecting_hazards: List[int]  # Entity IDs of hazards affecting this edge
    directional_safety: int  # Bitmask of safe approach directions
    time_to_hazard: float  # Time until path becomes dangerous (-1 if static)


class HazardClassificationSystem:
    """
    Classifies and tracks hazards for traversability analysis.
    
    This system maintains caches of static hazards and tracks dynamic hazards
    within a radius of the ninja for efficient real-time path updates.
    """
    
    # Constants for hazard detection
    HAZARD_UPDATE_RADIUS = 150.0  # Pixels from ninja position
    THWUMP_ACTIVATION_RANGE = 38.0  # Line-of-sight activation range
    SHOVE_THWUMP_CORE_RADIUS = 8.0  # Deadly core radius
    ONE_WAY_PLATFORM_THICKNESS = 12.0  # Platform collision thickness
    DRONE_PREDICTION_TIME = 60.0  # Frames to predict drone movement
    
    def __init__(self):
        """Initialize hazard classification system."""
        # Static hazard cache (never changes during level)
        self._static_hazard_cache: Dict[Tuple[int, int], HazardInfo] = {}
        # Dynamic hazard tracker (updated each frame)
        self._dynamic_hazards: Dict[int, HazardInfo] = {}
        # Edge hazard metadata
        self._edge_hazard_meta: Dict[int, EdgeHazardMeta] = {}
        self._current_level_id = None
        self._current_frame = 0
    
    def build_static_hazard_cache(
        self,
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any]
    ) -> Dict[Tuple[int, int], HazardInfo]:
        """
        Build cache of static hazards that create permanent path blocks.
        
        Args:
            entities: List of entity dictionaries
            level_data: Level data and structure
            
        Returns:
            Dictionary mapping sub-grid cells to hazard information
        """
        level_id = level_data.get('level_id', id(level_data))
        if self._current_level_id == level_id and self._static_hazard_cache:
            return self._static_hazard_cache
        
        self._current_level_id = level_id
        self._static_hazard_cache = {}
        
        for entity in entities:
            entity_type = entity.get('type', 0)
            entity_id = entity.get('id', -1)
            entity_x = entity.get('x', 0.0)
            entity_y = entity.get('y', 0.0)
            entity_state = entity.get('state', 0)
            entity_active = entity.get('active', True)
            
            hazard_info = None
            
            if entity_type == EntityType.TOGGLE_MINE:
                hazard_info = self._classify_toggle_mine(entity)
            elif entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                hazard_info = self._classify_toggle_mine_toggled(entity)
            elif entity_type == EntityType.THWUMP:
                hazard_info = self._classify_thwump_static(entity)
            elif entity_type == EntityType.SHWUMP:  # Shove Thwump
                hazard_info = self._classify_shove_thwump_static(entity)
            elif entity_type == EntityType.ONE_WAY:
                hazard_info = self._classify_one_way_platform(entity)
            
            if hazard_info:
                # Map hazard to affected sub-grid cells
                for cell in hazard_info.blocked_cells:
                    self._static_hazard_cache[cell] = hazard_info
        
        return self._static_hazard_cache
    
    def get_dynamic_hazards_in_range(
        self,
        entities: List[Dict[str, Any]],
        ninja_pos: Tuple[float, float],
        radius: float = None
    ) -> List[HazardInfo]:
        """
        Get dynamic hazards within update radius of ninja position.
        
        Args:
            entities: List of entity dictionaries
            ninja_pos: Current ninja position (x, y)
            radius: Update radius (default: HAZARD_UPDATE_RADIUS)
            
        Returns:
            List of dynamic hazard information
        """
        if radius is None:
            radius = self.HAZARD_UPDATE_RADIUS
        
        ninja_x, ninja_y = ninja_pos
        dynamic_hazards = []
        
        for entity in entities:
            entity_type = entity.get('type', 0)
            entity_id = entity.get('id', -1)
            entity_x = entity.get('x', 0.0)
            entity_y = entity.get('y', 0.0)
            
            # Check if entity is within update radius
            distance = math.sqrt((entity_x - ninja_x)**2 + (entity_y - ninja_y)**2)
            if distance > radius:
                continue
            
            hazard_info = None
            
            if entity_type == EntityType.DRONE_ZAP:
                hazard_info = self._classify_drone(entity)
            elif entity_type == EntityType.MINI_DRONE:
                hazard_info = self._classify_mini_drone(entity)
            elif entity_type == EntityType.DEATH_BALL:
                hazard_info = self._classify_death_ball(entity)
            elif entity_type == EntityType.THWUMP:
                # Thwumps can be dynamic when charging
                hazard_info = self._classify_thwump_dynamic(entity)
            elif entity_type == EntityType.SHWUMP:
                # Shove thwumps can be dynamic when triggered
                hazard_info = self._classify_shove_thwump_dynamic(entity)
            
            if hazard_info:
                dynamic_hazards.append(hazard_info)
                self._dynamic_hazards[entity_id] = hazard_info
        
        return dynamic_hazards
    
    def check_path_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo
    ) -> bool:
        """
        Check if a movement path intersects with a hazard.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            hazard_info: Hazard information
            
        Returns:
            True if path intersects hazard, False if safe
        """
        if hazard_info.hazard_type == HazardType.STATIC_BLOCKING:
            return self._check_static_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info)
        elif hazard_info.hazard_type == HazardType.DIRECTIONAL_BLOCKING:
            return self._check_directional_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info)
        elif hazard_info.hazard_type == HazardType.DYNAMIC_THREAT:
            return self._check_dynamic_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info)
        elif hazard_info.hazard_type == HazardType.ACTIVATION_TRIGGER:
            return self._check_activation_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info)
        
        return False
    
    def _classify_toggle_mine(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify toggle mine hazard."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_active = entity.get('active', True)
        
        if not entity_active:
            return None  # Inactive toggle mines are safe
        
        # Active toggle mines block movement
        blocked_cells = set()
        center_sub_x = int(entity_x // 12)  # Sub-cell size
        center_sub_y = int(entity_y // 12)
        
        # Block 3x3 area around toggle mine
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                blocked_cells.add((center_sub_x + dx, center_sub_y + dy))
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.STATIC_BLOCKING,
            entity_type=EntityType.TOGGLE_MINE,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=set(range(8)),  # All directions blocked
            danger_radius=18.0,  # 1.5 tiles
            activation_range=0.0,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=0,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_toggle_mine_toggled(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify toggled toggle mine hazard (inactive state)."""
        # Toggled toggle mines are safe
        return None
    
    def _classify_thwump_static(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify thwump as static hazard (immobile state)."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_state = entity.get('state', 0)
        orientation = entity.get('orientation', 0)
        
        if entity_state != 0:  # Not in immobile state
            return None
        
        # Thwumps in immobile state create static blocks in their charge direction
        charge_dx, charge_dy = self._get_orientation_vector(orientation)
        blocked_cells = set()
        
        # Block cells in charge direction (up to 5 tiles)
        for i in range(1, 6):
            block_x = entity_x + i * TILE_PIXEL_SIZE * charge_dx
            block_y = entity_y + i * TILE_PIXEL_SIZE * charge_dy
            sub_x = int(block_x // 12)
            sub_y = int(block_y // 12)
            blocked_cells.add((sub_x, sub_y))
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.ACTIVATION_TRIGGER,
            entity_type=EntityType.THWUMP,
            position=(entity_x, entity_y),
            state=HazardState.INACTIVE,
            blocked_directions=set(),  # No directions blocked when inactive
            danger_radius=0.0,
            activation_range=self.THWUMP_ACTIVATION_RANGE,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=orientation,
            charge_direction=(charge_dx, charge_dy),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_thwump_dynamic(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify thwump as dynamic hazard (charging/retreating state)."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_state = entity.get('state', 0)
        orientation = entity.get('orientation', 0)
        
        if entity_state == 0:  # Immobile state
            return None
        
        # Determine hazard state
        hazard_state = HazardState.ACTIVE
        if entity_state == 1:  # Charging
            hazard_state = HazardState.CHARGING
        elif entity_state == 2:  # Retreating
            hazard_state = HazardState.RETREATING
        
        charge_dx, charge_dy = self._get_orientation_vector(orientation)
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DYNAMIC_THREAT,
            entity_type=EntityType.THWUMP,
            position=(entity_x, entity_y),
            state=hazard_state,
            blocked_directions=set(range(8)),  # All directions dangerous when active
            danger_radius=12.0,  # Thwump body radius
            activation_range=0.0,
            blocked_cells=set(),
            velocity=(charge_dx * 2.0, charge_dy * 2.0),  # Charging velocity
            predicted_positions=self._predict_thwump_movement(entity_x, entity_y, charge_dx, charge_dy, entity_state),
            orientation=orientation,
            charge_direction=(charge_dx, charge_dy),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_shove_thwump_static(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify shove thwump as static hazard (untriggered state)."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_state = entity.get('state', 0)
        
        if entity_state != 0:  # Not in untriggered state
            return None
        
        # Untriggered shove thwumps are safe to touch from any side
        return None
    
    def _classify_shove_thwump_dynamic(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify shove thwump as dynamic hazard (triggered state)."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_state = entity.get('state', 0)
        
        if entity_state == 0:  # Untriggered state
            return None
        
        # Triggered shove thwumps have deadly core
        blocked_cells = set()
        center_sub_x = int(entity_x // 12)
        center_sub_y = int(entity_y // 12)
        blocked_cells.add((center_sub_x, center_sub_y))
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.STATIC_BLOCKING,
            entity_type=EntityType.SHWUMP,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=set(range(8)),  # All directions dangerous
            danger_radius=self.SHOVE_THWUMP_CORE_RADIUS,
            activation_range=0.0,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=0,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_one_way_platform(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify one-way platform hazard."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        orientation = entity.get('orientation', 0)
        
        # One-way platforms block movement from specific direction
        blocked_direction = orientation
        blocked_directions = {blocked_direction}
        
        # Calculate blocked cells based on platform orientation
        blocked_cells = set()
        platform_dx, platform_dy = self._get_orientation_vector(orientation)
        
        # Block cells in the platform area
        for i in range(-1, 2):  # 3-cell wide platform
            for j in range(-1, 2):
                block_x = entity_x + i * 12 * platform_dy  # Perpendicular to orientation
                block_y = entity_y + j * 12 * platform_dx
                sub_x = int(block_x // 12)
                sub_y = int(block_y // 12)
                blocked_cells.add((sub_x, sub_y))
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DIRECTIONAL_BLOCKING,
            entity_type=EntityType.ONE_WAY,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=blocked_directions,
            danger_radius=self.ONE_WAY_PLATFORM_THICKNESS,
            activation_range=0.0,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=orientation,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_drone(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify drone as dynamic hazard."""
        entity_id = entity.get('id', -1)
        entity_x = entity.get('x', 0.0)
        entity_y = entity.get('y', 0.0)
        entity_vx = entity.get('vx', 0.0)
        entity_vy = entity.get('vy', 0.0)
        
        # Predict drone movement
        predicted_positions = self._predict_drone_movement(
            entity_x, entity_y, entity_vx, entity_vy, self.DRONE_PREDICTION_TIME
        )
        
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DYNAMIC_THREAT,
            entity_type=EntityType.DRONE_ZAP,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=set(range(8)),  # All directions dangerous
            danger_radius=12.0,  # Drone collision radius
            activation_range=0.0,
            blocked_cells=set(),
            velocity=(entity_vx, entity_vy),
            predicted_positions=predicted_positions,
            orientation=0,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[]
        )
    
    def _classify_mini_drone(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify mini drone as dynamic hazard."""
        # Similar to regular drone but smaller
        return self._classify_drone(entity)  # Reuse drone logic for now
    
    def _classify_death_ball(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify death ball as dynamic hazard."""
        # Similar to drone but with different movement pattern
        return self._classify_drone(entity)  # Reuse drone logic for now
    
    def _get_orientation_vector(self, orientation: int) -> Tuple[float, float]:
        """Convert orientation (0-7) to unit vector."""
        # 8-directional orientation: 0=right, 1=down-right, 2=down, etc.
        angle = orientation * math.pi / 4.0
        return (math.cos(angle), math.sin(angle))
    
    def _predict_thwump_movement(
        self,
        start_x: float,
        start_y: float,
        charge_dx: float,
        charge_dy: float,
        state: int
    ) -> List[Tuple[float, float]]:
        """Predict thwump movement based on current state."""
        positions = []
        
        if state == 1:  # Charging
            # Predict charging movement (5 tiles in charge direction)
            for i in range(1, 6):
                pred_x = start_x + i * TILE_PIXEL_SIZE * charge_dx
                pred_y = start_y + i * TILE_PIXEL_SIZE * charge_dy
                positions.append((pred_x, pred_y))
        elif state == 2:  # Retreating
            # Predict retreating movement (back to original position)
            for i in range(4, 0, -1):
                pred_x = start_x + i * TILE_PIXEL_SIZE * charge_dx
                pred_y = start_y + i * TILE_PIXEL_SIZE * charge_dy
                positions.append((pred_x, pred_y))
        
        return positions
    
    def _predict_drone_movement(
        self,
        start_x: float,
        start_y: float,
        vx: float,
        vy: float,
        prediction_time: float
    ) -> List[Tuple[float, float]]:
        """Predict drone movement over time."""
        positions = []
        
        # Simple linear prediction (could be enhanced with patrol route detection)
        for t in range(1, int(prediction_time) + 1, 5):  # Every 5 frames
            pred_x = start_x + vx * t
            pred_y = start_y + vy * t
            positions.append((pred_x, pred_y))
        
        return positions
    
    def _check_static_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo
    ) -> bool:
        """Check intersection with static hazard."""
        hazard_x, hazard_y = hazard_info.position
        
        # Check if path passes through hazard danger radius
        return self._line_intersects_circle(
            src_x, src_y, tgt_x, tgt_y,
            hazard_x, hazard_y, hazard_info.danger_radius + NINJA_RADIUS
        )
    
    def _check_directional_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo
    ) -> bool:
        """Check intersection with directional hazard (one-way platform)."""
        # Calculate approach direction
        dx = tgt_x - src_x
        dy = tgt_y - src_y
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return False  # No movement
        
        # Convert movement direction to orientation (0-7)
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        approach_orientation = int((angle + math.pi/8) / (math.pi/4)) % 8
        
        # Check if approaching from blocked direction
        if approach_orientation in hazard_info.blocked_directions:
            return self._check_static_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info)
        
        return False  # Safe approach direction
    
    def _check_dynamic_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo
    ) -> bool:
        """Check intersection with dynamic hazard."""
        # Check current position
        if self._check_static_hazard_intersection(src_x, src_y, tgt_x, tgt_y, hazard_info):
            return True
        
        # Check predicted positions
        for pred_x, pred_y in hazard_info.predicted_positions:
            if self._line_intersects_circle(
                src_x, src_y, tgt_x, tgt_y,
                pred_x, pred_y, hazard_info.danger_radius + NINJA_RADIUS
            ):
                return True
        
        return False
    
    def _check_activation_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo
    ) -> bool:
        """Check intersection with activation hazard (thwump activation zone)."""
        hazard_x, hazard_y = hazard_info.position
        
        # Check if path enters activation range
        if self._line_intersects_circle(
            src_x, src_y, tgt_x, tgt_y,
            hazard_x, hazard_y, hazard_info.activation_range + NINJA_RADIUS
        ):
            # If activated, check charge direction for collision
            charge_dx, charge_dy = hazard_info.charge_direction
            
            # Check if path crosses charge lane
            for i in range(1, 6):  # 5 tiles in charge direction
                charge_x = hazard_x + i * TILE_PIXEL_SIZE * charge_dx
                charge_y = hazard_y + i * TILE_PIXEL_SIZE * charge_dy
                
                if self._line_intersects_circle(
                    src_x, src_y, tgt_x, tgt_y,
                    charge_x, charge_y, 12.0 + NINJA_RADIUS  # Thwump body radius
                ):
                    return True
        
        return False
    
    def _line_intersects_circle(
        self,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        circle_x: float,
        circle_y: float,
        radius: float
    ) -> bool:
        """Check if line segment intersects circle."""
        # Vector from line start to circle center
        to_circle_x = circle_x - line_x1
        to_circle_y = circle_y - line_y1
        
        # Line direction vector
        line_dx = line_x2 - line_x1
        line_dy = line_y2 - line_y1
        line_length_sq = line_dx*line_dx + line_dy*line_dy
        
        if line_length_sq < 1e-6:  # Degenerate line (point)
            dist_sq = to_circle_x*to_circle_x + to_circle_y*to_circle_y
            return dist_sq <= radius*radius
        
        # Project circle center onto line
        t = (to_circle_x*line_dx + to_circle_y*line_dy) / line_length_sq
        t = max(0.0, min(1.0, t))  # Clamp to line segment
        
        # Closest point on line to circle center
        closest_x = line_x1 + t * line_dx
        closest_y = line_y1 + t * line_dy
        
        # Distance from circle center to closest point
        dist_sq = (circle_x - closest_x)**2 + (circle_y - closest_y)**2
        
        return dist_sq <= radius*radius