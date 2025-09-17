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
from ..graph.level_data import LevelData
# Removed legacy precise collision - using simplified collision detection
from ..constants.physics_constants import (
    NINJA_RADIUS,
    TILE_PIXEL_SIZE,
    DRONE_RADIUS,
    MINI_DRONE_RADIUS,
    DRONE_GRID_SIZE,
    MINI_DRONE_GRID_SIZE,
    DRONE_LAUNCH_SPEED,
    HAZARD_UPDATE_RADIUS,
    THWUMP_ACTIVATION_RANGE,
    SHOVE_THWUMP_CORE_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
    TOGGLE_MINE_RADIUS_UNTOGGLED,
    TOGGLE_MINE_RADIUS_TOGGLING,
    THWUMP_DANGER_RADIUS,
    SHOVE_THWUMP_DANGER_RADIUS,
    ONE_WAY_PLATFORM_SEMI_SIDE,
    BOUNCE_BLOCK_SEMI_SIDE,
)
from ..utils.physics_utils import calculate_clearance_directions


class HazardType(IntEnum):
    """Types of hazards that can block movement."""

    STATIC_BLOCKING = 0  # Permanent blocks (active toggle mines)
    DIRECTIONAL_BLOCKING = 1  # Direction-specific blocks (one-way platforms)
    DYNAMIC_THREAT = 2  # Moving threats (drones)
    ACTIVATION_TRIGGER = 3  # Triggered hazards (thwumps, shove thwumps)


class HazardState(IntEnum):
    """States of dynamic hazards."""

    INACTIVE = 0  # Safe, not threatening
    ACTIVE = 1  # Currently dangerous
    CHARGING = 2  # Building up to dangerous state
    RETREATING = 3  # Moving away from dangerous state
    LAUNCHING = 4  # Launching/moving phase (shove thwumps)


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
    core_position: Tuple[float, float]  # For shove thwumps
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

    # Constants for hazard detection (imported from physics_constants)
    # HAZARD_UPDATE_RADIUS, THWUMP_ACTIVATION_RANGE, SHOVE_THWUMP_CORE_RADIUS,
    # ONE_WAY_PLATFORM_THICKNESS, DRONE_PREDICTION_TIME are imported

    # Toggle mine radii by state (using imported constants)
    TOGGLE_MINE_RADII = {
        0: TOGGLE_MINE_RADIUS_TOGGLED,  # toggled
        1: TOGGLE_MINE_RADIUS_UNTOGGLED,  # untoggled
        2: TOGGLE_MINE_RADIUS_TOGGLING,  # toggling
    }

    def __init__(self):
        """Initialize simplified hazard classification system."""
        # Static hazard cache (never changes during level)
        self._static_hazard_cache: Dict[Tuple[int, int], HazardInfo] = {}
        # Dynamic hazard tracker (updated each frame)
        self._dynamic_hazards: Dict[int, HazardInfo] = {}
        # Current level tiles for collision checking
        self._current_tiles: Optional[np.ndarray] = None
        # Edge hazard metadata
        self._edge_hazard_meta: Dict[int, EdgeHazardMeta] = {}
        self._current_level_id = None
        self._current_frame = 0

    def set_tile_data(self, tiles: np.ndarray):
        """Set current level tiles for collision checking."""
        self._current_tiles = tiles

    def build_static_hazard_cache(
        self, level_data: LevelData
    ) -> Dict[Tuple[int, int], HazardInfo]:
        """
        Build cache of static hazards that create permanent path blocks.

        Args:
            entities: List of entity dictionaries
            level_data: Level data and structure

        Returns:
            Dictionary mapping sub-grid cells to hazard information
        """
        level_id = id(level_data)
        if self._current_level_id == level_id and self._static_hazard_cache:
            return self._static_hazard_cache

        self._current_level_id = level_id
        self._static_hazard_cache = {}

        for entity in level_data.entities:
            entity_type = entity.get("type", 0)

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
        radius: float = HAZARD_UPDATE_RADIUS,
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
            radius = HAZARD_UPDATE_RADIUS

        ninja_x, ninja_y = ninja_pos
        dynamic_hazards = []

        for entity in entities:
            entity_type = entity.get("type", 0)
            entity_id = entity.get("id", -1)
            entity_x = entity.get("x", 0.0)
            entity_y = entity.get("y", 0.0)

            # Check if entity is within update radius
            distance = math.sqrt((entity_x - ninja_x) ** 2 + (entity_y - ninja_y) ** 2)
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
        hazard_info: HazardInfo,
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
            return self._check_static_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            )
        elif hazard_info.hazard_type == HazardType.DIRECTIONAL_BLOCKING:
            return self._check_directional_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            )
        elif hazard_info.hazard_type == HazardType.DYNAMIC_THREAT:
            return self._check_dynamic_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            )
        elif hazard_info.hazard_type == HazardType.ACTIVATION_TRIGGER:
            return self._check_activation_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            )

        return False

    def _classify_toggle_mine(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify toggle mine hazard based on state."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)

        # Handle both old 'active' field and new 'state' field for backward compatibility
        entity_state = entity.get("state")
        entity_active = entity.get("active")

        # Determine if toggle mine is deadly and get correct radius
        is_deadly = False
        danger_radius = TOGGLE_MINE_RADIUS_TOGGLED  # Default to toggled state radius

        if entity_state is not None:
            # New state-based logic:
            # 0: Toggled (deadly) - blocks movement
            # 1: Untoggled (safe) - can be touched to toggle
            # 2: Toggling (safe) - in process of being toggled
            is_deadly = entity_state == 0
            danger_radius = self.TOGGLE_MINE_RADII.get(
                entity_state, TOGGLE_MINE_RADIUS_TOGGLED
            )
        elif entity_active is not None:
            # Old active-based logic for backward compatibility
            is_deadly = entity_active
            danger_radius = (
                TOGGLE_MINE_RADIUS_TOGGLED
                if entity_active
                else TOGGLE_MINE_RADIUS_UNTOGGLED
            )

        if is_deadly:
            return self._create_static_blocking_hazard(
                entity_id=entity_id,
                entity_type=EntityType.TOGGLE_MINE,
                position=(entity_x, entity_y),
                danger_radius=danger_radius,
                block_area_size=3,  # 3x3 blocking area
            )
        else:
            return None

    def _classify_toggle_mine_toggled(
        self, entity: Dict[str, Any]
    ) -> Optional[HazardInfo]:
        """Classify toggled toggle mine hazard (deadly state)."""
        # This method handles the TOGGLE_MINE_TOGGLED entity type
        # which represents the deadly toggled state (state 0)
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)

        return self._create_static_blocking_hazard(
            entity_id=entity_id,
            entity_type=EntityType.TOGGLE_MINE_TOGGLED,
            position=(entity_x, entity_y),
            danger_radius=TOGGLE_MINE_RADIUS_TOGGLED,  # Toggled state radius
            block_area_size=3,  # 3x3 blocking area
        )

    def _classify_thwump_static(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify thwump as static hazard (immobile or retreating state)."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        entity_state = entity.get("state", 0)
        orientation = entity.get("orientation", 0)

        # Handle different thwump states:
        # 0: Immobile (at rest) - activation trigger
        # -1: Retreating - static blocking hazard
        if entity_state == 0:
            # Immobile thwump - activation trigger with charge lane blocking
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
                blocked_directions=set(),
                danger_radius=0.0,
                activation_range=THWUMP_ACTIVATION_RANGE,  # 2 * (9 + 10) from entity class
                blocked_cells=blocked_cells,
                velocity=(0.0, 0.0),
                predicted_positions=[],
                orientation=orientation,
                charge_direction=(charge_dx, charge_dy),
                core_position=(entity_x, entity_y),
                launch_trajectories=[],
            )
        elif entity_state == -1:
            # Retreating thwump - static blocking hazard
            return self._create_static_blocking_hazard(
                entity_id=entity_id,
                entity_type=EntityType.THWUMP,
                position=(entity_x, entity_y),
                danger_radius=THWUMP_DANGER_RADIUS,  # 1.5 tiles
                block_area_size=3,  # 3x3 blocking area
                state=HazardState.RETREATING,
            )

        return None  # State 1 (charging) handled by dynamic classification

    def _classify_thwump_dynamic(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify thwump as dynamic hazard (charging/retreating state)."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        entity_state = entity.get("state", 0)
        orientation = entity.get("orientation", 0)

        if entity_state == 0:  # Immobile state
            return None

        # Determine hazard state
        hazard_state = HazardState.ACTIVE
        if entity_state == 1:  # Charging
            hazard_state = HazardState.CHARGING
        elif entity_state == 2:  # Retreating
            hazard_state = HazardState.RETREATING

        charge_dx, charge_dy = self._get_orientation_vector(orientation)
        predicted_positions = self._predict_thwump_movement(
            entity_x, entity_y, charge_dx, charge_dy, entity_state
        )

        return self._create_dynamic_threat_hazard(
            entity_id=entity_id,
            entity_type=EntityType.THWUMP,
            position=(entity_x, entity_y),
            velocity=(charge_dx * 2.0, charge_dy * 2.0),  # Charging velocity
            danger_radius=12.0,  # Thwump body radius
            predicted_positions=predicted_positions,
            state=hazard_state,
            charge_direction=(charge_dx, charge_dy),
            orientation=orientation,
        )

    def _classify_shove_thwump_static(
        self, entity: Dict[str, Any]
    ) -> Optional[HazardInfo]:
        """Classify shove thwump as static hazard (immobile or retreating state)."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        entity_state = entity.get("state", 0)

        # Handle different shove thwump states:
        # 0: Immobile (at rest) - safe to approach but can be activated
        # 3: Retreating - static blocking hazard
        if entity_state == 0:
            # Immobile shove thwump - safe but can be activated on contact
            return None  # Not a static hazard when immobile
        elif entity_state == 3:
            # Retreating shove thwump - static blocking hazard
            return self._create_static_blocking_hazard(
                entity_id=entity_id,
                entity_type=EntityType.SHWUMP,
                position=(entity_x, entity_y),
                danger_radius=SHOVE_THWUMP_DANGER_RADIUS,  # Outer size from entity class
                block_area_size=3,  # 3x3 blocking area
                state=HazardState.RETREATING,
            )

        return None  # States 1 and 2 (activated/launching) handled by dynamic classification

    def _classify_shove_thwump_dynamic(
        self, entity: Dict[str, Any]
    ) -> Optional[HazardInfo]:
        """Classify shove thwump as dynamic hazard (activated/launching state)."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        entity_state = entity.get("state", 0)
        xdir = entity.get("xdir", 0.0)
        ydir = entity.get("ydir", 0.0)

        # Handle different shove thwump states:
        # 1: Activated (contact triggered) - deadly core active
        # 2: Launching (moving away) - dynamic threat with movement prediction
        if entity_state == 1:
            # Activated shove thwump - deadly core active (static blocking)
            return self._create_static_blocking_hazard(
                entity_id=entity_id,
                entity_type=EntityType.SHWUMP,
                position=(entity_x, entity_y),
                danger_radius=SHOVE_THWUMP_CORE_RADIUS,
                block_area_size=1,  # Core only
                state=HazardState.ACTIVE,
            )
        elif entity_state == 2:
            # Launching shove thwump - dynamic threat with movement prediction
            predicted_positions = self._predict_shove_thwump_movement(
                entity_x, entity_y, xdir, ydir, entity_state
            )

            return self._create_dynamic_threat_hazard(
                entity_id=entity_id,
                entity_type=EntityType.SHWUMP,
                position=(entity_x, entity_y),
                velocity=(
                    xdir * DRONE_LAUNCH_SPEED,
                    ydir * DRONE_LAUNCH_SPEED,
                ),  # Launch speed from entity class
                danger_radius=SHOVE_THWUMP_CORE_RADIUS,
                predicted_positions=predicted_positions,
                state=HazardState.LAUNCHING,
                charge_direction=(xdir, ydir),
            )

        return None  # States 0 and 3 handled by static classification

    def _classify_one_way_platform(
        self, entity: Dict[str, Any]
    ) -> Optional[HazardInfo]:
        """
        Classify one-way platform hazard with continuous coordinate support.

        One-way platforms:
        - Can have continuous coordinates (not grid-aligned)
        - Block movement from specific direction based on orientation
        - Size: 12*12 pixel square (ONE_WAY_PLATFORM_SIZE total)
        - Complex collision detection based on approach angle and velocity
        """
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        orientation = entity.get("orientation", 0)

        # Get platform normal vector (direction it blocks from)
        normal_x, normal_y = self._get_orientation_vector(orientation)

        # One-way platforms block movement from the direction of their normal
        blocked_direction = orientation
        blocked_directions = {blocked_direction}

        # Platform is 24x24 pixels (SEMI_SIDE = ONE_WAY_PLATFORM_SEMI_SIDE)
        platform_semi_side = ONE_WAY_PLATFORM_SEMI_SIDE
        blocked_cells = self._create_blocked_cells_area(
            entity_x,
            entity_y,
            int(platform_semi_side * 2),  # ONE_WAY_PLATFORM_SIZE pixels
        )

        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DIRECTIONAL_BLOCKING,
            entity_type=EntityType.ONE_WAY,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=blocked_directions,
            danger_radius=ONE_WAY_PLATFORM_SEMI_SIDE
            * 2,  # ONE_WAY_PLATFORM_SIZE pixels
            activation_range=0.0,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=orientation,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[],
        )

    def _classify_drone(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify drone as dynamic hazard with comprehensive movement prediction."""
        entity_id = entity.get("id", -1)
        entity_x = entity.get("x", 0.0)
        entity_y = entity.get("y", 0.0)
        entity_type = entity.get("type", EntityType.DRONE_ZAP)

        # Get drone-specific properties
        orientation = entity.get("orientation", 0)
        mode = entity.get("mode", 0)
        direction = entity.get("dir", 0)
        speed = entity.get("speed", 8 / 7)  # Default zap drone speed

        # Determine drone properties based on type
        if entity_type == EntityType.MINI_DRONE:
            danger_radius = MINI_DRONE_RADIUS  # Mini drone radius
            grid_width = MINI_DRONE_GRID_SIZE  # Mini drone grid
            speed = entity.get("speed", 1.3)  # Mini drone speed
        else:
            danger_radius = DRONE_RADIUS  # Regular drone radius
            grid_width = DRONE_GRID_SIZE  # Regular drone grid
            speed = entity.get("speed", 8 / 7)  # Regular drone speed

        # Predict drone movement using comprehensive patrol logic
        predicted_positions = self._predict_drone_patrol_movement(
            entity_x,
            entity_y,
            direction,
            mode,
            speed,
            grid_width,
            self.DRONE_PREDICTION_TIME,
            entity,
        )

        # Calculate current velocity - use vx/vy if available, otherwise calculate from direction/speed
        vx = entity.get("vx")
        vy = entity.get("vy")
        if vx is not None and vy is not None:
            current_velocity = (vx, vy)
        else:
            # Calculate from direction and speed
            dir_vectors = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            dir_vec = dir_vectors.get(direction, (0, 0))
            current_velocity = (speed * dir_vec[0], speed * dir_vec[1])

        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DYNAMIC_THREAT,
            entity_type=entity_type,
            position=(entity_x, entity_y),
            state=HazardState.ACTIVE,
            blocked_directions=set(range(8)),  # All directions dangerous
            danger_radius=danger_radius,
            activation_range=0.0,
            blocked_cells=set(),
            velocity=current_velocity,
            predicted_positions=predicted_positions,
            orientation=orientation,
            charge_direction=(0.0, 0.0),
            core_position=(entity_x, entity_y),
            launch_trajectories=[],
        )

    def _classify_mini_drone(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify mini drone as dynamic hazard."""
        # Similar to regular drone but smaller
        return self._classify_drone(entity)  # Reuse drone logic for now

    def _classify_death_ball(self, entity: Dict[str, Any]) -> Optional[HazardInfo]:
        """Classify death ball as dynamic hazard."""
        # Similar to drone but with different movement pattern
        return self._classify_drone(entity)  # Reuse drone logic for now

    def analyze_bounce_block_traversal_blocking(
        self,
        bounce_block: Dict[str, Any],
        all_entities: List[Dict[str, Any]],
        path_start: Tuple[float, float],
        path_end: Tuple[float, float],
    ) -> bool:
        """
        Analyze if a bounce block blocks traversal in a narrow passage.

        Bounce blocks can block traversal if they're positioned in the center
        of a one-tile (24px) path and the ninja cannot displace them enough
        horizontally or vertically to get clearance.

        Args:
            bounce_block: Bounce block entity data
            all_entities: All entities in the level for clearance calculation
            path_start: Start position of the path being checked
            path_end: End position of the path being checked

        Returns:
            True if bounce block blocks the path, False otherwise
        """
        block_x = bounce_block.get("x", 0.0)
        block_y = bounce_block.get("y", 0.0)

        # Bounce block size: 9*9 pixel square
        bounce_block_semi_side = (
            BOUNCE_BLOCK_SEMI_SIDE  # 4.5 pixels (half the side length)
        )
        bounce_block_radius = bounce_block_semi_side * math.sqrt(2)  # Diagonal radius

        # Check if bounce block intersects with the path
        if not self._point_intersects_path(
            (block_x, block_y), path_start, path_end, bounce_block_radius + NINJA_RADIUS
        ):
            return False  # Doesn't intersect path

        # Calculate clearance in all directions
        clearance_dirs = calculate_clearance_directions(
            (block_x, block_y), all_entities
        )

        # Determine path direction
        path_dx = path_end[0] - path_start[0]
        path_dy = path_end[1] - path_start[1]
        path_length = math.sqrt(path_dx * path_dx + path_dy * path_dy)

        if path_length < 1e-6:
            return False  # No meaningful path

        path_dx /= path_length
        path_dy /= path_length

        # Check if ninja can displace bounce block enough to get clearance
        required_clearance = (
            NINJA_RADIUS + bounce_block_semi_side + 2.0
        )  # 2px safety margin

        # Check displacement options perpendicular to path direction
        if abs(path_dx) > abs(path_dy):  # Horizontal path
            # For horizontal path, ninja needs vertical clearance to go around
            up_clearance = clearance_dirs.get("up", 0.0)
            down_clearance = clearance_dirs.get("down", 0.0)

            # If bounce block can't be displaced vertically enough, it blocks
            if (
                up_clearance < required_clearance
                and down_clearance < required_clearance
            ):
                return True
        else:  # Vertical path
            # For vertical path, ninja needs horizontal clearance to go around
            left_clearance = clearance_dirs.get("left", 0.0)
            right_clearance = clearance_dirs.get("right", 0.0)

            # If bounce block can't be displaced horizontally enough, it blocks
            if (
                left_clearance < required_clearance
                and right_clearance < required_clearance
            ):
                return True

        return False  # Bounce block can be displaced, doesn't block traversal

    def _point_intersects_path(
        self,
        point: Tuple[float, float],
        path_start: Tuple[float, float],
        path_end: Tuple[float, float],
        radius: float,
    ) -> bool:
        """Check if a point (with radius) intersects a path line segment."""
        px, py = point
        x1, y1 = path_start
        x2, y2 = path_end

        # Calculate distance from point to line segment
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-6:
            # Path is a point
            dist_sq = (px - x1) * (px - x1) + (py - y1) * (py - y1)
            return dist_sq <= radius * radius

        # Project point onto line segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

        # Find closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Check distance
        dist_sq = (px - closest_x) * (px - closest_x) + (py - closest_y) * (
            py - closest_y
        )
        return dist_sq <= radius * radius

    def _get_orientation_vector(self, orientation: int) -> Tuple[float, float]:
        """Convert orientation (0-7) to unit vector."""
        # 8-directional orientation: 0=right, 1=down-right, 2=down, etc.
        angle = orientation * math.pi / 4.0
        return (math.cos(angle), math.sin(angle))

    def _create_blocked_cells_area(
        self, center_x: float, center_y: float, size: int
    ) -> Set[Tuple[int, int]]:
        """Create blocked cells for a square area around a center point."""
        blocked_cells = set()
        center_sub_x = int(center_x // 12)
        center_sub_y = int(center_y // 12)

        half_size = size // 2
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                blocked_cells.add((center_sub_x + dx, center_sub_y + dy))

        return blocked_cells

    def _create_static_blocking_hazard(
        self,
        entity_id: int,
        entity_type: int,
        position: Tuple[float, float],
        danger_radius: float,
        block_area_size: int = 1,
        state: HazardState = HazardState.ACTIVE,
        blocked_directions: Set[int] = None,
    ) -> HazardInfo:
        """Create a standard static blocking hazard."""
        if blocked_directions is None:
            blocked_directions = set(range(8))  # All directions blocked by default

        blocked_cells = self._create_blocked_cells_area(
            position[0], position[1], block_area_size
        )

        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.STATIC_BLOCKING,
            entity_type=entity_type,
            position=position,
            state=state,
            blocked_directions=blocked_directions,
            danger_radius=danger_radius,
            activation_range=0.0,
            blocked_cells=blocked_cells,
            velocity=(0.0, 0.0),
            predicted_positions=[],
            orientation=0,
            charge_direction=(0.0, 0.0),
            core_position=position,
            launch_trajectories=[],
        )

    def _create_dynamic_threat_hazard(
        self,
        entity_id: int,
        entity_type: int,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        danger_radius: float,
        predicted_positions: List[Tuple[float, float]],
        state: HazardState = HazardState.ACTIVE,
        charge_direction: Tuple[float, float] = (0.0, 0.0),
        orientation: int = 0,
    ) -> HazardInfo:
        """Create a standard dynamic threat hazard."""
        return HazardInfo(
            entity_id=entity_id,
            hazard_type=HazardType.DYNAMIC_THREAT,
            entity_type=entity_type,
            position=position,
            state=state,
            blocked_directions=set(range(8)),  # All directions dangerous
            danger_radius=danger_radius,
            activation_range=0.0,
            blocked_cells=set(),
            velocity=velocity,
            predicted_positions=predicted_positions,
            orientation=orientation,
            charge_direction=charge_direction,
            core_position=position,
            launch_trajectories=[],
        )

    def _predict_thwump_movement(
        self,
        start_x: float,
        start_y: float,
        charge_dx: float,
        charge_dy: float,
        state: int,
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
        prediction_time: float,
    ) -> List[Tuple[float, float]]:
        """Predict simple linear drone movement over time."""
        positions = []

        # Simple linear prediction (fallback for basic velocity-based prediction)
        for t in range(1, int(prediction_time) + 1, 5):  # Every 5 frames
            pred_x = start_x + vx * t
            pred_y = start_y + vy * t
            positions.append((pred_x, pred_y))

        return positions

    def _predict_drone_patrol_movement(
        self,
        start_x: float,
        start_y: float,
        direction: int,
        mode: int,
        speed: float,
        grid_width: int,
        prediction_time: float,
        entity: Dict[str, Any],
    ) -> List[Tuple[float, float]]:
        """
        Predict drone patrol movement using comprehensive movement logic.

        Drones follow grid-based patrol patterns:
        - Mode 0: Follow wall clockwise
        - Mode 1: Follow wall counter-clockwise
        - Mode 2: Wander clockwise
        - Mode 3: Wander counter-clockwise

        Args:
            start_x: Starting x position
            start_y: Starting y position
            direction: Current direction (0=right, 1=down, 2=left, 3=up)
            mode: Patrol mode (0-3)
            speed: Movement speed
            grid_width: Grid cell size (DRONE_GRID_SIZE for regular, MINI_DRONE_GRID_SIZE for mini)
            prediction_time: Time to predict (frames)
            entity: Full entity data for collision checking

        Returns:
            List of predicted positions
        """
        positions = []
        current_x, current_y = start_x, start_y
        current_dir = direction

        # Direction vectors
        dir_vectors = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        # Direction preference lists for each mode
        # {0:follow wall CW, 1:follow wall CCW, 2:wander CW, 3:wander CCW}
        # {0:keep forward, 1:turn right, 2:go backward, 3:turn left}
        dir_lists = {
            0: [1, 0, 3, 2],  # Follow wall CW: prefer right, forward, left, back
            1: [3, 0, 1, 2],  # Follow wall CCW: prefer left, forward, right, back
            2: [0, 1, 3, 2],  # Wander CW: prefer forward, right, left, back
            3: [0, 3, 1, 2],  # Wander CCW: prefer forward, left, right, back
        }

        # Get target position (drone moves between grid centers)
        target_x = entity.get("xtarget", start_x)
        target_y = entity.get("ytarget", start_y)

        for frame in range(int(prediction_time)):
            # Calculate movement toward target
            dx = target_x - current_x
            dy = target_y - current_y
            dist = math.sqrt(dx * dx + dy * dy)

            # If close to target, choose next direction and target
            if dist < 0.1:
                current_x, current_y = target_x, target_y

                # Try directions in mode preference order
                new_dir = None
                for i in range(4):
                    test_dir = (current_dir + dir_lists[mode][i]) % 4
                    if self._can_drone_move_direction(
                        current_x,
                        current_y,
                        test_dir,
                        grid_width,
                        entity.get(
                            "RADIUS",
                            DRONE_RADIUS
                            if grid_width == DRONE_GRID_SIZE
                            else MINI_DRONE_RADIUS,
                        ),
                    ):
                        new_dir = test_dir
                        break

                if new_dir is not None:
                    current_dir = new_dir
                    # Set new target
                    dir_vec = dir_vectors[current_dir]
                    target_x = current_x + grid_width * dir_vec[0]
                    target_y = current_y + grid_width * dir_vec[1]
                else:
                    # No valid direction, stay in place
                    target_x, target_y = current_x, current_y

            # Move toward target
            if dist > 0.1:
                move_x = speed * dx / dist
                move_y = speed * dy / dist
                current_x += move_x
                current_y += move_y

            positions.append((current_x, current_y))

        return positions

    def _can_drone_move_direction(
        self, x: float, y: float, direction: int, grid_width: int, radius: float
    ) -> bool:
        """
        Check if drone can move in given direction using precise collision detection.

        Args:
            x: Current x position
            y: Current y position
            direction: Direction to test (0=right, 1=down, 2=left, 3=up)
            grid_width: Grid cell size
            radius: Drone radius

        Returns:
            True if movement is possible
        """
        if not self._precise_collision or not self._current_tiles:
            # Fallback to simplified check if no collision system available
            return True

        # Calculate target position
        dir_vectors = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        dir_vec = dir_vectors.get(direction, (0, 0))
        target_x = x + grid_width * dir_vec[0]
        target_y = y + grid_width * dir_vec[1]

        # Check if path is traversable using precise collision detection
        return self._precise_collision.is_path_traversable(
            src_x=x,
            src_y=y,
            tgt_x=target_x,
            tgt_y=target_y,
            tiles=self._current_tiles,
            ninja_radius=radius,  # Use drone radius instead of ninja radius
        )

    def _predict_shove_thwump_movement(
        self, start_x: float, start_y: float, xdir: float, ydir: float, state: int
    ) -> List[Tuple[float, float]]:
        """
        Predict shove thwump movement during launch phase.

        Args:
            start_x: Starting x position
            start_y: Starting y position
            xdir: X direction (-1, 0, or 1)
            ydir: Y direction (-1, 0, or 1)
            state: Current state (2 = launching)

        Returns:
            List of predicted positions
        """
        positions = []
        current_x, current_y = start_x, start_y

        if state == 2:  # Launching
            launch_speed = DRONE_LAUNCH_SPEED  # From entity class

            # Predict movement until wall collision (simplified)
            for frame in range(60):  # Predict up to 60 frames
                current_x += xdir * launch_speed
                current_y += ydir * launch_speed
                positions.append((current_x, current_y))

                # Stop prediction if moved too far (would hit wall)
                if abs(current_x - start_x) > 240 or abs(current_y - start_y) > 240:
                    break

        return positions

    def _check_static_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo,
    ) -> bool:
        """Check intersection with static hazard."""
        hazard_x, hazard_y = hazard_info.position

        # Check if path passes through hazard danger radius
        return self._line_intersects_circle(
            src_x,
            src_y,
            tgt_x,
            tgt_y,
            hazard_x,
            hazard_y,
            hazard_info.danger_radius + NINJA_RADIUS,
        )

    def _check_directional_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo,
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
        approach_orientation = int((angle + math.pi / 8) / (math.pi / 4)) % 8

        # Check if approaching from blocked direction
        if approach_orientation in hazard_info.blocked_directions:
            return self._check_static_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            )

        return False  # Safe approach direction

    def _check_dynamic_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo,
    ) -> bool:
        """Check intersection with dynamic hazard."""
        # Check current position
        if self._check_static_hazard_intersection(
            src_x, src_y, tgt_x, tgt_y, hazard_info
        ):
            return True

        # Check predicted positions
        for pred_x, pred_y in hazard_info.predicted_positions:
            if self._line_intersects_circle(
                src_x,
                src_y,
                tgt_x,
                tgt_y,
                pred_x,
                pred_y,
                hazard_info.danger_radius + NINJA_RADIUS,
            ):
                return True

        return False

    def _check_activation_hazard_intersection(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        hazard_info: HazardInfo,
    ) -> bool:
        """Check intersection with activation hazard (thwump activation zone)."""
        hazard_x, hazard_y = hazard_info.position

        # Check if path enters activation range
        if self._line_intersects_circle(
            src_x,
            src_y,
            tgt_x,
            tgt_y,
            hazard_x,
            hazard_y,
            hazard_info.activation_range + NINJA_RADIUS,
        ):
            # If activated, check charge direction for collision
            charge_dx, charge_dy = hazard_info.charge_direction

            # Check if path crosses charge lane
            for i in range(1, 6):  # 5 tiles in charge direction
                charge_x = hazard_x + i * TILE_PIXEL_SIZE * charge_dx
                charge_y = hazard_y + i * TILE_PIXEL_SIZE * charge_dy

                if self._line_intersects_circle(
                    src_x,
                    src_y,
                    tgt_x,
                    tgt_y,
                    charge_x,
                    charge_y,
                    12.0 + NINJA_RADIUS,  # Thwump body radius
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
        radius: float,
    ) -> bool:
        """Check if line segment intersects circle."""
        # Vector from line start to circle center
        to_circle_x = circle_x - line_x1
        to_circle_y = circle_y - line_y1

        # Line direction vector
        line_dx = line_x2 - line_x1
        line_dy = line_y2 - line_y1
        line_length_sq = line_dx * line_dx + line_dy * line_dy

        if line_length_sq < 1e-6:  # Degenerate line (point)
            dist_sq = to_circle_x * to_circle_x + to_circle_y * to_circle_y
            return dist_sq <= radius * radius

        # Project circle center onto line
        t = (to_circle_x * line_dx + to_circle_y * line_dy) / line_length_sq
        t = max(0.0, min(1.0, t))  # Clamp to line segment

        # Closest point on line to circle center
        closest_x = line_x1 + t * line_dx
        closest_y = line_y1 + t * line_dy

        # Distance from circle center to closest point
        dist_sq = (circle_x - closest_x) ** 2 + (circle_y - closest_y) ** 2

        return dist_sq <= radius * radius
