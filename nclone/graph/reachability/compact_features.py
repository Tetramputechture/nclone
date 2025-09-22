"""
Compact reachability features for RL integration.

This module implements the 64-dimensional compact feature encoding specified in TASK_003.
It builds on the tiered reachability system and simplified completion strategy to provide
efficient, RL-friendly feature vectors for the HGT-based RL architecture.

Feature Vector Layout (64 dimensions):
[0-7]:    Objective distances (8 closest objectives)
[8-23]:   Switch states and dependencies (16 switches max)
[24-39]:  Hazard proximities and threat levels (16 hazards max)
[40-47]:  Area connectivity metrics (8 directional areas)
[48-55]:  Movement capability indicators (8 movement types)
[56-63]:  Meta-features (confidence, timing, complexity)
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass

from .reachability_types import ReachabilityResult
from ..simple_objective_system import SimplifiedCompletionStrategy, SimpleObjective


@dataclass
class FeatureConfig:
    """Configuration for compact feature encoding."""

    objective_slots: int = 8
    switch_slots: int = 16
    hazard_slots: int = 16
    area_slots: int = 8
    movement_slots: int = 8
    meta_slots: int = 8
    total_features: int = 64

    def __post_init__(self):
        """Validate configuration."""
        expected_total = (
            self.objective_slots
            + self.switch_slots
            + self.hazard_slots
            + self.area_slots
            + self.movement_slots
            + self.meta_slots
        )
        if expected_total != self.total_features:
            raise ValueError(
                f"Feature slots sum to {expected_total}, expected {self.total_features}"
            )


class CompactReachabilityFeatures:
    """
    Compact encoding of reachability analysis for RL integration.

    This class implements the 64-dimensional feature encoding specified in TASK_003,
    designed for integration with HGT-based RL architectures. It provides efficient,
    informative features that capture essential reachability information while
    maintaining fast extraction times suitable for real-time RL training.

    The encoding leverages the simplified completion strategy to provide clear
    objective-based features that are easy for RL agents to learn from.
    """

    def __init__(self, config: Optional[FeatureConfig] = None, debug: bool = False):
        """
        Initialize compact feature encoder.

        Args:
            config: Feature configuration (uses default if None)
            debug: Enable debug output
        """
        self.config = config or FeatureConfig()
        self.debug = debug
        self.completion_strategy = SimplifiedCompletionStrategy(debug=debug)

        # Feature dimension validation
        if self.config.total_features != 64:
            raise ValueError(f"Expected 64 features, got {self.config.total_features}")

        # Cache for expensive computations
        self._level_diagonal_cache = {}
        self._sector_cache = {}

    def encode_reachability(
        self,
        reachability_result: ReachabilityResult,
        level_data: Any,
        entities: List[Any],
        ninja_position: Tuple[float, float],
        switch_states: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """
        Encode reachability analysis into compact 64-dimensional feature vector.

        This is the main entry point for feature extraction. It combines information
        from reachability analysis, level structure, and the simplified completion
        strategy to create a comprehensive feature representation.

        Args:
            reachability_result: Result from tiered reachability analysis
            level_data: Level tile data and structure
            entities: List of game entities (switches, doors, hazards, etc.)
            ninja_position: Current ninja position (x, y)
            switch_states: Current switch activation states

        Returns:
            64-dimensional numpy array with encoded features
        """
        if switch_states is None:
            switch_states = {}

        # Initialize feature vector
        features = np.zeros(self.config.total_features, dtype=np.float32)
        # [0-7] Objective distances - leverages simplified completion strategy
        objective_features = self._encode_objective_distances(
            reachability_result, level_data, entities, ninja_position, switch_states
        )
        features[0:8] = objective_features

        # [8-23] Switch states and dependencies
        switch_features = self._encode_switch_states(
            reachability_result, entities, switch_states
        )
        features[8:24] = switch_features

        # [24-39] Hazard proximities and threat levels
        hazard_features = self._encode_hazard_proximities(
            reachability_result, entities, ninja_position
        )
        features[24:40] = hazard_features

        # [40-47] Area connectivity metrics
        area_features = self._encode_area_connectivity(
            reachability_result, level_data, ninja_position
        )
        features[40:48] = area_features

        # [48-55] Movement capability indicators
        movement_features = self._encode_movement_capabilities(
            reachability_result, level_data, ninja_position
        )
        features[48:56] = movement_features

        # [56-63] Meta-features (confidence, timing, complexity)
        meta_features = self._encode_meta_features(
            reachability_result, level_data, entities
        )
        features[56:64] = meta_features

        if self.debug:
            self._debug_feature_summary(features, ninja_position)

        return features

    def _encode_objective_distances(
        self,
        reachability_result: ReachabilityResult,
        level_data: Any,
        entities: List[Any],
        ninja_position: Tuple[float, float],
        switch_states: Dict[str, bool],
    ) -> np.ndarray:
        """
        Encode distances to key objectives using simplified completion strategy.

        This method leverages our simplified completion strategy to identify the
        most relevant objectives and encode their distances. This provides clear,
        RL-friendly objective guidance.

        Encoding Strategy:
        - Use simplified completion strategy to get current objective
        - Encode distance to current objective in feature [0]
        - Fill remaining slots with distances to other reachable objectives
        - Normalize distances by level diagonal for consistent scaling
        - Use log scaling for better gradient properties
        - Unreachable objectives encoded as 1.0 (maximum distance)
        """
        distances = np.ones(
            self.config.objective_slots, dtype=np.float32
        )  # Default: unreachable

        # Get current objective from simplified completion strategy
        current_objective = self.completion_strategy.get_next_objective(
            ninja_position, level_data, entities, switch_states
        )

        # Get level diagonal for normalization
        level_diagonal = self._get_level_diagonal(level_data)

        if current_objective:
            # Encode current objective distance in feature [0]
            normalized_distance = current_objective.distance / level_diagonal
            # Log scaling for better gradient properties
            distances[0] = min(math.log(1 + normalized_distance) / math.log(2), 1.0)

            if self.debug:
                print(
                    f"DEBUG: Current objective: {current_objective.objective_type.value} "
                    f"at {current_objective.position}, distance: {normalized_distance:.3f}"
                )

        # Fill remaining slots with other objectives
        other_objectives = self._identify_other_objectives(entities, current_objective)

        for i, objective_info in enumerate(
            other_objectives[: self.config.objective_slots - 1], 1
        ):
            if self._is_position_reachable(
                objective_info["position"], reachability_result
            ):
                raw_distance = self._calculate_distance(
                    ninja_position, objective_info["position"]
                )
                normalized_distance = raw_distance / level_diagonal
                distances[i] = min(math.log(1 + normalized_distance) / math.log(2), 1.0)
            # else: keep default 1.0 (unreachable)

        return distances

    def _encode_switch_states(
        self,
        reachability_result: ReachabilityResult,
        entities: List[Any],
        switch_states: Dict[str, bool],
    ) -> np.ndarray:
        """
        Encode switch states and dependencies.

        Encoding Strategy:
        - 0.0: Switch unreachable or doesn't exist
        - 0.5: Switch reachable but not activated
        - 1.0: Switch activated
        - Add small bonus for switches with dependencies (0.1-0.4)
        """
        switch_features = np.zeros(self.config.switch_slots, dtype=np.float32)

        switches = self._get_switches_from_entities(entities)

        for i, switch_info in enumerate(switches[: self.config.switch_slots]):
            switch_id = switch_info.get("id")
            position = switch_info.get("position")

            if switch_id and switch_states.get(switch_id, False):
                # Switch is activated
                switch_features[i] = 1.0
            elif position and self._is_position_reachable(
                position, reachability_result
            ):
                # Switch is reachable but not activated
                switch_features[i] = 0.5
            else:
                # Switch is unreachable
                switch_features[i] = 0.0

            # Add dependency bonus for important switches
            if self._switch_has_dependencies(switch_info):
                dependency_bonus = min(
                    0.1 * self._count_switch_dependencies(switch_info), 0.4
                )
                switch_features[i] = min(switch_features[i] + dependency_bonus, 1.4)

        return switch_features

    def _encode_hazard_proximities(
        self,
        reachability_result: ReachabilityResult,
        entities: List[Any],
        ninja_position: Tuple[float, float],
    ) -> np.ndarray:
        """
        Encode proximity and threat level of hazards.

        Encoding Strategy:
        - Distance-based threat encoding (closer = higher threat)
        - Hazard type weighting (drones > mines > static hazards)
        - Reachability-aware encoding (unreachable hazards have lower threat)
        - Exponential decay for distant hazards
        """
        hazard_features = np.zeros(self.config.hazard_slots, dtype=np.float32)

        hazards = self._get_hazards_from_entities(entities, ninja_position)

        for i, hazard_info in enumerate(hazards[: self.config.hazard_slots]):
            position = hazard_info.get("position")
            hazard_type = hazard_info.get("type", "unknown")

            if position:
                # Calculate base threat based on distance
                distance = self._calculate_distance(ninja_position, position)
                threat_radius = self._get_hazard_threat_radius(hazard_type)

                if distance <= threat_radius:
                    # Immediate threat (linear decay within threat radius)
                    base_threat = 1.0 - (distance / threat_radius)
                else:
                    # Distant threat (exponential decay)
                    base_threat = math.exp(-(distance - threat_radius) / threat_radius)

                # Weight by hazard type
                type_weight = self._get_hazard_type_weight(hazard_type)

                # Adjust for reachability (unreachable hazards less threatening)
                if not self._is_position_reachable(position, reachability_result):
                    type_weight *= 0.3

                hazard_features[i] = base_threat * type_weight

        return hazard_features

    def _encode_area_connectivity(
        self,
        reachability_result: ReachabilityResult,
        level_data: Any,
        ninja_position: Tuple[float, float],
    ) -> np.ndarray:
        """
        Encode connectivity to different areas of the level.

        Encoding Strategy:
        - Divide level into 8 directional sectors (N, NE, E, SE, S, SW, W, NW)
        - Encode reachable area percentage in each sector
        - Weight by objective density in each sector
        """
        area_features = np.zeros(self.config.area_slots, dtype=np.float32)

        sectors = self._define_level_sectors(level_data, ninja_position)
        reachable_positions = set(reachability_result.reachable_positions)

        for i, sector in enumerate(sectors):
            sector_positions = self._get_positions_in_sector(sector, level_data)

            if sector_positions:
                # Calculate reachable ratio in this sector
                sector_reachable = len(sector_positions & reachable_positions)
                reachability_ratio = sector_reachable / len(sector_positions)

                # Weight by objective density (if we had objective info)
                # For now, use simple reachability ratio
                area_features[i] = reachability_ratio
            else:
                area_features[i] = 0.0

        return area_features

    def _encode_movement_capabilities(
        self,
        reachability_result: ReachabilityResult,
        level_data: Any,
        ninja_position: Tuple[float, float],
    ) -> np.ndarray:
        """
        Encode available movement capabilities from current position.

        Encoding Strategy:
        - Test each movement type availability (walk, jump, wall_jump, etc.)
        - Encode as capability strength (0=impossible, 1=fully available)
        - Consider local tile types and physics constraints
        """
        movement_features = np.zeros(self.config.movement_slots, dtype=np.float32)

        movement_types = [
            "walk_left",
            "walk_right",
            "jump_up",
            "jump_left",
            "jump_right",
            "wall_jump",
            "fall",
            "special",
        ]

        for i, movement_type in enumerate(movement_types):
            capability = self._assess_movement_capability(
                ninja_position, movement_type, level_data, reachability_result
            )
            movement_features[i] = capability

        return movement_features

    def _encode_meta_features(
        self,
        reachability_result: ReachabilityResult,
        level_data: Any,
        entities: List[Any],
    ) -> np.ndarray:
        """
        Encode meta-information about the reachability analysis.

        Features:
        [0] Analysis confidence (0-1)
        [1] Computation time (normalized)
        [2] Level complexity estimate (0-1)
        [3] Reachable area ratio (0-1)
        [4] Switch dependency complexity (0-1)
        [5] Hazard density (0-1)
        [6] Analysis method indicator (tier 1/2/3)
        [7] Cache hit indicator (0=miss, 1=hit)
        """
        meta_features = np.zeros(self.config.meta_slots, dtype=np.float32)

        # Analysis confidence
        meta_features[0] = getattr(reachability_result, "confidence", 1.0)

        # Computation time (log-normalized)
        computation_time = getattr(reachability_result, "computation_time_ms", 1.0)
        meta_features[1] = min(math.log(1 + computation_time) / math.log(100), 1.0)

        # Level complexity estimate
        meta_features[2] = self._estimate_level_complexity(level_data, entities)

        # Reachable area ratio
        if hasattr(level_data, "shape"):
            total_positions = level_data.shape[0] * level_data.shape[1]
        else:
            total_positions = getattr(level_data, "width", 100) * getattr(
                level_data, "height", 100
            )

        reachable_count = len(reachability_result.reachable_positions)
        meta_features[3] = min(reachable_count / total_positions, 1.0)

        # Switch dependency complexity
        meta_features[4] = self._calculate_switch_complexity(entities)

        # Hazard density
        meta_features[5] = self._calculate_hazard_density(entities, level_data)

        # Analysis method indicator
        method = getattr(reachability_result, "method", "flood_fill")
        method_encoding = {
            "flood_fill": 0.3,
            "tier1": 0.3,
            "tier2": 0.6,
            "tier3": 1.0,
        }
        meta_features[6] = method_encoding.get(method, 0.5)

        # Cache hit indicator
        cache_hit = getattr(reachability_result, "from_cache", False)
        meta_features[7] = 1.0 if cache_hit else 0.0

        return meta_features

    # Helper methods for feature encoding

    def _get_level_diagonal(self, level_data: Any) -> float:
        """Get level diagonal for distance normalization."""
        if hasattr(level_data, "shape"):
            height, width = level_data.shape
        else:
            width = getattr(level_data, "width", 100)
            height = getattr(level_data, "height", 100)

        # Convert to pixels (assuming 24 pixels per tile)
        pixel_width = width * 24
        pixel_height = height * 24

        diagonal = math.sqrt(pixel_width**2 + pixel_height**2)

        # Cache for efficiency
        cache_key = (width, height)
        self._level_diagonal_cache[cache_key] = diagonal

        return diagonal

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _is_position_reachable(
        self, position: Tuple[float, float], reachability_result: ReachabilityResult
    ) -> bool:
        """Check if a position is reachable according to reachability analysis."""
        # Convert position to tile coordinates for comparison
        tile_x = int(position[0] // 24)
        tile_y = int(position[1] // 24)
        tile_center = (tile_x * 24 + 12, tile_y * 24 + 12)

        return tile_center in reachability_result.reachable_positions

    def _identify_other_objectives(
        self, entities: List[Any], current_objective: Optional[SimpleObjective]
    ) -> List[Dict[str, Any]]:
        """Identify other objectives beyond the current one."""
        objectives = []
        current_pos = current_objective.position if current_objective else None

        for entity in entities:
            if hasattr(entity, "entity_type"):
                entity_type = entity.entity_type
                position = (entity.x, entity.y)

                # Skip current objective
                if current_pos and position == current_pos:
                    continue

                if "switch" in entity_type or "door" in entity_type:
                    priority = self._get_objective_priority(entity_type)
                    objectives.append(
                        {
                            "position": position,
                            "type": entity_type,
                            "priority": priority,
                            "entity": entity,
                        }
                    )

        # Sort by priority
        objectives.sort(key=lambda x: x["priority"], reverse=True)
        return objectives

    def _get_objective_priority(self, entity_type: str) -> float:
        """Get priority for different objective types."""
        priority_map = {
            "exit_door": 1.0,
            "exit_switch": 0.9,
            "locked_door_switch": 0.8,
            "door_switch": 0.7,
            "trap_door_switch": 0.3,
            "switch": 0.5,
        }
        return priority_map.get(entity_type, 0.4)

    def _get_switches_from_entities(self, entities: List[Any]) -> List[Dict[str, Any]]:
        """Extract switch information from entities."""
        switches = []

        for entity in entities:
            if hasattr(entity, "entity_type") and "switch" in entity.entity_type:
                switches.append(
                    {
                        "position": (entity.x, entity.y),
                        "type": entity.entity_type,
                        "id": getattr(entity, "id", None),
                        "entity": entity,
                    }
                )

        return switches

    def _switch_has_dependencies(self, switch_info: Dict[str, Any]) -> bool:
        """Check if switch has dependencies (controls doors)."""
        switch_type = switch_info.get("type", "")

        # Enhanced dependency detection based on switch type
        dependency_indicators = ["door", "exit", "locked", "trap", "gate"]

        return any(
            indicator in switch_type.lower() for indicator in dependency_indicators
        )

    def _count_switch_dependencies(self, switch_info: Dict[str, Any]) -> int:
        """Count number of dependencies for a switch with enhanced analysis."""
        switch_type = switch_info.get("type", "").lower()
        position = switch_info.get("position", (0, 0))

        # Base dependency count based on switch type
        base_dependencies = 0

        if "exit" in switch_type:
            base_dependencies = 3  # Exit switches are critical
        elif "locked_door" in switch_type or "door_locked" in switch_type:
            base_dependencies = 2  # Locked doors block progress
        elif "door" in switch_type:
            base_dependencies = 2  # Regular doors
        elif "trap" in switch_type:
            base_dependencies = 1  # Trap doors (negative impact)
        elif "gate" in switch_type:
            base_dependencies = 2  # Gates block areas
        else:
            base_dependencies = 1  # Generic switches

        # Additional factors that could increase dependency count:
        # 1. Switch position (central switches likely more important)
        # 2. Switch accessibility (harder to reach = more important when reached)

        # Position-based importance (switches in central areas are more critical)
        # This is a heuristic - in a full implementation, you'd analyze the level graph
        x, y = position
        if 200 <= x <= 800 and 200 <= y <= 400:  # Rough center area
            base_dependencies += 1

        return min(base_dependencies, 5)  # Cap at 5 dependencies

    def _get_hazards_from_entities(
        self, entities: List[Any], ninja_position: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Extract and prioritize hazard information from entities."""
        hazards = []

        for entity in entities:
            if hasattr(entity, "entity_type"):
                entity_type = entity.entity_type
                if self._is_hazard_type(entity_type):
                    position = (entity.x, entity.y)
                    distance = self._calculate_distance(ninja_position, position)

                    hazards.append(
                        {
                            "position": position,
                            "type": entity_type,
                            "distance": distance,
                            "entity": entity,
                        }
                    )

        # Sort by distance (closest first)
        hazards.sort(key=lambda x: x["distance"])
        return hazards

    def _is_hazard_type(self, entity_type: str) -> bool:
        """Check if entity type represents a hazard."""
        hazard_types = ["drone", "mine", "thwump", "laser", "rocket", "chaingun"]
        return any(hazard in entity_type.lower() for hazard in hazard_types)

    def _get_hazard_threat_radius(self, hazard_type: str) -> float:
        """Get threat radius for different hazard types."""
        radius_map = {
            "drone": 48.0,  # 2 tiles
            "mine": 36.0,  # 1.5 tiles
            "thwump": 72.0,  # 3 tiles
            "laser": 120.0,  # 5 tiles
            "rocket": 60.0,  # 2.5 tiles
            "chaingun": 96.0,  # 4 tiles
        }

        for hazard_key, radius in radius_map.items():
            if hazard_key in hazard_type.lower():
                return radius

        return 48.0  # Default threat radius

    def _get_hazard_type_weight(self, hazard_type: str) -> float:
        """Get threat weight for different hazard types."""
        weight_map = {
            "drone": 1.0,  # Highest threat (moving)
            "mine": 0.8,  # High threat (explosive)
            "thwump": 0.6,  # Medium threat (predictable)
            "laser": 0.9,  # High threat (instant)
            "rocket": 0.7,  # Medium-high threat
            "chaingun": 0.5,  # Medium threat
        }

        for hazard_key, weight in weight_map.items():
            if hazard_key in hazard_type.lower():
                return weight

        return 0.4  # Default weight for unknown hazards

    def _define_level_sectors(
        self, level_data: Any, ninja_position: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Define 8 directional sectors around ninja position."""
        if hasattr(level_data, "shape"):
            height, width = level_data.shape
        else:
            width = getattr(level_data, "width", 100)
            height = getattr(level_data, "height", 100)

        # Convert to pixel coordinates
        level_width = width * 24
        level_height = height * 24

        ninja_x, ninja_y = ninja_position

        # Define 8 sectors: N, NE, E, SE, S, SW, W, NW
        sectors = []
        sector_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        for i, name in enumerate(sector_names):
            angle_start = i * 45 - 22.5  # Each sector is 45 degrees
            angle_end = angle_start + 45

            sectors.append(
                {
                    "name": name,
                    "angle_start": angle_start,
                    "angle_end": angle_end,
                    "center_x": ninja_x,
                    "center_y": ninja_y,
                    "level_width": level_width,
                    "level_height": level_height,
                }
            )

        return sectors

    def _get_positions_in_sector(
        self, sector: Dict[str, Any], level_data: Any
    ) -> Set[Tuple[int, int]]:
        """Get all positions within a directional sector using proper angular calculations."""
        positions = set()

        if hasattr(level_data, "shape"):
            height, width = level_data.shape
        else:
            width = getattr(level_data, "width", 100)
            height = getattr(level_data, "height", 100)

        sector_name = sector["name"]
        ninja_x = sector.get("center_x", width * 12)
        ninja_y = sector.get("center_y", height * 12)

        # Define sector angles (in radians, 0 = right, π/2 = up)
        sector_angles = {
            "E": (7 * math.pi / 4, math.pi / 4),  # -45° to 45°
            "NE": (math.pi / 4, 3 * math.pi / 4),  # 45° to 135°
            "N": (3 * math.pi / 4, 5 * math.pi / 4),  # 135° to 225°
            "NW": (5 * math.pi / 4, 7 * math.pi / 4),  # 225° to 315°
            "W": (7 * math.pi / 4, math.pi / 4),  # 315° to 45° (wraps around)
            "SW": (math.pi / 4, 3 * math.pi / 4),  # Same as NE but mirrored
            "S": (3 * math.pi / 4, 5 * math.pi / 4),  # Same as N but mirrored
            "SE": (5 * math.pi / 4, 7 * math.pi / 4),  # Same as NW but mirrored
        }

        # Correct sector angle mapping
        angle_map = {
            "E": (0, math.pi / 4),  # 0° to 45°
            "NE": (math.pi / 4, 3 * math.pi / 4),  # 45° to 135°
            "N": (3 * math.pi / 4, 5 * math.pi / 4),  # 135° to 225°
            "NW": (5 * math.pi / 4, 7 * math.pi / 4),  # 225° to 315°
            "W": (7 * math.pi / 4, 2 * math.pi),  # 315° to 360°
            "SW": (math.pi, 5 * math.pi / 4),  # 180° to 225°
            "S": (3 * math.pi / 2, 7 * math.pi / 4),  # 270° to 315°
            "SE": (7 * math.pi / 4, 2 * math.pi),  # 315° to 360°
        }

        # Use proper directional angles
        if sector_name in angle_map:
            start_angle, end_angle = angle_map[sector_name]

            # Calculate maximum distance to consider (level diagonal)
            max_distance = math.sqrt((width * 24) ** 2 + (height * 24) ** 2)

            # Sample positions in the sector
            for x in range(0, width):
                for y in range(0, height):
                    pixel_x = x * 24 + 12
                    pixel_y = y * 24 + 12

                    # Calculate angle from ninja position to this position
                    dx = pixel_x - ninja_x
                    dy = pixel_y - ninja_y

                    if dx == 0 and dy == 0:
                        continue  # Skip ninja's current position

                    angle = math.atan2(-dy, dx)  # Negative dy for screen coordinates
                    if angle < 0:
                        angle += 2 * math.pi  # Normalize to [0, 2π]

                    # Check if angle is within sector
                    in_sector = False
                    if start_angle <= end_angle:
                        in_sector = start_angle <= angle <= end_angle
                    else:  # Wraps around (e.g., 315° to 45°)
                        in_sector = angle >= start_angle or angle <= end_angle

                    if in_sector:
                        # Also check distance to avoid including the entire level
                        distance = math.sqrt(dx * dx + dy * dy)
                        if (
                            distance <= max_distance * 0.7
                        ):  # Limit to 70% of level diagonal
                            positions.add((pixel_x, pixel_y))

        return positions

    def _assess_movement_capability(
        self,
        ninja_position: Tuple[float, float],
        movement_type: str,
        level_data: Any,
        reachability_result: ReachabilityResult,
    ) -> float:
        """Assess capability for specific movement type using physics-based calculations."""
        ninja_x, ninja_y = ninja_position
        reachable_positions = reachability_result.reachable_positions

        # Physics constants (from ninja.py constants)
        GRAVITY = 0.21  # Approximate gravity constant
        MAX_HORIZONTAL_SPEED = 4.5
        JUMP_VELOCITY = -9.0  # Negative for upward movement
        WALL_JUMP_X = 4.0
        WALL_JUMP_Y = -7.0

        capability_score = 0.0
        test_count = 0

        if movement_type == "walk_left":
            # Test horizontal walking capability to the left
            for distance in [24, 48, 72]:  # 1, 2, 3 tiles
                test_pos = (ninja_x - distance, ninja_y)
                if (
                    self._is_position_walkable(test_pos, level_data)
                    and test_pos in reachable_positions
                ):
                    capability_score += 1.0 / (
                        distance / 24
                    )  # Closer positions weighted higher
                test_count += 1

        elif movement_type == "walk_right":
            # Test horizontal walking capability to the right
            for distance in [24, 48, 72]:
                test_pos = (ninja_x + distance, ninja_y)
                if (
                    self._is_position_walkable(test_pos, level_data)
                    and test_pos in reachable_positions
                ):
                    capability_score += 1.0 / (distance / 24)
                test_count += 1

        elif movement_type == "jump_up":
            # Test vertical jumping capability
            for height in [24, 48, 72, 96]:  # Different jump heights
                test_pos = (ninja_x, ninja_y - height)
                if self._can_reach_by_jumping(ninja_position, test_pos, level_data):
                    if test_pos in reachable_positions:
                        capability_score += 1.0 / (height / 24)
                test_count += 1

        elif movement_type == "jump_left":
            # Test diagonal jumping to the left
            for dx, dy in [(-24, -24), (-48, -24), (-24, -48), (-48, -48)]:
                test_pos = (ninja_x + dx, ninja_y + dy)
                if self._can_reach_by_jumping(ninja_position, test_pos, level_data):
                    if test_pos in reachable_positions:
                        distance = math.sqrt(dx * dx + dy * dy)
                        capability_score += 1.0 / (distance / 24)
                test_count += 1

        elif movement_type == "jump_right":
            # Test diagonal jumping to the right
            for dx, dy in [(24, -24), (48, -24), (24, -48), (48, -48)]:
                test_pos = (ninja_x + dx, ninja_y + dy)
                if self._can_reach_by_jumping(ninja_position, test_pos, level_data):
                    if test_pos in reachable_positions:
                        distance = math.sqrt(dx * dx + dy * dy)
                        capability_score += 1.0 / (distance / 24)
                test_count += 1

        elif movement_type == "wall_jump":
            # Test wall jumping capability
            wall_jump_positions = [
                (
                    ninja_x - WALL_JUMP_X * 24,
                    ninja_y + WALL_JUMP_Y * 24,
                ),  # Left wall jump
                (
                    ninja_x + WALL_JUMP_X * 24,
                    ninja_y + WALL_JUMP_Y * 24,
                ),  # Right wall jump
            ]

            for test_pos in wall_jump_positions:
                if self._can_wall_jump_to(ninja_position, test_pos, level_data):
                    if test_pos in reachable_positions:
                        capability_score += 1.0
                test_count += 1

        elif movement_type == "fall":
            # Test falling capability
            for distance in [24, 48, 96, 144]:  # Different fall distances
                test_pos = (ninja_x, ninja_y + distance)
                if self._can_fall_to(ninja_position, test_pos, level_data):
                    if test_pos in reachable_positions:
                        capability_score += 1.0 / (
                            distance / 48
                        )  # Falling is easier than jumping
                test_count += 1

        elif movement_type == "special":
            # Test special movement capabilities (launch pads, bounce blocks, etc.)
            capability_score = self._assess_special_movement_capability(
                ninja_position, level_data, reachability_result
            )
            test_count = 1

        return capability_score / test_count if test_count > 0 else 0.0

    def _is_position_walkable(
        self, position: Tuple[float, float], level_data: Any
    ) -> bool:
        """Check if a position is walkable (has ground support)."""
        x, y = position

        # Convert to tile coordinates
        tile_x = int(x // 24)
        tile_y = int(y // 24)

        # Check if position is within level bounds
        if hasattr(level_data, "shape"):
            height, width = level_data.shape
            if tile_x < 0 or tile_x >= width or tile_y < 0 or tile_y >= height:
                return False

            # Check if current position is not solid
            if level_data[tile_y, tile_x] != 0:  # 0 = empty space
                return False

            # Check if there's ground support below
            if tile_y + 1 < height:
                return level_data[tile_y + 1, tile_x] != 0  # Has ground below

        return True  # Default to walkable if we can't determine

    def _can_reach_by_jumping(
        self,
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Any,
    ) -> bool:
        """Check if target position can be reached by jumping using physics simulation."""
        start_x, start_y = start_pos
        target_x, target_y = target_pos

        # Calculate required velocity for jump
        dx = target_x - start_x
        dy = target_y - start_y

        # Simple physics check - can we reach this with a reasonable jump?
        GRAVITY = 0.21
        MAX_JUMP_VELOCITY = 9.0
        MAX_HORIZONTAL_VELOCITY = 4.5

        # Check if horizontal distance is achievable
        if abs(dx) > MAX_HORIZONTAL_VELOCITY * 60:  # 60 frames max flight time
            return False

        # Check if vertical distance is achievable
        if dy > 0:  # Jumping down
            # Can always fall down (with some limits)
            if dy > 300:  # Max reasonable fall distance
                return False
        else:  # Jumping up
            # Check if we can jump high enough
            max_jump_height = (MAX_JUMP_VELOCITY * MAX_JUMP_VELOCITY) / (2 * GRAVITY)
            if abs(dy) > max_jump_height:
                return False

        # Simple collision check - ensure path is mostly clear
        return self._is_jump_path_clear(start_pos, target_pos, level_data)

    def _can_wall_jump_to(
        self,
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Any,
    ) -> bool:
        """Check if target position can be reached by wall jumping."""
        start_x, start_y = start_pos
        target_x, target_y = target_pos

        # Check if there's a wall nearby to jump off
        wall_positions = [
            (start_x - 24, start_y),  # Left wall
            (start_x + 24, start_y),  # Right wall
        ]

        has_wall = False
        for wall_pos in wall_positions:
            if self._is_position_solid(wall_pos, level_data):
                has_wall = True
                break

        if not has_wall:
            return False

        # Check if target is within wall jump range
        dx = abs(target_x - start_x)
        dy = abs(target_y - start_y)

        # Wall jump physics constraints
        if dx > 96 or dy > 168:  # Reasonable wall jump limits
            return False

        return self._is_jump_path_clear(start_pos, target_pos, level_data)

    def _can_fall_to(
        self,
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Any,
    ) -> bool:
        """Check if target position can be reached by falling."""
        start_x, start_y = start_pos
        target_x, target_y = target_pos

        # Can only fall down
        if target_y <= start_y:
            return False

        # Check if horizontal distance is reasonable for falling
        dx = abs(target_x - start_x)
        if dx > 120:  # Max horizontal drift while falling
            return False

        # Check if path is clear
        return self._is_fall_path_clear(start_pos, target_pos, level_data)

    def _assess_special_movement_capability(
        self,
        ninja_position: Tuple[float, float],
        level_data: Any,
        reachability_result: ReachabilityResult,
    ) -> float:
        """Assess special movement capabilities (launch pads, bounce blocks, etc.)."""
        # This would need entity information to properly assess
        # For now, return a basic assessment based on reachable positions
        ninja_x, ninja_y = ninja_position
        reachable_positions = reachability_result.reachable_positions

        # Check for positions that would require special movement
        special_positions = 0
        total_positions = 0

        # Look for positions that are far from ninja (indicating special movement)
        for pos in reachable_positions:
            distance = math.sqrt((pos[0] - ninja_x) ** 2 + (pos[1] - ninja_y) ** 2)
            if distance > 150:  # Far positions likely need special movement
                special_positions += 1
            total_positions += 1

        return special_positions / total_positions if total_positions > 0 else 0.0

    def _is_position_solid(
        self, position: Tuple[float, float], level_data: Any
    ) -> bool:
        """Check if a position contains solid terrain."""
        x, y = position
        tile_x = int(x // 24)
        tile_y = int(y // 24)

        if hasattr(level_data, "shape"):
            height, width = level_data.shape
            if 0 <= tile_x < width and 0 <= tile_y < height:
                return level_data[tile_y, tile_x] != 0

        return False

    def _is_jump_path_clear(
        self,
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Any,
    ) -> bool:
        """Check if jump path is reasonably clear of obstacles."""
        start_x, start_y = start_pos
        target_x, target_y = target_pos

        # Sample points along the jump arc
        num_samples = 5
        for i in range(1, num_samples):
            t = i / num_samples
            # Simple linear interpolation (could be improved with actual arc)
            sample_x = start_x + t * (target_x - start_x)
            sample_y = start_y + t * (target_y - start_y)

            if self._is_position_solid((sample_x, sample_y), level_data):
                return False

        return True

    def _is_fall_path_clear(
        self,
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Any,
    ) -> bool:
        """Check if fall path is clear of obstacles."""
        start_x, start_y = start_pos
        target_x, target_y = target_pos

        # Check vertical path with slight horizontal drift
        steps = int((target_y - start_y) // 12)  # Check every half tile

        for i in range(1, steps):
            t = i / steps
            sample_x = start_x + t * (target_x - start_x)
            sample_y = start_y + t * (target_y - start_y)

            if self._is_position_solid((sample_x, sample_y), level_data):
                return False

        return True

    def _estimate_level_complexity(self, level_data: Any, entities: List[Any]) -> float:
        """Estimate overall level complexity."""
        complexity = 0.0

        # Factor 1: Number of entities
        entity_count = len(entities)
        complexity += min(entity_count / 20.0, 0.3)  # Max 0.3 from entities

        # Factor 2: Switch count
        switch_count = len(
            [
                e
                for e in entities
                if hasattr(e, "entity_type") and "switch" in e.entity_type
            ]
        )
        complexity += min(switch_count / 10.0, 0.3)  # Max 0.3 from switches

        # Factor 3: Hazard count
        hazard_count = len(
            [
                e
                for e in entities
                if hasattr(e, "entity_type") and self._is_hazard_type(e.entity_type)
            ]
        )
        complexity += min(hazard_count / 15.0, 0.2)  # Max 0.2 from hazards

        # Factor 4: Level size
        if hasattr(level_data, "shape"):
            level_size = level_data.shape[0] * level_data.shape[1]
        else:
            level_size = getattr(level_data, "width", 100) * getattr(
                level_data, "height", 100
            )

        complexity += min(level_size / 2000.0, 0.2)  # Max 0.2 from size

        return min(complexity, 1.0)

    def _calculate_switch_complexity(self, entities: List[Any]) -> float:
        """Calculate switch dependency complexity."""
        switches = self._get_switches_from_entities(entities)

        if not switches:
            return 0.0

        # Simple complexity based on switch types
        complexity = 0.0
        for switch in switches:
            switch_type = switch.get("type", "")
            if "exit" in switch_type:
                complexity += 0.3
            elif "door" in switch_type:
                complexity += 0.2
            else:
                complexity += 0.1

        return min(complexity, 1.0)

    def _calculate_hazard_density(self, entities: List[Any], level_data: Any) -> float:
        """Calculate hazard density in the level."""
        hazard_count = len(
            [
                e
                for e in entities
                if hasattr(e, "entity_type") and self._is_hazard_type(e.entity_type)
            ]
        )

        if hasattr(level_data, "shape"):
            level_area = level_data.shape[0] * level_data.shape[1]
        else:
            level_area = getattr(level_data, "width", 100) * getattr(
                level_data, "height", 100
            )

        density = hazard_count / (level_area / 100.0)  # Hazards per 100 tiles
        return min(density, 1.0)

    def _debug_feature_summary(
        self, features: np.ndarray, ninja_position: Tuple[float, float]
    ):
        """Print debug summary of encoded features."""
        print(f"DEBUG: Feature encoding summary for ninja at {ninja_position}")
        print(f"  Objectives [0-7]: {features[0:8]}")
        print(f"  Switches [8-23]: {features[8:24]}")
        print(f"  Hazards [24-39]: {features[24:40]}")
        print(f"  Areas [40-47]: {features[40:48]}")
        print(f"  Movement [48-55]: {features[48:56]}")
        print(f"  Meta [56-63]: {features[56:64]}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Non-zero features: {np.count_nonzero(features)}/64")

    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.

        Returns:
            List of 64 feature names for debugging and analysis
        """
        names = []

        # Objective distances [0-7]
        for i in range(self.config.objective_slots):
            names.append(f"objective_distance_{i}")

        # Switch states [8-23]
        for i in range(self.config.switch_slots):
            names.append(f"switch_state_{i}")

        # Hazard proximities [24-39]
        for i in range(self.config.hazard_slots):
            names.append(f"hazard_proximity_{i}")

        # Area connectivity [40-47]
        area_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for i, area_name in enumerate(area_names):
            names.append(f"area_connectivity_{area_name}")

        # Movement capabilities [48-55]
        movement_names = [
            "walk_left",
            "walk_right",
            "jump_up",
            "jump_left",
            "jump_right",
            "wall_jump",
            "fall",
            "special",
        ]
        for movement_name in movement_names:
            names.append(f"movement_{movement_name}")

        # Meta features [56-63]
        meta_names = [
            "confidence",
            "computation_time",
            "level_complexity",
            "reachable_ratio",
            "switch_complexity",
            "hazard_density",
            "analysis_method",
            "cache_hit",
        ]
        for meta_name in meta_names:
            names.append(f"meta_{meta_name}")

        return names
