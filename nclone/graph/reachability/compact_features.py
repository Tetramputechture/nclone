"""
Simplified compact reachability features for RL integration.

This module implements simplified 8-dimensional feature encoding that relies on
the fast OpenCV flood fill reachability system. It removes complex physics
calculations and focuses on strategic connectivity information.

Feature Vector Layout (8 dimensions):
[0]:   Reachable area ratio (0-1)
[1]:   Current objective distance (normalized)
[2]:   Switch accessibility (0-1)
[3]:   Exit accessibility (0-1)
[4]:   Hazard proximity (0-1)
[5]:   Connectivity score (0-1)
[6]:   Analysis confidence (0-1)
[7]:   Computation time (normalized)
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .reachability_types import ReachabilityApproximation
from ...planning import LevelCompletionPlanner


@dataclass
class FeatureConfig:
    """Configuration for simplified feature encoding."""

    total_features: int = 8


class CompactReachabilityFeatures:
    """
    Simplified encoding of reachability analysis for RL integration.

    This class implements 8-dimensional feature encoding that relies on the fast
    OpenCV flood fill reachability system. It removes complex physics calculations
    and focuses on strategic connectivity information that helps RL agents make
    decisions while letting them learn movement patterns emergently.
    """

    def __init__(self, config: Optional[FeatureConfig] = None, debug: bool = False):
        """
        Initialize simplified feature encoder.

        Args:
            config: Feature configuration (uses default if None)
            debug: Enable debug output
        """
        self.config = config or FeatureConfig()
        self.debug = debug
        self.completion_planner = LevelCompletionPlanner()

        # Feature dimension validation
        if self.config.total_features != 8:
            raise ValueError(f"Expected 8 features, got {self.config.total_features}")

        # Simple cache for level diagonal
        self._level_diagonal_cache = {}

    def encode_reachability(
        self,
        reachability_result: ReachabilityApproximation,
        level_data: Any,
        entities: List[Any],
        ninja_position: Tuple[float, float],
        switch_states: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """
        Encode reachability analysis into simplified 8-dimensional feature vector.

        This is the main entry point for simplified feature extraction. It uses only
        the fast flood fill results to create strategic connectivity features.

        Args:
            reachability_result: Result from fast flood fill analysis
            level_data: Level tile data and structure
            entities: List of game entities (switches, doors, hazards, etc.)
            ninja_position: Current ninja position (x, y)
            switch_states: Current switch activation states

        Returns:
            8-dimensional numpy array with encoded features
        """
        if switch_states is None:
            switch_states = {}

        # Initialize simplified feature vector
        features = np.zeros(self.config.total_features, dtype=np.float32)

        # [0] Reachable area ratio
        features[0] = self._calculate_reachable_area_ratio(reachability_result, level_data)

        # [1] Current objective distance
        features[1] = self._calculate_objective_distance(
            reachability_result, level_data, entities, ninja_position, switch_states
        )

        # [2] Switch accessibility
        features[2] = self._calculate_switch_accessibility(
            reachability_result, entities, switch_states
        )

        # [3] Exit accessibility
        features[3] = self._calculate_exit_accessibility(
            reachability_result, entities
        )

        # [4] Hazard proximity
        features[4] = self._calculate_hazard_proximity(
            reachability_result, entities, ninja_position
        )

        # [5] Connectivity score
        features[5] = self._calculate_connectivity_score(reachability_result)

        # [6] Analysis confidence
        features[6] = getattr(reachability_result, "confidence", 1.0)

        # [7] Computation time (normalized)
        computation_time = getattr(reachability_result, "computation_time_ms", 1.0)
        features[7] = min(math.log(1 + computation_time) / math.log(10), 1.0)

        if self.debug:
            self._debug_feature_summary(features, ninja_position)

        return features

    def _calculate_reachable_area_ratio(
        self, reachability_result: ReachabilityApproximation, level_data: Any
    ) -> float:
        """Calculate the ratio of reachable area to total traversable area."""
        reachable_count = len(reachability_result.reachable_positions)
        
        # Estimate total traversable area
        tiles = level_data.tiles if hasattr(level_data, 'tiles') else level_data
        if hasattr(tiles, "shape"):
            total_area = tiles.shape[0] * tiles.shape[1]
            # Rough estimate: assume 60% of tiles are traversable
            traversable_area = total_area * 0.6
        else:
            traversable_area = 1000  # Default fallback
        
        return min(reachable_count / traversable_area, 1.0)

    def _calculate_objective_distance(
        self,
        reachability_result: ReachabilityApproximation,
        level_data: Any,
        entities: List[Any],
        ninja_position: Tuple[float, float],
        switch_states: Dict[str, bool],
    ) -> float:
        """Calculate normalized distance to current objective."""
        # Get current objective from completion planner
        current_objective = self.completion_planner.get_next_objective(
            ninja_position, level_data, entities, switch_states
        )
        
        if not current_objective:
            return 1.0  # No objective found
        
        # Check if objective is reachable
        if not self._is_position_reachable(current_objective.position, reachability_result):
            return 1.0  # Unreachable
        
        # Calculate normalized distance
        level_diagonal = self._get_level_diagonal(level_data)
        normalized_distance = current_objective.distance / level_diagonal
        return min(normalized_distance, 1.0)

    def _calculate_switch_accessibility(
        self,
        reachability_result: ReachabilityApproximation,
        entities: List[Any],
        switch_states: Dict[str, bool],
    ) -> float:
        """Calculate accessibility of important switches."""
        switches = self._get_switches_from_entities(entities)
        if not switches:
            return 1.0  # No switches to worry about
        
        accessible_switches = 0
        important_switches = 0
        
        for switch_info in switches:
            switch_id = switch_info.get("id")
            position = switch_info.get("position")
            
            # Check if this is an important switch (not already activated)
            if switch_id and not switch_states.get(switch_id, False):
                important_switches += 1
                if position and self._is_position_reachable(position, reachability_result):
                    accessible_switches += 1
        
        if important_switches == 0:
            return 1.0  # All switches activated
        
        return accessible_switches / important_switches

    def _calculate_exit_accessibility(
        self, reachability_result: ReachabilityApproximation, entities: List[Any]
    ) -> float:
        """Calculate accessibility of exit."""
        # Find exit entities
        exits = [e for e in entities if hasattr(e, "entity_type") and "exit" in str(e.entity_type).lower()]
        
        if not exits:
            return 0.0  # No exit found
        
        # Check if any exit is reachable
        for exit_entity in exits:
            if hasattr(exit_entity, "x") and hasattr(exit_entity, "y"):
                exit_pos = (exit_entity.x, exit_entity.y)
            else:
                exit_pos = (exit_entity.get("x", 0), exit_entity.get("y", 0))
            
            if self._is_position_reachable(exit_pos, reachability_result):
                return 1.0  # Exit is reachable
        
        return 0.0  # No reachable exit

    def _calculate_hazard_proximity(
        self,
        reachability_result: ReachabilityApproximation,
        entities: List[Any],
        ninja_position: Tuple[float, float],
    ) -> float:
        """
        Calculate proximity to nearest hazard with mine state awareness.
        
        For toggle mines, only considers dangerous states (toggled/toggling).
        """
        hazards = [e for e in entities if hasattr(e, "entity_type") and self._is_hazard_type(str(e.entity_type))]
        
        if not hazards:
            return 0.0  # No hazards
        
        min_distance = float('inf')
        for hazard in hazards:
            if hasattr(hazard, "x") and hasattr(hazard, "y"):
                hazard_pos = (hazard.x, hazard.y)
            else:
                hazard_pos = (hazard.get("x", 0), hazard.get("y", 0))
            
            # For toggle mines, check state - only consider dangerous states
            if self._is_mine_entity(hazard):
                mine_state = getattr(hazard, 'state', 1)  # Default to untoggled (safe)
                # Skip if mine is untoggled (safe state)
                if mine_state == 1:  # 1 = untoggled = safe
                    continue
            
            # Only consider reachable hazards as threats
            if self._is_position_reachable(hazard_pos, reachability_result):
                distance = self._calculate_distance(ninja_position, hazard_pos)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0  # No reachable hazards
        
        # Convert to proximity score (closer = higher score)
        threat_radius = 72.0  # 3 tiles
        if min_distance <= threat_radius:
            return 1.0 - (min_distance / threat_radius)
        else:
            return 0.0

    def _calculate_connectivity_score(
        self, reachability_result: ReachabilityApproximation
    ) -> float:
        """Calculate overall connectivity score based on reachable positions."""
        reachable_count = len(reachability_result.reachable_positions)
        
        # Simple connectivity score based on number of reachable positions
        # More reachable positions = better connectivity
        if reachable_count < 100:
            return reachable_count / 100.0
        else:
            return 1.0

    # Helper methods (simplified versions)
    
    def _get_level_diagonal(self, level_data: Any) -> float:
        """Get level diagonal for distance normalization."""
        tiles = level_data.tiles if hasattr(level_data, 'tiles') else level_data
        if hasattr(tiles, "shape"):
            height, width = tiles.shape
        else:
            width = getattr(level_data, "width", 42)
            height = getattr(level_data, "height", 23)

        # Convert to pixels (24 pixels per tile)
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
        self, position: Tuple[float, float], reachability_result: ReachabilityApproximation
    ) -> bool:
        """Check if a position is reachable according to reachability analysis."""
        # Convert position to tile coordinates for comparison
        tile_x = int(position[0] // 24)
        tile_y = int(position[1] // 24)
        tile_center = (tile_x * 24 + 12, tile_y * 24 + 12)
        return tile_center in reachability_result.reachable_positions

    def _get_switches_from_entities(self, entities: List[Any]) -> List[Dict[str, Any]]:
        """Extract switch information from entities."""
        switches = []
        for entity in entities:
            if hasattr(entity, "entity_type") and "switch" in str(entity.entity_type).lower():
                if hasattr(entity, "x") and hasattr(entity, "y"):
                    position = (entity.x, entity.y)
                    entity_id = getattr(entity, "id", None)
                else:
                    position = (entity.get("x", 0), entity.get("y", 0))
                    entity_id = entity.get("id", None)
                
                switches.append({
                    "position": position,
                    "type": str(entity.entity_type),
                    "id": entity_id,
                    "entity": entity,
                })
        return switches

    def _is_hazard_type(self, entity_type: str) -> bool:
        """Check if entity type represents a hazard."""
        hazard_types = ["drone", "mine", "thwump", "laser", "rocket", "chaingun"]
        return any(hazard in entity_type.lower() for hazard in hazard_types)
    
    def _is_mine_entity(self, entity: Any) -> bool:
        """Check if entity is a toggle mine."""
        if not hasattr(entity, 'entity_type'):
            return False
        entity_type_str = str(entity.entity_type).lower()
        return "mine" in entity_type_str

    def _debug_feature_summary(
        self, features: np.ndarray, ninja_position: Tuple[float, float]
    ):
        """Print debug summary of encoded features."""
        print(f"DEBUG: Simplified feature encoding for ninja at {ninja_position}")
        print(f"  [0] Reachable area ratio: {features[0]:.3f}")
        print(f"  [1] Objective distance: {features[1]:.3f}")
        print(f"  [2] Switch accessibility: {features[2]:.3f}")
        print(f"  [3] Exit accessibility: {features[3]:.3f}")
        print(f"  [4] Hazard proximity: {features[4]:.3f}")
        print(f"  [5] Connectivity score: {features[5]:.3f}")
        print(f"  [6] Analysis confidence: {features[6]:.3f}")
        print(f"  [7] Computation time: {features[7]:.3f}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")

    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.

        Returns:
            List of 8 feature names for debugging and analysis
        """
        return [
            "reachable_area_ratio",
            "objective_distance",
            "switch_accessibility", 
            "exit_accessibility",
            "hazard_proximity",
            "connectivity_score",
            "analysis_confidence",
            "computation_time"
        ]

