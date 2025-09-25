"""
Hierarchical graph builder for multi-resolution N++ level processing.

This module creates hierarchical graph representations with multiple resolution levels
optimized for Deep RL. It focuses on strategic information rather than detailed
physics calculations, designed to work with heterogeneous graph transformers,
3D CNNs, and MLPs.

This is the simplified, production-ready version that replaces the complex
physics-based approach with connectivity-focused strategic information.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .common import GraphData
from .level_data import LevelData, PlayerState, ensure_level_data
from .edge_building import EdgeBuilder, create_simplified_graph_data
from .reachability.reachability_system import ReachabilitySystem


class ResolutionLevel(Enum):
    """Multi-resolution levels for hierarchical analysis."""

    FINE = "6px"  # 6px resolution (detailed local movement)
    MEDIUM = "24px"  # 24px resolution (tile-level planning)
    COARSE = "96px"  # 96px resolution (strategic overview)


@dataclass
class HierarchicalGraphData:
    """Container for multi-resolution simplified graph data."""

    fine_graph: GraphData
    medium_graph: GraphData
    coarse_graph: GraphData
    reachability_info: Dict
    strategic_features: Dict

    @property
    def total_nodes(self) -> int:
        """Total nodes across all resolution levels."""
        return (
            self.fine_graph.num_nodes
            + self.medium_graph.num_nodes
            + self.coarse_graph.num_nodes
        )

    @property
    def total_edges(self) -> int:
        """Total edges across all resolution levels."""
        return (
            self.fine_graph.num_edges
            + self.medium_graph.num_edges
            + self.coarse_graph.num_edges
        )


class HierarchicalGraphBuilder:
    """
    Simplified hierarchical graph builder for Deep RL.

    This builder creates multi-resolution graphs that focus on strategic
    information rather than detailed physics. It's optimized for:
    - Heterogeneous Graph Transformers (structural relationships)
    - 3D CNNs (spatial patterns)
    - MLPs (action decisions)
    - Reward-guided learning (strategic optimization)
    """

    def __init__(self, debug: bool = False):
        """Initialize simplified hierarchical builder."""
        self.debug = debug
        self.edge_builder = EdgeBuilder(debug=debug)
        self.reachability_system = ReachabilitySystem(debug=debug)
        # self.edge_builder.set_reachability_system(self.reachability_system)

    def build_graph(
        self, level_data, entities: List = None, ninja_pos: Tuple[int, int] = None
    ) -> HierarchicalGraphData:
        """
        Build simplified hierarchical graph for strategic RL.

        Args:
            level_data: Complete level data (LevelData object preferred)
            entities: Optional entities list (for backward compatibility)
            ninja_pos: Optional ninja position (for backward compatibility)

        Returns:
            HierarchicalGraphData with multi-resolution graphs
        """
        # Consolidate all data into LevelData object
        level_data = ensure_level_data(level_data, ninja_pos, entities)

        if self.debug:
            print("Building simplified hierarchical graph...")
            print(f"  Level size: {level_data.width}x{level_data.height}")
            print(f"  Entities: {len(level_data.entities)}")
            if level_data.player:
                print(
                    f"  Player position: ({level_data.player.x:.1f}, {level_data.player.y:.1f})"
                )

        # Build graphs at different resolutions using consolidated data
        fine_graph = self._build_resolution_graph(level_data, ResolutionLevel.FINE)
        medium_graph = self._build_resolution_graph(level_data, ResolutionLevel.MEDIUM)
        coarse_graph = self._build_resolution_graph(level_data, ResolutionLevel.COARSE)

        # Extract reachability information using consolidated data
        reachability_info = self._extract_reachability_info(level_data)

        # Extract strategic features using consolidated data
        strategic_features = self._extract_strategic_features(level_data)

        result = HierarchicalGraphData(
            fine_graph=fine_graph,
            medium_graph=medium_graph,
            coarse_graph=coarse_graph,
            reachability_info=reachability_info,
            strategic_features=strategic_features,
        )

        if self.debug:
            print("Built hierarchical graph:")
            print(f"  Fine: {fine_graph.num_nodes} nodes, {fine_graph.num_edges} edges")
            print(
                f"  Medium: {medium_graph.num_nodes} nodes, {medium_graph.num_edges} edges"
            )
            print(
                f"  Coarse: {coarse_graph.num_nodes} nodes, {coarse_graph.num_edges} edges"
            )
            print(f"  Total: {result.total_nodes} nodes, {result.total_edges} edges")

        return result

    def _build_resolution_graph(
        self, level_data: LevelData, resolution: ResolutionLevel
    ) -> GraphData:
        """Build graph at specific resolution level using consolidated data."""

        # Downsample level data based on resolution (includes player position)
        downsampled_level = self._downsample_level_data(level_data, resolution)

        # Build edges using consolidated data approach
        edges = self.edge_builder.build_edges(downsampled_level)

        # Convert to GraphData
        graph_data = create_simplified_graph_data(edges, downsampled_level)

        return graph_data

    def _downsample_level_data(
        self, level_data: LevelData, resolution: ResolutionLevel
    ) -> LevelData:
        """Downsample level data to target resolution, preserving player state."""

        # Filter entities based on resolution
        filtered_entities = self._filter_entities_for_resolution(
            level_data.entities, resolution
        )

        # Adjust player position for resolution
        adjusted_player = None
        if level_data.player:
            adjusted_pos = self._adjust_position_for_resolution(
                level_data.player.position, resolution
            )
            adjusted_player = PlayerState(
                position=adjusted_pos,
                velocity=level_data.player.velocity,
                on_ground=level_data.player.on_ground,
                facing_right=level_data.player.facing_right,
                health=level_data.player.health,
                frame=level_data.player.frame,
            )

        if resolution == ResolutionLevel.FINE:
            # 6px resolution - higher detail than tile level
            scale_factor = 4  # 24px / 6px = 4
            upsampled_tiles = np.repeat(
                np.repeat(level_data.tiles, scale_factor, axis=0), scale_factor, axis=1
            )
            return LevelData(
                tiles=upsampled_tiles,
                entities=filtered_entities,
                player=adjusted_player,
                level_id=level_data.level_id,
                metadata=level_data.metadata,
                switch_states=level_data.switch_states,
            )

        elif resolution == ResolutionLevel.MEDIUM:
            # 24px resolution - tile level (no change needed for tiles)
            return LevelData(
                tiles=level_data.tiles,
                entities=filtered_entities,
                player=adjusted_player,
                level_id=level_data.level_id,
                metadata=level_data.metadata,
                switch_states=level_data.switch_states,
            )

        elif resolution == ResolutionLevel.COARSE:
            # 96px resolution - 4x4 tile blocks
            scale_factor = 4  # 96px / 24px = 4
            new_width = level_data.width // scale_factor
            new_height = level_data.height // scale_factor

            # Downsample by taking max value in each block (wall if any wall present)
            downsampled_tiles = np.zeros(
                (new_height, new_width), dtype=level_data.tiles.dtype
            )

            for row in range(new_height):
                for col in range(new_width):
                    block = level_data.tiles[
                        row * scale_factor : (row + 1) * scale_factor,
                        col * scale_factor : (col + 1) * scale_factor,
                    ]
                    # Take max value (wall takes precedence over empty)
                    downsampled_tiles[row, col] = np.max(block)

            return LevelData(
                tiles=downsampled_tiles,
                entities=filtered_entities,
                player=adjusted_player,
                level_id=level_data.level_id,
                metadata=level_data.metadata,
                switch_states=level_data.switch_states,
            )

    def _filter_entities_for_resolution(
        self, entities: List, resolution: ResolutionLevel
    ) -> List:
        """Filter entities based on resolution level."""
        if resolution == ResolutionLevel.FINE:
            # Include all entities for detailed analysis
            return entities

        elif resolution == ResolutionLevel.MEDIUM:
            # Include important entities (switches, doors, exit)
            important_types = ["switch", "door", "exit", "spawn"]
            return [
                e
                for e in entities
                if any(t in type(e).__name__.lower() for t in important_types)
            ]

        elif resolution == ResolutionLevel.COARSE:
            # Include only strategic entities (exit, major switches)
            strategic_types = ["exit", "spawn"]
            return [
                e
                for e in entities
                if any(t in type(e).__name__.lower() for t in strategic_types)
            ]

    def _adjust_position_for_resolution(
        self, pos: Tuple[float, float], resolution: ResolutionLevel
    ) -> Tuple[float, float]:
        """Adjust position coordinates for target resolution."""
        x, y = pos

        if resolution == ResolutionLevel.FINE:
            # 6px resolution - increase precision
            return (x * 4, y * 4)  # 24px -> 6px

        elif resolution == ResolutionLevel.MEDIUM:
            # 24px resolution - no change
            return pos

        elif resolution == ResolutionLevel.COARSE:
            # 96px resolution - decrease precision
            return (x / 4, y / 4)  # 24px -> 96px

    def _extract_reachability_info(self, level_data: LevelData) -> Dict:
        """Extract reachability information for strategic planning."""

        # Get reachability analysis from the simplified ultra-fast system
        ninja_pos = level_data.player.position if level_data.player else (0, 0)
        switch_states = level_data.switch_states if level_data.switch_states else {}

        try:
            result = self.reachability_system.analyze_reachability(
                level_data, ninja_pos, switch_states
            )

            reachable_count = result.get_reachable_count()
            is_completable = result.is_level_completable()

            return {
                "reachable_count": reachable_count,
                "total_reachable_area": float(reachable_count),
                "is_level_completable": is_completable,
                "connectivity_score": min(reachable_count / 1000.0, 1.0),  # Normalize
                "computation_time_ms": result.computation_time_ms,
                "confidence": result.confidence,
                "method": result.method,
            }
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Reachability analysis failed: {e}")
            return {
                "reachable_count": 0,
                "total_reachable_area": 0.0,
                "is_level_completable": False,
                "connectivity_score": 0.0,
                "computation_time_ms": 0.0,
                "confidence": 0.0,
                "method": "failed",
            }

    def _extract_strategic_features(self, level_data: LevelData) -> Dict:
        """Extract strategic features for RL decision making using consolidated data."""
        features = {}

        # Get player position
        player_pos = level_data.player.position if level_data.player else (0, 0)

        # Entity counts by type using EntityType constants
        from nclone.constants.entity_types import EntityType

        entity_counts = {}
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        features["entity_counts"] = entity_counts

        # Distance to key entities with switch proximity logic
        distances = {}
        key_entity_types = [
            EntityType.EXIT_DOOR,
            EntityType.EXIT_SWITCH,
            EntityType.LOCKED_DOOR,
        ]

        # Check if locked doors are present for switch proximity logic
        locked_doors_present = any(
            entity.get("type", 0) == EntityType.LOCKED_DOOR
            for entity in level_data.entities
        )

        # Always initialize switch_distances for safety
        switch_distances = []

        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)

            # Calculate distance for key entity types
            if entity_type in key_entity_types:
                dist = np.sqrt(
                    (entity_x - player_pos[0]) ** 2 + (entity_y - player_pos[1]) ** 2
                )
                if entity_type not in distances:
                    distances[entity_type] = []
                distances[entity_type].append(dist)

                # For locked doors, also check if this is the switch part
                if entity_type == EntityType.LOCKED_DOOR and not entity.get(
                    "is_door_part", False
                ):
                    # This is the switch part of a locked door
                    switch_distances.append(dist)

            # Track other switch types when locked doors are present
            elif locked_doors_present and entity_type in [
                EntityType.EXIT_SWITCH,
                EntityType.TRAP_DOOR,
            ]:
                # Include exit switches and trap door switches in switch proximity
                if not entity.get(
                    "is_door_part", False
                ):  # Only switch parts, not door parts
                    dist = np.sqrt(
                        (entity_x - player_pos[0]) ** 2
                        + (entity_y - player_pos[1]) ** 2
                    )
                    switch_distances.append(dist)

        # Take minimum distance for each type
        for entity_type, dist_list in distances.items():
            features[f"min_distance_to_type_{entity_type}"] = (
                min(dist_list) if dist_list else float("inf")
            )

        # Add switch proximity features when locked doors are present
        if locked_doors_present and switch_distances:
            features["min_distance_to_nearest_switch"] = min(switch_distances)
            features["switch_count_near_locked_doors"] = len(switch_distances)
        else:
            features["min_distance_to_nearest_switch"] = float("inf")
            features["switch_count_near_locked_doors"] = 0

        # Level complexity metrics
        features["level_width"] = level_data.width
        features["level_height"] = level_data.height
        features["total_entities"] = len(level_data.entities)
        features["wall_density"] = np.mean(level_data.tiles > 0)  # Assuming 0 is empty

        return features
