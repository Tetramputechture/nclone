"""
Optimized single-resolution graph builder for efficient RL training.

This module creates a single tile-level (24px) graph with hybrid static-dynamic
architecture. It builds the graph structure once per level and updates only
changed features when entity states change.

Key optimizations:
- Single resolution (tile-level) instead of 3 hierarchical levels
- 19-dimensional node features (down from 55)
- Hybrid updates: static topology, dynamic features
- Only critical entities: toggle mines, locked doors, exit switch/door
- NO physics pre-computation (agent learns from experience)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .common import (
    GraphData,
    StaticGraphStructure,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from .level_data import LevelData, ensure_level_data
from .edge_building import EdgeBuilder
from .feature_builder import NodeFeatureBuilder, EdgeFeatureBuilder
from .update_tracker import StateChangeDetector, GraphUpdateInfo
from .reachability.reachability_system import ReachabilitySystem
from ..constants.entity_types import EntityType


@dataclass
class HierarchicalGraphData:
    """
    Container for graph data (maintains interface compatibility).

    All resolution fields now point to the same single tile-level graph.
    """

    fine_graph: GraphData
    medium_graph: GraphData
    coarse_graph: GraphData
    reachability_info: Dict
    strategic_features: Dict

    @property
    def total_nodes(self) -> int:
        """Total nodes (same for all resolutions now)."""
        return self.fine_graph.num_nodes

    @property
    def total_edges(self) -> int:
        """Total edges (same for all resolutions now)."""
        return self.fine_graph.num_edges


class GraphBuilder:
    """
    Optimized single-resolution graph builder.

    Builds single tile-level (24px) graph with hybrid static-dynamic updates.
    Maintains HierarchicalGraphData interface but returns same graph for all resolutions.
    """

    def __init__(self, debug: bool = False):
        """Initialize graph builder."""
        self.debug = debug
        self.reachability_system = ReachabilitySystem(debug=debug)
        self.edge_builder = EdgeBuilder(
            debug=debug, reachability_system=self.reachability_system
        )
        self.node_builder = NodeFeatureBuilder()
        self.edge_builder_features = EdgeFeatureBuilder()
        self.state_detector = StateChangeDetector()

        # Cache for static structure
        self.static_structure: Optional[StaticGraphStructure] = None
        self.current_graph_data: Optional[GraphData] = None
        self.current_level_id: Optional[str] = None

    def build_graph(
        self, level_data, entities: List = None, ninja_pos: Tuple[int, int] = None
    ) -> HierarchicalGraphData:
        """
        Build graph using optimized hybrid system.

        Args:
            level_data: Complete level data (LevelData object preferred)
            entities: Optional entities list (for backward compatibility)
            ninja_pos: Optional ninja position (for backward compatibility)

        Returns:
            HierarchicalGraphData with all resolution fields pointing to same graph
        """
        level_data = ensure_level_data(level_data, ninja_pos, entities)

        # Check if we need to rebuild structure (new level)
        if self._is_new_level(level_data):
            if self.debug:
                print(
                    f"Building static graph structure for level {level_data.level_id}"
                )

            # Build static structure once
            self.static_structure = self.edge_builder.build_static_structure(level_data)
            self.current_level_id = level_data.level_id
            self.state_detector.reset()

            # Initialize features
            graph_data = self._initialize_features(level_data)
            self.current_graph_data = graph_data

            if self.debug:
                print(
                    f"  Initialized graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges"
                )
        else:
            # Selective update
            update_info = self.state_detector.detect_changes(
                level_data.entities,
                self.static_structure.entity_node_indices,
            )

            if update_info.updated_node_indices or update_info.connectivity_changed:
                graph_data = self._update_features(level_data, update_info)
                self.current_graph_data = graph_data

                if self.debug:
                    print(f"  Updated {len(update_info.updated_node_indices)} nodes")
            else:
                # No changes, return cached graph
                graph_data = self.current_graph_data

        # Extract strategic info
        reachability_info = self._extract_reachability_info(level_data)
        strategic_features = self._extract_strategic_features(level_data)

        # Return compatible format (all 3 resolutions point to same graph)
        return HierarchicalGraphData(
            fine_graph=graph_data,
            medium_graph=graph_data,  # Same graph
            coarse_graph=graph_data,  # Same graph
            reachability_info=reachability_info,
            strategic_features=strategic_features,
        )

    def _is_new_level(self, level_data: LevelData) -> bool:
        """Check if this is a new level requiring structure rebuild."""
        return (
            self.static_structure is None
            or self.current_level_id != level_data.level_id
        )

    def _initialize_features(self, level_data: LevelData) -> GraphData:
        """Initialize all node features from scratch."""
        # Initialize fixed-size arrays
        node_features = np.zeros((N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32)
        edge_features = np.zeros((E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.int32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.int32)

        # Get ninja and goal positions
        ninja_pos = level_data.player.position if level_data.player else None
        goal_pos = self._get_goal_position(level_data)

        # Run reachability analysis
        reachability_result = self._analyze_reachability(level_data)

        # Build features for each node
        for i in range(self.static_structure.num_nodes):
            node_pos = tuple(self.static_structure.node_positions[i])
            node_type = self.static_structure.node_types[i]
            tile_category = self.static_structure.tile_categories[i]

            # Check if node is reachable
            is_reachable = False
            if reachability_result:
                is_reachable = reachability_result.is_position_reachable(node_pos)

            # Get entity info if this is an entity node
            entity_info = self._get_entity_info_for_node(i, level_data)

            # Check if this is ninja's current node
            is_ninja_node = False
            if ninja_pos:
                ninja_tile_pos = self._pos_to_tile_center(ninja_pos)
                node_tile_pos = self._pos_to_tile_center(node_pos)
                is_ninja_node = ninja_tile_pos == node_tile_pos

            # Build node features
            node_features[i] = self.node_builder.build_node_features(
                node_pos=node_pos,
                node_type=node_type,
                tile_category=tile_category,
                entity_info=entity_info,
                reachability=is_reachable,
                ninja_pos=ninja_pos,
                goal_pos=goal_pos,
                is_ninja_node=is_ninja_node,
            )
            node_mask[i] = 1

        # Build edge features
        for i in range(self.static_structure.num_edges):
            edge_type = self.static_structure.edge_types[i]
            # Simple edge features (weight=1.0, reachability=1.0 for now)
            edge_features[i] = self.edge_builder_features.build_edge_features(
                edge_type=edge_type,
                weight=1.0,
                reachability_confidence=1.0,
            )
            edge_mask[i] = 1

        # Pad edge_index to fixed size
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_index[:, : self.static_structure.num_edges] = (
            self.static_structure.edge_index
        )

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=self.static_structure.node_types,
            edge_types=self.static_structure.edge_types,
            num_nodes=self.static_structure.num_nodes,
            num_edges=self.static_structure.num_edges,
        )

    def _update_features(
        self, level_data: LevelData, update_info: GraphUpdateInfo
    ) -> GraphData:
        """Selectively update only changed node features."""
        # Start with cached graph
        graph_data = self.current_graph_data

        # Get current ninja and goal positions
        ninja_pos = level_data.player.position if level_data.player else None
        goal_pos = self._get_goal_position(level_data)

        # Run reachability if connectivity changed
        reachability_result = None
        if update_info.connectivity_changed:
            reachability_result = self._analyze_reachability(level_data)

        # Update changed nodes
        for node_idx in update_info.updated_node_indices:
            if node_idx >= self.static_structure.num_nodes:
                continue

            node_pos = tuple(self.static_structure.node_positions[node_idx])
            node_type = self.static_structure.node_types[node_idx]
            tile_category = self.static_structure.tile_categories[node_idx]

            # Check if node is reachable
            is_reachable = False
            if reachability_result:
                is_reachable = reachability_result.is_position_reachable(node_pos)

            # Get updated entity info
            entity_info = self._get_entity_info_for_node(node_idx, level_data)

            # Check if this is ninja's current node
            is_ninja_node = False
            if ninja_pos:
                ninja_tile_pos = self._pos_to_tile_center(ninja_pos)
                node_tile_pos = self._pos_to_tile_center(node_pos)
                is_ninja_node = ninja_tile_pos == node_tile_pos

            # Rebuild node features
            graph_data.node_features[node_idx] = self.node_builder.build_node_features(
                node_pos=node_pos,
                node_type=node_type,
                tile_category=tile_category,
                entity_info=entity_info,
                reachability=is_reachable,
                ninja_pos=ninja_pos,
                goal_pos=goal_pos,
                is_ninja_node=is_ninja_node,
            )

        return graph_data

    def _get_entity_info_for_node(
        self, node_idx: int, level_data: LevelData
    ) -> Optional[Dict]:
        """Get entity info for a node if it's an entity node."""
        # Find entity_id for this node
        entity_id = None
        for eid, nidx in self.static_structure.entity_node_indices.items():
            if nidx == node_idx:
                entity_id = eid
                break

        if not entity_id:
            return None

        # Find entity in level_data
        for entity in level_data.entities:
            if entity.get("entity_id") == entity_id:
                return entity

        return None

    def _get_goal_position(
        self, level_data: LevelData
    ) -> Optional[Tuple[float, float]]:
        """Get goal position (exit door or exit switch)."""
        # Look for exit switch first, then exit door
        for entity in level_data.entities:
            entity_type = entity.get("type")
            if entity_type == EntityType.EXIT_SWITCH:
                return (entity.get("x", 0), entity.get("y", 0))

        for entity in level_data.entities:
            entity_type = entity.get("type")
            if entity_type == EntityType.EXIT_DOOR:
                return (entity.get("x", 0), entity.get("y", 0))

        return None

    def _pos_to_tile_center(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert position to tile center coordinates."""
        from ..constants.physics_constants import TILE_PIXEL_SIZE

        x, y = pos
        tile_x = int(x // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        tile_y = int(y // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        return (tile_x, tile_y)

    def _analyze_reachability(self, level_data: LevelData):
        """Run reachability analysis."""
        ninja_pos = level_data.player.position if level_data.player else (0, 0)
        switch_states = getattr(level_data, "switch_states", {})

        try:
            return self.reachability_system.analyze_reachability(
                level_data=level_data,
                ninja_position=ninja_pos,
                switch_states=switch_states,
            )
        except Exception as e:
            if self.debug:
                print(f"Reachability analysis failed: {e}")
            return None

    def _extract_reachability_info(self, level_data: LevelData) -> Dict:
        """Extract reachability information for strategic planning."""
        ninja_pos = level_data.player.position if level_data.player else (0, 0)
        switch_states = getattr(level_data, "switch_states", {})

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
                "connectivity_score": min(reachable_count / 1000.0, 1.0),
                "computation_time_ms": result.computation_time_ms,
                "confidence": result.confidence,
                "method": result.method,
            }
        except Exception as e:
            if self.debug:
                print(f"Reachability analysis failed: {e}")
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
        """Extract strategic features for RL decision making."""
        features = {}

        # Get player position
        player_pos = level_data.player.position if level_data.player else (0, 0)

        # Entity counts by type
        entity_counts = {}
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        features["entity_counts"] = entity_counts

        # Distance to key entities
        distances = {}
        key_entity_types = [
            EntityType.EXIT_DOOR,
            EntityType.EXIT_SWITCH,
            EntityType.LOCKED_DOOR,
        ]

        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            if entity_type in key_entity_types:
                entity_x = entity.get("x", 0)
                entity_y = entity.get("y", 0)
                dist = np.sqrt(
                    (entity_x - player_pos[0]) ** 2 + (entity_y - player_pos[1]) ** 2
                )
                if entity_type not in distances:
                    distances[entity_type] = []
                distances[entity_type].append(dist)

        # Take minimum distance for each type
        for entity_type, dist_list in distances.items():
            features[f"min_distance_to_type_{entity_type}"] = (
                min(dist_list) if dist_list else float("inf")
            )

        # Level complexity metrics
        features["level_width"] = level_data.width
        features["level_height"] = level_data.height
        features["total_entities"] = len(level_data.entities)
        features["wall_density"] = np.mean(level_data.tiles > 0)

        return features


# Maintain backward compatibility alias
HierarchicalGraphBuilder = GraphBuilder
