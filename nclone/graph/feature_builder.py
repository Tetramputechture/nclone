"""
Optimized feature builder for graph nodes and edges.

This module creates compact 19-dimensional node features and 6-dimensional
edge features for efficient deep RL with GNNs. Features are designed to provide
essential information for level completion with minimal overhead.

Key design principles:
- Spatial information from positions and distances
- Entity state tracking (toggle mines, locked doors, exit switch/door)
- Reachability from flood-fill analysis (not physics simulation)
- Compact tile category encoding (3 categories instead of 38-dim one-hot)
- Agent learns movement dynamics from temporal frames and game state
- NO physics pre-computation (too complex for continuous movement)

Supported critical entities:
- Toggle mines (types 1, 21): state-dependent hazards
- Locked doors (type 6): gated progression
- Exit switch (type 4): primary objective
- Exit door (type 3): level completion

Feature dimensions: 19 total (optimized from 55)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..constants.entity_types import EntityType
from ..constants.physics_constants import (
    FULL_MAP_WIDTH_PX,
    FULL_MAP_HEIGHT_PX,
    NINJA_RADIUS,
)
from .common import NodeType, EdgeType, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


class NodeFeatureBuilder:
    """
    Builds compact 19-dimensional node feature vectors.

    Feature breakdown:
    - Spatial (2): position (x, y)
    - Type (6): one-hot node type
    - Tile Category (3): compact tile encoding (empty, solid, navigable)
    - Entity (4): entity-specific information (type, state, active, radius)
    - Reachability (1): from reachability system
    - Proximity (3): distances to key points + ninja node flag

    Total: 2 + 6 + 3 + 4 + 1 + 3 = 19 features
    """

    def __init__(self):
        """Initialize the node feature builder."""
        self.screen_diagonal = np.sqrt(FULL_MAP_WIDTH_PX**2 + FULL_MAP_HEIGHT_PX**2)

        # Feature index boundaries
        self.SPATIAL_START = 0
        self.SPATIAL_END = 2
        self.TYPE_START = 2
        self.TYPE_END = 8
        self.TILE_CAT_START = 8
        self.TILE_CAT_END = 11
        self.ENTITY_START = 11
        self.ENTITY_END = 15
        self.REACHABILITY_START = 15
        self.REACHABILITY_END = 16
        self.PROXIMITY_START = 16
        self.PROXIMITY_END = 19

        assert self.PROXIMITY_END == NODE_FEATURE_DIM, "Feature dimensions mismatch!"

    def build_node_features(
        self,
        node_pos: Tuple[float, float],
        node_type: NodeType,
        tile_category: int = 0,  # 0=empty, 1=solid, 2=navigable_complex
        entity_info: Optional[Dict[str, Any]] = None,
        reachability: bool = False,
        ninja_pos: Optional[Tuple[float, float]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
        is_ninja_node: bool = False,
    ) -> np.ndarray:
        """
        Build compact 19-dimensional node features.

        Args:
            node_pos: (x, y) position of node in pixels
            node_type: NodeType enum value (EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT)
            tile_category: Tile category (0=empty, 1=solid, 2=navigable slopes/curves)
            entity_info: Dict with entity-specific information (type, state, active, radius)
            reachability: Boolean flag indicating if reachable from ninja
            ninja_pos: Current ninja position for distance calculation
            goal_pos: Current goal position (switch or exit) for distance
            is_ninja_node: Boolean flag indicating if this is the ninja's current node

        Returns:
            np.ndarray of shape (19,) with normalized features
        """
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

        # ===== Spatial Features (0-1) =====
        features[0] = np.clip(node_pos[0] / FULL_MAP_WIDTH_PX, 0.0, 1.0)
        features[1] = np.clip(node_pos[1] / FULL_MAP_HEIGHT_PX, 0.0, 1.0)

        # ===== Type Encoding (2-7) =====
        # [EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT]
        if 0 <= node_type < 6:
            features[self.TYPE_START + int(node_type)] = 1.0

        # ===== Tile Category (8-10) =====
        if 0 <= tile_category < 3:
            features[self.TILE_CAT_START + tile_category] = 1.0

        # ===== Entity-Specific Features (11-14) =====
        if entity_info is not None:
            self._add_entity_features(features, entity_info)

        # ===== Reachability (15) =====
        features[self.REACHABILITY_START] = 1.0 if reachability else 0.0

        # ===== Proximity Features (16-18) =====
        if ninja_pos is not None:
            dist = np.linalg.norm(np.array(node_pos) - np.array(ninja_pos))
            features[self.PROXIMITY_START] = min(dist / self.screen_diagonal, 1.0)

        if goal_pos is not None:
            dist = np.linalg.norm(np.array(node_pos) - np.array(goal_pos))
            features[self.PROXIMITY_START + 1] = min(dist / self.screen_diagonal, 1.0)

        features[self.PROXIMITY_START + 2] = 1.0 if is_ninja_node else 0.0

        return features

    def _add_entity_features(self, features: np.ndarray, entity_info: Dict[str, Any]):
        """
        Add entity-specific features (indices 11-14, 4 features).

        Entity types from entity files:
        - Toggle Mine (types 1, 21): state (0=toggled/deadly, 1=untoggled/safe, 2=toggling), xpos, ypos, RADIUS
        - Locked Door (type 6): xpos, ypos, active, closed, RADIUS
        - Exit Switch (type 4): xpos, ypos, active, RADIUS
        - Exit Door (type 3): xpos, ypos, switch_hit, RADIUS

        Args:
            features: Feature array to modify
            entity_info: Dict with entity-specific information
        """
        entity_type = entity_info.get("type", 0)

        # Index 11: entity_type encoding
        type_encoding = 0.0
        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            type_encoding = 0.25  # toggle mine
        elif entity_type == EntityType.EXIT_SWITCH:
            type_encoding = 0.5  # exit switch
        elif entity_type == EntityType.LOCKED_DOOR:
            type_encoding = 0.75  # locked door
        elif entity_type == EntityType.EXIT_DOOR:
            type_encoding = 1.0  # exit door
        features[self.ENTITY_START] = type_encoding

        # Index 12: entity state (normalized)
        entity_state = 0.0
        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            # Toggle mine state: 0/1/2 → 0.0/0.5/1.0
            mine_state = entity_info.get("state", 1)
            entity_state = mine_state / 2.0
        elif entity_type == EntityType.LOCKED_DOOR:
            # Locked door: closed → 1.0, open → 0.0
            entity_state = 1.0 if entity_info.get("closed", True) else 0.0
        elif entity_type == EntityType.EXIT_DOOR:
            # Exit door: switch_hit → 0.0 (inactive), not hit → 1.0 (active)
            entity_state = 0.0 if entity_info.get("switch_hit", False) else 1.0
        features[self.ENTITY_START + 1] = entity_state

        # Index 13: entity active
        features[self.ENTITY_START + 2] = (
            1.0 if entity_info.get("active", True) else 0.0
        )

        # Index 14: entity radius (normalized)
        entity_radius = entity_info.get("radius", 0.0)
        max_radius = NINJA_RADIUS * 2
        features[self.ENTITY_START + 3] = np.clip(entity_radius / max_radius, 0.0, 1.0)


class EdgeFeatureBuilder:
    """
    Builds comprehensive 6-dimensional edge feature vectors.

    Feature breakdown:
    - Edge type (4): one-hot encoding
    - Connectivity (2): weight and reachability confidence

    Total: 4 + 2 = 6 features

    Note: Movement requirements (jump/walljump/momentum) are NOT included
    as physics is too complex to pre-compute reliably.
    """

    def __init__(self):
        """Initialize the edge feature builder."""
        self.TYPE_START = 0
        self.TYPE_END = 4
        self.CONNECTIVITY_START = 4
        self.CONNECTIVITY_END = 6

        assert self.CONNECTIVITY_END == EDGE_FEATURE_DIM, (
            "Edge feature dimensions mismatch!"
        )

    def build_edge_features(
        self,
        edge_type: EdgeType,
        weight: float = 1.0,
        reachability_confidence: float = 1.0,
    ) -> np.ndarray:
        """
        Build comprehensive 6-dimensional edge features.

        Args:
            edge_type: EdgeType enum value
                - ADJACENT (0): Neighboring nodes in grid
                - REACHABLE (1): Connected by reachability system
                - FUNCTIONAL (2): Entity relationship (switch-door)
                - BLOCKED (3): Currently blocked (locked door)
            weight: Graph traversal weight [0, 1], typically distance-based
            reachability_confidence: Confidence from reachability system [0, 1]

        Returns:
            np.ndarray of shape (6,) with normalized features
        """
        features = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)

        # ===== Edge Type (4 one-hot, indices 0-3) =====
        if 0 <= edge_type < 4:
            features[self.TYPE_START + int(edge_type)] = 1.0

        # ===== Connectivity Features (indices 4-5) =====
        features[self.CONNECTIVITY_START] = np.clip(weight, 0.0, 1.0)
        features[self.CONNECTIVITY_START + 1] = np.clip(
            reachability_confidence, 0.0, 1.0
        )

        return features


def create_default_node_features(num_nodes: int) -> np.ndarray:
    """
    Create default node features array with all zeros.

    Args:
        num_nodes: Number of nodes

    Returns:
        np.ndarray of shape (num_nodes, NODE_FEATURE_DIM)
    """
    return np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)


def create_default_edge_features(num_edges: int) -> np.ndarray:
    """
    Create default edge features array with all zeros.

    Args:
        num_edges: Number of edges

    Returns:
        np.ndarray of shape (num_edges, EDGE_FEATURE_DIM)
    """
    return np.zeros((num_edges, EDGE_FEATURE_DIM), dtype=np.float32)


def validate_node_features(features: np.ndarray) -> bool:
    """
    Validate node features array.

    Args:
        features: Node features array

    Returns:
        True if valid, False otherwise
    """
    if features.shape[-1] != NODE_FEATURE_DIM:
        return False
    if not np.all(np.isfinite(features)):
        return False
    # Check that one-hot encodings are valid (sum to 0 or 1)
    type_onehot = features[..., 3:9]
    tile_onehot = features[..., 19:57]
    if not np.all((type_onehot.sum(axis=-1) == 0) | (type_onehot.sum(axis=-1) == 1)):
        return False
    if not np.all((tile_onehot.sum(axis=-1) == 0) | (tile_onehot.sum(axis=-1) == 1)):
        return False
    return True


def validate_edge_features(features: np.ndarray) -> bool:
    """
    Validate edge features array.

    Args:
        features: Edge features array

    Returns:
        True if valid, False otherwise
    """
    if features.shape[-1] != EDGE_FEATURE_DIM:
        return False
    if not np.all(np.isfinite(features)):
        return False
    # Check that one-hot encoding is valid
    type_onehot = features[..., 0:4]
    if not np.all((type_onehot.sum(axis=-1) == 0) | (type_onehot.sum(axis=-1) == 1)):
        return False
    return True


# Convenience function for batch building
def build_node_features_batch(
    node_positions: List[Tuple[float, float]],
    node_types: List[NodeType],
    resolution_levels: Optional[List[int]] = None,
    tile_types: Optional[List[int]] = None,
    entity_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    reachability_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    ninja_pos: Optional[Tuple[float, float]] = None,
    goal_pos: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Build node features for multiple nodes in batch.

    Args:
        node_positions: List of (x, y) positions
        node_types: List of NodeType enum values
        resolution_levels: List of resolution levels (0, 1, 2)
        tile_types: List of tile type IDs
        entity_infos: List of entity info dicts (None for non-entity nodes)
        reachability_infos: List of reachability info dicts
        ninja_pos: Current ninja position
        goal_pos: Current goal position

    Returns:
        np.ndarray of shape (num_nodes, NODE_FEATURE_DIM)
    """
    num_nodes = len(node_positions)

    # Fill defaults
    if resolution_levels is None:
        resolution_levels = [0] * num_nodes
    if tile_types is None:
        tile_types = [0] * num_nodes
    if entity_infos is None:
        entity_infos = [None] * num_nodes
    if reachability_infos is None:
        reachability_infos = [None] * num_nodes

    # Build features for each node
    builder = NodeFeatureBuilder()
    features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    for i in range(num_nodes):
        features[i] = builder.build_node_features(
            node_pos=node_positions[i],
            node_type=node_types[i],
            resolution_level=resolution_levels[i],
            tile_type=tile_types[i],
            entity_info=entity_infos[i],
            reachability_info=reachability_infos[i],
            ninja_pos=ninja_pos,
            goal_pos=goal_pos,
        )

    return features


def build_edge_features_batch(
    edge_types: List[EdgeType],
    weights: Optional[List[float]] = None,
    reachability_confidences: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build edge features for multiple edges in batch.

    Args:
        edge_types: List of EdgeType enum values
        weights: List of edge weights
        reachability_confidences: List of reachability confidences

    Returns:
        np.ndarray of shape (num_edges, EDGE_FEATURE_DIM)
    """
    num_edges = len(edge_types)

    # Fill defaults
    if weights is None:
        weights = [1.0] * num_edges
    if reachability_confidences is None:
        reachability_confidences = [1.0] * num_edges

    # Build features for each edge
    builder = EdgeFeatureBuilder()
    features = np.zeros((num_edges, EDGE_FEATURE_DIM), dtype=np.float32)

    for i in range(num_edges):
        features[i] = builder.build_edge_features(
            edge_type=edge_types[i],
            weight=weights[i],
            reachability_confidence=reachability_confidences[i],
        )

    return features
