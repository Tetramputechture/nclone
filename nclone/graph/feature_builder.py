"""
Feature builder for graph nodes and edges.

This module creates comprehensive 56-dimensional node features and 6-dimensional
edge features for deep RL with GNNs. Features are designed to provide all necessary
information for level completion while avoiding complex physics pre-computation.

Key design principles:
- Spatial information from positions and distances
- Entity state tracking (mines, doors, switches)
- Reachability from flood-fill analysis (not physics simulation)
- Tile type encoding for terrain understanding
- Agent learns movement dynamics from frames and game state

Supported entities:
- Ninja (player)
- Exit switch and door
- Locked doors (up to 16) with switches
- Toggle mines (up to 256 total: 128 toggled + 128 untoggled)
- Tile types 0-33 (glitched tiles 34-37 treated as 0)

Feature dimensions: 50 total (3 spatial + 7 type + 5 entity + 34 tile + 1 reachability)
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
    Builds comprehensive 50-dimensional node feature vectors.

    Feature breakdown:
    - Spatial (3): position (x, y) + resolution level
    - Type (7): one-hot node type (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)
    - Entity (5): entity-specific information (reduced from 10, removed unused)
    - Tile (34): one-hot tile type (0-33, glitched tiles 34-37 treated as 0)
    - Reachability (1): from reachability system

    Total: 3 + 7 + 5 + 34 + 1 = 50 features
    """

    def __init__(self):
        """Initialize the node feature builder."""
        # Feature index boundaries
        self.SPATIAL_START = 0
        self.SPATIAL_END = 3
        self.TYPE_START = 3
        self.TYPE_END = 10  # 7 node types: EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR
        self.ENTITY_START = 10
        self.ENTITY_END = 15
        self.TILE_START = 15
        self.TILE_END = 49  # 34 tile types: 0-33 (glitched tiles 34-37 treated as 0)
        self.REACHABILITY_START = 49
        self.REACHABILITY_END = 50

        assert self.REACHABILITY_END == NODE_FEATURE_DIM, "Feature dimensions mismatch!"

    def build_node_features(
        self,
        node_pos: Tuple[float, float],
        node_type: NodeType,
        resolution_level: int = 0,  # 0=fine, 1=medium, 2=coarse
        tile_type: int = 0,
        entity_info: Optional[Dict[str, Any]] = None,
        reachability_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Build comprehensive 50-dimensional node features.

        Args:
            node_pos: (x, y) position of node in pixels
            node_type: NodeType enum value (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)
            resolution_level: 0=fine(6px), 1=medium(24px), 2=coarse(96px)
            tile_type: Tile type ID (0-33, glitched tiles 34-37 should be normalized to 0)
            entity_info: Dict with entity-specific information (type, state, etc.)
            reachability_info: Dict with reachability from flood-fill system

        Returns:
            np.ndarray of shape (50,) with normalized features
        """
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

        # ===== Spatial Features =====
        features[0] = np.clip(node_pos[0] / FULL_MAP_WIDTH_PX, 0.0, 1.0)
        features[1] = np.clip(node_pos[1] / FULL_MAP_HEIGHT_PX, 0.0, 1.0)
        features[2] = resolution_level / 2.0  # 0=fine, 0.5=medium, 1.0=coarse

        # ===== Type Encoding =====
        # [EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR]
        if 0 <= node_type < 7:
            features[self.TYPE_START + int(node_type)] = 1.0

        # ===== Entity-Specific Features =====
        if entity_info is not None:
            self._add_entity_features(features, entity_info)

        # ===== Tile Information =====
        # Clamp tile types 34-37 to 0 (treat glitched tiles as empty)
        if tile_type > 33:
            tile_type = 0
        if 0 <= tile_type <= 33:
            features[self.TILE_START + tile_type] = 1.0

        # ===== Reachability Features =====
        if reachability_info is not None:
            self._add_reachability_features(features, reachability_info)

        return features

    def _add_entity_features(self, features: np.ndarray, entity_info: Dict[str, Any]):
        """
        Add entity-specific features (indices 9-13, 5 features).

        Args:
            features: Feature array to modify
            entity_info: Dict with keys from entity_extractor.py:
                - type: EntityType enum value
                - active: bool (entity is active)
                - state: float (normalized entity state)
                - radius: float (collision radius in pixels)
                - closed: bool (for doors, True if closed)

        Entity-specific attributes from actual classes:
            Toggle Mine: type, active, state (0/1/2), radius, xpos, ypos
            Exit Switch: type, active, state, radius, xpos, ypos
            Exit Door: type, active, state, radius, xpos, ypos
            Locked Door: type, active, state, closed, radius, xpos, ypos
        """
        # Index 9: entity_type (0=none, 0.25=mine, 0.5=switch, 0.75=door, 1.0=exit)
        entity_type = entity_info.get("type", 0)
        type_encoding = 0.0
        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            type_encoding = 0.25  # mine
        elif entity_type == EntityType.EXIT_SWITCH:
            type_encoding = 0.5  # switch
        elif entity_type == EntityType.LOCKED_DOOR:
            type_encoding = 0.75  # door
        elif entity_type == EntityType.EXIT_DOOR:
            type_encoding = 1.0  # exit
        features[self.ENTITY_START] = type_encoding

        # Index 10: entity_subtype (for locked doors: closed status, for mines: state value)
        # For toggle mines, state is 0 (toggled/deadly), 1 (untoggled/safe), or 2 (toggling)
        entity_subtype = 0.0
        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            # Use the mine's actual state (0, 1, or 2) normalized
            mine_state = entity_info.get("state", 0.0)
            entity_subtype = mine_state / 2.0  # Normalize to [0, 1]
        elif entity_type == EntityType.LOCKED_DOOR:
            # For doors, use closed status (0=open, 1=closed)
            entity_subtype = 1.0 if entity_info.get("closed", True) else 0.0
        features[self.ENTITY_START + 1] = np.clip(entity_subtype, 0.0, 1.0)

        # Index 11: entity_active (from entity_extractor)
        features[self.ENTITY_START + 2] = (
            1.0 if entity_info.get("active", True) else 0.0
        )

        # Index 12: entity_state (normalized state from entity_extractor)
        entity_state = entity_info.get("state", 0.0)
        features[self.ENTITY_START + 3] = np.clip(entity_state, 0.0, 1.0)

        # Index 13: entity_radius_norm (collision radius from entity_extractor)
        entity_radius = entity_info.get("radius", 0.0)
        max_radius = NINJA_RADIUS * 2  # Assume max entity radius is 2x ninja radius
        features[self.ENTITY_START + 4] = np.clip(entity_radius / max_radius, 0.0, 1.0)

    def _add_reachability_features(
        self, features: np.ndarray, reachability_info: Dict[str, Any]
    ):
        """
        Add reachability features from flood-fill system (index 49, 1 feature).

        The reachability system uses OpenCV flood-fill (<1ms) for connectivity analysis.
        This is NOT physics simulation - it's simple connectivity checking.

        **Usage Example**:
        ```python
        from nclone.graph.reachability.reachability_system import ReachabilitySystem

        # Initialize system
        reachability_sys = ReachabilitySystem()

        # Analyze reachability
        result = reachability_sys.analyze_reachability(
            level_data=level_data,
            ninja_position=(ninja_x, ninja_y),
            switch_states=switch_states
        )

        # Check if node position is reachable
        node_pos = (x, y)  # in pixels
        is_reachable = result.is_position_reachable(node_pos)

        # Use in node features
        reachability_info = {
            'reachable_from_ninja': is_reachable,
        }
        ```

        Args:
            features: Feature array to modify
            reachability_info: Dict with keys:
                - reachable_from_ninja: bool (from ReachabilityApproximation.is_position_reachable)

        Note: Movement requirements (jump/walljump) are NOT included as physics
        is too complex to pre-compute. Agent learns from frames and game state.
        """
        # Index 49: reachable_from_ninja (from reachability system's flood-fill)
        features[self.REACHABILITY_START] = (
            1.0 if reachability_info.get("reachable_from_ninja", False) else 0.0
        )


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

    # Use feature builder constants for maintainability
    builder = NodeFeatureBuilder()

    # Check that one-hot encodings are valid (sum to 0 or 1)
    # Type one-hot: indices 3-9 (7 node types)
    type_onehot = features[..., builder.TYPE_START : builder.TYPE_END]
    # Tile one-hot: indices 15-48 (34 tile types)
    tile_onehot = features[..., builder.TILE_START : builder.TILE_END]

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
) -> np.ndarray:
    """
    Build node features for multiple nodes in batch.

    Args:
        node_positions: List of (x, y) positions
        node_types: List of NodeType enum values
        resolution_levels: List of resolution levels (0, 1, 2)
        tile_types: List of tile type IDs (0-33, glitched tiles 34-37 should be normalized to 0)
        entity_infos: List of entity info dicts (None for non-entity nodes)
        reachability_infos: List of reachability info dicts

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
        # Normalize glitched tiles to 0
        tile_type = tile_types[i]
        if tile_type > 33:
            tile_type = 0

        features[i] = builder.build_node_features(
            node_pos=node_positions[i],
            node_type=node_types[i],
            resolution_level=resolution_levels[i],
            tile_type=tile_type,
            entity_info=entity_infos[i],
            reachability_info=reachability_infos[i],
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
