"""
Feature builder for graph nodes and edges.

This module creates optimized node features and edge features for deep RL with GNNs.
Features are designed to provide all necessary information for level completion,
especially mine navigation, while avoiding redundancy.

Key design principles:
- Spatial information from positions (tile types removed - redundant with position)
- Clear mine state encoding for navigation
- Entity state tracking (doors, switches)
- Agent learns movement dynamics and spatial patterns from graph structure

Supported entities:
- Ninja (player)
- Exit switch and door
- Locked doors (up to 16) with switches
- Toggle mines (up to 256 total: 128 toggled + 128 untoggled)

Feature dimensions (Phase 1): 15 total (2 spatial + 7 type + 3 mine + 2 entity_state)
Feature dimensions (Phase 5): 21 total (adds 6 topological features)
Feature dimensions (Phase 6 - MEMORY OPTIMIZED): 16 total (removed: is_mine, in_degree, out_degree, betweenness)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

from ..constants.entity_types import EntityType
from ..constants.physics_constants import (
    FULL_MAP_WIDTH_PX,
    FULL_MAP_HEIGHT_PX,
    NINJA_RADIUS,
)
from .common import NodeType, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


class NodeFeatureBuilder:
    """
    Builds GCN-optimized 6-dimensional node feature vectors.

    Feature breakdown (GCN doesn't use type embeddings or topological features):
    - Spatial (2): position (x, y) normalized
    - Mine-specific (2): mine_state (-1/0/+1), mine_radius
    - Entity state (2): entity_active, door_closed

    Total: 2 + 2 + 2 = 6 features (was 17)

    Removed for GCN optimization:
    - Type one-hot (7): Redundant, GCN doesn't use type embeddings
    - Topological (3): Redundant with PBRS shortest paths in reward
    - Reachability (1): Redundant, all nodes in graph are reachable (flood fill filtered)

    Memory savings: 65% reduction in node feature storage (17 → 6 dims)
    """

    def __init__(self):
        """Initialize the node feature builder."""
        # Feature index boundaries (6 features total - GCN optimized)
        self.SPATIAL_START = 0
        self.SPATIAL_END = 2
        self.MINE_START = 2
        self.MINE_END = 4  # 2 mine features: mine_state, mine_radius
        self.ENTITY_STATE_START = 4
        self.ENTITY_STATE_END = 6  # 2 entity state features: active, door_closed

        assert self.ENTITY_STATE_END == NODE_FEATURE_DIM, "Feature dimensions mismatch!"

    def build_node_features(
        self,
        node_pos: Tuple[float, float],
        node_type: NodeType,
        entity_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Build GCN-optimized 6-dimensional node features.

        Args:
            node_pos: (x, y) position of node in pixels
            node_type: NodeType enum value (used for mine detection, not encoded)
            entity_info: Dict with entity-specific information (type, state, radius, closed for doors)

        Returns:
            np.ndarray of shape (6,) with normalized features
        """
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

        # ===== Spatial Features (2 dims) =====
        features[0] = np.clip(node_pos[0] / FULL_MAP_WIDTH_PX, 0.0, 1.0)
        features[1] = np.clip(node_pos[1] / FULL_MAP_HEIGHT_PX, 0.0, 1.0)

        # ===== Mine-Specific Features (2 dims) =====
        # Always compute mine features (will be zeros for non-mines)
        if entity_info is not None:
            self._add_mine_features(features, entity_info, node_type)

        # ===== Entity State Features (2 dims) =====
        if entity_info is not None:
            self._add_entity_state_features(features, entity_info)

        return features

    def _add_mine_features(
        self, features: np.ndarray, entity_info: Dict[str, Any], node_type: NodeType
    ):
        """
        Add mine-specific features (indices 2-3, 2 features).

        Args:
            features: Feature array to modify
            entity_info: Dict with keys:
                - type: EntityType enum value
                - state: float (for toggle mines: 0=deadly/toggled, 1=safe/untoggled, 2=transitioning)
                - radius: float (collision radius in pixels)
            node_type: NodeType to check if this is a mine node

        Mine state encoding:
        - mine_state (index 2): -1.0=deadly, 0.0=transitioning, +1.0=safe (0.0 for non-mines)
        - mine_radius (index 3): normalized collision radius (0.0 for non-mines)
        """
        entity_type = entity_info.get("type", 0)
        is_mine = (
            entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]
            or node_type == NodeType.TOGGLE_MINE
        )

        # Index 9: mine_state (-1/0/+1 for deadly/transitioning/safe)
        if is_mine:
            raw_state = entity_info.get("state", 0.0)
            # Toggle mine states: 0=toggled (deadly), 1=untoggled (safe), 2=transitioning
            if raw_state == 0.0:
                mine_state = -1.0  # Deadly
            elif raw_state == 2.0:
                mine_state = 0.0  # Transitioning
            else:  # raw_state == 1.0
                mine_state = 1.0  # Safe
            features[self.MINE_START] = mine_state
        else:
            features[self.MINE_START] = 0.0

        # Index 10: mine_radius (normalized)
        if is_mine:
            mine_radius = entity_info.get("radius", 0.0)
            max_radius = NINJA_RADIUS * 2  # Max expected mine radius
            features[self.MINE_START + 1] = np.clip(mine_radius / max_radius, 0.0, 1.0)
        else:
            features[self.MINE_START + 1] = 0.0

    def _add_entity_state_features(
        self, features: np.ndarray, entity_info: Dict[str, Any]
    ):
        """
        Add general entity state features (indices 4-5, 2 features).

        Args:
            features: Feature array to modify
            entity_info: Dict with keys:
                - active: bool (entity is active, for switches/doors)
                - closed: bool (for doors, True if closed)

        Features:
        - entity_active (index 4): 1.0 if active, 0.0 otherwise
        - door_closed (index 5): 1.0 if door is closed, 0.0 if open or not a door
        """
        # Index 11: entity_active (for switches and doors)
        features[self.ENTITY_STATE_START] = (
            1.0 if entity_info.get("active", True) else 0.0
        )

        # Index 12: door_closed (for locked doors only)
        entity_type = entity_info.get("type", 0)
        if entity_type == EntityType.LOCKED_DOOR:
            features[self.ENTITY_STATE_START + 1] = (
                1.0 if entity_info.get("closed", True) else 0.0
            )
        else:
            features[self.ENTITY_STATE_START + 1] = 0.0


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
    Create default edge features array (empty - no edge features).

    Args:
        num_edges: Number of edges (ignored when EDGE_FEATURE_DIM = 0)

    Returns:
        np.ndarray of shape (num_edges, 0) - empty array
    """
    if EDGE_FEATURE_DIM == 0:
        return np.zeros((num_edges, 0), dtype=np.float32)
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
    # Type one-hot: indices 2-9 (7 node types)
    type_onehot = features[..., builder.TYPE_START : builder.TYPE_END]

    if not np.all((type_onehot.sum(axis=-1) == 0) | (type_onehot.sum(axis=-1) == 1)):
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

    # No validation needed when EDGE_FEATURE_DIM = 0
    if EDGE_FEATURE_DIM == 0:
        return True

    if not np.all(np.isfinite(features)):
        return False

    return True
