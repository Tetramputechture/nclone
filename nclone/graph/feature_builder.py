"""
Feature builder for graph nodes and edges.

This module creates optimized node features and edge features for deep RL with GNNs.
Features are designed to provide all necessary information for level completion,
especially mine navigation, while avoiding redundancy.

Key design principles:
- Spatial information from positions (tile types removed - redundant with position)
- Clear mine state encoding for navigation
- Entity state tracking (doors, switches)
- Reachability from flood-fill analysis (not physics simulation)
- Agent learns movement dynamics and spatial patterns from graph structure

Supported entities:
- Ninja (player)
- Exit switch and door
- Locked doors (up to 16) with switches
- Toggle mines (up to 256 total: 128 toggled + 128 untoggled)

Feature dimensions (Phase 1): 15 total (2 spatial + 7 type + 3 mine + 2 entity_state + 1 reachability)
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
from .common import NodeType, EdgeType, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


class NodeFeatureBuilder:
    """
    Builds optimized 17-dimensional node feature vectors (Phase 6 - Memory Optimized).

    Feature breakdown:
    - Spatial (2): position (x, y) normalized
    - Type (7): one-hot node type (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)
    - Mine-specific (2): mine_state (-1/0/+1), mine_radius [REMOVED: is_mine - redundant with type]
    - Entity state (2): entity_active, door_closed
    - Reachability (1): from reachability system
    - Topological (3): objective_dx, objective_dy, objective_hops [REMOVED: in_degree, out_degree, betweenness]

    Total: 2 + 7 + 2 + 2 + 1 + 3 = 17 features (was 21)

    Memory savings: ~40% reduction in node feature storage
    """

    def __init__(self):
        """Initialize the node feature builder."""
        # Feature index boundaries (16 features total - memory optimized)
        self.SPATIAL_START = 0
        self.SPATIAL_END = 2
        self.TYPE_START = 2
        self.TYPE_END = 9  # 7 node types: EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR
        self.MINE_START = 9
        self.MINE_END = (
            11  # 2 mine features: mine_state, mine_radius (removed: is_mine)
        )
        self.ENTITY_STATE_START = 11
        self.ENTITY_STATE_END = 13  # 2 entity state features: active, door_closed
        self.REACHABILITY_START = 13
        self.REACHABILITY_END = 14
        self.TOPOLOGICAL_START = 14
        self.TOPOLOGICAL_END = 17  # 3 topological features: objective_dx, objective_dy, objective_hops (removed: in_degree, out_degree, betweenness)

        assert self.TOPOLOGICAL_END == NODE_FEATURE_DIM, "Feature dimensions mismatch!"

    def build_node_features(
        self,
        node_pos: Tuple[float, float],
        node_type: NodeType,
        entity_info: Optional[Dict[str, Any]] = None,
        reachability_info: Optional[Dict[str, Any]] = None,
        topological_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Build optimized 17-dimensional node features (Phase 6 - Memory Optimized).

        Args:
            node_pos: (x, y) position of node in pixels
            node_type: NodeType enum value (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)
            entity_info: Dict with entity-specific information (type, state, radius, closed for doors)
            reachability_info: Dict with reachability from flood-fill system
            topological_info: Dict with topological features (objective-relative only)

        Returns:
            np.ndarray of shape (17,) with normalized features
        """
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

        # ===== Spatial Features (2 dims) =====
        features[0] = np.clip(node_pos[0] / FULL_MAP_WIDTH_PX, 0.0, 1.0)
        features[1] = np.clip(node_pos[1] / FULL_MAP_HEIGHT_PX, 0.0, 1.0)

        # ===== Type Encoding (7 dims) =====
        # [EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR]
        if 0 <= node_type < 7:
            features[self.TYPE_START + int(node_type)] = 1.0

        # ===== Mine-Specific Features (2 dims) =====
        # Always compute mine features (will be zeros for non-mines)
        if entity_info is not None:
            self._add_mine_features(features, entity_info, node_type)

        # ===== Entity State Features (2 dims) =====
        if entity_info is not None:
            self._add_entity_state_features(features, entity_info)

        # ===== Reachability Features (1 dim) =====
        if reachability_info is not None:
            self._add_reachability_features(features, reachability_info)

        # ===== Topological Features (3 dims) =====
        # Only objective-relative features (removed: in_degree, out_degree, betweenness)
        if topological_info is not None:
            self._add_topological_features(features, topological_info)

        return features

    def _add_mine_features(
        self, features: np.ndarray, entity_info: Dict[str, Any], node_type: NodeType
    ):
        """
        Add mine-specific features (indices 9-10, 2 features).

        MEMORY OPTIMIZATION: Removed is_mine flag (redundant with node_type TOGGLE_MINE bit).

        Args:
            features: Feature array to modify
            entity_info: Dict with keys:
                - type: EntityType enum value
                - state: float (for toggle mines: 0=deadly/toggled, 1=safe/untoggled, 2=transitioning)
                - radius: float (collision radius in pixels)
            node_type: NodeType to check if this is a mine node

        Mine state encoding:
        - mine_state (index 9): -1.0=deadly, 0.0=transitioning, +1.0=safe (0.0 for non-mines)
        - mine_radius (index 10): normalized collision radius (0.0 for non-mines)
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
        Add general entity state features (indices 11-12, 2 features).

        Args:
            features: Feature array to modify
            entity_info: Dict with keys:
                - active: bool (entity is active, for switches/doors)
                - closed: bool (for doors, True if closed)

        Features:
        - entity_active (index 11): 1.0 if active, 0.0 otherwise
        - door_closed (index 12): 1.0 if door is closed, 0.0 if open or not a door
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

    def _add_reachability_features(
        self, features: np.ndarray, reachability_info: Dict[str, Any]
    ):
        """
        Add reachability features from flood-fill system (index 13, 1 feature).

        Args:
            features: Feature array to modify
            reachability_info: Dict with keys:
                - reachable_from_ninja: bool (from ReachabilityResult.is_position_reachable)

        Note: Movement requirements (jump/walljump) are NOT included as physics
        is too complex to pre-compute. Agent learns from graph structure and experience.
        """
        # Index 13: reachable_from_ninja (from reachability system's flood-fill)
        features[self.REACHABILITY_START] = (
            1.0 if reachability_info.get("reachable_from_ninja", False) else 0.0
        )

    def _add_topological_features(
        self, features: np.ndarray, topological_info: Dict[str, Any]
    ):
        """
        Add topological features from graph analysis (indices 14-16, 3 features).

        MEMORY OPTIMIZATION: Removed in_degree, out_degree, betweenness (not critical for shortest-path navigation).

        Args:
            features: Feature array to modify
            topological_info: Dict with keys:
                - objective_dx: Normalized x-distance to objective [-1, 1]
                - objective_dy: Normalized y-distance to objective [-1, 1]
                - objective_hops: Normalized graph hops to objective [0, 1]
        """
        # Index 14: objective_dx
        features[self.TOPOLOGICAL_START] = np.clip(
            topological_info.get("objective_dx", 0.0), -1.0, 1.0
        )
        # Index 15: objective_dy
        features[self.TOPOLOGICAL_START + 1] = np.clip(
            topological_info.get("objective_dy", 0.0), -1.0, 1.0
        )
        # Index 16: objective_hops
        features[self.TOPOLOGICAL_START + 2] = np.clip(
            topological_info.get("objective_hops", 0.0), 0.0, 1.0
        )


class EdgeFeatureBuilder:
    """
    Builds optimized edge feature vectors (Phase 6 - Memory Optimized: 12 dims).

    Feature breakdown:
    - Edge type (4): one-hot encoding
    - Connectivity (2): weight and reachability confidence
    - Geometric (4): dx, dy, distance, movement_category
    - Mine danger (2): nearest_mine_dist, passes_deadly_mine [REMOVED: threat_level, mines_nearby]

    Total: 4 + 2 + 4 + 2 = 12 features (was 14)

    Memory savings: ~32% reduction in edge feature storage
    """

    def __init__(self):
        """Initialize the edge feature builder."""
        self.TYPE_START = 0
        self.TYPE_END = 4
        self.CONNECTIVITY_START = 4
        self.CONNECTIVITY_END = 6
        self.GEOMETRIC_START = 6
        self.GEOMETRIC_END = 10
        self.MINE_START = 10
        self.MINE_END = 12  # 2 mine features: nearest_mine_distance, passes_deadly_mine (removed: threat_level, mines_nearby)

        assert self.MINE_END == EDGE_FEATURE_DIM, "Edge feature dimensions mismatch!"

    def build_edge_features(
        self,
        edge_type: EdgeType,
        weight: float = 1.0,
        reachability_confidence: float = 1.0,
        geometric_features: Optional[Dict[str, float]] = None,
        mine_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Build optimized edge features (12 dimensions in Phase 6 - Memory Optimized).

        Args:
            edge_type: EdgeType enum value
                - ADJACENT (0): Neighboring nodes in grid
                - REACHABLE (1): Connected by reachability system
                - FUNCTIONAL (2): Entity relationship (switch-door)
                - BLOCKED (3): Currently blocked (locked door)
            weight: Graph traversal weight [0, 1], typically distance-based
            reachability_confidence: Confidence from reachability system [0, 1]
            geometric_features: Dict with keys:
                - dx_norm: normalized x-direction [-1, 1]
                - dy_norm: normalized y-direction [-1, 1]
                - distance: normalized Euclidean distance [0, 1]
                - movement_category: movement type [0, 1]
            mine_features: Dict with keys:
                - nearest_mine_distance: normalized distance to nearest mine [0, 1]
                - passes_deadly_mine: binary flag [0, 1]

        Returns:
            np.ndarray of shape (12,) with normalized features
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

        # ===== Geometric Features (indices 6-9) =====
        if geometric_features is not None:
            features[self.GEOMETRIC_START] = np.clip(
                geometric_features.get("dx_norm", 0.0), -1.0, 1.0
            )
            features[self.GEOMETRIC_START + 1] = np.clip(
                geometric_features.get("dy_norm", 0.0), -1.0, 1.0
            )
            features[self.GEOMETRIC_START + 2] = np.clip(
                geometric_features.get("distance", 0.0), 0.0, 1.0
            )
            features[self.GEOMETRIC_START + 3] = np.clip(
                geometric_features.get("movement_category", 0.0), 0.0, 1.0
            )

        # ===== Mine Danger Features (indices 10-11) =====
        # MEMORY OPTIMIZATION: Removed mine_threat_level and num_mines_nearby (redundant)
        if mine_features is not None:
            features[self.MINE_START] = np.clip(
                mine_features.get("nearest_mine_distance", 1.0), 0.0, 1.0
            )
            features[self.MINE_START + 1] = np.clip(
                mine_features.get("passes_deadly_mine", 0.0), 0.0, 1.0
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
    if not np.all(np.isfinite(features)):
        return False
    # Check that one-hot encoding is valid
    type_onehot = features[..., 0:4]
    if not np.all((type_onehot.sum(axis=-1) == 0) | (type_onehot.sum(axis=-1) == 1)):
        return False
    return True
