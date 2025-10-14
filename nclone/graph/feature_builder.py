"""
Production-ready feature builder for graph nodes and edges.

This module creates comprehensive 61-dimensional node features and 6-dimensional
edge features for deep RL with GNNs. Features are designed to provide all necessary 
information for level completion while avoiding complex physics pre-computation.

Key design principles:
- Spatial information from positions and distances
- Entity state tracking (mines, doors, switches)
- Reachability from flood-fill analysis (not physics simulation)
- Tile type encoding for terrain understanding
- Agent learns movement dynamics from temporal frames and game state

Supported entities:
- Ninja (player)
- Exit switch and door
- Locked doors (up to 16) with switches
- Toggle mines (up to 256 total: 128 toggled + 128 untoggled)
- All tile types (38 types)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..constants.entity_types import EntityType
from ..constants.physics_constants import (
    MAP_TILE_WIDTH, 
    MAP_TILE_HEIGHT, 
    FULL_MAP_WIDTH_PX, 
    FULL_MAP_HEIGHT_PX,
    NINJA_RADIUS
)
from .common import (
    NodeType, 
    EdgeType, 
    NODE_FEATURE_DIM, 
    EDGE_FEATURE_DIM,
    MAX_MINES_TOTAL,
    MAX_LOCKED_DOORS
)


class NodeFeatureBuilder:
    """
    Builds comprehensive 61-dimensional node feature vectors.
    
    Feature breakdown:
    - Spatial (3): position (x, y) + resolution level
    - Type (6): one-hot node type
    - Entity (10): entity-specific information
    - Tile (38): one-hot tile type
    - Reachability (2): from reachability system
    - Proximity (2): distances to key points
    
    Total: 3 + 6 + 10 + 38 + 2 + 2 = 61 features
    """
    
    def __init__(self):
        """Initialize the node feature builder."""
        self.screen_diagonal = np.sqrt(FULL_MAP_WIDTH_PX**2 + FULL_MAP_HEIGHT_PX**2)
        
        # Feature index boundaries for clarity
        self.SPATIAL_START = 0
        self.SPATIAL_END = 3
        self.TYPE_START = 3
        self.TYPE_END = 9
        self.ENTITY_START = 9
        self.ENTITY_END = 19
        self.TILE_START = 19
        self.TILE_END = 57
        self.REACHABILITY_START = 57
        self.REACHABILITY_END = 59
        self.PROXIMITY_START = 59
        self.PROXIMITY_END = 61
        
        assert self.PROXIMITY_END == NODE_FEATURE_DIM, "Feature dimensions mismatch!"
    
    def build_node_features(
        self,
        node_pos: Tuple[float, float],
        node_type: NodeType,
        resolution_level: int = 0,  # 0=fine, 1=medium, 2=coarse
        tile_type: int = 0,
        entity_info: Optional[Dict[str, Any]] = None,
        reachability_info: Optional[Dict[str, Any]] = None,
        ninja_pos: Optional[Tuple[float, float]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Build comprehensive 61-dimensional node features.
        
        Args:
            node_pos: (x, y) position of node in pixels
            node_type: NodeType enum value (EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT)
            resolution_level: 0=fine(6px), 1=medium(24px), 2=coarse(96px)
            tile_type: Tile type ID (0-37), if applicable
            entity_info: Dict with entity-specific information (type, state, etc.)
            reachability_info: Dict with reachability from flood-fill system
            ninja_pos: Current ninja position for distance calculation
            goal_pos: Current goal position (switch or exit) for distance
            
        Returns:
            np.ndarray of shape (61,) with normalized features
        """
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        
        # ===== Spatial Features (indices 0-2) =====
        features[0] = np.clip(node_pos[0] / FULL_MAP_WIDTH_PX, 0.0, 1.0)
        features[1] = np.clip(node_pos[1] / FULL_MAP_HEIGHT_PX, 0.0, 1.0)
        features[2] = resolution_level / 2.0  # 0=fine, 0.5=medium, 1.0=coarse
        
        # ===== Type Encoding (indices 3-8, 6 one-hot) =====
        # [EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT]
        if 0 <= node_type < 6:
            features[self.TYPE_START + int(node_type)] = 1.0
        
        # ===== Entity-Specific Features (indices 9-18, 10 features) =====
        if entity_info is not None:
            self._add_entity_features(features, entity_info)
        
        # ===== Tile Information (indices 19-56, 38 one-hot) =====
        if 0 <= tile_type <= 37:
            features[self.TILE_START + tile_type] = 1.0
        
        # ===== Reachability Features (indices 57-58, 2 features) =====
        if reachability_info is not None:
            self._add_reachability_features(features, reachability_info)
        
        # ===== Proximity Features (indices 59-60, 2 features) =====
        if ninja_pos is not None:
            dx = node_pos[0] - ninja_pos[0]
            dy = node_pos[1] - ninja_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            features[self.PROXIMITY_START] = min(dist / self.screen_diagonal, 1.0)
        
        if goal_pos is not None:
            dx = node_pos[0] - goal_pos[0]
            dy = node_pos[1] - goal_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            features[self.PROXIMITY_START + 1] = min(dist / self.screen_diagonal, 1.0)
        
        return features
    
    def _add_entity_features(self, features: np.ndarray, entity_info: Dict[str, Any]):
        """
        Add entity-specific features (indices 9-18, 10 features).
        
        Args:
            features: Feature array to modify
            entity_info: Dict with keys:
                - type: EntityType enum value
                - subtype: int (door ID, mine toggle state)
                - active: bool (entity is active)
                - state: float (0-1, for toggles/animations)
                - radius: float (collision radius in pixels)
                - switch_activated: bool (for switches)
                - door_open: bool (for doors)
                - door_locked: bool (for locked doors)
                - requires_switch: bool (for locked doors)
                - cooldown: int (frames until next interaction)
        """
        # Index 9: entity_type (0=none, 1=mine, 2=switch, 3=door, 4=exit)
        entity_type = entity_info.get('type', 0)
        type_encoding = 0.0
        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            type_encoding = 0.25  # mine
        elif entity_type in [EntityType.EXIT_SWITCH, EntityType.LOCKED_DOOR_SWITCH]:
            type_encoding = 0.5  # switch
        elif entity_type in [EntityType.LOCKED_DOOR, EntityType.REGULAR_DOOR, EntityType.TRAP_DOOR]:
            type_encoding = 0.75  # door
        elif entity_type == EntityType.EXIT_DOOR:
            type_encoding = 1.0  # exit
        features[self.ENTITY_START] = type_encoding
        
        # Index 10: entity_subtype (for locked doors: door ID 0-15, for mines: toggle state)
        entity_subtype = entity_info.get('subtype', 0)
        features[self.ENTITY_START + 1] = np.clip(entity_subtype / MAX_LOCKED_DOORS, 0.0, 1.0)
        
        # Index 11: entity_active
        features[self.ENTITY_START + 2] = 1.0 if entity_info.get('active', True) else 0.0
        
        # Index 12: entity_state (for mines: 0=safe, 1=deadly; for doors: 0=closed, 1=open)
        entity_state = entity_info.get('state', 0.0)
        features[self.ENTITY_START + 3] = np.clip(entity_state, 0.0, 1.0)
        
        # Index 13: entity_radius_norm (collision radius)
        entity_radius = entity_info.get('radius', 0.0)
        max_radius = NINJA_RADIUS * 2  # Assume max entity radius is 2x ninja radius
        features[self.ENTITY_START + 4] = np.clip(entity_radius / max_radius, 0.0, 1.0)
        
        # Index 14: switch_activated (for switches)
        features[self.ENTITY_START + 5] = 1.0 if entity_info.get('switch_activated', False) else 0.0
        
        # Index 15: door_open (for doors)
        features[self.ENTITY_START + 6] = 1.0 if entity_info.get('door_open', False) else 0.0
        
        # Index 16: door_locked (for doors)
        features[self.ENTITY_START + 7] = 1.0 if entity_info.get('door_locked', False) else 0.0
        
        # Index 17: door_requires_switch (for locked doors)
        features[self.ENTITY_START + 8] = 1.0 if entity_info.get('requires_switch', False) else 0.0
        
        # Index 18: interaction_cooldown
        cooldown = entity_info.get('cooldown', 0)
        max_cooldown = 60  # Assume max cooldown is 1 second at 60fps
        features[self.ENTITY_START + 9] = np.clip(cooldown / max_cooldown, 0.0, 1.0)
    
    def _add_reachability_features(self, features: np.ndarray, reachability_info: Dict[str, Any]):
        """
        Add reachability features from flood-fill system (indices 57-58, 2 features).
        
        Args:
            features: Feature array to modify
            reachability_info: Dict with keys:
                - reachable_from_ninja: bool (from flood-fill)
                - on_critical_path: bool (shortest path to goal)
                
        Note: Movement requirements (jump/walljump) are NOT included as physics
        is too complex to pre-compute. Agent learns from temporal frames.
        """
        # Index 57: reachable_from_ninja (from reachability system's flood-fill)
        features[self.REACHABILITY_START] = 1.0 if reachability_info.get('reachable_from_ninja', False) else 0.0
        
        # Index 58: on_critical_path (on shortest path from ninja to goal)
        features[self.REACHABILITY_START + 1] = 1.0 if reachability_info.get('on_critical_path', False) else 0.0


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
        
        assert self.CONNECTIVITY_END == EDGE_FEATURE_DIM, "Edge feature dimensions mismatch!"
    
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
        features[self.CONNECTIVITY_START + 1] = np.clip(reachability_confidence, 0.0, 1.0)
        
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
