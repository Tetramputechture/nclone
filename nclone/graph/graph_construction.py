"""
Graph construction utilities for hierarchical graph building.

This module provides core graph construction methods for building
sub-cell graphs and managing entity sorting.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from .common import (
    GraphData, NodeType, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, 
    N_MAX_NODES, E_MAX_EDGES
)
from .feature_extraction import FeatureExtractor
from .edge_building import EdgeBuilder


class GraphConstructor:
    """
    Handles core graph construction with bounce block awareness.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor, edge_builder: EdgeBuilder):
        """
        Initialize graph constructor.
        
        Args:
            feature_extractor: Feature extraction utility instance
            edge_builder: Edge building utility instance
        """
        self.feature_extractor = feature_extractor
        self.edge_builder = edge_builder
    
    def build_sub_cell_graph(
        self,
        level_data: np.ndarray,
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        ninja_velocity: Tuple[float, float],
        ninja_state: int,
        node_feature_dim: int,
        edge_feature_dim: int
    ) -> GraphData:
        """
        Build sub-cell graph with bounce block awareness.
        
        This method builds the finest resolution graph with comprehensive
        bounce block mechanics integrated directly.
        
        Args:
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            entities: List of entity dictionaries
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            node_feature_dim: Node feature dimension
            edge_feature_dim: Edge feature dimension
            
        Returns:
            GraphData with sub-cell resolution graph
        """
        # Initialize arrays
        node_features = np.zeros((N_MAX_NODES, node_feature_dim), dtype=np.float32)
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_features = np.zeros((E_MAX_EDGES, edge_feature_dim), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.float32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.float32)
        node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
        edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)
        
        node_count = 0
        edge_count = 0
        
        # Build sub-grid nodes (canonical ordering: sub-row by sub-row)
        sub_grid_node_map = {}  # (sub_row, sub_col) -> node_index
        
        for sub_row in range(SUB_GRID_HEIGHT):
            for sub_col in range(SUB_GRID_WIDTH):
                if node_count >= N_MAX_NODES:
                    break
                    
                node_idx = node_count
                sub_grid_node_map[(sub_row, sub_col)] = node_idx
                
                # Extract sub-cell features
                sub_cell_features = self.feature_extractor.extract_sub_cell_features(
                    level_data, sub_row, sub_col, ninja_position, ninja_velocity, ninja_state, entities, node_feature_dim
                )
                
                node_features[node_idx] = sub_cell_features
                node_mask[node_idx] = 1.0
                node_types[node_idx] = NodeType.GRID_CELL
                node_count += 1
        
        # Build entity nodes (sorted by type then position for determinism)
        entity_nodes = []
        sorted_entities = sorted(entities, key=self.entity_sort_key)
        
        for entity in sorted_entities:
            if node_count >= N_MAX_NODES:
                break
                
            node_idx = node_count
            entity_nodes.append((node_idx, entity))
            
            # Extract entity features
            entity_features = self.feature_extractor.extract_entity_features(
                entity, ninja_position, ninja_velocity, node_feature_dim
            )
            
            node_features[node_idx] = entity_features
            node_mask[node_idx] = 1.0
            node_types[node_idx] = NodeType.ENTITY
            node_count += 1
        
        # Build edges with bounce block awareness
        edge_count = self.edge_builder.build_edges(
            sub_grid_node_map, entity_nodes, entities, level_data,
            ninja_position, ninja_velocity, ninja_state,
            edge_index, edge_features, edge_mask, edge_types, edge_count, edge_feature_dim
        )
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=node_types,
            edge_types=edge_types,
            num_nodes=node_count,
            num_edges=edge_count
        )
    
    def entity_sort_key(self, entity: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Generate sort key for entity ordering.
        
        Args:
            entity: Entity dictionary with type, position, state
            
        Returns:
            Sort key tuple (type, x, y)
        """
        entity_type = entity.get('type', 0)
        x = entity.get('x', 0.0)
        y = entity.get('y', 0.0)
        return (entity_type, x, y)