"""
Hierarchical graph builder for multi-resolution N++ level processing.

This module creates hierarchical graph representations with multiple resolution levels:
- Level 0: Sub-cell (6px) - Fine-grained movement precision
- Level 1: Tile (24px) - Standard game tile resolution  
- Level 2: Region (96px) - Strategic planning resolution

The hierarchical structure enables both precise local movement decisions
and strategic global pathfinding through multi-scale graph processing.
"""

import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import IntEnum

# Use shared constants from the simulator
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE
from .common import GraphData, NodeType, EdgeType, SUB_CELL_SIZE


class ResolutionLevel(IntEnum):
    """Resolution levels for hierarchical processing."""
    SUB_CELL = 0    # 6px resolution - fine movement
    TILE = 1        # 24px resolution - standard tiles
    REGION = 2      # 96px resolution - strategic areas


@dataclass
class HierarchicalGraphData:
    """Container for multi-resolution graph data."""
    # Individual level graphs
    sub_cell_graph: GraphData      # Level 0: 6px resolution
    tile_graph: GraphData          # Level 1: 24px resolution  
    region_graph: GraphData        # Level 2: 96px resolution
    
    # Inter-level connectivity
    sub_to_tile_mapping: np.ndarray    # [N_sub_nodes] -> tile_node_idx
    tile_to_region_mapping: np.ndarray # [N_tile_nodes] -> region_node_idx
    
    # Cross-scale edges for information flow
    cross_scale_edges: Dict[str, np.ndarray]  # 'up' and 'down' connectivity
    
    # Metadata
    resolution_info: Dict[str, Any]


class HierarchicalGraphBuilder:
    """
    Builds multi-resolution hierarchical graph representations.
    
    Creates three levels of spatial resolution with inter-level connectivity
    to enable both precise local movement and strategic global planning.
    """
    
    def __init__(self):
        """Initialize hierarchical graph builder."""
        # Base graph builder for sub-cell level
        self.base_builder = GraphBuilder()
        
        # Resolution parameters
        self.resolutions = {
            ResolutionLevel.SUB_CELL: 6,   # 6px per cell
            ResolutionLevel.TILE: 24,      # 24px per cell (4x4 sub-cells)
            ResolutionLevel.REGION: 96     # 96px per cell (16x16 sub-cells, 4x4 tiles)
        }
        
        # Grid dimensions for each level
        self.grid_dimensions = self._calculate_grid_dimensions()
        
        # Maximum nodes per level (with padding for entities)
        self.max_nodes_per_level = {
            ResolutionLevel.SUB_CELL: self.grid_dimensions[ResolutionLevel.SUB_CELL][0] * 
                                     self.grid_dimensions[ResolutionLevel.SUB_CELL][1] + 400,
            ResolutionLevel.TILE: self.grid_dimensions[ResolutionLevel.TILE][0] * 
                                 self.grid_dimensions[ResolutionLevel.TILE][1] + 100,
            ResolutionLevel.REGION: self.grid_dimensions[ResolutionLevel.REGION][0] * 
                                   self.grid_dimensions[ResolutionLevel.REGION][1] + 50
        }
        
        # Feature dimensions (inherit from base builder, adjust for higher levels)
        self.node_feature_dims = {
            ResolutionLevel.SUB_CELL: self.base_builder.node_feature_dim,
            ResolutionLevel.TILE: self.base_builder.node_feature_dim + 8,  # Add aggregation stats
            ResolutionLevel.REGION: self.base_builder.node_feature_dim + 16  # Add more aggregation stats
        }
        
        self.edge_feature_dims = {
            ResolutionLevel.SUB_CELL: self.base_builder.edge_feature_dim,
            ResolutionLevel.TILE: self.base_builder.edge_feature_dim + 4,  # Add multi-path info
            ResolutionLevel.REGION: self.base_builder.edge_feature_dim + 8  # Add strategic info
        }
    
    def _calculate_grid_dimensions(self) -> Dict[ResolutionLevel, Tuple[int, int]]:
        """Calculate grid dimensions for each resolution level."""
        level_pixel_width = MAP_TILE_WIDTH * TILE_PIXEL_SIZE  # 1008px
        level_pixel_height = MAP_TILE_HEIGHT * TILE_PIXEL_SIZE  # 552px
        
        dimensions = {}
        for level, resolution in self.resolutions.items():
            width = level_pixel_width // resolution
            height = level_pixel_height // resolution
            dimensions[level] = (width, height)
        
        return dimensions
    
    def build_hierarchical_graph(
        self,
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[int] = None
    ) -> HierarchicalGraphData:
        """
        Build hierarchical graph representation with multiple resolution levels.
        
        Args:
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            entities: List of entity dictionaries
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            
        Returns:
            HierarchicalGraphData with multi-resolution graphs and connectivity
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If graph building fails
        """
        # Input validation
        if not isinstance(level_data, dict):
            raise ValueError("level_data must be a dictionary")
        if not isinstance(ninja_position, (tuple, list)) or len(ninja_position) != 2:
            raise ValueError("ninja_position must be a tuple/list of length 2")
        if not isinstance(entities, list):
            raise ValueError("entities must be a list")
        
        try:
            # Build base sub-cell graph using existing builder
            sub_cell_graph = self.base_builder.build_graph(
                level_data, ninja_position, entities, ninja_velocity, ninja_state
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build sub-cell graph: {e}") from e
        
        # Build tile-level graph through coarsening
        tile_graph, sub_to_tile_mapping = self._build_coarsened_graph(
            sub_cell_graph, ResolutionLevel.TILE, level_data, ninja_position, entities
        )
        
        # Build region-level graph through further coarsening
        region_graph, tile_to_region_mapping = self._build_coarsened_graph(
            tile_graph, ResolutionLevel.REGION, level_data, ninja_position, entities
        )
        
        # Build cross-scale connectivity
        cross_scale_edges = self._build_cross_scale_edges(
            sub_cell_graph, tile_graph, region_graph,
            sub_to_tile_mapping, tile_to_region_mapping
        )
        
        # Collect resolution metadata
        resolution_info = {
            'resolutions': self.resolutions,
            'grid_dimensions': self.grid_dimensions,
            'node_counts': {
                ResolutionLevel.SUB_CELL: sub_cell_graph.num_nodes,
                ResolutionLevel.TILE: tile_graph.num_nodes,
                ResolutionLevel.REGION: region_graph.num_nodes
            },
            'edge_counts': {
                ResolutionLevel.SUB_CELL: sub_cell_graph.num_edges,
                ResolutionLevel.TILE: tile_graph.num_edges,
                ResolutionLevel.REGION: region_graph.num_edges
            }
        }
        
        return HierarchicalGraphData(
            sub_cell_graph=sub_cell_graph,
            tile_graph=tile_graph,
            region_graph=region_graph,
            sub_to_tile_mapping=sub_to_tile_mapping,
            tile_to_region_mapping=tile_to_region_mapping,
            cross_scale_edges=cross_scale_edges,
            resolution_info=resolution_info
        )
    
    def _build_coarsened_graph(
        self,
        fine_graph: GraphData,
        target_level: ResolutionLevel,
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]]
    ) -> Tuple[GraphData, np.ndarray]:
        """
        Build coarsened graph at target resolution level.
        
        Args:
            fine_graph: Higher resolution graph to coarsen
            target_level: Target resolution level
            level_data: Original level data
            ninja_position: Ninja position
            entities: Entity list
            
        Returns:
            Tuple of (coarsened_graph, node_mapping)
        """
        target_width, target_height = self.grid_dimensions[target_level]
        max_nodes = self.max_nodes_per_level[target_level]
        node_feat_dim = self.node_feature_dims[target_level]
        edge_feat_dim = self.edge_feature_dims[target_level]
        
        # Calculate coarsening factor
        if target_level == ResolutionLevel.TILE:
            coarsening_factor = 4  # 4x4 sub-cells per tile
            source_width, source_height = self.grid_dimensions[ResolutionLevel.SUB_CELL]
        else:  # REGION level
            coarsening_factor = 4  # 4x4 tiles per region
            source_width, source_height = self.grid_dimensions[ResolutionLevel.TILE]
        
        # Initialize coarsened graph arrays
        node_features = np.zeros((max_nodes, node_feat_dim), dtype=np.float32)
        edge_index = np.zeros((2, max_nodes * 8), dtype=np.int32)  # Conservative edge estimate
        edge_features = np.zeros((max_nodes * 8, edge_feat_dim), dtype=np.float32)
        node_mask = np.zeros(max_nodes, dtype=np.float32)
        edge_mask = np.zeros(max_nodes * 8, dtype=np.float32)
        
        # Node mapping from fine to coarse
        node_mapping = np.full(fine_graph.num_nodes, -1, dtype=np.int32)
        
        node_count = 0
        edge_count = 0
        
        # Build coarsened nodes by aggregating fine-level nodes
        coarse_node_map = {}  # (coarse_row, coarse_col) -> node_idx
        
        for coarse_row in range(target_height):
            for coarse_col in range(target_width):
                if node_count >= max_nodes:
                    break
                
                # Aggregate features from fine-level nodes in this coarse cell
                aggregated_features = self._aggregate_node_features(
                    fine_graph, coarse_row, coarse_col, coarsening_factor,
                    source_width, source_height, target_level
                )
                
                # Skip empty regions
                if aggregated_features is None:
                    continue
                
                node_idx = node_count
                coarse_node_map[(coarse_row, coarse_col)] = node_idx
                
                # Update node mapping for all fine nodes in this coarse cell
                self._update_node_mapping(
                    node_mapping, coarse_row, coarse_col, coarsening_factor,
                    source_width, source_height, node_idx
                )
                
                node_features[node_idx] = aggregated_features
                node_mask[node_idx] = 1.0
                node_count += 1
        
        # Build coarsened edges
        edges_to_add = []
        
        # Spatial connectivity edges
        for (coarse_row, coarse_col), src_idx in coarse_node_map.items():
            # Check 8-connected neighbors
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            
            for dr, dc in directions:
                new_row, new_col = coarse_row + dr, coarse_col + dc
                
                if (new_row, new_col) in coarse_node_map:
                    tgt_idx = coarse_node_map[(new_row, new_col)]
                    
                    # Aggregate edge features from fine level
                    edge_features_agg = self._aggregate_edge_features(
                        fine_graph, coarse_row, coarse_col, new_row, new_col,
                        coarsening_factor, source_width, source_height, target_level
                    )
                    
                    if edge_features_agg is not None:
                        # Normalize direction
                        norm = np.sqrt(dr*dr + dc*dc)
                        dx, dy = dc / norm, dr / norm
                        edges_to_add.append((src_idx, tgt_idx, edge_features_agg, dx, dy))
        
        # Add entity connectivity at coarse level
        entity_edges = self._build_coarse_entity_edges(
            entities, coarse_node_map, target_level, coarsening_factor
        )
        edges_to_add.extend(entity_edges)
        
        # Sort and add edges
        edges_to_add.sort(key=lambda e: (e[0], e[1]))
        
        for src, tgt, edge_feat, dx, dy in edges_to_add:
            if edge_count >= max_nodes * 8:
                break
            
            edge_index[0, edge_count] = src
            edge_index[1, edge_count] = tgt
            edge_features[edge_count] = edge_feat
            edge_mask[edge_count] = 1.0
            edge_count += 1
        
        # Trim edge arrays to actual size
        edge_index = edge_index[:, :edge_count]
        edge_features = edge_features[:edge_count]
        edge_mask = edge_mask[:edge_count]
        
        coarsened_graph = GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            num_nodes=node_count,
            num_edges=edge_count
        )
        
        return coarsened_graph, node_mapping
    
    def _aggregate_node_features(
        self,
        fine_graph: GraphData,
        coarse_row: int,
        coarse_col: int,
        coarsening_factor: int,
        source_width: int,
        source_height: int,
        target_level: ResolutionLevel
    ) -> Optional[np.ndarray]:
        """
        Aggregate node features from fine level to coarse level.
        
        Args:
            fine_graph: Fine-level graph
            coarse_row, coarse_col: Coarse cell coordinates
            coarsening_factor: How many fine cells per coarse cell
            source_width, source_height: Fine level dimensions
            target_level: Target resolution level
            
        Returns:
            Aggregated feature vector or None if empty region
        """
        # Calculate fine-level region bounds
        fine_row_start = coarse_row * coarsening_factor
        fine_row_end = min(fine_row_start + coarsening_factor, source_height)
        fine_col_start = coarse_col * coarsening_factor
        fine_col_end = min(fine_col_start + coarsening_factor, source_width)
        
        # Collect features from fine nodes in this region
        region_features = []
        region_masks = []
        
        for fine_row in range(fine_row_start, fine_row_end):
            for fine_col in range(fine_col_start, fine_col_end):
                fine_node_idx = fine_row * source_width + fine_col
                
                if fine_node_idx < fine_graph.num_nodes and fine_graph.node_mask[fine_node_idx] > 0:
                    region_features.append(fine_graph.node_features[fine_node_idx])
                    region_masks.append(1.0)
        
        if not region_features:
            return None
        
        region_features = np.array(region_features)
        region_masks = np.array(region_masks)
        
        # Aggregate features using multiple statistics
        base_features = np.mean(region_features, axis=0)  # Mean aggregation
        
        # Add level-specific aggregation statistics
        if target_level == ResolutionLevel.TILE:
            # Add basic statistics for tile level
            max_features = np.max(region_features, axis=0)
            min_features = np.min(region_features, axis=0)
            std_features = np.std(region_features, axis=0)
            
            # Select key statistics to avoid dimension explosion
            key_stats = np.array([
                np.mean(max_features[:10]),  # Max of first 10 features
                np.mean(min_features[:10]),  # Min of first 10 features
                np.mean(std_features[:10]),  # Std of first 10 features
                len(region_features),        # Count of nodes in region
                np.sum(region_masks),        # Active node count
                np.mean(region_features[:, -5:].flatten()),  # Mean of physics features
                np.max(region_features[:, -5:].flatten()),   # Max of physics features
                np.std(region_features[:, -5:].flatten())    # Std of physics features
            ])
            
            aggregated = np.concatenate([base_features, key_stats])
            
        elif target_level == ResolutionLevel.REGION:
            # Add comprehensive statistics for region level
            max_features = np.max(region_features, axis=0)
            min_features = np.min(region_features, axis=0)
            std_features = np.std(region_features, axis=0)
            median_features = np.median(region_features, axis=0)
            
            # Strategic-level statistics
            strategic_stats = np.array([
                np.mean(max_features[:20]),     # Max of key features
                np.mean(min_features[:20]),     # Min of key features
                np.mean(std_features[:20]),     # Std of key features
                np.mean(median_features[:20]),  # Median of key features
                len(region_features),           # Total node count
                np.sum(region_masks),           # Active node count
                np.mean(region_features[:, -10:].flatten()),  # Mean of physics features
                np.max(region_features[:, -10:].flatten()),   # Max of physics features
                np.min(region_features[:, -10:].flatten()),   # Min of physics features
                np.std(region_features[:, -10:].flatten()),   # Std of physics features
                # Connectivity statistics
                np.mean(region_features[:, 40:50].flatten()) if region_features.shape[1] > 50 else 0.0,
                np.max(region_features[:, 40:50].flatten()) if region_features.shape[1] > 50 else 0.0,
                # Hazard density
                np.mean(region_features[:, 38:42].flatten()) if region_features.shape[1] > 42 else 0.0,
                # Entity density
                np.sum(region_features[:, 42:62].flatten()) if region_features.shape[1] > 62 else 0.0,
                # Movement complexity
                np.std(region_features[:, -18:].flatten()) if region_features.shape[1] > 18 else 0.0,
                # Strategic value (placeholder for future enhancement)
                0.0
            ])
            
            aggregated = np.concatenate([base_features, strategic_stats])
        else:
            aggregated = base_features
        
        return aggregated.astype(np.float32)
    
    def _update_node_mapping(
        self,
        node_mapping: np.ndarray,
        coarse_row: int,
        coarse_col: int,
        coarsening_factor: int,
        source_width: int,
        source_height: int,
        coarse_node_idx: int
    ):
        """Update node mapping from fine to coarse level."""
        fine_row_start = coarse_row * coarsening_factor
        fine_row_end = min(fine_row_start + coarsening_factor, source_height)
        fine_col_start = coarse_col * coarsening_factor
        fine_col_end = min(fine_col_start + coarsening_factor, source_width)
        
        for fine_row in range(fine_row_start, fine_row_end):
            for fine_col in range(fine_col_start, fine_col_end):
                fine_node_idx = fine_row * source_width + fine_col
                if fine_node_idx < len(node_mapping):
                    node_mapping[fine_node_idx] = coarse_node_idx
    
    def _aggregate_edge_features(
        self,
        fine_graph: GraphData,
        src_coarse_row: int,
        src_coarse_col: int,
        tgt_coarse_row: int,
        tgt_coarse_col: int,
        coarsening_factor: int,
        source_width: int,
        source_height: int,
        target_level: ResolutionLevel
    ) -> Optional[np.ndarray]:
        """
        Aggregate edge features between coarse regions.
        
        Finds all fine-level edges that cross between the two coarse regions
        and aggregates their features.
        """
        # Get fine node indices for source and target regions (as sets for O(1) lookup)
        src_fine_nodes = set(self._get_region_fine_nodes(
            src_coarse_row, src_coarse_col, coarsening_factor, source_width, source_height
        ))
        tgt_fine_nodes = set(self._get_region_fine_nodes(
            tgt_coarse_row, tgt_coarse_col, coarsening_factor, source_width, source_height
        ))
        
        # Find edges between these regions (optimized with set lookup)
        cross_region_edges = []
        
        for edge_idx in range(fine_graph.num_edges):
            if fine_graph.edge_mask[edge_idx] == 0:
                continue
                
            src_node = fine_graph.edge_index[0, edge_idx]
            tgt_node = fine_graph.edge_index[1, edge_idx]
            
            if src_node in src_fine_nodes and tgt_node in tgt_fine_nodes:
                cross_region_edges.append(fine_graph.edge_features[edge_idx])
        
        if not cross_region_edges:
            return None
        
        cross_region_edges = np.array(cross_region_edges)
        
        # Aggregate edge features
        base_features = np.mean(cross_region_edges, axis=0)
        
        # Add level-specific edge statistics
        if target_level == ResolutionLevel.TILE:
            edge_stats = np.array([
                len(cross_region_edges),  # Number of connections
                np.max(cross_region_edges[:, 2]),  # Max traversal cost
                np.min(cross_region_edges[:, 2]),  # Min traversal cost
                np.std(cross_region_edges[:, 2])   # Cost variability
            ])
            aggregated = np.concatenate([base_features, edge_stats])
            
        elif target_level == ResolutionLevel.REGION:
            edge_stats = np.array([
                len(cross_region_edges),  # Number of connections
                np.max(cross_region_edges[:, 2]),  # Max traversal cost
                np.min(cross_region_edges[:, 2]),  # Min traversal cost
                np.std(cross_region_edges[:, 2]),  # Cost variability
                np.mean(cross_region_edges[:, 9:12]),  # Mean trajectory features
                np.max(cross_region_edges[:, 9:12]),   # Max trajectory features
                np.sum(cross_region_edges[:, 13:15]),  # Movement requirements
                np.mean(cross_region_edges[:, 6:9])    # Mean direction features
            ])
            aggregated = np.concatenate([base_features, edge_stats])
        else:
            aggregated = base_features
        
        return aggregated.astype(np.float32)
    
    def _get_region_fine_nodes(
        self,
        coarse_row: int,
        coarse_col: int,
        coarsening_factor: int,
        source_width: int,
        source_height: int
    ) -> set:
        """Get set of fine node indices in a coarse region."""
        fine_nodes = set()
        
        fine_row_start = coarse_row * coarsening_factor
        fine_row_end = min(fine_row_start + coarsening_factor, source_height)
        fine_col_start = coarse_col * coarsening_factor
        fine_col_end = min(fine_col_start + coarsening_factor, source_width)
        
        for fine_row in range(fine_row_start, fine_row_end):
            for fine_col in range(fine_col_start, fine_col_end):
                fine_node_idx = fine_row * source_width + fine_col
                fine_nodes.add(fine_node_idx)
        
        return fine_nodes
    
    def _build_coarse_entity_edges(
        self,
        entities: List[Dict[str, Any]],
        coarse_node_map: Dict[Tuple[int, int], int],
        target_level: ResolutionLevel,
        coarsening_factor: int
    ) -> List[Tuple[int, int, np.ndarray, float, float]]:
        """Build entity-related edges at coarse resolution."""
        entity_edges = []
        
        # For now, create simple connectivity based on entity proximity
        # This can be enhanced with more sophisticated entity relationship modeling
        
        for entity in entities:
            entity_pos = entity.get('position', (0, 0))
            entity_type = entity.get('type', 'unknown')
            
            # Map entity position to coarse grid
            resolution = self.resolutions[target_level]
            coarse_col = int(entity_pos[0] // resolution)
            coarse_row = int(entity_pos[1] // resolution)
            
            if (coarse_row, coarse_col) in coarse_node_map:
                entity_node = coarse_node_map[(coarse_row, coarse_col)]
                
                # Create edges to nearby coarse nodes for functional relationships
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                            
                        neighbor_pos = (coarse_row + dr, coarse_col + dc)
                        if neighbor_pos in coarse_node_map:
                            neighbor_node = coarse_node_map[neighbor_pos]
                            
                            # Create functional edge features
                            edge_feat = np.zeros(self.edge_feature_dims[target_level], dtype=np.float32)
                            edge_feat[EdgeType.FUNCTIONAL] = 1.0  # Functional edge type
                            
                            # Direction
                            norm = np.sqrt(dr*dr + dc*dc)
                            dx, dy = dc / norm, dr / norm
                            edge_feat[len(EdgeType)] = dx
                            edge_feat[len(EdgeType) + 1] = dy
                            
                            # Cost based on entity type
                            if entity_type in ['switch', 'key']:
                                edge_feat[len(EdgeType) + 2] = 0.5  # Low cost for interactive
                            elif entity_type in ['hazard', 'enemy']:
                                edge_feat[len(EdgeType) + 2] = 5.0  # High cost for dangerous
                            else:
                                edge_feat[len(EdgeType) + 2] = 1.0  # Default cost
                            
                            entity_edges.append((entity_node, neighbor_node, edge_feat, dx, dy))
        
        return entity_edges
    
    def _build_cross_scale_edges(
        self,
        sub_cell_graph: GraphData,
        tile_graph: GraphData,
        region_graph: GraphData,
        sub_to_tile_mapping: np.ndarray,
        tile_to_region_mapping: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Build cross-scale edges for information flow between resolution levels.
        
        Creates 'up' edges (fine to coarse) and 'down' edges (coarse to fine)
        to enable hierarchical information propagation.
        """
        cross_scale_edges = {}
        
        # Sub-cell to tile edges (up)
        sub_to_tile_edges = []
        for sub_idx in range(sub_cell_graph.num_nodes):
            if sub_cell_graph.node_mask[sub_idx] > 0:
                tile_idx = sub_to_tile_mapping[sub_idx]
                if tile_idx >= 0 and tile_idx < tile_graph.num_nodes:
                    sub_to_tile_edges.append([sub_idx, tile_idx])
        
        cross_scale_edges['sub_to_tile'] = np.array(sub_to_tile_edges).T if sub_to_tile_edges else np.zeros((2, 0))
        
        # Tile to region edges (up)
        tile_to_region_edges = []
        for tile_idx in range(tile_graph.num_nodes):
            if tile_graph.node_mask[tile_idx] > 0:
                region_idx = tile_to_region_mapping[tile_idx]
                if region_idx >= 0 and region_idx < region_graph.num_nodes:
                    tile_to_region_edges.append([tile_idx, region_idx])
        
        cross_scale_edges['tile_to_region'] = np.array(tile_to_region_edges).T if tile_to_region_edges else np.zeros((2, 0))
        
        # Reverse mappings for down edges
        cross_scale_edges['tile_to_sub'] = cross_scale_edges['sub_to_tile'][[1, 0]] if cross_scale_edges['sub_to_tile'].size > 0 else np.zeros((2, 0))
        cross_scale_edges['region_to_tile'] = cross_scale_edges['tile_to_region'][[1, 0]] if cross_scale_edges['tile_to_region'].size > 0 else np.zeros((2, 0))
        
        return cross_scale_edges