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
from .level_data import LevelData, ensure_level_data
from .edge_building import EdgeBuilder, create_simplified_graph_data
# from .reachability.tiered_system import TieredReachabilitySystem  # Disabled for simplified approach


class ResolutionLevel(Enum):
    """Multi-resolution levels for hierarchical analysis."""
    FINE = "6px"      # 6px resolution (detailed local movement)
    MEDIUM = "24px"   # 24px resolution (tile-level planning)  
    COARSE = "96px"   # 96px resolution (strategic overview)


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
        return (self.fine_graph.num_nodes + 
                self.medium_graph.num_nodes + 
                self.coarse_graph.num_nodes)
    
    @property
    def total_edges(self) -> int:
        """Total edges across all resolution levels."""
        return (self.fine_graph.num_edges + 
                self.medium_graph.num_edges + 
                self.coarse_graph.num_edges)


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
        # self.reachability_system = TieredReachabilitySystem(debug=debug)  # Disabled for simplified approach
        self.reachability_system = None
        # self.edge_builder.set_reachability_system(self.reachability_system)
    
    def build_graph(self, level_data, entities: List, ninja_pos: Tuple[int, int]) -> HierarchicalGraphData:
        """
        Build simplified hierarchical graph for strategic RL.
        
        Args:
            level_data: Level tile data (dict or LevelData)
            entities: List of game entities
            ninja_pos: Current ninja position
            
        Returns:
            HierarchicalGraphData with multi-resolution graphs
        """
        # Ensure level_data is in correct format
        level_data = ensure_level_data(level_data)
        
        if self.debug:
            print("Building simplified hierarchical graph...")
            print(f"  Level size: {level_data.width}x{level_data.height}")
            print(f"  Entities: {len(entities)}")
            print(f"  Ninja position: {ninja_pos}")
        
        # Build graphs at different resolutions
        fine_graph = self._build_resolution_graph(level_data, entities, ninja_pos, ResolutionLevel.FINE)
        medium_graph = self._build_resolution_graph(level_data, entities, ninja_pos, ResolutionLevel.MEDIUM)
        coarse_graph = self._build_resolution_graph(level_data, entities, ninja_pos, ResolutionLevel.COARSE)
        
        # Extract reachability information
        reachability_info = self._extract_reachability_info(level_data, entities, ninja_pos)
        
        # Extract strategic features
        strategic_features = self._extract_strategic_features(level_data, entities, ninja_pos)
        
        result = HierarchicalGraphData(
            fine_graph=fine_graph,
            medium_graph=medium_graph,
            coarse_graph=coarse_graph,
            reachability_info=reachability_info,
            strategic_features=strategic_features
        )
        
        if self.debug:
            print("Built hierarchical graph:")
            print(f"  Fine: {fine_graph.num_nodes} nodes, {fine_graph.num_edges} edges")
            print(f"  Medium: {medium_graph.num_nodes} nodes, {medium_graph.num_edges} edges")
            print(f"  Coarse: {coarse_graph.num_nodes} nodes, {coarse_graph.num_edges} edges")
            print(f"  Total: {result.total_nodes} nodes, {result.total_edges} edges")
        
        return result
    
    def _build_resolution_graph(self, level_data: LevelData, entities: List, 
                              ninja_pos: Tuple[int, int], resolution: ResolutionLevel) -> GraphData:
        """Build graph at specific resolution level."""
        
        # Downsample level data based on resolution
        downsampled_level = self._downsample_level_data(level_data, resolution)
        
        # Filter entities based on resolution
        filtered_entities = self._filter_entities_for_resolution(entities, resolution)
        
        # Adjust ninja position for resolution
        adjusted_ninja_pos = self._adjust_position_for_resolution(ninja_pos, resolution)
        
        # Build edges using simplified approach
        edges = self.edge_builder.build_edges(downsampled_level, filtered_entities, adjusted_ninja_pos)
        
        # Convert to GraphData
        graph_data = create_simplified_graph_data(edges, downsampled_level, filtered_entities)
        
        return graph_data
    
    def _downsample_level_data(self, level_data: LevelData, resolution: ResolutionLevel) -> LevelData:
        """Downsample level data to target resolution."""
        if resolution == ResolutionLevel.FINE:
            # 6px resolution - higher detail than tile level
            scale_factor = 4  # 24px / 6px = 4
            new_width = level_data.width * scale_factor
            new_height = level_data.height * scale_factor
            
            # Upsample tiles array
            upsampled_tiles = np.repeat(np.repeat(level_data.tiles, scale_factor, axis=0), scale_factor, axis=1)
            
            return LevelData(
                tiles=upsampled_tiles,
                entities=[]
            )
            
        elif resolution == ResolutionLevel.MEDIUM:
            # 24px resolution - tile level (no change)
            return level_data
            
        elif resolution == ResolutionLevel.COARSE:
            # 96px resolution - 4x4 tile blocks
            scale_factor = 4  # 96px / 24px = 4
            new_width = level_data.width // scale_factor
            new_height = level_data.height // scale_factor
            
            # Downsample by taking max value in each block (wall if any wall present)
            downsampled_tiles = np.zeros((new_height, new_width), dtype=level_data.tiles.dtype)
            
            for row in range(new_height):
                for col in range(new_width):
                    block = level_data.tiles[row*scale_factor:(row+1)*scale_factor,
                                           col*scale_factor:(col+1)*scale_factor]
                    # Take max value (wall takes precedence over empty)
                    downsampled_tiles[row, col] = np.max(block)
            
            return LevelData(
                tiles=downsampled_tiles,
                entities=[]
            )
    
    def _filter_entities_for_resolution(self, entities: List, resolution: ResolutionLevel) -> List:
        """Filter entities based on resolution level."""
        if resolution == ResolutionLevel.FINE:
            # Include all entities for detailed analysis
            return entities
            
        elif resolution == ResolutionLevel.MEDIUM:
            # Include important entities (switches, doors, gold, exit)
            important_types = ['switch', 'door', 'gold', 'exit', 'spawn']
            return [e for e in entities if any(t in type(e).__name__.lower() for t in important_types)]
            
        elif resolution == ResolutionLevel.COARSE:
            # Include only strategic entities (exit, major switches)
            strategic_types = ['exit', 'spawn']
            return [e for e in entities if any(t in type(e).__name__.lower() for t in strategic_types)]
    
    def _adjust_position_for_resolution(self, pos: Tuple[int, int], resolution: ResolutionLevel) -> Tuple[int, int]:
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
            return (x // 4, y // 4)  # 24px -> 96px
    
    def _extract_reachability_info(self, level_data: LevelData, entities: List, 
                                 ninja_pos: Tuple[int, int]) -> Dict:
        """Extract reachability information for strategic planning."""
        try:
            # Get reachability analysis from tiered system
            tier1_result = self.reachability_system.tier1_system.quick_check(level_data, entities, ninja_pos)
            tier2_result = self.reachability_system.tier2_system.quick_check(level_data, entities, ninja_pos)
            tier3_result = self.reachability_system.tier3_system.quick_check(level_data, entities, ninja_pos)
            
            return {
                'tier1_reachable_count': getattr(tier1_result, 'reachable_count', 0),
                'tier2_reachable_count': getattr(tier2_result, 'reachable_count', 0),
                'tier3_reachable_count': getattr(tier3_result, 'reachable_count', 0),
                'is_level_completable': getattr(tier3_result, 'is_level_completable', lambda: False)(),
                'connectivity_score': self._calculate_connectivity_score(tier1_result, tier2_result, tier3_result)
            }
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not extract reachability info: {e}")
            return {
                'tier1_reachable_count': 0,
                'tier2_reachable_count': 0,
                'tier3_reachable_count': 0,
                'is_level_completable': False,
                'connectivity_score': 0.0
            }
    
    def _extract_strategic_features(self, level_data: LevelData, entities: List, 
                                  ninja_pos: Tuple[int, int]) -> Dict:
        """Extract strategic features for RL decision making."""
        features = {}
        
        # Entity counts by type
        entity_counts = {}
        for entity in entities:
            entity_type = type(entity).__name__.lower()
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        features['entity_counts'] = entity_counts
        
        # Distance to key entities
        distances = {}
        for entity in entities:
            entity_type = type(entity).__name__.lower()
            if entity_type in ['exit', 'gold', 'switch']:
                dist = np.sqrt((entity.x - ninja_pos[0])**2 + (entity.y - ninja_pos[1])**2)
                if entity_type not in distances:
                    distances[entity_type] = []
                distances[entity_type].append(dist)
        
        # Take minimum distance for each type
        for entity_type, dist_list in distances.items():
            features[f'min_distance_to_{entity_type}'] = min(dist_list) if dist_list else float('inf')
        
        # Level complexity metrics
        features['level_width'] = level_data.width
        features['level_height'] = level_data.height
        features['total_entities'] = len(entities)
        features['wall_density'] = np.mean(level_data.tiles > 0)  # Assuming 0 is empty
        
        return features
    
    def _calculate_connectivity_score(self, tier1_result, tier2_result, tier3_result) -> float:
        """Calculate overall connectivity score for the level."""
        try:
            # Combine reachability counts from all tiers
            total_reachable = (
                getattr(tier1_result, 'reachable_count', 0) +
                getattr(tier2_result, 'reachable_count', 0) +
                getattr(tier3_result, 'reachable_count', 0)
            )
            
            # Normalize by some reasonable maximum
            max_possible = 1000  # Rough estimate
            return min(total_reachable / max_possible, 1.0)
            
        except Exception:
            return 0.0


# Convenience function for backward compatibility
def build_simplified_hierarchical_graph(level_data, entities: List, ninja_pos: Tuple[int, int], 
                                       debug: bool = False) -> HierarchicalGraphData:
    """
    Convenience function to build simplified hierarchical graph.
    
    This function provides a simple interface for building graphs without
    needing to instantiate the builder class directly.
    """
    builder = SimplifiedHierarchicalGraphBuilder(debug=debug)
    return builder.build_graph(level_data, entities, ninja_pos)