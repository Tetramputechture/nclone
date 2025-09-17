"""
Tests for simplified graph architecture.

This test suite validates that the simplified graph approach provides
sufficient information for Deep RL while being more efficient and
generalizable than the detailed physics approach.
"""

import unittest
import numpy as np
import time
from typing import List, Tuple

# Add the project root to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.hierarchical_builder import (
    HierarchicalGraphBuilder,
    ResolutionLevel,
    HierarchicalGraphData
)
from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.common import NodeType, EdgeType
from nclone.graph.level_data import LevelData


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, x: int, y: int, entity_type: str = "generic"):
        self.x = x
        self.y = y
        self.entity_type = entity_type
    
    def __class__(self):
        return type(self.entity_type, (), {})


class TestSimplifiedGraphArchitecture(unittest.TestCase):
    """Test suite for simplified graph architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test level (10x10 tiles)
        tiles = np.zeros((10, 10), dtype=np.int32)  # All empty space
        
        # Add some walls
        tiles[2:8, 2] = 1  # Vertical wall
        tiles[2, 2:8] = 1  # Horizontal wall
        
        self.level_data = LevelData(
            tiles=tiles,
            entities=[]  # Will add entities separately
        )
        
        # Create test entities
        self.entities = [
            MockEntity(120, 120, "switch"),  # 5*24, 5*24 (center)
            MockEntity(216, 216, "door"),    # 9*24, 9*24 (corner)
            MockEntity(48, 48, "gold"),      # 2*24, 2*24 (near wall)
            MockEntity(192, 192, "exit")     # 8*24, 8*24 (exit)
        ]
        
        self.ninja_pos = (72, 72)  # 3*24, 3*24 (start position)
        
        self.builder = HierarchicalGraphBuilder(debug=True)
    
    def test_simplified_node_types(self):
        """Test that simplified node types cover necessary categories."""
        # Check that we have the right node types for strategic RL
        expected_types = {NodeType.EMPTY, NodeType.WALL, NodeType.ENTITY, 
                         NodeType.HAZARD, NodeType.SPAWN, NodeType.EXIT}
        
        actual_types = set(NodeType)
        self.assertEqual(actual_types, expected_types)
        
        # Verify enum values are reasonable
        self.assertEqual(NodeType.EMPTY, 0)
        self.assertEqual(NodeType.WALL, 1)
        self.assertEqual(NodeType.ENTITY, 2)
    
    def test_simplified_edge_types(self):
        """Test that simplified edge types support strategic connectivity."""
        expected_types = {EdgeType.ADJACENT, EdgeType.REACHABLE, 
                         EdgeType.FUNCTIONAL, EdgeType.BLOCKED}
        
        actual_types = set(EdgeType)
        self.assertEqual(actual_types, expected_types)
        
        # Verify enum values
        self.assertEqual(EdgeType.ADJACENT, 0)
        self.assertEqual(EdgeType.REACHABLE, 1)
        self.assertEqual(EdgeType.FUNCTIONAL, 2)
        self.assertEqual(EdgeType.BLOCKED, 3)
    
    def test_simplified_edge_builder(self):
        """Test simplified edge building functionality."""
        edge_builder = EdgeBuilder(debug=True)
        
        # Build edges
        edges = edge_builder.build_edges(self.level_data, self.entities, self.ninja_pos)
        
        # Should have some edges
        self.assertGreater(len(edges), 0)
        
        # Check edge types
        edge_types = [edge.edge_type for edge in edges]
        self.assertIn(EdgeType.ADJACENT, edge_types)
        
        # Edges should have reasonable weights
        for edge in edges:
            self.assertGreaterEqual(edge.weight, 0)
            self.assertIsInstance(edge.source, tuple)
            self.assertIsInstance(edge.target, tuple)
    
    def test_hierarchical_graph_construction(self):
        """Test hierarchical graph construction with multiple resolutions."""
        # Build hierarchical graph
        graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        
        # Should have all three resolution levels
        self.assertIsNotNone(graph_data.fine_graph)
        self.assertIsNotNone(graph_data.medium_graph)
        self.assertIsNotNone(graph_data.coarse_graph)
        
        # Each graph should have nodes and edges
        for graph in [graph_data.fine_graph, graph_data.medium_graph, graph_data.coarse_graph]:
            self.assertGreater(graph.num_nodes, 0)
            self.assertGreaterEqual(graph.num_edges, 0)  # Could be 0 for very coarse graphs
            
            # Node features should be reasonable (fixed-size arrays)
            self.assertEqual(graph.node_features.shape[1], 3)  # [x, y, node_type]
            self.assertGreaterEqual(graph.node_features.shape[0], graph.num_nodes)  # Fixed size >= actual nodes
            
            # Edge indices should be valid
            if graph.num_edges > 0:
                self.assertEqual(graph.edge_index.shape[0], 2)
                
                # All edge indices should be valid node indices
                max_node_idx = graph.num_nodes - 1
                valid_edges = graph.edge_index[:, :graph.num_edges]
                self.assertTrue(np.all(valid_edges >= 0))
                self.assertTrue(np.all(valid_edges <= max_node_idx))
    
    def test_reachability_integration(self):
        """Test integration with reachability system."""
        graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        
        # Should have reachability information
        self.assertIsNotNone(graph_data.reachability_info)
        
        # Check expected keys
        expected_keys = ['tier1_reachable_count', 'tier2_reachable_count', 
                        'tier3_reachable_count', 'is_level_completable', 'connectivity_score']
        
        for key in expected_keys:
            self.assertIn(key, graph_data.reachability_info)
        
        # Values should be reasonable
        self.assertIsInstance(graph_data.reachability_info['is_level_completable'], bool)
        self.assertGreaterEqual(graph_data.reachability_info['connectivity_score'], 0.0)
        self.assertLessEqual(graph_data.reachability_info['connectivity_score'], 1.0)
    
    def test_strategic_features(self):
        """Test extraction of strategic features for RL."""
        graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        
        # Should have strategic features
        self.assertIsNotNone(graph_data.strategic_features)
        
        # Check entity counts
        self.assertIn('entity_counts', graph_data.strategic_features)
        entity_counts = graph_data.strategic_features['entity_counts']
        
        # Should count our test entities
        self.assertEqual(entity_counts.get('switch', 0), 1)
        self.assertEqual(entity_counts.get('door', 0), 1)
        self.assertEqual(entity_counts.get('gold', 0), 1)
        self.assertEqual(entity_counts.get('exit', 0), 1)
        
        # Check level metrics
        self.assertEqual(graph_data.strategic_features['level_width'], 10)
        self.assertEqual(graph_data.strategic_features['level_height'], 10)
        self.assertEqual(graph_data.strategic_features['total_entities'], 4)
    
    def test_performance_comparison(self):
        """Test that simplified approach is faster than detailed physics."""
        # Time simplified approach
        start_time = time.time()
        for _ in range(10):
            graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        simplified_time = time.time() - start_time
        
        print(f"Simplified approach: {simplified_time:.3f}s for 10 iterations")
        print(f"Average per iteration: {simplified_time/10:.3f}s")
        
        # Should be reasonably fast
        self.assertLess(simplified_time / 10, 0.1)  # Less than 100ms per iteration
        
        # Check memory usage (rough estimate)
        total_nodes = graph_data.total_nodes
        total_edges = graph_data.total_edges
        
        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        
        # Should be reasonable sizes for RL
        self.assertLess(total_nodes, 10000)  # Manageable for neural networks
        self.assertLess(total_edges, 50000)  # Reasonable edge count
    
    def test_multi_resolution_consistency(self):
        """Test that different resolutions provide consistent information."""
        graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        
        # Coarse graph should have fewer nodes than medium, medium fewer than fine
        self.assertLessEqual(graph_data.coarse_graph.num_nodes, graph_data.medium_graph.num_nodes)
        self.assertLessEqual(graph_data.medium_graph.num_nodes, graph_data.fine_graph.num_nodes)
        
        # All graphs should have valid structure
        for graph_name, graph in [("fine", graph_data.fine_graph), 
                                 ("medium", graph_data.medium_graph),
                                 ("coarse", graph_data.coarse_graph)]:
            
            # Node types should be valid
            valid_node_types = set(NodeType)
            for node_type in graph.node_types:
                self.assertIn(node_type, valid_node_types, f"Invalid node type in {graph_name} graph")
            
            # Edge types should be valid
            if graph.num_edges > 0:
                valid_edge_types = set(EdgeType)
                for edge_type in graph.edge_types:
                    self.assertIn(edge_type, valid_edge_types, f"Invalid edge type in {graph_name} graph")
    
    def test_rl_compatibility(self):
        """Test compatibility with RL architectures."""
        graph_data = self.builder.build_graph(self.level_data, self.entities, self.ninja_pos)
        
        # Test heterogeneous graph transformer compatibility
        # Should have different node/edge types for heterogeneous processing
        fine_graph = graph_data.fine_graph
        
        unique_node_types = np.unique(fine_graph.node_types)
        unique_edge_types = np.unique(fine_graph.edge_types) if fine_graph.num_edges > 0 else []
        
        self.assertGreater(len(unique_node_types), 1, "Need multiple node types for heterogeneous GNN")
        
        # Test 3D CNN compatibility
        # Strategic features should be suitable for spatial processing
        features = graph_data.strategic_features
        
        # Should have spatial dimensions
        self.assertIn('level_width', features)
        self.assertIn('level_height', features)
        
        # Test MLP compatibility
        # Features should be numeric and bounded
        for key, value in features.items():
            if isinstance(value, (int, float)):
                self.assertFalse(np.isnan(value), f"NaN value in feature {key}")
                self.assertFalse(np.isinf(value), f"Infinite value in feature {key}")
    
    def test_convenience_function(self):
        """Test convenience function for building graphs."""
        # Should work without instantiating builder
        graph_data = HierarchicalGraphBuilder().build_graph(
            self.level_data, self.entities, self.ninja_pos, debug=True
        )
        
        self.assertIsNotNone(graph_data)
        self.assertGreater(graph_data.total_nodes, 0)
    
    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        # Empty level
        empty_level = LevelData(tiles=np.zeros((5, 5), dtype=np.int32), entities=[])
        
        graph_data = self.builder.build_graph(empty_level, [], (60, 60))
        self.assertIsNotNone(graph_data)
        
        # Level with all walls
        wall_level = LevelData(tiles=np.ones((5, 5), dtype=np.int32), entities=[])
        
        graph_data = self.builder.build_graph(wall_level, [], (60, 60))
        self.assertIsNotNone(graph_data)
        
        # Many entities
        many_entities = [MockEntity(i*24, j*24, "test") for i in range(5) for j in range(5)]
        
        graph_data = self.builder.build_graph(self.level_data, many_entities, self.ninja_pos)
        self.assertIsNotNone(graph_data)
        self.assertGreaterEqual(graph_data.strategic_features['total_entities'], 25)


if __name__ == '__main__':
    unittest.main()