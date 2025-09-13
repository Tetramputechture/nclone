#!/usr/bin/env python3
"""
Comprehensive unit tests for the graph visualization fixes.

This test suite ensures that all three issues remain resolved:
1. Functional edges (switch-door connections)
2. Walkable edges in solid tiles
3. Ninja navigation from solid spawn tile
"""

import os
import sys
import unittest

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from nclone.nclone_environments.npp_environment import NppEnvironment
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.navigation import PathfindingEngine
from nclone.graph.common import EdgeType


class TestGraphVisualizationFixes(unittest.TestCase):
    """Test suite for graph visualization fixes."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create environment
        cls.env = NppEnvironment(
            render_mode="rgb_array",
            enable_frame_stack=False,
            enable_debug_overlay=False,
            eval_mode=False,
            seed=42
        )
        
        # Reset to load the map
        cls.env.reset()
        
        # Get level data and ninja position
        cls.level_data = cls.env.level_data
        cls.ninja_pos = cls.env.nplay_headless.ninja_position()
        
        # Build graph
        cls.graph_builder = HierarchicalGraphBuilder()
        cls.hierarchical_data = cls.graph_builder.build_graph(cls.level_data, cls.ninja_pos)
        cls.graph_data = cls.hierarchical_data.sub_cell_graph
        cls.navigation_engine = PathfindingEngine()
        
        # Find ninja node
        cls.ninja_node = cls.navigation_engine._find_node_at_position(cls.graph_data, cls.ninja_pos)
    
    def test_issue1_functional_edges_exist(self):
        """Test that functional edges exist between switches and doors."""
        functional_edges = []
        
        for edge_idx in range(self.graph_data.num_edges):
            if self.graph_data.edge_mask[edge_idx] == 0:
                continue
            
            edge_type = int(self.graph_data.edge_types[edge_idx])
            if edge_type == EdgeType.FUNCTIONAL:
                src_idx = int(self.graph_data.edge_index[0, edge_idx])
                dst_idx = int(self.graph_data.edge_index[1, edge_idx])
                
                src_pos = self.navigation_engine._get_node_position(self.graph_data, src_idx)
                dst_pos = self.navigation_engine._get_node_position(self.graph_data, dst_idx)
                
                functional_edges.append((src_pos, dst_pos))
        
        # Should have at least 2 functional edges (switch-door connections)
        self.assertGreaterEqual(len(functional_edges), 2, 
                               "Should have at least 2 functional edges for switch-door connections")
        
        # Verify specific expected connections exist
        expected_connections = [
            ((396.0, 204.0), (480.0, 276.0)),  # Switch to door
            ((468.0, 204.0), (504.0, 276.0)),  # Switch to door
        ]
        
        for expected_src, expected_dst in expected_connections:
            found = any(
                (abs(src[0] - expected_src[0]) < 1 and abs(src[1] - expected_src[1]) < 1 and
                 abs(dst[0] - expected_dst[0]) < 1 and abs(dst[1] - expected_dst[1]) < 1)
                for src, dst in functional_edges
            )
            self.assertTrue(found, f"Expected functional edge {expected_src} -> {expected_dst} not found")
    
    def test_issue2_no_invalid_solid_tile_edges(self):
        """Test that there are no invalid walkable edges in solid tiles."""
        invalid_solid_edges = 0
        ninja_escape_edges = 0
        
        for edge_idx in range(self.graph_data.num_edges):
            if self.graph_data.edge_mask[edge_idx] == 0:
                continue
            
            edge_type = int(self.graph_data.edge_types[edge_idx])
            if edge_type == EdgeType.WALK:
                src_idx = int(self.graph_data.edge_index[0, edge_idx])
                dst_idx = int(self.graph_data.edge_index[1, edge_idx])
                
                src_pos = self.navigation_engine._get_node_position(self.graph_data, src_idx)
                dst_pos = self.navigation_engine._get_node_position(self.graph_data, dst_idx)
                
                # Check if source is in solid tile
                src_tile_x = int(src_pos[0] // 24)
                src_tile_y = int(src_pos[1] // 24)
                
                if (0 <= src_tile_x < self.level_data.width and 
                    0 <= src_tile_y < self.level_data.height):
                    tile_value = self.level_data.get_tile(src_tile_y, src_tile_x)
                    
                    if tile_value == 1:  # Solid tile
                        # Check if this is a ninja escape edge (within 24px of ninja)
                        ninja_distance_src = ((src_pos[0] - self.ninja_pos[0])**2 + (src_pos[1] - self.ninja_pos[1])**2)**0.5
                        ninja_distance_dst = ((dst_pos[0] - self.ninja_pos[0])**2 + (dst_pos[1] - self.ninja_pos[1])**2)**0.5
                        
                        if ninja_distance_src <= 24 or ninja_distance_dst <= 24:
                            ninja_escape_edges += 1
                        else:
                            invalid_solid_edges += 1
        
        # Should have no invalid solid tile edges
        self.assertEqual(invalid_solid_edges, 0, 
                        "Should have no invalid walkable edges in solid tiles")
        
        # Should have ninja escape edges (intentional)
        self.assertGreater(ninja_escape_edges, 0, 
                          "Should have ninja escape edges for spawn tile navigation")
    
    def test_issue3_ninja_navigation_local(self):
        """Test that ninja can navigate to nearby empty areas."""
        self.assertIsNotNone(self.ninja_node, "Ninja node should be found")
        
        # Test navigation to nearby empty tile positions
        nearby_targets = [
            (129, 429),  # Empty tile
            (135, 429),  # Empty tile
            (123, 429),  # Empty tile
            (141, 429),  # Empty tile
        ]
        
        successful_paths = 0
        
        for target_pos in nearby_targets:
            target_node = self.navigation_engine._find_node_at_position(self.graph_data, target_pos)
            
            if target_node is not None:
                path_result = self.navigation_engine.find_shortest_path(
                    self.graph_data, self.ninja_node, target_node
                )
                
                if path_result and path_result.success and len(path_result.path) > 0:
                    successful_paths += 1
        
        # Should achieve high success rate for local navigation
        success_rate = (successful_paths / len(nearby_targets)) * 100
        self.assertGreaterEqual(success_rate, 75, 
                               f"Local navigation success rate should be >= 75%, got {success_rate:.1f}%")
    
    def test_issue3_ninja_connectivity_improvement(self):
        """Test that ninja's connected component is significantly larger."""
        # Find ninja's connected component
        visited = set()
        stack = [self.ninja_node]
        ninja_component = []
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            ninja_component.append(current)
            
            # Find neighbors
            for edge_idx in range(self.graph_data.num_edges):
                if self.graph_data.edge_mask[edge_idx] == 0:
                    continue
                
                src = int(self.graph_data.edge_index[0, edge_idx])
                dst = int(self.graph_data.edge_index[1, edge_idx])
                
                if src == current and dst not in visited:
                    stack.append(dst)
                elif dst == current and src not in visited:
                    stack.append(src)
        
        # Should have significantly more than the original 44 nodes
        self.assertGreater(len(ninja_component), 100, 
                          f"Ninja's connected component should be > 100 nodes, got {len(ninja_component)}")
        
        # Should be at least 5x improvement from original 44 nodes
        improvement_factor = len(ninja_component) / 44
        self.assertGreaterEqual(improvement_factor, 5.0, 
                               f"Connectivity improvement should be >= 5x, got {improvement_factor:.1f}x")
    
    def test_corridor_connections_exist(self):
        """Test that corridor connections were added to the graph."""
        # Count total edges
        total_edges = sum(1 for i in range(self.graph_data.num_edges) if self.graph_data.edge_mask[i] == 1)
        
        # Should have significantly more edges than the original ~3246
        self.assertGreater(total_edges, 3500, 
                          f"Should have > 3500 edges with corridor connections, got {total_edges}")
    
    def test_long_distance_navigation_improvement(self):
        """Test that some long-distance navigation is now possible."""
        # Test targets in different empty tile clusters
        distant_targets = [
            (168, 324),  # Cluster 5 center
            (192, 396),  # Cluster 6 center
            (120, 420),  # Cluster 7 center
            (156, 252),  # Cluster 8 center
        ]
        
        successful_paths = 0
        
        for target_pos in distant_targets:
            target_node = self.navigation_engine._find_node_at_position(self.graph_data, target_pos)
            
            if target_node is not None:
                path_result = self.navigation_engine.find_shortest_path(
                    self.graph_data, self.ninja_node, target_node
                )
                
                if path_result and path_result.success and len(path_result.path) > 0:
                    successful_paths += 1
        
        # Should achieve some success for long-distance navigation
        success_rate = (successful_paths / len(distant_targets)) * 100
        self.assertGreaterEqual(success_rate, 25, 
                               f"Long-distance navigation success rate should be >= 25%, got {success_rate:.1f}%")
    
    def test_graph_structure_integrity(self):
        """Test that the graph structure is valid and consistent."""
        # Check that all edges have valid node indices
        for edge_idx in range(self.graph_data.num_edges):
            if self.graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src_idx = int(self.graph_data.edge_index[0, edge_idx])
            dst_idx = int(self.graph_data.edge_index[1, edge_idx])
            
            # Indices should be within valid range
            self.assertGreaterEqual(src_idx, 0, f"Source index {src_idx} should be >= 0")
            self.assertLess(src_idx, self.graph_data.num_nodes, 
                           f"Source index {src_idx} should be < {self.graph_data.num_nodes}")
            
            self.assertGreaterEqual(dst_idx, 0, f"Destination index {dst_idx} should be >= 0")
            self.assertLess(dst_idx, self.graph_data.num_nodes, 
                           f"Destination index {dst_idx} should be < {self.graph_data.num_nodes}")
            
            # Nodes should be active
            self.assertEqual(self.graph_data.node_mask[src_idx], 1, 
                           f"Source node {src_idx} should be active")
            self.assertEqual(self.graph_data.node_mask[dst_idx], 1, 
                           f"Destination node {dst_idx} should be active")
    
    def test_edge_types_valid(self):
        """Test that all edge types are valid."""
        valid_edge_types = {EdgeType.WALK, EdgeType.JUMP, EdgeType.FALL, EdgeType.FUNCTIONAL}
        
        for edge_idx in range(self.graph_data.num_edges):
            if self.graph_data.edge_mask[edge_idx] == 0:
                continue
            
            edge_type = int(self.graph_data.edge_types[edge_idx])
            self.assertIn(edge_type, valid_edge_types, 
                         f"Edge type {edge_type} should be valid")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)