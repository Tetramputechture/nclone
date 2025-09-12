#!/usr/bin/env python3
"""
Final comprehensive test validating all graph visualization fixes.

This test validates that all three original issues have been resolved:
1. Functional edges between switches and doors are working
2. No walkable edges in solid tiles (ninja radius clearance enforced)
3. Pathfinding working on traversable paths

Tests are run on both the original bblock_test map and a custom map with solid tiles.
"""

import os
import sys
import unittest

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.navigation import PathfindingEngine
from nclone.graph.common import EdgeType
from nclone.constants import TILE_PIXEL_SIZE
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
import numpy as np


class TestAllGraphFixes(unittest.TestCase):
    """Comprehensive test suite for all graph visualization fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph_builder = HierarchicalGraphBuilder()
        self.navigation_engine = PathfindingEngine()

    def load_bblock_test_map(self):
        """Load the original bblock_test map."""
        map_path = os.path.join('nclone', 'test_maps', 'bblock_test')
        with open(map_path, 'rb') as f:
            map_data = f.read()
        
        sim = Simulator()
        sim.load_map_from_bytes(map_data)
        
        level_data = LevelData()
        level_data.extract_from_simulator(sim)
        
        return level_data, sim

    def load_custom_solid_tiles_map(self):
        """Load the custom map with solid tiles."""
        map_path = os.path.join('debug', 'test_map_with_solid_tiles.dat')
        if not os.path.exists(map_path):
            # Create the map if it doesn't exist
            from debug.create_test_map import create_test_map_with_solid_tiles
            create_test_map_with_solid_tiles()
        
        with open(map_path, 'rb') as f:
            map_data = f.read()
        
        sim = Simulator()
        sim.load_map_from_bytes(map_data)
        
        # Use custom level data extraction for this map
        from debug.test_with_solid_tiles import create_level_data_from_custom_simulator
        level_data = create_level_data_from_custom_simulator(sim)
        
        return level_data, sim

    def test_issue_1_functional_edges_bblock_test(self):
        """Test Issue #1: Functional edges in bblock_test map."""
        print("\n=== Testing Issue #1: Functional edges (bblock_test) ===")
        
        level_data, sim = self.load_bblock_test_map()
        ninja_pos = (36, 564)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Count functional edges
        edge_types = graph_data.sub_cell_graph.edge_types
        functional_edges = np.sum(edge_types == EdgeType.FUNCTIONAL.value)
        
        print(f"Functional edges found: {functional_edges}")
        self.assertGreater(functional_edges, 0, "Should have functional edges in bblock_test map")
        self.assertGreater(functional_edges, 500, "Should have substantial number of functional edges")

    def test_issue_2_no_walkable_edges_in_solid_tiles_bblock_test(self):
        """Test Issue #2: No walkable edges in solid tiles (bblock_test)."""
        print("\n=== Testing Issue #2: No walkable edges in solid tiles (bblock_test) ===")
        
        level_data, sim = self.load_bblock_test_map()
        ninja_pos = (36, 564)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Check for walkable edges in solid tiles
        edge_index = graph_data.sub_cell_graph.edge_index
        edge_types = graph_data.sub_cell_graph.edge_types
        node_features = graph_data.sub_cell_graph.node_features
        
        walkable_edges_in_solid = 0
        
        for i in range(graph_data.sub_cell_graph.num_edges):
            if edge_types[i] == EdgeType.WALK.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                # Get node positions
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                # Check if either node is in a solid tile
                for pos in [node1_pos, node2_pos]:
                    tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                    tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                    
                    if (0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width and
                        level_data.get_tile(tile_y, tile_x) == 1):
                        walkable_edges_in_solid += 1
                        break
        
        print(f"Walkable edges in solid tiles: {walkable_edges_in_solid}")
        self.assertEqual(walkable_edges_in_solid, 0, "Should have no walkable edges in solid tiles")

    def test_issue_2_no_walkable_edges_in_solid_tiles_custom_map(self):
        """Test Issue #2: No walkable edges in solid tiles (custom map)."""
        print("\n=== Testing Issue #2: No walkable edges in solid tiles (custom map) ===")
        
        level_data, sim = self.load_custom_solid_tiles_map()
        ninja_pos = (120, 480)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Check for walkable edges in solid tiles
        edge_index = graph_data.sub_cell_graph.edge_index
        edge_types = graph_data.sub_cell_graph.edge_types
        node_features = graph_data.sub_cell_graph.node_features
        
        walkable_edges_in_solid = 0
        
        for i in range(graph_data.sub_cell_graph.num_edges):
            if edge_types[i] == EdgeType.WALK.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                # Get node positions
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                # Check if either node is in a solid tile
                for pos in [node1_pos, node2_pos]:
                    tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                    tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                    
                    if (0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width and
                        level_data.get_tile(tile_y, tile_x) == 1):
                        walkable_edges_in_solid += 1
                        break
        
        print(f"Walkable edges in solid tiles: {walkable_edges_in_solid}")
        self.assertEqual(walkable_edges_in_solid, 0, "Should have no walkable edges in solid tiles")

    def test_issue_3_navigation_bblock_test(self):
        """Test Issue #3: Pathfinding working (bblock_test)."""
        print("\n=== Testing Issue #3: Pathfinding (bblock_test) ===")
        
        level_data, sim = self.load_bblock_test_map()
        ninja_pos = (36, 564)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Test navigation between nearby nodes
        start_pos = (150, 446)
        end_pos = (129, 446)
        
        start_node = self.navigation_engine._find_node_at_position(graph_data.sub_cell_graph, start_pos)
        end_node = self.navigation_engine._find_node_at_position(graph_data.sub_cell_graph, end_pos)
        
        self.assertIsNotNone(start_node, f"Should find node at {start_pos}")
        self.assertIsNotNone(end_node, f"Should find node at {end_pos}")
        
        path_result = self.navigation_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, end_node)
        
        print(f"Path found: {path_result.success}, nodes: {len(path_result.path)}, cost: {path_result.total_cost:.2f}")
        self.assertTrue(path_result.success, "Should find path between nearby traversable positions")
        self.assertGreater(len(path_result.path), 0, "Path should have nodes")

    def test_issue_3_navigation_custom_map(self):
        """Test Issue #3: Pathfinding working (custom map)."""
        print("\n=== Testing Issue #3: Pathfinding (custom map) ===")
        
        level_data, sim = self.load_custom_solid_tiles_map()
        ninja_pos = (120, 480)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Test navigation in open area
        start_pos = (120, 480)
        end_pos = (200, 480)
        
        start_node = self.navigation_engine._find_node_at_position(graph_data.sub_cell_graph, start_pos)
        end_node = self.navigation_engine._find_node_at_position(graph_data.sub_cell_graph, end_pos)
        
        self.assertIsNotNone(start_node, f"Should find node at {start_pos}")
        self.assertIsNotNone(end_node, f"Should find node at {end_pos}")
        
        path_result = self.navigation_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, end_node)
        
        print(f"Path found: {path_result.success}, nodes: {len(path_result.path)}, cost: {path_result.total_cost:.2f}")
        self.assertTrue(path_result.success, "Should find path between nearby traversable positions")
        self.assertGreater(len(path_result.path), 0, "Path should have nodes")

    def test_node_positions_stored_correctly(self):
        """Test that node positions are stored correctly in feature vectors."""
        print("\n=== Testing node positions in feature vectors ===")
        
        level_data, sim = self.load_bblock_test_map()
        ninja_pos = (36, 564)
        
        # Build graph
        graph_data = self.graph_builder.build_hierarchical_graph(level_data, ninja_pos)
        
        # Check that node positions are stored in features[0:2]
        node_features = graph_data.sub_cell_graph.node_features
        
        # Check first few nodes have reasonable positions
        for i in range(min(10, graph_data.sub_cell_graph.num_nodes)):
            if graph_data.sub_cell_graph.node_mask[i]:
                x_pos = float(node_features[i, 0])
                y_pos = float(node_features[i, 1])
                
                # Positions should be reasonable (not all 1.0 or 0.0)
                self.assertNotEqual(x_pos, 1.0, f"Node {i} x position should not be default 1.0")
                self.assertNotEqual(y_pos, 0.0, f"Node {i} y position should not be default 0.0")
                self.assertGreater(x_pos, 0, f"Node {i} x position should be positive")
                self.assertGreater(y_pos, 0, f"Node {i} y position should be positive")
                
                print(f"Node {i}: position ({x_pos}, {y_pos})")
                break


if __name__ == '__main__':
    print("=" * 80)
    print("COMPREHENSIVE GRAPH VISUALIZATION FIXES VALIDATION")
    print("=" * 80)
    print("Testing all three original issues:")
    print("1. Functional edges between switches and doors")
    print("2. Walkable edges in solid tiles (ninja radius clearance)")
    print("3. Pathfinding not working on traversable paths")
    print("=" * 80)
    
    unittest.main(verbosity=2)