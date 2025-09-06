#!/usr/bin/env python3
"""
Unit tests for pathfinding functionality to ensure it works correctly.
"""

import unittest
import numpy as np
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants import TILE_PIXEL_SIZE


class TestPathfindingFunctionality(unittest.TestCase):
    """Test cases for pathfinding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple level with clear paths
        width, height = 7, 7
        self.tiles = np.zeros((height, width), dtype=int)
        
        # Create borders
        self.tiles[0, :] = 1  # Top border
        self.tiles[-1, :] = 1  # Bottom border
        self.tiles[:, 0] = 1  # Left border
        self.tiles[:, -1] = 1  # Right border
        
        # Add a simple obstacle in the middle
        self.tiles[3, 3] = 1
        
        # No entities for these tests
        entities = []
        
        self.level_data = LevelData(
            tiles=self.tiles,
            entities=entities,
            level_id='test_pathfinding'
        )
        
        # Build graph
        builder = HierarchicalGraphBuilder()
        ninja_position = (3.5 * TILE_PIXEL_SIZE, 3.5 * TILE_PIXEL_SIZE)
        ninja_velocity = (0.0, 0.0)
        ninja_state = 0
        
        hierarchical_graph_data = builder.build_graph(
            self.level_data, ninja_position, ninja_velocity, ninja_state
        )
        
        self.graph_data = hierarchical_graph_data.sub_cell_graph
        self.pathfinder = PathfindingEngine()
        
        # Find connected nodes for testing
        self.connected_nodes = self._find_connected_nodes()
    
    def _find_connected_nodes(self):
        """Find nodes that have connections for testing."""
        adjacency = {}
        for i in range(self.graph_data.num_nodes):
            if self.graph_data.node_mask[i] > 0:
                adjacency[i] = []
        
        for i in range(self.graph_data.num_edges):
            if self.graph_data.edge_mask[i] > 0:
                src = self.graph_data.edge_index[0, i]
                dst = self.graph_data.edge_index[1, i]
                if src in adjacency and dst in adjacency:
                    adjacency[src].append(dst)
        
        return [node for node, neighbors in adjacency.items() if len(neighbors) > 0]
    
    def test_pathfinding_between_adjacent_nodes_succeeds(self):
        """Test that pathfinding between adjacent nodes succeeds."""
        if len(self.connected_nodes) < 2:
            self.skipTest("Not enough connected nodes for test")
        
        start_node = self.connected_nodes[0]
        goal_node = self.connected_nodes[1]
        
        result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        self.assertTrue(result.success, "Pathfinding between connected nodes should succeed")
        self.assertGreater(len(result.path), 0, "Path should contain nodes")
        self.assertEqual(result.path[0], start_node, "Path should start with start node")
        self.assertEqual(result.path[-1], goal_node, "Path should end with goal node")
        self.assertGreater(result.nodes_explored, 0, "Should explore at least one node")
    
    def test_pathfinding_with_dijkstra_succeeds(self):
        """Test that Dijkstra pathfinding works correctly."""
        if len(self.connected_nodes) < 2:
            self.skipTest("Not enough connected nodes for test")
        
        start_node = self.connected_nodes[0]
        goal_node = self.connected_nodes[1]
        
        result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, PathfindingAlgorithm.DIJKSTRA
        )
        
        self.assertTrue(result.success, "Dijkstra pathfinding should succeed")
        self.assertGreater(len(result.path), 0, "Path should contain nodes")
        self.assertEqual(result.path[0], start_node, "Path should start with start node")
        self.assertEqual(result.path[-1], goal_node, "Path should end with goal node")
    
    def test_astar_and_dijkstra_find_same_optimal_path(self):
        """Test that A* and Dijkstra find paths with the same cost."""
        if len(self.connected_nodes) < 2:
            self.skipTest("Not enough connected nodes for test")
        
        start_node = self.connected_nodes[0]
        goal_node = self.connected_nodes[1]
        
        astar_result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        dijkstra_result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, PathfindingAlgorithm.DIJKSTRA
        )
        
        if astar_result.success and dijkstra_result.success:
            self.assertAlmostEqual(
                astar_result.total_cost, dijkstra_result.total_cost, places=3,
                msg="A* and Dijkstra should find paths with same cost"
            )
    
    def test_pathfinding_to_same_node_succeeds_immediately(self):
        """Test that pathfinding to the same node succeeds immediately."""
        if len(self.connected_nodes) < 1:
            self.skipTest("No connected nodes for test")
        
        node = self.connected_nodes[0]
        
        result = self.pathfinder.find_shortest_path(
            self.graph_data, node, node, PathfindingAlgorithm.A_STAR
        )
        
        self.assertTrue(result.success, "Pathfinding to same node should succeed")
        self.assertEqual(len(result.path), 1, "Path to same node should have length 1")
        self.assertEqual(result.path[0], node, "Path should contain only the node itself")
        self.assertEqual(result.total_cost, 0.0, "Cost to same node should be 0")
        self.assertEqual(result.nodes_explored, 1, "Should explore exactly 1 node")
    
    def test_pathfinding_returns_valid_path_coordinates(self):
        """Test that pathfinding returns valid path coordinates."""
        if len(self.connected_nodes) < 2:
            self.skipTest("Not enough connected nodes for test")
        
        start_node = self.connected_nodes[0]
        goal_node = self.connected_nodes[1]
        
        result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, PathfindingAlgorithm.A_STAR
        )
        
        if result.success:
            self.assertEqual(
                len(result.path_coordinates), len(result.path),
                "Should have coordinates for each node in path"
            )
            
            for coord in result.path_coordinates:
                self.assertEqual(len(coord), 2, "Each coordinate should be (x, y)")
                self.assertIsInstance(coord[0], (int, float), "X coordinate should be numeric")
                self.assertIsInstance(coord[1], (int, float), "Y coordinate should be numeric")
    
    def test_pathfinding_with_max_nodes_limit(self):
        """Test that pathfinding respects the max nodes exploration limit."""
        if len(self.connected_nodes) < 2:
            self.skipTest("Not enough connected nodes for test")
        
        start_node = self.connected_nodes[0]
        goal_node = self.connected_nodes[-1]  # Use a distant node
        
        # Set a very low limit
        result = self.pathfinder.find_shortest_path(
            self.graph_data, start_node, goal_node, 
            algorithm=PathfindingAlgorithm.A_STAR, max_nodes_to_explore=5
        )
        
        # Should either succeed or fail gracefully
        self.assertIsInstance(result.success, bool, "Should return boolean success")
        self.assertLessEqual(
            result.nodes_explored, 10,  # Allow some tolerance
            "Should not explore significantly more than the limit"
        )
    
    def test_graph_has_reasonable_connectivity(self):
        """Test that the graph has reasonable connectivity."""
        # Count total connected nodes
        total_connected = len(self.connected_nodes)
        total_nodes = sum(1 for i in range(self.graph_data.num_nodes) 
                         if self.graph_data.node_mask[i] > 0)
        
        # Should have a significant portion of nodes connected
        connectivity_ratio = total_connected / total_nodes if total_nodes > 0 else 0
        
        self.assertGreater(
            connectivity_ratio, 0.5,
            f"Graph should have reasonable connectivity (got {connectivity_ratio:.2%})"
        )
        
        # Should have a reasonable number of edges
        total_edges = sum(1 for i in range(self.graph_data.num_edges) 
                         if self.graph_data.edge_mask[i] > 0)
        
        self.assertGreater(total_edges, 0, "Graph should have edges")
        self.assertGreater(
            total_edges, total_connected,
            "Should have more edges than connected nodes"
        )


if __name__ == '__main__':
    unittest.main()