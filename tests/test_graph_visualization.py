"""
Comprehensive tests for graph visualization system.

Tests navigation, rendering, overlay functionality, and API integration.
"""

import unittest
import pygame
import numpy as np
from typing import Dict, List, Any
import tempfile
import os

# Import visualization components
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig, VisualizationMode
from nclone.graph.navigation import PathfindingEngine, PathfindingAlgorithm, PathResult
from nclone.graph.visualization_api import (
    GraphVisualizationAPI, VisualizationRequest, visualize_level_graph, find_path_and_visualize
)
from nclone.graph.enhanced_debug_overlay import EnhancedDebugOverlay, OverlayMode
from nclone.graph.common import GraphData


class MockSimulator:
    """Mock simulator for testing overlay functionality."""
    
    def __init__(self):
        self.ninja = MockNinja()
        self.entities = []
        self.level_data = self._create_test_level_data()
    
    def _create_test_level_data(self) -> Dict[str, Any]:
        """Create test level data."""
        # Create a simple 10x10 level with some walls
        tiles = np.zeros((10, 10), dtype=int)
        
        # Add some walls
        tiles[2:8, 2] = 1  # Vertical wall
        tiles[2, 2:8] = 1  # Horizontal wall
        tiles[7, 2:8] = 1  # Another horizontal wall
        
        return {
            'level_id': 'test_level',
            'tiles': tiles,
            'width': 10,
            'height': 10
        }


class MockNinja:
    """Mock ninja for testing."""
    
    def __init__(self):
        self.x = 100.0
        self.y = 100.0
        self.vx = 0.0
        self.vy = 0.0
        self.movement_state = 0


class TestGraphVisualization(unittest.TestCase):
    """Test graph visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        pygame.init()
        
        # Create test graph data
        self.graph_data = self._create_test_graph_data()
        self.level_data = self._create_test_level_data()
        self.entities = self._create_test_entities()
        
        # Create visualizer
        self.config = VisualizationConfig()
        self.visualizer = GraphVisualizer(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        pygame.quit()
    
    def _create_test_graph_data(self) -> GraphData:
        """Create test graph data."""
        num_nodes = 20
        num_edges = 30
        node_feature_dim = 16
        edge_feature_dim = 8
        
        # Create node features (first two are x, y coordinates)
        node_features = np.random.rand(num_nodes, node_feature_dim).astype(np.float32)
        node_features[:, 0] = np.linspace(0, 200, num_nodes)  # X coordinates
        node_features[:, 1] = np.linspace(0, 200, num_nodes)  # Y coordinates
        
        # Create edges
        edge_index = np.zeros((2, num_edges), dtype=np.int32)
        edge_features = np.random.rand(num_edges, edge_feature_dim).astype(np.float32)
        
        # Create simple chain of edges
        for i in range(min(num_edges, num_nodes - 1)):
            edge_index[0, i] = i
            edge_index[1, i] = i + 1
        
        # Add some random edges
        for i in range(num_nodes - 1, num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            edge_index[0, i] = src
            edge_index[1, i] = dst
        
        # Create masks and types
        node_mask = np.ones(num_nodes, dtype=np.float32)
        edge_mask = np.ones(num_edges, dtype=np.float32)
        node_types = np.random.randint(0, 3, num_nodes, dtype=np.int32)
        edge_types = np.random.randint(0, 6, num_edges, dtype=np.int32)
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=node_types,
            edge_types=edge_types,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
    
    def _create_test_level_data(self) -> Dict[str, Any]:
        """Create test level data."""
        tiles = np.zeros((10, 10), dtype=int)
        tiles[2:8, 2] = 1  # Vertical wall
        tiles[2, 2:8] = 1  # Horizontal wall
        
        return {
            'level_id': 'test_level',
            'tiles': tiles,
            'width': 10,
            'height': 10
        }
    
    def _create_test_entities(self) -> List[Dict[str, Any]]:
        """Create test entities."""
        return [
            {'type': 1, 'x': 50.0, 'y': 50.0, 'state': 0},
            {'type': 2, 'x': 150.0, 'y': 150.0, 'state': 1},
            {'type': 1, 'x': 100.0, 'y': 200.0, 'state': 0},
        ]
    
    def test_visualization_config(self):
        """Test visualization configuration."""
        config = VisualizationConfig()
        
        # Test default values
        self.assertTrue(config.show_nodes)
        self.assertTrue(config.show_edges)
        self.assertEqual(config.node_size, 3.0)
        self.assertEqual(config.edge_width, 1.0)
        
        # Test custom configuration
        custom_config = VisualizationConfig(
            show_nodes=False,
            show_edges=True,
            node_size=5.0,
            edge_width=2.0
        )
        
        self.assertFalse(custom_config.show_nodes)
        self.assertTrue(custom_config.show_edges)
        self.assertEqual(custom_config.node_size, 5.0)
        self.assertEqual(custom_config.edge_width, 2.0)
    
    def test_standalone_visualization(self):
        """Test standalone graph visualization."""
        surface = self.visualizer.create_standalone_visualization(
            self.graph_data,
            width=800,
            height=600
        )
        
        self.assertIsInstance(surface, pygame.Surface)
        self.assertEqual(surface.get_size(), (800, 600))
    
    def test_overlay_visualization(self):
        """Test overlay visualization."""
        # Create base simulator surface
        sim_surface = pygame.Surface((800, 600))
        sim_surface.fill((50, 50, 50))
        
        overlay_surface = self.visualizer.create_overlay_visualization(
            self.graph_data,
            sim_surface,
            goal_position=(150.0, 150.0),
            ninja_position=(50.0, 50.0)
        )
        
        self.assertIsInstance(overlay_surface, pygame.Surface)
        self.assertEqual(overlay_surface.get_size(), (800, 600))
    
    def test_side_by_side_visualization(self):
        """Test side-by-side visualization."""
        sim_surface = pygame.Surface((400, 600))
        sim_surface.fill((50, 50, 50))
        
        combined_surface = self.visualizer.create_side_by_side_visualization(
            self.graph_data,
            sim_surface,
            goal_position=(150.0, 150.0),
            ninja_position=(50.0, 50.0)
        )
        
        self.assertIsInstance(combined_surface, pygame.Surface)
        self.assertEqual(combined_surface.get_size(), (800, 600))  # Double width


class TestPathfinding(unittest.TestCase):
    """Test navigation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.navigation_engine = PathfindingEngine()
        self.graph_data = self._create_navigation_test_graph()
    
    def _create_navigation_test_graph(self) -> GraphData:
        """Create graph data suitable for navigation tests."""
        # Create a simple grid graph
        grid_size = 5
        num_nodes = grid_size * grid_size
        
        # Node features (x, y coordinates)
        node_features = np.zeros((num_nodes, 16), dtype=np.float32)
        node_idx = 0
        for y in range(grid_size):
            for x in range(grid_size):
                node_features[node_idx, 0] = x * 50.0  # X coordinate
                node_features[node_idx, 1] = y * 50.0  # Y coordinate
                node_idx += 1
        
        # Create edges (4-connected grid)
        edges = []
        for y in range(grid_size):
            for x in range(grid_size):
                current_node = y * grid_size + x
                
                # Right neighbor
                if x < grid_size - 1:
                    right_node = y * grid_size + (x + 1)
                    edges.append((current_node, right_node))
                
                # Down neighbor
                if y < grid_size - 1:
                    down_node = (y + 1) * grid_size + x
                    edges.append((current_node, down_node))
        
        num_edges = len(edges)
        edge_index = np.zeros((2, num_edges), dtype=np.int32)
        edge_features = np.ones((num_edges, 8), dtype=np.float32)
        
        for i, (src, dst) in enumerate(edges):
            edge_index[0, i] = src
            edge_index[1, i] = dst
        
        # Create masks and types
        node_mask = np.ones(num_nodes, dtype=np.float32)
        edge_mask = np.ones(num_edges, dtype=np.float32)
        node_types = np.zeros(num_nodes, dtype=np.int32)  # All grid cells
        edge_types = np.zeros(num_edges, dtype=np.int32)  # All walk edges
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=node_types,
            edge_types=edge_types,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
    
    def test_dijkstra_navigation(self):
        """Test Dijkstra navigation algorithm."""
        start_node = 0  # Top-left corner
        goal_node = 24  # Bottom-right corner (5x5 grid)
        
        result = self.navigation_engine.find_shortest_path(
            self.graph_data,
            start_node,
            goal_node,
            PathfindingAlgorithm.DIJKSTRA
        )
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.path), 0)
        self.assertEqual(result.path[0], start_node)
        self.assertEqual(result.path[-1], goal_node)
        self.assertGreater(result.nodes_explored, 0)
    
    def test_astar_navigation(self):
        """Test A* navigation algorithm."""
        start_node = 0  # Top-left corner
        goal_node = 24  # Bottom-right corner
        
        result = self.navigation_engine.find_shortest_path(
            self.graph_data,
            start_node,
            goal_node,
            PathfindingAlgorithm.A_STAR
        )
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.path), 0)
        self.assertEqual(result.path[0], start_node)
        self.assertEqual(result.path[-1], goal_node)
        self.assertGreater(result.nodes_explored, 0)
    
    def test_navigation_same_node(self):
        """Test navigation when start and goal are the same."""
        node = 5
        
        result = self.navigation_engine.find_shortest_path(
            self.graph_data,
            node,
            node,
            PathfindingAlgorithm.A_STAR
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.path), 1)
        self.assertEqual(result.path[0], node)
        self.assertEqual(result.total_cost, 0.0)
    
    def test_navigation_unreachable_goal(self):
        """Test navigation with unreachable goal."""
        # Create graph with disconnected components
        graph_data = self.graph_data
        
        # Remove all edges to make nodes disconnected
        graph_data.edge_mask.fill(0)
        graph_data.num_edges = 0
        
        result = self.navigation_engine.find_shortest_path(
            graph_data,
            0,
            24,
            PathfindingAlgorithm.A_STAR
        )
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.path), 0)
        self.assertEqual(result.total_cost, float('inf'))


class TestVisualizationAPI(unittest.TestCase):
    """Test unified visualization API."""
    
    def setUp(self):
        """Set up test fixtures."""
        pygame.init()
        self.api = GraphVisualizationAPI()
        self.level_data = self._create_test_level_data()
        self.entities = self._create_test_entities()
    
    def tearDown(self):
        """Clean up after tests."""
        pygame.quit()
    
    def _create_test_level_data(self) -> Dict[str, Any]:
        """Create test level data."""
        tiles = np.zeros((10, 10), dtype=int)
        tiles[2:8, 2] = 1  # Vertical wall
        tiles[2, 2:8] = 1  # Horizontal wall
        
        return {
            'level_id': 'test_level',
            'tiles': tiles,
            'width': 10,
            'height': 10
        }
    
    def _create_test_entities(self) -> List[Dict[str, Any]]:
        """Create test entities."""
        return [
            {'type': 1, 'x': 50.0, 'y': 50.0, 'state': 0},
            {'type': 2, 'x': 150.0, 'y': 150.0, 'state': 1},
        ]
    
    def test_visualization_request(self):
        """Test visualization request creation and processing."""
        request = VisualizationRequest(
            level_data=self.level_data,
            entities=self.entities,
            ninja_position=(100.0, 100.0),
            goal_position=(200.0, 200.0),
            mode=VisualizationMode.STANDALONE,
            output_size=(800, 600)
        )
        
        result = self.api.visualize_graph(request)
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.surface, pygame.Surface)
        self.assertEqual(result.surface.get_size(), (800, 600))
        self.assertIsNotNone(result.graph_stats)
        self.assertGreater(result.render_time_ms, 0)
    
    def test_navigation_integration(self):
        """Test navigation integration with visualization."""
        start_pos = (50.0, 50.0)
        goal_pos = (200.0, 200.0)
        
        path_result = self.api.find_shortest_path(
            self.level_data,
            self.entities,
            start_pos,
            goal_pos
        )
        
        self.assertIsInstance(path_result, PathResult)
        # Note: Path may not be successful due to test level layout
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        initial_stats = self.api.get_performance_stats()
        
        # Perform some operations
        request = VisualizationRequest(
            level_data=self.level_data,
            entities=self.entities,
            ninja_position=(100.0, 100.0)
        )
        
        self.api.visualize_graph(request)
        
        updated_stats = self.api.get_performance_stats()
        
        self.assertGreater(updated_stats['total_visualizations'], 
                          initial_stats['total_visualizations'])
    
    def test_cache_functionality(self):
        """Test graph data caching."""
        request = VisualizationRequest(
            level_data=self.level_data,
            entities=self.entities,
            ninja_position=(100.0, 100.0)
        )
        
        # First request
        result1 = self.api.visualize_graph(request)
        initial_cache_hits = self.api.get_performance_stats()['cache_hits']
        
        # Second request with same data
        result2 = self.api.visualize_graph(request)
        updated_cache_hits = self.api.get_performance_stats()['cache_hits']
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertGreater(updated_cache_hits, initial_cache_hits)
    
    def test_config_export_import(self):
        """Test configuration export and import."""
        config = VisualizationConfig(
            show_nodes=False,
            show_edges=True,
            node_size=5.0
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export config
            self.api.export_visualization_config(config, temp_path)
            
            # Import config
            imported_config = self.api.import_visualization_config(temp_path)
            
            self.assertEqual(config.show_nodes, imported_config.show_nodes)
            self.assertEqual(config.show_edges, imported_config.show_edges)
            self.assertEqual(config.node_size, imported_config.node_size)
        
        finally:
            os.unlink(temp_path)


class TestEnhancedDebugOverlay(unittest.TestCase):
    """Test enhanced debug overlay functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        pygame.init()
        self.screen = pygame.Surface((800, 600))
        self.sim = MockSimulator()
        self.overlay = EnhancedDebugOverlay(
            self.sim, self.screen, 1.0, 0.0, 0.0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pygame.quit()
    
    def test_overlay_modes(self):
        """Test different overlay modes."""
        # Test mode switching
        self.overlay.set_overlay_mode(OverlayMode.BASIC_GRAPH)
        self.assertEqual(self.overlay.overlay_mode, OverlayMode.BASIC_GRAPH)
        
        self.overlay.set_overlay_mode(OverlayMode.PATHFINDING)
        self.assertEqual(self.overlay.overlay_mode, OverlayMode.PATHFINDING)
        
        self.overlay.set_overlay_mode(OverlayMode.FULL_ANALYSIS)
        self.assertEqual(self.overlay.overlay_mode, OverlayMode.FULL_ANALYSIS)
    
    def test_goal_position_setting(self):
        """Test goal position setting for navigation."""
        goal_pos = (200.0, 200.0)
        self.overlay.set_goal_position(goal_pos)
        self.assertEqual(self.overlay.goal_position, goal_pos)
    
    def test_overlay_rendering(self):
        """Test overlay rendering."""
        self.overlay.set_overlay_mode(OverlayMode.BASIC_GRAPH)
        
        overlay_surface = self.overlay.draw_overlay()
        
        self.assertIsInstance(overlay_surface, pygame.Surface)
        self.assertEqual(overlay_surface.get_size(), self.screen.get_size())
    
    def test_key_handling(self):
        """Test keyboard input handling."""
        # Test mode cycling
        handled = self.overlay.handle_key_press(pygame.K_g)
        self.assertTrue(handled)
        
        # Test hierarchical toggle
        handled = self.overlay.handle_key_press(pygame.K_h)
        self.assertTrue(handled)
        
        # Test reset
        handled = self.overlay.handle_key_press(pygame.K_r)
        self.assertTrue(handled)
        
        # Test unhandled key
        handled = self.overlay.handle_key_press(pygame.K_z)
        self.assertFalse(handled)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        pygame.init()
        self.level_data = self._create_test_level_data()
        self.entities = self._create_test_entities()
    
    def tearDown(self):
        """Clean up after tests."""
        pygame.quit()
    
    def _create_test_level_data(self) -> Dict[str, Any]:
        """Create test level data."""
        tiles = np.zeros((5, 5), dtype=int)
        return {
            'level_id': 'test_level',
            'tiles': tiles,
            'width': 5,
            'height': 5
        }
    
    def _create_test_entities(self) -> List[Dict[str, Any]]:
        """Create test entities."""
        return [
            {'type': 1, 'x': 50.0, 'y': 50.0, 'state': 0},
        ]
    
    def test_visualize_level_graph(self):
        """Test level graph visualization convenience function."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            success = visualize_level_graph(
                self.level_data,
                self.entities,
                temp_path,
                size=(400, 300)
            )
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_path))
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_find_path_and_visualize(self):
        """Test navigation and visualization convenience function."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            success, path_result = find_path_and_visualize(
                self.level_data,
                self.entities,
                (25.0, 25.0),
                (100.0, 100.0),
                temp_path,
                size=(400, 300)
            )
            
            self.assertTrue(success)
            self.assertIsInstance(path_result, PathResult)
            self.assertTrue(os.path.exists(temp_path))
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)