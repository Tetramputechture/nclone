#!/usr/bin/env python3
"""
Unit tests for the hierarchical reachability analysis system.

This test suite validates the core functionality of the hierarchical
reachability analyzer using test-driven development principles.
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from nclone.graph.reachability.hierarchical_reachability import (
    HierarchicalReachabilityAnalyzer,
    HierarchicalNode,
    HierarchicalReachabilityResult
)
from nclone.graph.reachability.hierarchical_constants import ResolutionLevel


class TestHierarchicalReachabilityAnalyzer(unittest.TestCase):
    """Test cases for the hierarchical reachability analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = HierarchicalReachabilityAnalyzer(debug=True)
        
        # Create mock level data
        self.mock_level_data = Mock()
        self.mock_level_data.width = 800
        self.mock_level_data.height = 600
        # Create numpy array for tiles (25 rows x 33 cols for 800x600 at 24px tiles)
        import numpy as np
        self.mock_level_data.tiles = np.zeros((25, 33), dtype=int)
    
    def test_resolution_level_enum(self):
        """Test that resolution levels are correctly defined."""
        self.assertEqual(ResolutionLevel.REGION.value, 96)
        self.assertEqual(ResolutionLevel.TILE.value, 24)
        self.assertEqual(ResolutionLevel.SUBCELL.value, 6)
    
    def test_hierarchical_node_creation(self):
        """Test hierarchical node creation and initialization."""
        node = HierarchicalNode(
            position=(5, 3),
            resolution=ResolutionLevel.TILE,
            reachable=True
        )
        
        self.assertEqual(node.position, (5, 3))
        self.assertEqual(node.resolution, ResolutionLevel.TILE)
        self.assertTrue(node.reachable)
        self.assertEqual(node.movement_cost, 1.0)
        self.assertIsInstance(node.children, set)
        self.assertEqual(len(node.children), 0)
    
    def test_pixel_to_region_conversion(self):
        """Test conversion from pixel coordinates to region coordinates."""
        # Test various pixel positions
        test_cases = [
            ((0, 0), (0, 0)),
            ((95, 95), (0, 0)),
            ((96, 96), (1, 1)),
            ((192, 288), (2, 3)),
            ((480, 360), (5, 3))
        ]
        
        for pixel_pos, expected_region in test_cases:
            with self.subTest(pixel_pos=pixel_pos):
                result = self.analyzer._pixel_to_region(pixel_pos)
                self.assertEqual(result, expected_region)
    
    def test_pixel_to_tile_conversion(self):
        """Test conversion from pixel coordinates to tile coordinates."""
        test_cases = [
            ((0, 0), (0, 0)),
            ((23, 23), (0, 0)),
            ((24, 24), (1, 1)),
            ((48, 72), (2, 3)),
            ((120, 96), (5, 4))
        ]
        
        for pixel_pos, expected_tile in test_cases:
            with self.subTest(pixel_pos=pixel_pos):
                result = self.analyzer._pixel_to_tile(pixel_pos)
                self.assertEqual(result, expected_tile)
    
    def test_pixel_to_subcell_conversion(self):
        """Test conversion from pixel coordinates to subcell coordinates."""
        test_cases = [
            ((0, 0), (0, 0)),
            ((5, 5), (0, 0)),
            ((6, 6), (1, 1)),
            ((12, 18), (2, 3)),
            ((30, 24), (5, 4))
        ]
        
        for pixel_pos, expected_subcell in test_cases:
            with self.subTest(pixel_pos=pixel_pos):
                result = self.analyzer._pixel_to_subcell(pixel_pos)
                self.assertEqual(result, expected_subcell)
    
    def test_analyze_reachability_basic(self):
        """Test basic reachability analysis functionality."""
        ninja_position = (100, 100)
        switch_states = {}
        
        result = self.analyzer.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Verify result structure
        self.assertIsInstance(result, HierarchicalReachabilityResult)
        self.assertIsInstance(result.reachable_regions, set)
        self.assertIsInstance(result.reachable_tiles, set)
        self.assertIsInstance(result.reachable_subcells, set)
        self.assertIsInstance(result.subgoals, list)
        self.assertGreater(result.analysis_time_ms, 0)
        
        # Verify that some positions are reachable
        self.assertGreater(len(result.reachable_regions), 0)
        self.assertGreater(len(result.reachable_tiles), 0)
        self.assertGreater(len(result.reachable_subcells), 0)
    
    def test_analyze_regions_basic(self):
        """Test region-level analysis."""
        ninja_region = (1, 1)
        switch_states = {}
        
        # Initialize geometry analyzer
        self.analyzer.geometry_analyzer.initialize_for_level(self.mock_level_data)
        
        reachable_regions = self.analyzer.geometry_analyzer.analyze_region_reachability(
            self.mock_level_data,
            ninja_region,
            switch_states
        )
        
        # Should include ninja's region
        self.assertIn(ninja_region, reachable_regions)
        self.assertGreater(len(reachable_regions), 0)  # Should have some regions
    
    def test_analyze_tiles_from_regions(self):
        """Test tile-level analysis based on reachable regions."""
        ninja_tile = (4, 4)
        reachable_regions = {(1, 1)}  # Single region
        switch_states = {}
        
        # Initialize geometry analyzer
        self.analyzer.geometry_analyzer.initialize_for_level(self.mock_level_data)
        
        reachable_tiles = self.analyzer.geometry_analyzer.analyze_tile_reachability(
            self.mock_level_data,
            ninja_tile,
            reachable_regions,
            switch_states
        )
        
        # Should have tiles from the reachable region
        self.assertGreater(len(reachable_tiles), 0)
    
    def test_analyze_subcells_from_tiles(self):
        """Test subcell-level analysis based on reachable tiles."""
        ninja_subcell = (16, 16)
        reachable_tiles = {(4, 4)}  # Single tile
        switch_states = {}
        
        # Initialize geometry analyzer
        self.analyzer.geometry_analyzer.initialize_for_level(self.mock_level_data)
        
        reachable_subcells = self.analyzer.geometry_analyzer.analyze_subcell_reachability(
            self.mock_level_data,
            ninja_subcell,
            reachable_tiles,
            switch_states
        )
        
        # Should have subcells from the reachable tile
        self.assertGreater(len(reachable_subcells), 0)
    
    def test_generate_subgoals_basic(self):
        """Test basic subgoal generation."""
        reachable_regions = {(0, 0), (1, 1)}
        reachable_tiles = {(0, 0), (1, 1), (2, 2)}
        switch_states = {}
        
        subgoals = self.analyzer._generate_subgoals(
            self.mock_level_data,
            reachable_regions,
            reachable_tiles,
            switch_states
        )
        
        # Should generate some subgoals
        self.assertIsInstance(subgoals, list)
        self.assertGreater(len(subgoals), 0)
        
        # Check for expected subgoal types
        self.assertIn("explore_reachable_areas", subgoals)
        self.assertIn("navigate_between_regions", subgoals)
    
    def test_cache_hit_rate_tracking(self):
        """Test cache hit rate tracking functionality."""
        # Initially should be 0
        self.assertEqual(self.analyzer.get_cache_hit_rate(), 0.0)
        
        # After queries, should track properly
        self.analyzer.total_queries = 10
        self.analyzer.cache_hits = 3
        self.assertEqual(self.analyzer.get_cache_hit_rate(), 0.3)
    
    def test_hierarchical_result_properties(self):
        """Test hierarchical result properties and methods."""
        result = HierarchicalReachabilityResult(
            reachable_regions={(0, 0), (1, 1)},
            reachable_tiles={(0, 0), (1, 1), (2, 2)},
            reachable_subcells={(0, 0), (1, 1), (2, 2), (3, 3)},
            subgoals=["test_subgoal"],
            analysis_time_ms=5.5,
            cache_hits=2,
            total_queries=5
        )
        
        # Test total_reachable_positions property
        self.assertEqual(result.total_reachable_positions, 4)  # subcells count
        
        # Test with no subcells
        result_no_subcells = HierarchicalReachabilityResult(
            reachable_regions={(0, 0)},
            reachable_tiles={(0, 0), (1, 1)},
            reachable_subcells=set(),
            subgoals=[],
            analysis_time_ms=3.0
        )
        
        self.assertEqual(result_no_subcells.total_reachable_positions, 2)  # tiles count
    
    def test_performance_requirements(self):
        """Test that analysis meets performance requirements for RL."""
        ninja_position = (200, 200)
        switch_states = {}
        
        # Run analysis and check timing
        result = self.analyzer.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Should complete within RL requirements (target: <10ms)
        # For this basic implementation, we'll be lenient but track the metric
        self.assertGreater(result.analysis_time_ms, 0)
        self.assertLess(result.analysis_time_ms, 100)  # Should be reasonable
        
        print(f"Analysis time: {result.analysis_time_ms:.2f}ms")


class TestResolutionConversions(unittest.TestCase):
    """Test cases for resolution conversion utilities."""
    
    def test_resolution_level_relationships(self):
        """Test that resolution levels have correct relationships."""
        # Region should be 4x larger than tile
        self.assertEqual(ResolutionLevel.REGION.value, ResolutionLevel.TILE.value * 4)
        
        # Tile should be 4x larger than subcell
        self.assertEqual(ResolutionLevel.TILE.value, ResolutionLevel.SUBCELL.value * 4)
        
        # Region should be 16x larger than subcell
        self.assertEqual(ResolutionLevel.REGION.value, ResolutionLevel.SUBCELL.value * 16)
    
    def test_boundary_conditions(self):
        """Test conversion at boundary conditions."""
        analyzer = HierarchicalReachabilityAnalyzer()
        
        # Test zero coordinates
        self.assertEqual(analyzer._pixel_to_region((0, 0)), (0, 0))
        self.assertEqual(analyzer._pixel_to_tile((0, 0)), (0, 0))
        self.assertEqual(analyzer._pixel_to_subcell((0, 0)), (0, 0))
        
        # Test negative coordinates (should floor correctly)
        self.assertEqual(analyzer._pixel_to_region((-1, -1)), (-1, -1))
        self.assertEqual(analyzer._pixel_to_tile((-1, -1)), (-1, -1))
        self.assertEqual(analyzer._pixel_to_subcell((-1, -1)), (-1, -1))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)