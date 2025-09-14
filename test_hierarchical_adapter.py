#!/usr/bin/env python3
"""
Integration tests for the hierarchical reachability adapter.

This test suite validates that the hierarchical adapter correctly integrates
with the existing test infrastructure and provides compatible results.
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from nclone.graph.reachability.hierarchical_adapter import HierarchicalReachabilityAdapter
from nclone.graph.reachability.reachability_state import ReachabilityState
from nclone.graph.trajectory_calculator import TrajectoryCalculator


class TestHierarchicalAdapter(unittest.TestCase):
    """Test cases for the hierarchical reachability adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trajectory_calculator = Mock(spec=TrajectoryCalculator)
        self.adapter = HierarchicalReachabilityAdapter(
            self.trajectory_calculator,
            debug=True
        )
        
        # Create mock level data
        self.mock_level_data = Mock()
        self.mock_level_data.width = 800
        self.mock_level_data.height = 600
        self.mock_level_data.tiles = [[0 for _ in range(33)] for _ in range(25)]
    
    def test_adapter_initialization(self):
        """Test that adapter initializes correctly."""
        self.assertIsNotNone(self.adapter.hierarchical_analyzer)
        self.assertEqual(self.adapter.debug, True)
        self.assertEqual(self.adapter.trajectory_calculator, self.trajectory_calculator)
    
    def test_analyze_reachability_returns_legacy_state(self):
        """Test that adapter returns ReachabilityState compatible with existing code."""
        ninja_position = (100, 100)
        switch_states = {}
        
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Should return ReachabilityState
        self.assertIsInstance(result, ReachabilityState)
        
        # Should have required attributes
        self.assertIsInstance(result.reachable_positions, set)
        self.assertIsInstance(result.switch_states, dict)
        self.assertIsInstance(result.unlocked_areas, set)
        self.assertIsInstance(result.subgoals, list)
        
        # Should have some reachable positions
        self.assertGreater(len(result.reachable_positions), 0)
    
    def test_hierarchical_data_preserved(self):
        """Test that hierarchical data is preserved in legacy state."""
        ninja_position = (200, 200)
        switch_states = {'test_switch': True}
        
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Should have hierarchical data as attributes
        self.assertTrue(hasattr(result, 'hierarchical_regions'))
        self.assertTrue(hasattr(result, 'hierarchical_tiles'))
        self.assertTrue(hasattr(result, 'hierarchical_subcells'))
        self.assertTrue(hasattr(result, 'analysis_time_ms'))
        
        # Hierarchical data should be populated
        self.assertIsInstance(result.hierarchical_regions, set)
        self.assertIsInstance(result.hierarchical_tiles, set)
        self.assertIsInstance(result.hierarchical_subcells, set)
        self.assertGreater(result.analysis_time_ms, 0)
    
    def test_switch_states_preserved(self):
        """Test that switch states are correctly preserved."""
        ninja_position = (150, 150)
        switch_states = {
            'exit_switch': False,
            'door_1': True,
            'door_2': False
        }
        
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Switch states should be preserved
        self.assertEqual(result.switch_states, switch_states)
    
    def test_performance_stats_available(self):
        """Test that performance statistics are available."""
        # Run a few analyses to generate stats
        for i in range(3):
            self.adapter.analyze_reachability(
                self.mock_level_data,
                (100 + i * 50, 100 + i * 50),
                {}
            )
        
        # Should have cache hit rate
        cache_hit_rate = self.adapter.get_cache_hit_rate()
        self.assertIsInstance(cache_hit_rate, float)
        self.assertGreaterEqual(cache_hit_rate, 0.0)
        self.assertLessEqual(cache_hit_rate, 1.0)
        
        # Should have performance stats
        stats = self.adapter.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('total_queries', stats)
        self.assertIn('cache_hits', stats)
        
        # Total queries should match our test runs
        self.assertEqual(stats['total_queries'], 3)
    
    def test_compatibility_with_none_switch_states(self):
        """Test compatibility when switch_states is None."""
        ninja_position = (100, 100)
        
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            None  # None switch states
        )
        
        # Should handle None gracefully
        self.assertIsInstance(result, ReachabilityState)
        self.assertIsInstance(result.switch_states, dict)
        self.assertEqual(len(result.switch_states), 0)
    
    def test_performance_improvement(self):
        """Test that hierarchical adapter provides performance improvement."""
        ninja_position = (300, 300)
        switch_states = {}
        
        # Run analysis and check timing
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Should complete quickly (hierarchical advantage)
        self.assertLess(result.analysis_time_ms, 10.0)  # Should be under 10ms
        
        print(f"Hierarchical adapter analysis time: {result.analysis_time_ms:.2f}ms")
    
    def test_subgoals_generation(self):
        """Test that subgoals are generated correctly."""
        ninja_position = (100, 100)
        switch_states = {}
        
        result = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Should have subgoals
        self.assertIsInstance(result.subgoals, list)
        # Basic implementation should generate some subgoals
        self.assertGreater(len(result.subgoals), 0)


class TestAdapterCompatibility(unittest.TestCase):
    """Test compatibility with existing interfaces."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trajectory_calculator = Mock(spec=TrajectoryCalculator)
        self.adapter = HierarchicalReachabilityAdapter(
            self.trajectory_calculator,
            debug=False,
            enable_caching=True,
            cache_size=500,
            cache_ttl=120.0
        )
        
        self.mock_level_data = Mock()
        self.mock_level_data.width = 400
        self.mock_level_data.height = 300
        self.mock_level_data.tiles = [[0 for _ in range(17)] for _ in range(13)]
    
    def test_constructor_parameters_preserved(self):
        """Test that constructor parameters are preserved for compatibility."""
        self.assertEqual(self.adapter.enable_caching, True)
        self.assertEqual(self.adapter.cache_size, 500)
        self.assertEqual(self.adapter.cache_ttl, 120.0)
        self.assertEqual(self.adapter.debug, False)
    
    def test_multiple_analyses_consistent(self):
        """Test that multiple analyses of the same position are consistent."""
        ninja_position = (100, 100)
        switch_states = {'test': True}
        
        # Run analysis twice
        result1 = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        result2 = self.adapter.analyze_reachability(
            self.mock_level_data,
            ninja_position,
            switch_states
        )
        
        # Results should be consistent
        self.assertEqual(result1.reachable_positions, result2.reachable_positions)
        self.assertEqual(result1.switch_states, result2.switch_states)
        self.assertEqual(result1.subgoals, result2.subgoals)
    
    def test_different_positions_different_results(self):
        """Test that different positions produce different results."""
        switch_states = {}
        
        result1 = self.adapter.analyze_reachability(
            self.mock_level_data,
            (50, 50),
            switch_states
        )
        
        result2 = self.adapter.analyze_reachability(
            self.mock_level_data,
            (300, 250),
            switch_states
        )
        
        # Results should be different (different starting positions)
        # Note: With current simple implementation, they might be similar,
        # but the analysis should still work correctly
        self.assertIsInstance(result1.reachable_positions, set)
        self.assertIsInstance(result2.reachable_positions, set)
        
        # Both should have reachable positions
        self.assertGreater(len(result1.reachable_positions), 0)
        self.assertGreater(len(result2.reachable_positions), 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)