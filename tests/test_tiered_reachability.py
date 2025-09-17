"""
Unit tests for the tiered reachability system.

This module provides comprehensive testing for all three tiers of the reachability
system, including performance tests, accuracy tests, and integration tests.
"""

import unittest
import time
import numpy as np
from typing import Set, Tuple, Dict, Any
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.reachability_types import (
    PerformanceTarget, ReachabilityApproximation, ReachabilityResult
)
from nclone.graph.reachability.opencv_flood_fill import OpenCVFloodFill


class MockLevelData:
    """Mock level data for testing."""
    
    def __init__(self, width: int = 42, height: int = 23, fill_value: int = 0):
        """
        Create mock level data.
        
        Args:
            width: Level width in tiles
            height: Level height in tiles
            fill_value: Default tile value
        """
        self.tiles = np.full((width, height), fill_value, dtype=int)
        self.width = width
        self.height = height
    
    def add_solid_wall(self, x: int, y_start: int, y_end: int):
        """Add a vertical solid wall."""
        for y in range(y_start, y_end + 1):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.tiles[x, y] = 1  # Solid tile
    
    def add_platform(self, x_start: int, x_end: int, y: int):
        """Add a horizontal platform."""
        for x in range(x_start, x_end + 1):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.tiles[x, y] = 1  # Solid tile


class TestTieredReachabilitySystem(unittest.TestCase):
    """Test cases for the main tiered reachability system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tiered_system = TieredReachabilitySystem(debug=True)
        self.test_levels = self._create_test_levels()
    
    def _create_test_levels(self) -> list:
        """Create test levels with known characteristics."""
        levels = []
        
        # Simple open level
        open_level = MockLevelData()
        levels.append(('open_level', open_level, (100, 100), {}))
        
        # Level with walls
        walled_level = MockLevelData()
        walled_level.add_solid_wall(10, 0, 22)  # Vertical wall
        walled_level.add_platform(0, 9, 15)     # Platform before wall
        walled_level.add_platform(11, 20, 15)   # Platform after wall
        levels.append(('walled_level', walled_level, (50, 350), {}))
        
        # Complex level with multiple areas
        complex_level = MockLevelData()
        complex_level.add_platform(0, 15, 20)   # Ground level
        complex_level.add_platform(5, 10, 15)   # Upper platform
        complex_level.add_platform(20, 30, 18)  # Separate area
        levels.append(('complex_level', complex_level, (120, 480), {'switch1': True}))
        
        return levels
    
    def test_tier1_performance(self):
        """Test that Tier 1 meets performance requirements (<1ms)."""
        for level_name, level_data, ninja_pos, switch_states in self.test_levels:
            with self.subTest(level=level_name):
                start_time = time.perf_counter()
                
                result = self.tiered_system.tier1.quick_check(
                    ninja_pos, level_data, switch_states
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # Performance requirements
                self.assertLess(elapsed_ms, 1.0, 
                    f"Tier 1 too slow for {level_name}: {elapsed_ms:.3f}ms")
                
                # Accuracy requirements
                self.assertGreater(result.confidence, 0.80, 
                    f"Tier 1 accuracy too low for {level_name}: {result.confidence}")
                
                # Result validity
                self.assertIsInstance(result, ReachabilityApproximation)
                self.assertGreater(len(result.reachable_positions), 0)
                self.assertEqual(result.tier_used, 1)
    
    def test_tier2_performance(self):
        """Test that Tier 2 meets performance requirements (<10ms)."""
        for level_name, level_data, ninja_pos, switch_states in self.test_levels:
            with self.subTest(level=level_name):
                start_time = time.perf_counter()
                
                result = self.tiered_system.tier2.medium_analysis(
                    ninja_pos, level_data, switch_states
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # Performance requirements
                self.assertLess(elapsed_ms, 10.0, 
                    f"Tier 2 too slow for {level_name}: {elapsed_ms:.3f}ms")
                
                # Accuracy requirements
                self.assertGreater(result.confidence, 0.90, 
                    f"Tier 2 accuracy too low for {level_name}: {result.confidence}")
                
                # Result validity
                self.assertIsInstance(result, ReachabilityResult)
                self.assertGreater(len(result.reachable_positions), 0)
                self.assertEqual(result.tier_used, 2)
    
    def test_adaptive_tier_selection(self):
        """Test adaptive tier selection based on performance targets."""
        level_name, level_data, ninja_pos, switch_states = self.test_levels[0]
        
        # Test different performance targets
        targets_and_expected_tiers = [
            (PerformanceTarget.ULTRA_FAST, 1),
            (PerformanceTarget.FAST, [1, 2]),  # Could be either
            (PerformanceTarget.BALANCED, [1, 2, 3]),  # Could be any
            (PerformanceTarget.ACCURATE, 3)
        ]
        
        for target, expected_tiers in targets_and_expected_tiers:
            with self.subTest(target=target):
                result = self.tiered_system.analyze_reachability(
                    level_data, ninja_pos, switch_states, target
                )
                
                if isinstance(expected_tiers, list):
                    self.assertIn(result.tier_used, expected_tiers)
                else:
                    self.assertEqual(result.tier_used, expected_tiers)
    
    def test_performance_tracking(self):
        """Test performance tracking and history."""
        level_name, level_data, ninja_pos, switch_states = self.test_levels[0]
        
        # Clear history
        self.tiered_system.reset_performance_history()
        
        # Run multiple analyses
        for _ in range(5):
            self.tiered_system.analyze_reachability(
                level_data, ninja_pos, switch_states, PerformanceTarget.ULTRA_FAST
            )
        
        # Check performance summary
        summary = self.tiered_system.get_performance_summary()
        
        self.assertIn('tier1', summary)
        self.assertEqual(summary['tier1']['sample_count'], 5)
        self.assertGreater(summary['tier1']['avg_time_ms'], 0)


class TestFloodFillApproximator(unittest.TestCase):
    """Test cases for Tier 1 flood fill approximator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.approximator = FloodFillApproximator(debug=True)
        self.test_level = MockLevelData()
    
    def test_binary_grid_conversion(self):
        """Test conversion of level data to binary grid."""
        # Add some solid tiles
        self.test_level.add_solid_wall(5, 0, 10)
        
        binary_grid = self.approximator._get_or_create_binary_grid(self.test_level)
        
        # Check shape
        self.assertEqual(binary_grid.shape, (23, 42))  # Note: (height, width)
        
        # Check that solid tiles are marked as non-traversable
        self.assertFalse(binary_grid[5, 5])  # Should be solid
        self.assertTrue(binary_grid[0, 0])   # Should be traversable
    
    def test_vectorized_flood_fill(self):
        """Test vectorized flood fill algorithm."""
        # Create a simple level with a wall
        self.test_level.add_solid_wall(10, 5, 15)
        
        binary_grid = self.approximator._get_or_create_binary_grid(self.test_level)
        
        # Start from left side
        start_tile = (5, 10)
        reachable_mask = self.approximator._vectorized_flood_fill(start_tile, binary_grid)
        
        # Should reach left side but not right side (blocked by wall)
        self.assertTrue(reachable_mask[10, 5])   # Start position
        self.assertTrue(reachable_mask[10, 8])   # Left of wall
        self.assertFalse(reachable_mask[10, 12]) # Right of wall (blocked)
    
    def test_caching_behavior(self):
        """Test caching of binary grids and flood fill results."""
        # Clear cache
        self.approximator.clear_cache()
        
        # First call should be cache miss
        initial_stats = self.approximator.get_cache_stats()
        self.assertEqual(initial_stats['cache_misses'], 0)
        
        # Perform analysis
        result1 = self.approximator.quick_check((100, 100), self.test_level, {})
        
        stats_after_first = self.approximator.get_cache_stats()
        self.assertEqual(stats_after_first['cache_misses'], 1)
        
        # Second call with same level should be cache hit
        result2 = self.approximator.quick_check((150, 150), self.test_level, {})
        
        stats_after_second = self.approximator.get_cache_stats()
        self.assertEqual(stats_after_second['cache_hits'], 1)


class TestSimplifiedPhysicsAnalyzer(unittest.TestCase):
    """Test cases for Tier 2 simplified physics analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SimplifiedPhysicsAnalyzer(debug=True)
        self.test_level = MockLevelData()
    
    def test_jump_pattern_precomputation(self):
        """Test pre-computation of jump patterns."""
        patterns = self.analyzer.physics_model.jump_reach_patterns
        
        # Should have patterns for various distances and heights
        self.assertGreater(len(patterns), 100)
        
        # Test some basic patterns
        # Short horizontal jump should be possible
        self.assertTrue(patterns.get((2, 0, 'clear'), False))
        
        # Very long jump should not be possible
        self.assertFalse(patterns.get((50, 0, 'clear'), True))
    
    def test_tile_group_classification(self):
        """Test tile group classification system."""
        tile_groups = self.analyzer.physics_model.tile_groups
        
        # Solid tiles should be marked as solid
        solid_group = tile_groups[1]  # Solid tile
        self.assertTrue(solid_group.is_solid)
        self.assertTrue(solid_group.allows_wall_jump)
        
        # Empty tiles should not be solid
        empty_group = tile_groups[0]  # Empty tile
        self.assertFalse(empty_group.is_solid)
    
    def test_performance_within_limits(self):
        """Test that analysis completes within performance limits."""
        # Create a moderately complex level
        for i in range(0, 40, 5):
            self.test_level.add_platform(i, i + 3, 20 - (i // 10))
        
        start_time = time.perf_counter()
        
        result = self.analyzer.medium_analysis((100, 400), self.test_level, {})
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Should complete within 10ms target
        self.assertLess(elapsed_ms, 15.0)  # Allow some margin
        self.assertIsInstance(result, ReachabilityResult)


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("Running Tiered Reachability Performance Benchmark")
    print("=" * 50)
    
    tiered_system = TieredReachabilitySystem(debug=False)
    
    # Create test levels of varying complexity
    levels = [
        ("Simple", MockLevelData()),
        ("Complex", _create_complex_level()),
    ]
    
    results = {
        'tier1': {'times': [], 'accuracies': []},
        'tier2': {'times': [], 'accuracies': []}
    }
    
    # Run benchmarks
    for level_name, level_data in levels:
        print(f"\nTesting {level_name} level:")
        
        ninja_pos = (240, 240)
        switch_states = {}
        
        # Test each tier
        for tier_num in [1, 2]:
            times = []
            
            # Run multiple iterations
            for _ in range(10):
                start_time = time.perf_counter()
                
                if tier_num == 1:
                    result = tiered_system.tier1.quick_check(ninja_pos, level_data, switch_states)
                else:
                    result = tiered_system.tier2.medium_analysis(ninja_pos, level_data, switch_states)
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            p95_time = np.percentile(times, 95)
            
            results[f'tier{tier_num}']['times'].extend(times)
            results[f'tier{tier_num}']['accuracies'].append(result.confidence)
            
            print(f"  Tier {tier_num}: {avg_time:.3f}ms avg, {p95_time:.3f}ms p95, "
                  f"{result.confidence:.2f} confidence, {len(result.reachable_positions)} positions")
    
    # Print summary
    print("\nOverall Performance Summary:")
    print("-" * 30)
    
    for tier_num in [1, 2]:
        times = results[f'tier{tier_num}']['times']
        accuracies = results[f'tier{tier_num}']['accuracies']
        
        if times:
            print(f"Tier {tier_num}:")
            print(f"  Average time: {np.mean(times):.3f}ms")
            print(f"  95th percentile: {np.percentile(times, 95):.3f}ms")
            print(f"  Max time: {np.max(times):.3f}ms")
            print(f"  Average accuracy: {np.mean(accuracies):.3f}")


def _create_complex_level():
    """Create a complex level for testing."""
    level = MockLevelData()
    level.add_platform(0, 15, 20)
    level.add_platform(5, 10, 15)
    level.add_platform(20, 35, 18)
    level.add_platform(25, 30, 10)
    return level


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    print("\n" + "=" * 60)
    run_performance_benchmark()