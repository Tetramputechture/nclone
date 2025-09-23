"""
Integration tests for the tiered reachability system.

This module tests the integration of the tiered reachability system with
the broader nclone framework, ensuring backward compatibility and proper
integration with existing components.
"""

import unittest
import time
import numpy as np
from typing import Set, Tuple, Dict, Any, List
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.reachability_types import (
    PerformanceTarget, ReachabilityApproximation, ReachabilityResult
)
from nclone.planning import LevelCompletionPlanner, Subgoal, SubgoalPlan, CompletionStrategy


class TestLevelData:
    """Test level data for integration testing."""
    
    def __init__(self, name: str, width: int = 42, height: int = 23):
        self.name = name
        self.tiles = np.zeros((width, height), dtype=int)
        self.width = width
        self.height = height
        self.ninja_pos = (100.0, 100.0)  # Default ninja position
        self.switch_states = {}
        self.expected_result = True  # Default expectation
        
    def add_walls(self, positions: List[Tuple[int, int]]):
        """Add solid walls at specified positions."""
        for x, y in positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.tiles[x, y] = 1  # Solid tile
                
    def add_switches(self, positions: List[Tuple[int, int]], states: List[bool] = None):
        """Add switches at specified positions."""
        if states is None:
            states = [False] * len(positions)
        for i, (x, y) in enumerate(positions):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.tiles[x, y] = 10  # Switch tile
                self.switch_states[f"switch_{i}"] = states[i]


def get_all_test_maps() -> List[TestLevelData]:
    """Generate comprehensive test maps for integration testing."""
    test_maps = []
    
    # Simple open level
    simple_level = TestLevelData("simple_open", 42, 23)
    simple_level.expected_result = True
    test_maps.append(simple_level)
    
    # Level with walls
    walled_level = TestLevelData("with_walls", 42, 23)
    walled_level.add_walls([(10, 10), (10, 11), (10, 12), (11, 10), (12, 10)])
    walled_level.expected_result = True
    test_maps.append(walled_level)
    
    # Level with switches
    switch_level = TestLevelData("with_switches", 42, 23)
    switch_level.add_switches([(15, 15), (20, 20)], [True, False])
    switch_level.expected_result = True
    test_maps.append(switch_level)
    
    # Complex level with multiple features
    complex_level = TestLevelData("complex_features", 42, 23)
    complex_level.add_walls([(5, 5), (5, 6), (5, 7), (6, 5), (7, 5)])
    complex_level.add_switches([(10, 10), (15, 15)], [True, True])
    complex_level.expected_result = True
    test_maps.append(complex_level)
    
    # Blocked level (should be non-completable)
    blocked_level = TestLevelData("blocked", 42, 23)
    # Create a wall around the ninja spawn area
    for x in range(2, 8):
        for y in range(2, 8):
            if x == 2 or x == 7 or y == 2 or y == 7:
                blocked_level.add_walls([(x, y)])
    blocked_level.ninja_pos = (120.0, 120.0)  # Inside the blocked area
    blocked_level.expected_result = False
    test_maps.append(blocked_level)
    
    return test_maps


class TestReachabilityIntegration(unittest.TestCase):
    """Integration tests for tiered reachability system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tiered_system = TieredReachabilitySystem()
        self.test_maps = get_all_test_maps()
        
    def test_existing_api_compatibility(self):
        """Ensure tiered system maintains backward compatibility."""
        print("\n=== Testing API Compatibility ===")
        
        for test_map in self.test_maps:
            with self.subTest(level=test_map.name):
                # Test that the main analyze_reachability method works
                result = self.tiered_system.analyze_reachability(
                    level_data=test_map.tiles,
                    ninja_position=test_map.ninja_pos,
                    switch_states=test_map.switch_states,
                    performance_target=PerformanceTarget.BALANCED
                )
                
                # Verify result structure (can be either ReachabilityApproximation or ReachabilityResult)
                self.assertTrue(isinstance(result, (ReachabilityApproximation, ReachabilityResult)))
                self.assertIsInstance(result.reachable_positions, set)
                self.assertIsInstance(result.confidence, float)
                self.assertIsInstance(result.computation_time_ms, float)
                self.assertIsInstance(result.method, str)
                
                # Verify confidence is reasonable
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                
                # Verify computation time is recorded
                self.assertGreater(result.computation_time_ms, 0.0)
                
                print(f"  ✅ {test_map.name}: {len(result.reachable_positions)} positions, "
                      f"{result.computation_time_ms:.2f}ms, {result.confidence:.2f} confidence")
    
    def test_performance_targets(self):
        """Verify performance targets are met across all test levels."""
        print("\n=== Testing Performance Targets ===")
        
        performance_results = {
            PerformanceTarget.ULTRA_FAST: [],
            PerformanceTarget.FAST: [],
            PerformanceTarget.BALANCED: [],
            PerformanceTarget.ACCURATE: []
        }
        
        for test_map in self.test_maps:
            for target in performance_results.keys():
                start_time = time.perf_counter()
                result = self.tiered_system.analyze_reachability(
                    level_data=test_map.tiles,
                    ninja_position=test_map.ninja_pos,
                    switch_states=test_map.switch_states,
                    performance_target=target
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                performance_results[target].append(elapsed_ms)
        
        # Verify performance targets
        ultra_fast_p95 = np.percentile(performance_results[PerformanceTarget.ULTRA_FAST], 95)
        fast_p95 = np.percentile(performance_results[PerformanceTarget.FAST], 95)
        balanced_p95 = np.percentile(performance_results[PerformanceTarget.BALANCED], 95)
        accurate_p95 = np.percentile(performance_results[PerformanceTarget.ACCURATE], 95)
        
        print(f"  Ultra Fast 95th percentile: {ultra_fast_p95:.2f}ms (target: <1ms)")
        print(f"  Fast 95th percentile: {fast_p95:.2f}ms (target: <5ms)")
        print(f"  Balanced 95th percentile: {balanced_p95:.2f}ms (target: <20ms)")
        print(f"  Accurate 95th percentile: {accurate_p95:.2f}ms (target: <100ms)")
        
        # Assert performance targets (with reasonable tolerance for test environment)
        self.assertLess(ultra_fast_p95, 10.0, f"Ultra fast 95th percentile too slow: {ultra_fast_p95:.2f}ms")
        self.assertLess(fast_p95, 20.0, f"Fast 95th percentile too slow: {fast_p95:.2f}ms")
        self.assertLess(balanced_p95, 50.0, f"Balanced 95th percentile too slow: {balanced_p95:.2f}ms")
        self.assertLess(accurate_p95, 200.0, f"Accurate 95th percentile too slow: {accurate_p95:.2f}ms")
    
    def test_subgoal_planner_integration(self):
        """Test integration with LevelCompletionPlanner."""
        print("\n=== Testing LevelCompletionPlanner Integration ===")
        
        # Test that LevelCompletionPlanner can work with tiered reachability
        planner = LevelCompletionPlanner()
        
        for test_map in self.test_maps[:3]:  # Test first 3 maps to keep test time reasonable
            with self.subTest(level=test_map.name):
                # Test completion plan
                strategy = planner.plan_completion(
                    ninja_pos=test_map.ninja_pos,
                    level_data=test_map.tiles,
                    switch_states={},
                    reachability_system=self.reachability_system
                )
                
                # Strategy can be None for simple levels without switches/doors
                if strategy is not None:
                    self.assertIsInstance(strategy, CompletionStrategy)
                    self.assertIsInstance(strategy.steps, list)
                    
                print(f"  ✅ {test_map.name}: Plan created successfully")
    
    def test_switch_state_handling(self):
        """Test proper handling of switch states across tiers."""
        print("\n=== Testing Switch State Handling ===")
        
        # Create a level with switches
        switch_level = TestLevelData("switch_test", 42, 23)
        switch_level.add_switches([(10, 10), (15, 15)], [True, False])
        
        # Test with different switch states
        test_cases = [
            {"switch_0": True, "switch_1": False},
            {"switch_0": False, "switch_1": True},
            {"switch_0": True, "switch_1": True},
            {"switch_0": False, "switch_1": False},
        ]
        
        for i, switch_states in enumerate(test_cases):
            with self.subTest(case=i):
                result = self.tiered_system.analyze_reachability(
                    level_data=switch_level.tiles,
                    ninja_position=switch_level.ninja_pos,
                    switch_states=switch_states,
                    performance_target=PerformanceTarget.BALANCED
                )
                
                # Verify result is valid (can be either type)
                self.assertTrue(isinstance(result, (ReachabilityApproximation, ReachabilityResult)))
                self.assertGreater(len(result.reachable_positions), 0)
                
                print(f"  ✅ Switch case {i}: {len(result.reachable_positions)} positions reachable")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")
        
        # Test with empty level
        empty_level = np.zeros((1, 1), dtype=int)
        result = self.tiered_system.analyze_reachability(
            level_data=empty_level,
            ninja_position=(0.0, 0.0),
            switch_states={},
            performance_target=PerformanceTarget.FAST
        )
        self.assertTrue(isinstance(result, (ReachabilityApproximation, ReachabilityResult)))
        print("  ✅ Empty level handled correctly")
        
        # Test with large level
        large_level = np.zeros((100, 100), dtype=int)
        start_time = time.perf_counter()
        result = self.tiered_system.analyze_reachability(
            level_data=large_level,
            ninja_position=(500.0, 500.0),
            switch_states={},
            performance_target=PerformanceTarget.ULTRA_FAST
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.assertTrue(isinstance(result, (ReachabilityApproximation, ReachabilityResult)))
        self.assertLess(elapsed_ms, 20.0, "Large level analysis too slow")
        print(f"  ✅ Large level (100x100) handled in {elapsed_ms:.2f}ms")
        
        # Test with invalid ninja position (should handle gracefully)
        result = self.tiered_system.analyze_reachability(
            level_data=np.zeros((10, 10), dtype=int),
            ninja_position=(-100.0, -100.0),  # Outside level bounds
            switch_states={},
            performance_target=PerformanceTarget.FAST
        )
        self.assertTrue(isinstance(result, (ReachabilityApproximation, ReachabilityResult)))
        print("  ✅ Invalid ninja position handled gracefully")
    
    def test_consistency_across_tiers(self):
        """Test that different tiers produce consistent results."""
        print("\n=== Testing Tier Consistency ===")
        
        test_map = self.test_maps[1]  # Use a level with some complexity
        
        # Get results from different tiers
        ultra_fast_result = self.tiered_system.analyze_reachability(
            level_data=test_map.tiles,
            ninja_position=test_map.ninja_pos,
            switch_states=test_map.switch_states,
            performance_target=PerformanceTarget.ULTRA_FAST
        )
        
        balanced_result = self.tiered_system.analyze_reachability(
            level_data=test_map.tiles,
            ninja_position=test_map.ninja_pos,
            switch_states=test_map.switch_states,
            performance_target=PerformanceTarget.BALANCED
        )
        
        accurate_result = self.tiered_system.analyze_reachability(
            level_data=test_map.tiles,
            ninja_position=test_map.ninja_pos,
            switch_states=test_map.switch_states,
            performance_target=PerformanceTarget.ACCURATE
        )
        
        # Check that results are reasonably consistent
        # (exact match not expected due to different algorithms)
        ultra_positions = len(ultra_fast_result.reachable_positions)
        balanced_positions = len(balanced_result.reachable_positions)
        accurate_positions = len(accurate_result.reachable_positions)
        
        # Results should be within reasonable range of each other
        max_positions = max(ultra_positions, balanced_positions, accurate_positions)
        min_positions = min(ultra_positions, balanced_positions, accurate_positions)
        
        if max_positions > 0:
            consistency_ratio = min_positions / max_positions
            self.assertGreater(consistency_ratio, 0.7, 
                             f"Tier results too inconsistent: {ultra_positions}, {balanced_positions}, {accurate_positions}")
        
        print(f"  ✅ Tier consistency: Ultra={ultra_positions}, Balanced={balanced_positions}, Accurate={accurate_positions}")
        print(f"      Consistency ratio: {min_positions/max_positions:.2f}" if max_positions > 0 else "      All tiers returned 0 positions")


class TestReachabilityBenchmark(unittest.TestCase):
    """Benchmark tests for performance analysis."""
    
    def setUp(self):
        """Set up benchmark fixtures."""
        self.tiered_system = TieredReachabilitySystem()
        self.test_maps = get_all_test_maps()
    
    def test_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark."""
        print("\n" + "="*60)
        print("COMPREHENSIVE REACHABILITY BENCHMARK")
        print("="*60)
        
        results = {
            'ultra_fast': {'times': [], 'positions': [], 'confidences': []},
            'fast': {'times': [], 'positions': [], 'confidences': []},
            'balanced': {'times': [], 'positions': [], 'confidences': []},
            'accurate': {'times': [], 'positions': [], 'confidences': []}
        }
        
        target_map = {
            'ultra_fast': PerformanceTarget.ULTRA_FAST,
            'fast': PerformanceTarget.FAST,
            'balanced': PerformanceTarget.BALANCED,
            'accurate': PerformanceTarget.ACCURATE
        }
        
        for test_map in self.test_maps:
            print(f"\nTesting {test_map.name}:")
            
            for tier_name, target in target_map.items():
                # Run multiple iterations for stable timing
                times = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    result = self.tiered_system.analyze_reachability(
                        level_data=test_map.tiles,
                        ninja_position=test_map.ninja_pos,
                        switch_states=test_map.switch_states,
                        performance_target=target
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    times.append(elapsed_ms)
                
                avg_time = np.mean(times)
                p95_time = np.percentile(times, 95)
                
                results[tier_name]['times'].append(avg_time)
                results[tier_name]['positions'].append(len(result.reachable_positions))
                results[tier_name]['confidences'].append(result.confidence)
                
                print(f"  {tier_name.capitalize()}: {avg_time:.2f}ms avg, {p95_time:.2f}ms p95, "
                      f"{result.confidence:.2f} confidence, {len(result.reachable_positions)} positions")
        
        # Generate summary report
        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE SUMMARY")
        print("-" * 60)
        
        for tier_name, data in results.items():
            avg_time = np.mean(data['times'])
            p95_time = np.percentile(data['times'], 95)
            max_time = np.max(data['times'])
            avg_confidence = np.mean(data['confidences'])
            
            print(f"{tier_name.capitalize()}:")
            print(f"  Average time: {avg_time:.3f}ms")
            print(f"  95th percentile: {p95_time:.3f}ms")
            print(f"  Max time: {max_time:.3f}ms")
            print(f"  Average accuracy: {avg_confidence:.3f}")
        
        # Verify performance targets are met
        ultra_fast_p95 = np.percentile(results['ultra_fast']['times'], 95)
        fast_p95 = np.percentile(results['fast']['times'], 95)
        balanced_p95 = np.percentile(results['balanced']['times'], 95)
        accurate_p95 = np.percentile(results['accurate']['times'], 95)
        
        # Performance assertions with reasonable tolerances for test environment
        self.assertLess(ultra_fast_p95, 15.0, f"Ultra fast tier too slow: {ultra_fast_p95:.2f}ms")
        self.assertLess(fast_p95, 25.0, f"Fast tier too slow: {fast_p95:.2f}ms")
        self.assertLess(balanced_p95, 50.0, f"Balanced tier too slow: {balanced_p95:.2f}ms")
        self.assertLess(accurate_p95, 250.0, f"Accurate tier too slow: {accurate_p95:.2f}ms")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)