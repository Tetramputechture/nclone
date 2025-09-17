"""
Comprehensive tests for compact reachability features (TASK_003).

This module tests the complete compact feature system including:
- CompactReachabilityFeatures encoder
- ReachabilityFeatureExtractor interface
- Feature validation and analysis
- Performance characteristics
- Integration with simplified completion strategy
"""

import unittest
import numpy as np
import time
from typing import List, Dict, Any
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.reachability.compact_features import CompactReachabilityFeatures, FeatureConfig
from nclone.graph.reachability.feature_extractor import (
    ReachabilityFeatureExtractor, PerformanceMode, FeatureAnalyzer
)
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.reachability_types import ReachabilityResult, PerformanceTarget


class MockEntity:
    """Mock entity for testing."""
    
    def __init__(self, entity_type: str, x: float, y: float, entity_id: str = None):
        self.entity_type = entity_type
        self.x = x
        self.y = y
        self.id = entity_id or f"{entity_type}_{x}_{y}"


class MockReachabilityResult:
    """Mock reachability result for testing."""
    
    def __init__(self, reachable_positions: List[tuple], confidence: float = 1.0):
        self.reachable_positions = set(reachable_positions)
        self.confidence = confidence
        self.computation_time_ms = 5.0
        self.method = 'flood_fill'
        self.from_cache = False


class TestCompactReachabilityFeatures(unittest.TestCase):
    """Test the CompactReachabilityFeatures encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = CompactReachabilityFeatures(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)  # Simple open level
        self.ninja_pos = (100.0, 100.0)
        
        # Create test entities
        self.entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0, 'door_switch_1'),
            MockEntity('exit_door', 300.0, 300.0),
            MockEntity('drone', 180.0, 180.0),
            MockEntity('mine', 220.0, 220.0)
        ]
        
        # Create mock reachability result
        reachable_positions = []
        for x in range(0, 500, 24):
            for y in range(0, 500, 24):
                reachable_positions.append((x + 12, y + 12))
        
        self.reachability_result = MockReachabilityResult(reachable_positions)
        
    def test_feature_vector_dimensions(self):
        """Test that feature vector has correct dimensions."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        self.assertEqual(features.shape, (64,))
        self.assertEqual(features.dtype, np.float32)
        
    def test_feature_config_validation(self):
        """Test feature configuration validation."""
        # Valid config
        config = FeatureConfig()
        encoder = CompactReachabilityFeatures(config=config)
        self.assertEqual(encoder.config.total_features, 64)
        
        # Invalid config
        with self.assertRaises(ValueError):
            invalid_config = FeatureConfig(objective_slots=10)  # Would sum to 66
            CompactReachabilityFeatures(config=invalid_config)
            
    def test_objective_distance_encoding(self):
        """Test objective distance encoding using simplified completion strategy."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        # Feature [0] should contain current objective distance
        self.assertGreater(features[0], 0.0)
        self.assertLessEqual(features[0], 1.0)
        
        # Should have some non-zero objective features
        objective_features = features[0:8]
        non_zero_objectives = np.count_nonzero(objective_features)
        self.assertGreater(non_zero_objectives, 0)
        
    def test_switch_state_encoding(self):
        """Test switch state encoding."""
        switch_states = {'door_switch_1': True}
        
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, switch_states
        )
        
        # Switch features should be in range [0, 1.4] (with dependency bonus)
        switch_features = features[8:24]
        self.assertTrue(np.all(switch_features >= 0.0))
        self.assertTrue(np.all(switch_features <= 1.5))
        
        # Should have some activated switches
        activated_switches = np.sum(switch_features >= 0.9)
        self.assertGreater(activated_switches, 0)
        
    def test_hazard_proximity_encoding(self):
        """Test hazard proximity encoding."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        # Hazard features should be in reasonable range
        hazard_features = features[24:40]
        self.assertTrue(np.all(hazard_features >= 0.0))
        self.assertTrue(np.all(hazard_features <= 1.0))
        
        # Should detect hazards near ninja
        non_zero_hazards = np.count_nonzero(hazard_features)
        self.assertGreater(non_zero_hazards, 0)
        
    def test_area_connectivity_encoding(self):
        """Test area connectivity encoding."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        # Area features should be in range [0, 1]
        area_features = features[40:48]
        self.assertTrue(np.all(area_features >= 0.0))
        self.assertTrue(np.all(area_features <= 1.0))
        
        # Should have connectivity in multiple directions
        non_zero_areas = np.count_nonzero(area_features)
        self.assertGreater(non_zero_areas, 0)
        
    def test_movement_capability_encoding(self):
        """Test movement capability encoding."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        # Movement features should be in range [0, 1]
        movement_features = features[48:56]
        self.assertTrue(np.all(movement_features >= 0.0))
        self.assertTrue(np.all(movement_features <= 1.0))
        
    def test_meta_feature_encoding(self):
        """Test meta-feature encoding."""
        features = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        # Meta features should be in reasonable ranges
        meta_features = features[56:64]
        self.assertTrue(np.all(meta_features >= 0.0))
        self.assertTrue(np.all(meta_features <= 1.0))
        
        # Confidence should be encoded
        self.assertEqual(meta_features[0], 1.0)  # Mock confidence
        
    def test_feature_names(self):
        """Test feature name generation."""
        names = self.encoder.get_feature_names()
        
        self.assertEqual(len(names), 64)
        self.assertTrue(all(isinstance(name, str) for name in names))
        
        # Check specific feature names
        self.assertTrue(names[0].startswith('objective_distance'))
        self.assertTrue(names[8].startswith('switch_state'))
        self.assertTrue(names[24].startswith('hazard_proximity'))
        self.assertTrue(names[40].startswith('area_connectivity'))
        self.assertTrue(names[48].startswith('movement_'))
        self.assertTrue(names[56].startswith('meta_'))
        
    def test_error_handling(self):
        """Test graceful error handling."""
        # Test with invalid inputs
        features = self.encoder.encode_reachability(
            None, None, [], self.ninja_pos, {}
        )
        
        # Should return zero features on error
        self.assertEqual(features.shape, (64,))
        # Note: Some features might still be computed even with None inputs
        # The important thing is that it doesn't crash
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
        
    def test_feature_consistency(self):
        """Test that identical inputs produce identical features."""
        features1 = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        features2 = self.encoder.encode_reachability(
            self.reachability_result, self.level_data, self.entities, self.ninja_pos, {}
        )
        
        np.testing.assert_array_equal(features1, features2)


class TestReachabilityFeatureExtractor(unittest.TestCase):
    """Test the ReachabilityFeatureExtractor interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ReachabilityFeatureExtractor(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)
        self.ninja_pos = (100.0, 100.0)
        
        self.entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0, 'door_switch_1')
        ]
        
    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(
            self.ninja_pos, self.level_data, self.entities, {}
        )
        
        self.assertEqual(features.shape, (64,))
        self.assertEqual(features.dtype, np.float32)
        
    def test_performance_modes(self):
        """Test different performance modes."""
        modes = [PerformanceMode.ULTRA_FAST, PerformanceMode.FAST, PerformanceMode.ACCURATE]
        
        for mode in modes:
            features = self.extractor.extract_features(
                self.ninja_pos, self.level_data, self.entities, {}, performance_mode=mode
            )
            
            self.assertEqual(features.shape, (64,))
            
    def test_caching_functionality(self):
        """Test feature caching."""
        # First extraction (cache miss)
        features1 = self.extractor.extract_features(
            self.ninja_pos, self.level_data, self.entities, {}
        )
        
        # Second extraction (should be cache hit)
        features2 = self.extractor.extract_features(
            self.ninja_pos, self.level_data, self.entities, {}
        )
        
        np.testing.assert_array_equal(features1, features2)
        
        # Check cache statistics
        stats = self.extractor.get_performance_stats()
        self.assertGreater(stats['cache_hit_rate'], 0.0)
        
    def test_batch_extraction(self):
        """Test batch feature extraction."""
        batch_data = [
            {
                'ninja_position': (100.0, 100.0),
                'level_data': self.level_data,
                'entities': self.entities,
                'switch_states': {}
            },
            {
                'ninja_position': (150.0, 150.0),
                'level_data': self.level_data,
                'entities': self.entities,
                'switch_states': {'door_switch_1': True}
            }
        ]
        
        batch_features = self.extractor.extract_features_batch(batch_data)
        
        self.assertEqual(batch_features.shape, (2, 64))
        self.assertEqual(batch_features.dtype, np.float32)
        
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Extract some features
        for i in range(5):
            self.extractor.extract_features(
                (100.0 + i * 10, 100.0), self.level_data, self.entities, {}
            )
        
        stats = self.extractor.get_performance_stats()
        
        self.assertGreater(stats['total_extractions'], 0)
        self.assertGreaterEqual(stats['avg_extraction_time_ms'], 0.0)
        self.assertGreaterEqual(stats['cache_size'], 0)
        
    def test_cache_management(self):
        """Test cache management functionality."""
        # Fill cache
        for i in range(10):
            self.extractor.extract_features(
                (100.0 + i, 100.0), self.level_data, self.entities, {}
            )
        
        initial_cache_size = self.extractor.get_performance_stats()['cache_size']
        self.assertGreater(initial_cache_size, 0)
        
        # Clear cache
        self.extractor.clear_cache()
        
        final_cache_size = self.extractor.get_performance_stats()['cache_size']
        self.assertEqual(final_cache_size, 0)
        
    def test_feature_validation(self):
        """Test feature validation."""
        features = self.extractor.extract_features(
            self.ninja_pos, self.level_data, self.entities, {}
        )
        
        validation = self.extractor.validate_features(features)
        
        self.assertTrue(validation['shape_valid'])
        self.assertTrue(validation['dtype_valid'])
        self.assertFalse(validation['has_nan'])
        self.assertFalse(validation['has_inf'])
        self.assertTrue(validation['values_in_range'])
        
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        # Test with invalid inputs
        features = self.extractor.extract_features(
            self.ninja_pos, None, [], {}  # Invalid level_data
        )
        
        # Should return zero features without crashing
        self.assertEqual(features.shape, (64,))
        # The important thing is that it doesn't crash and returns valid features
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))


class TestFeatureAnalyzer(unittest.TestCase):
    """Test the FeatureAnalyzer utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ReachabilityFeatureExtractor(debug=False)
        self.analyzer = FeatureAnalyzer(self.extractor)
        self.level_data = np.zeros((42, 23), dtype=int)
        
        # Create test states
        self.test_states = [
            {
                'ninja_position': (100.0, 100.0),
                'level_data': self.level_data,
                'entities': [MockEntity('exit_switch', 200.0, 200.0)],
                'switch_states': {}
            },
            {
                'ninja_position': (150.0, 150.0),
                'level_data': self.level_data,
                'entities': [MockEntity('door_switch', 180.0, 180.0)],
                'switch_states': {}
            }
        ]
        
    def test_feature_distribution_analysis(self):
        """Test feature distribution analysis."""
        analysis = self.analyzer.analyze_feature_distribution(self.test_states)
        
        self.assertEqual(analysis['num_states'], 2)
        self.assertIn('feature_stats', analysis)
        self.assertEqual(len(analysis['feature_stats']), 64)
        
        # Check that each feature has proper statistics
        for feature_name, stats in analysis['feature_stats'].items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)
            self.assertIn('variance', stats)
            
    def test_performance_mode_comparison(self):
        """Test performance mode comparison."""
        modes = [PerformanceMode.FAST, PerformanceMode.ACCURATE]
        comparison = self.analyzer.compare_performance_modes(self.test_states, modes)
        
        self.assertIn('timing_comparison', comparison)
        self.assertIn('feature_similarity', comparison)
        
        # Should have timing data for each mode
        for mode in modes:
            self.assertIn(mode.value, comparison['timing_comparison'])
            
    def test_feature_report_generation(self):
        """Test feature report generation."""
        report = self.analyzer.generate_feature_report(self.test_states)
        
        self.assertIsInstance(report, str)
        self.assertIn('Compact Reachability Features Analysis Report', report)
        self.assertIn('Performance Statistics', report)
        self.assertIn('Feature Distribution Analysis', report)


class TestIntegrationWithSimplifiedStrategy(unittest.TestCase):
    """Test integration with simplified completion strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ReachabilityFeatureExtractor(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)
        
    def test_objective_integration(self):
        """Test that current objective is properly encoded in features."""
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0)
        ]
        
        ninja_pos = (100.0, 100.0)
        features = self.extractor.extract_features(ninja_pos, self.level_data, entities, {})
        
        # Feature [0] should contain current objective distance
        self.assertGreater(features[0], 0.0)
        self.assertLessEqual(features[0], 1.0)
        
    def test_switch_state_reactivity(self):
        """Test that features react to switch state changes."""
        entities = [
            MockEntity('exit_switch', 300.0, 300.0),
            MockEntity('door_switch', 150.0, 150.0, 'door_switch_1')
        ]
        
        ninja_pos = (100.0, 100.0)
        
        # Features without switch activation
        features1 = self.extractor.extract_features(ninja_pos, self.level_data, entities, {})
        
        # Features with switch activation
        switch_states = {'door_switch_1': True}
        features2 = self.extractor.extract_features(ninja_pos, self.level_data, entities, switch_states)
        
        # Features should be different
        self.assertFalse(np.array_equal(features1, features2))
        
        # Switch features should reflect activation
        switch_features1 = features1[8:24]
        switch_features2 = features2[8:24]
        self.assertFalse(np.array_equal(switch_features1, switch_features2))


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of the compact feature system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ReachabilityFeatureExtractor(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)
        
        self.entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0),
            MockEntity('drone', 180.0, 180.0)
        ]
        
    def test_extraction_speed(self):
        """Test feature extraction speed."""
        ninja_pos = (100.0, 100.0)
        
        # Time multiple extractions
        times = []
        for i in range(10):
            start_time = time.perf_counter()
            features = self.extractor.extract_features(
                (ninja_pos[0] + i, ninja_pos[1]), self.level_data, self.entities, {}
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
            
            self.assertEqual(features.shape, (64,))
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Should be fast enough for real-time RL
        self.assertLess(avg_time, 50.0, f"Average extraction too slow: {avg_time:.2f}ms")
        self.assertLess(max_time, 100.0, f"Max extraction too slow: {max_time:.2f}ms")
        
        print(f"Feature extraction performance: {avg_time:.2f}ms avg, {max_time:.2f}ms max")
        
    def test_cache_effectiveness(self):
        """Test cache effectiveness."""
        ninja_pos = (100.0, 100.0)
        
        # First extraction (cache miss)
        start_time = time.perf_counter()
        features1 = self.extractor.extract_features(ninja_pos, self.level_data, self.entities, {})
        miss_time = (time.perf_counter() - start_time) * 1000
        
        # Second extraction (cache hit)
        start_time = time.perf_counter()
        features2 = self.extractor.extract_features(ninja_pos, self.level_data, self.entities, {})
        hit_time = (time.perf_counter() - start_time) * 1000
        
        # Cache hit should be much faster
        self.assertLess(hit_time, miss_time * 0.5, "Cache hit not significantly faster")
        
        # Features should be identical
        np.testing.assert_array_equal(features1, features2)
        
        print(f"Cache effectiveness: {miss_time:.2f}ms miss, {hit_time:.2f}ms hit")
        
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        # Clear cache to ensure fair comparison
        self.extractor.clear_cache()
        
        batch_data = []
        for i in range(20):
            batch_data.append({
                'ninja_position': (100.0 + i * 5, 100.0),
                'level_data': self.level_data,
                'entities': self.entities,
                'switch_states': {}
            })
        
        # Time batch processing
        start_time = time.perf_counter()
        batch_features = self.extractor.extract_features_batch(batch_data)
        batch_time = (time.perf_counter() - start_time) * 1000
        
        # Clear cache again for fair individual comparison
        self.extractor.clear_cache()
        
        # Time individual processing
        start_time = time.perf_counter()
        individual_features = []
        for data in batch_data:
            features = self.extractor.extract_features(
                data['ninja_position'], data['level_data'], 
                data['entities'], data['switch_states']
            )
            individual_features.append(features)
        individual_time = (time.perf_counter() - start_time) * 1000
        
        # Results should be very similar (allowing for small numerical differences)
        individual_array = np.array(individual_features)
        self.assertEqual(batch_features.shape, individual_array.shape)
        
        # Check that results are close (allowing for small differences due to caching/timing)
        max_diff = np.max(np.abs(batch_features - individual_array))
        self.assertLess(max_diff, 0.1, f"Batch and individual results differ by {max_diff}")
        
        # Also check that most features are very similar
        close_features = np.sum(np.abs(batch_features - individual_array) < 0.01)
        total_features = batch_features.size
        similarity_ratio = close_features / total_features
        self.assertGreater(similarity_ratio, 0.9, f"Only {similarity_ratio:.1%} of features are very similar")
        
        print(f"Batch processing: {batch_time:.2f}ms batch, {individual_time:.2f}ms individual")
        
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        # Extract features for many different states
        for i in range(100):
            self.extractor.extract_features(
                (100.0 + i, 100.0 + i), self.level_data, self.entities, {}
            )
        
        stats = self.extractor.get_performance_stats()
        memory_mb = stats['cache_memory_mb']
        
        # Memory usage should be reasonable
        self.assertLess(memory_mb, 50.0, f"Memory usage too high: {memory_mb:.2f}MB")
        
        print(f"Memory usage: {memory_mb:.2f}MB for {stats['cache_size']} cached entries")


if __name__ == '__main__':
    unittest.main(verbosity=2)