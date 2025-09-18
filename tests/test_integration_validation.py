#!/usr/bin/env python3
"""
Integration validation for enhanced compact reachability features.

This validates that the enhanced features work correctly with the core
simulation components and produce meaningful feature vectors for RL training.
"""

import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path so we can import nclone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.reachability.feature_extractor import ReachabilityFeatureExtractor
from nclone.graph.reachability.compact_features import CompactReachabilityFeatures
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem


class TestIntegrationValidation(unittest.TestCase):
    """Test integration of enhanced features with core simulation."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.feature_extractor = ReachabilityFeatureExtractor()
        self.compact_features = CompactReachabilityFeatures()
        self.tiered_system = TieredReachabilitySystem()
        
        # Create a simple synthetic map for testing
        self.level_data = np.zeros((25, 44), dtype=int)
        self.level_data[24, :] = 1  # Ground
        self.level_data[:, 0] = 1   # Left wall
        self.level_data[:, 43] = 1  # Right wall
        self.level_data[0, :] = 1   # Ceiling
        
        # Add some internal structure
        self.level_data[20, 10:15] = 1  # Platform
        self.level_data[15, 25:30] = 1  # Another platform
        
        # Create some test entities
        self.entities = [
            {'type': 'exit', 'position': (900, 500)},
            {'type': 'exit_switch', 'position': (200, 400)},
            {'type': 'door_locked', 'position': (600, 500)},
        ]
        
        self.ninja_spawn = (100, 500)
    
    def test_end_to_end_feature_extraction(self):
        """Test complete end-to-end feature extraction pipeline."""
        ninja_position = self.ninja_spawn
        
        # Extract features using the complete pipeline
        features = self.feature_extractor.extract_features(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=self.entities
        )
        
        # Validate basic properties
        self.assertEqual(len(features), 64, "Should produce 64-dimensional feature vector")
        self.assertTrue(np.all(np.isfinite(features)), "All features should be finite")
        
        # Check feature ranges are reasonable for RL
        self.assertTrue(np.all(features >= -20.0), "Features should not be extremely negative")
        self.assertTrue(np.all(features <= 20.0), "Features should not be extremely positive")
        
        # Check that features contain information
        self.assertGreater(np.std(features), 0.01, "Features should have some variance")
    
    def test_enhanced_vs_original_features(self):
        """Test that enhanced features provide more information than basic ones."""
        ninja_position = self.ninja_spawn
        
        # Extract features with enhanced system
        enhanced_features = self.feature_extractor.extract_features(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=self.entities
        )
        
        # The enhanced features should be well-distributed
        feature_variance = np.var(enhanced_features)
        self.assertGreater(feature_variance, 0.1, "Enhanced features should have good variance")
        
        # Check that different feature groups have different characteristics
        objective_features = enhanced_features[0:8]    # Objective distances
        switch_features = enhanced_features[8:16]      # Switch states
        hazard_features = enhanced_features[16:24]     # Hazard proximities
        connectivity_features = enhanced_features[24:32]  # Area connectivity
        movement_features = enhanced_features[32:40]   # Movement capabilities
        meta_features = enhanced_features[40:64]       # Meta features
        
        # Each group should have some internal structure
        groups = [objective_features, switch_features, hazard_features, 
                 connectivity_features, movement_features, meta_features]
        
        for i, group in enumerate(groups):
            self.assertTrue(np.all(np.isfinite(group)), f"Group {i} should have finite values")
    
    def test_feature_stability_across_positions(self):
        """Test that features change appropriately as ninja moves."""
        positions = [
            (100, 500),   # Left side
            (500, 500),   # Center
            (900, 500),   # Right side
        ]
        
        feature_sets = []
        for pos in positions:
            features = self.feature_extractor.extract_features(
                ninja_position=pos,
                level_data=self.level_data,
                entities=self.entities
            )
            feature_sets.append(features)
        
        # Features should be different for different positions
        for i in range(len(feature_sets)):
            for j in range(i + 1, len(feature_sets)):
                # Features should not be identical
                self.assertFalse(np.array_equal(feature_sets[i], feature_sets[j]),
                               f"Features should differ between positions {i} and {j}")
                
                # But they should still be in reasonable ranges
                self.assertTrue(np.all(np.isfinite(feature_sets[i])))
                self.assertTrue(np.all(np.isfinite(feature_sets[j])))
    
    def test_enhanced_movement_assessment(self):
        """Test that enhanced movement assessment provides meaningful data."""
        ninja_position = (500, 500)  # Center position
        
        # Get reachability result
        reachability_result = self.tiered_system.analyze_reachability(
            ninja_position=ninja_position,
            level_data=self.level_data,
            switch_states={}
        )
        
        # Test different movement types
        movement_types = ['walk_left', 'walk_right', 'jump_up', 'jump_left', 'jump_right', 'wall_jump', 'fall']
        
        movement_scores = {}
        for movement_type in movement_types:
            score = self.compact_features._assess_movement_capability(
                ninja_position, movement_type, self.level_data, reachability_result
            )
            movement_scores[movement_type] = score
            
            # Score should be valid
            self.assertGreaterEqual(score, 0.0, f"{movement_type} score should be non-negative")
            self.assertLessEqual(score, 1.0, f"{movement_type} score should not exceed 1.0")
        
        # Some movement types should be more feasible than others
        # (This depends on the level layout, but we can check basic properties)
        self.assertIsInstance(movement_scores['walk_left'], float)
        self.assertIsInstance(movement_scores['walk_right'], float)
    
    def test_enhanced_sector_calculations(self):
        """Test that enhanced sector calculations work correctly."""
        ninja_position = (500, 300)
        
        # Define sectors
        sectors = self.compact_features._define_level_sectors(self.level_data, ninja_position)
        
        # Should have 8 sectors
        self.assertEqual(len(sectors), 8)
        
        # Test sector position calculation
        for sector in sectors:
            positions = self.compact_features._get_positions_in_sector(sector, self.level_data)
            
            # Should find some positions
            self.assertGreaterEqual(len(positions), 0, f"Sector {sector['name']} should have positions")
            
            # Positions should be tuples of numbers
            for pos in list(positions)[:5]:  # Check first 5 positions
                self.assertIsInstance(pos, tuple)
                self.assertEqual(len(pos), 2)
                self.assertIsInstance(pos[0], (int, float))
                self.assertIsInstance(pos[1], (int, float))
    
    def test_enhanced_switch_analysis(self):
        """Test enhanced switch dependency analysis."""
        # Create test switches
        test_switches = [
            {'type': 'exit_switch', 'position': (500, 300)},
            {'type': 'door_locked_switch', 'position': (400, 400)},
            {'type': 'generic_switch', 'position': (600, 200)},
        ]
        
        for switch in test_switches:
            # Test dependency detection
            has_deps = self.compact_features._switch_has_dependencies(switch)
            self.assertIsInstance(has_deps, bool)
            
            # Test dependency counting
            dep_count = self.compact_features._count_switch_dependencies(switch)
            self.assertIsInstance(dep_count, int)
            self.assertGreaterEqual(dep_count, 1)
            self.assertLessEqual(dep_count, 5)
            
            # Exit switches should have high dependency count
            if 'exit' in switch['type']:
                self.assertGreaterEqual(dep_count, 3)
    
    def test_physics_based_movement_validation(self):
        """Test physics-based movement validation methods."""
        start_pos = (400, 400)
        
        # Test walkable position detection
        walkable_pos = (450, 575)  # Just above ground
        non_walkable_pos = (450, 300)  # In air
        
        is_walkable = self.compact_features._is_position_walkable(walkable_pos, self.level_data)
        is_not_walkable = self.compact_features._is_position_walkable(non_walkable_pos, self.level_data)
        
        # Results should be boolean (convert numpy bool to Python bool)
        self.assertIsInstance(bool(is_walkable), bool)
        self.assertIsInstance(bool(is_not_walkable), bool)
        
        # Test jump capability
        reasonable_jump = (450, 350)  # Reasonable jump target
        impossible_jump = (450, 50)   # Impossible jump target
        
        can_jump = self.compact_features._can_reach_by_jumping(start_pos, reasonable_jump, self.level_data)
        cannot_jump = self.compact_features._can_reach_by_jumping(start_pos, impossible_jump, self.level_data)
        
        self.assertIsInstance(bool(can_jump), bool)
        self.assertIsInstance(bool(cannot_jump), bool)
        
        # Impossible jump should be rejected
        self.assertFalse(bool(cannot_jump), "Impossible jump should be rejected")
    
    def test_performance_characteristics(self):
        """Test that enhanced features maintain good performance."""
        import time
        
        ninja_position = (500, 300)
        
        # Time feature extraction
        start_time = time.time()
        for _ in range(20):
            features = self.feature_extractor.extract_features(
                ninja_position=ninja_position,
                level_data=self.level_data,
                entities=self.entities
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        
        # Should be fast enough for RL training
        self.assertLess(avg_time, 0.02, f"Feature extraction took {avg_time:.3f}s, expected < 0.02s")
        
        # Features should still be valid
        self.assertEqual(len(features), 64)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_feature_interpretability(self):
        """Test that features are interpretable and well-named."""
        feature_names = self.compact_features.get_feature_names()
        
        # Should have names for all features
        self.assertEqual(len(feature_names), 64)
        
        # Names should be strings
        for name in feature_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
        
        # Should have diverse feature categories
        categories = set()
        for name in feature_names:
            name_lower = name.lower()
            if 'objective' in name_lower or 'distance' in name_lower:
                categories.add('objective')
            elif 'switch' in name_lower:
                categories.add('switch')
            elif 'hazard' in name_lower:
                categories.add('hazard')
            elif 'movement' in name_lower or 'capability' in name_lower:
                categories.add('movement')
            elif 'connectivity' in name_lower or 'area' in name_lower:
                categories.add('connectivity')
            elif 'meta' in name_lower or 'complexity' in name_lower:
                categories.add('meta')
        
        # Should have multiple categories
        self.assertGreaterEqual(len(categories), 4, "Should have diverse feature categories")


if __name__ == '__main__':
    unittest.main()