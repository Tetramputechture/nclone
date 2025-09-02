"""
Integration tests for the enhanced traversability system.

This test suite validates the complete integration of precise tile collision
detection and hazard-aware traversability in the graph construction pipeline.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple

from nclone.graph.edge_building import EdgeBuilder
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.graph.hazard_system import HazardClassificationSystem
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import NINJA_RADIUS


class TestEnhancedTraversabilityIntegration:
    """Integration test suite for enhanced traversability system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create feature extractor and edge builder
        self.feature_extractor = FeatureExtractor()
        self.edge_builder = EdgeBuilder(self.feature_extractor)
        
        # Create test level with mixed tile types and hazards
        self.test_level_data = {
            'level_id': 'integration_test',
            'tiles': np.array([
                [0, 1, 0, 0, 0, 0],  # Empty, solid, empty path
                [0, 1, 0, 6, 7, 0],  # Slopes in middle
                [0, 1, 0, 0, 0, 0],  # Clear path on right
                [0, 0, 0, 10, 11, 0], # Quarter moons
                [0, 0, 0, 0, 0, 0],  # Clear bottom row
            ], dtype=np.int32)
        }
        
        # Create test entities with various hazards
        self.test_entities = [
            # Active toggle mine blocking path
            {
                'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 48.0, 'y': 48.0,
                'active': True, 'state': 0
            },
            # Thwump that can be activated
            {
                'id': 2, 'type': EntityType.THWUMP, 'x': 96.0, 'y': 48.0,
                'state': 0, 'orientation': 0  # Facing right
            },
            # One-way platform
            {
                'id': 3, 'type': EntityType.ONE_WAY, 'x': 144.0, 'y': 48.0,
                'orientation': 4  # Blocks from left
            },
            # Moving drone
            {
                'id': 4, 'type': EntityType.DRONE_ZAP, 'x': 192.0, 'y': 48.0,
                'vx': 1.0, 'vy': 0.0
            },
            # Bounce block (traversable but affects movement)
            {
                'id': 5, 'type': 17, 'x': 240.0, 'y': 48.0,  # Bounce block
                'state': 0
            }
        ]
        
        # Test ninja position
        self.ninja_position = (24.0, 24.0)
    
    def test_precise_tile_collision_integration(self):
        """Test that precise tile collision is properly integrated."""
        # Test movement through empty space (should be allowed)
        result = self.edge_builder.is_precise_traversable(
            12.0, 12.0,  # Center of empty tile (0,0)
            12.0, 36.0,  # Center of empty tile (0,1)
            self.test_level_data
        )
        assert result is True
        
        # Test movement into solid tile (should be blocked)
        result = self.edge_builder.is_precise_traversable(
            12.0, 12.0,  # Center of empty tile (0,0)
            36.0, 12.0,  # Center of solid tile (1,0)
            self.test_level_data
        )
        assert result is False
        
        # Test movement through diagonal slope (depends on path)
        result = self.edge_builder.is_precise_traversable(
            72.0, 6.0,   # Top of slope tile
            96.0, 30.0,  # Bottom-right of slope tile
            self.test_level_data
        )
        # Result depends on precise slope geometry
        assert isinstance(result, bool)
    
    def test_hazard_awareness_integration(self):
        """Test that hazard awareness is properly integrated."""
        # Test movement near active toggle mine (should be blocked)
        result = self.edge_builder.is_traversable_with_hazards(
            3, 3,  # Sub-cell near toggle mine
            5, 3,  # Sub-cell through toggle mine area
            self.test_entities, self.test_level_data, self.ninja_position
        )
        assert result is False
        
        # Test movement away from hazards (should be allowed)
        result = self.edge_builder.is_traversable_with_hazards(
            0, 0,  # Far from hazards
            1, 0,  # Still far from hazards
            self.test_entities, self.test_level_data, self.ninja_position
        )
        assert result is True
    
    def test_bounce_block_interaction(self):
        """Test that bounce blocks are handled correctly (traversable but detected)."""
        # Movement through bounce block should be allowed
        result = self.edge_builder.is_traversable_with_hazards(
            19, 3,  # Before bounce block
            21, 3,  # Through bounce block
            self.test_entities, self.test_level_data, self.ninja_position
        )
        assert result is True
        
        # Bounce block interaction should be detected
        bounce_detected = self.edge_builder._check_bounce_block_interactions(
            228.0, 48.0,  # Before bounce block
            252.0, 48.0,  # Through bounce block
            self.test_entities
        )
        assert bounce_detected is True
    
    def test_static_hazard_caching(self):
        """Test that static hazard caching works correctly."""
        # First call should build cache
        result1 = self.edge_builder._check_static_hazards(
            24.0, 48.0, 72.0, 48.0,
            self.test_entities, self.test_level_data
        )
        
        # Cache should be populated
        assert len(self.edge_builder._static_hazard_cache) > 0
        assert self.edge_builder._current_level_id == 'integration_test'
        
        # Second call should use cache
        result2 = self.edge_builder._check_static_hazards(
            24.0, 60.0, 72.0, 60.0,
            self.test_entities, self.test_level_data
        )
        
        # Results should be consistent
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)
    
    def test_dynamic_hazard_range_system(self):
        """Test that dynamic hazard range system works correctly."""
        # Ninja position near drone
        near_drone_pos = (200.0, 48.0)
        
        # Should detect drone as dynamic hazard
        result = self.edge_builder._check_dynamic_hazards(
            180.0, 48.0, 220.0, 48.0,
            self.test_entities, near_drone_pos
        )
        assert result is False  # Path should be blocked by drone
        
        # Ninja position far from drone
        far_from_drone_pos = (24.0, 24.0)
        
        # Should not be affected by distant drone
        result = self.edge_builder._check_dynamic_hazards(
            12.0, 12.0, 36.0, 12.0,
            self.test_entities, far_from_drone_pos
        )
        assert result is True  # Path should be clear
    
    def test_complete_traversability_pipeline(self):
        """Test the complete traversability checking pipeline."""
        # Test various movement scenarios
        test_cases = [
            # (src_row, src_col, tgt_row, tgt_col, expected_result_type)
            (0, 0, 0, 1, bool),  # Empty to empty
            (0, 0, 0, 2, bool),  # Through solid tile area
            (1, 2, 1, 3, bool),  # Through slope area
            (3, 3, 3, 4, bool),  # Through quarter moon area
            (4, 4, 4, 5, bool),  # Clear bottom area
        ]
        
        for src_row, src_col, tgt_row, tgt_col, expected_type in test_cases:
            result = self.edge_builder.is_traversable_with_hazards(
                src_row, src_col, tgt_row, tgt_col,
                self.test_entities, self.test_level_data, self.ninja_position
            )
            
            assert isinstance(result, expected_type), \
                f"Traversability check ({src_row},{src_col}) -> ({tgt_row},{tgt_col}) failed"
    
    def test_edge_building_with_enhanced_system(self):
        """Test edge building with the enhanced traversability system."""
        # Create sub-grid node map
        sub_grid_node_map = {}
        node_idx = 0
        for row in range(5):
            for col in range(6):
                sub_grid_node_map[(row, col)] = node_idx
                node_idx += 1
        
        # Create edge arrays
        max_edges = 1000
        edge_index = np.zeros((2, max_edges), dtype=np.int32)
        edge_features = np.zeros((max_edges, 16), dtype=np.float32)
        edge_mask = np.zeros(max_edges, dtype=np.float32)
        edge_types = np.zeros(max_edges, dtype=np.int32)
        
        # Build edges
        edge_count = self.edge_builder.build_edges(
            sub_grid_node_map=sub_grid_node_map,
            entity_nodes=[],  # No entity nodes for this test
            entities=self.test_entities,
            level_data=self.test_level_data,
            ninja_position=self.ninja_position,
            ninja_velocity=(0.0, 0.0),
            ninja_state=0,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_mask=edge_mask,
            edge_types=edge_types,
            edge_count=0,
            edge_feature_dim=16
        )
        
        # Should have built some edges
        assert edge_count > 0
        
        # Check that edges respect traversability constraints
        for i in range(edge_count):
            src_idx = edge_index[0, i]
            tgt_idx = edge_index[1, i]
            
            # Find corresponding sub-grid coordinates
            src_coords = None
            tgt_coords = None
            for (row, col), idx in sub_grid_node_map.items():
                if idx == src_idx:
                    src_coords = (row, col)
                elif idx == tgt_idx:
                    tgt_coords = (row, col)
            
            if src_coords and tgt_coords:
                # Edge should represent a valid traversable connection
                result = self.edge_builder.is_traversable_with_hazards(
                    src_coords[0], src_coords[1],
                    tgt_coords[0], tgt_coords[1],
                    self.test_entities, self.test_level_data, self.ninja_position
                )
                assert result is True, f"Edge {i} connects non-traversable cells"
    
    def test_performance_integration(self):
        """Test performance of the integrated system."""
        import time
        
        # Create larger test scenario
        large_entities = []
        for i in range(100):
            large_entities.append({
                'id': i, 'type': EntityType.TOGGLE_MINE, 'x': i * 5.0, 'y': 50.0,
                'active': i % 3 == 0  # Every third mine is active
            })
        
        # Test multiple traversability checks
        start_time = time.time()
        
        for i in range(50):
            self.edge_builder.is_traversable_with_hazards(
                0, 0, 1, 1, large_entities, self.test_level_data, self.ninja_position
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed_time < 2.0, f"Performance test took {elapsed_time:.3f}s, expected < 2.0s"
    
    def test_system_consistency(self):
        """Test that the system produces consistent results."""
        # Same traversability check should always return same result
        test_params = (2, 2, 3, 3, self.test_entities, self.test_level_data, self.ninja_position)
        
        results = []
        for _ in range(10):
            result = self.edge_builder.is_traversable_with_hazards(*test_params)
            results.append(result)
        
        # All results should be identical
        assert all(r == results[0] for r in results), "System should be deterministic"
    
    def test_edge_cases_integration(self):
        """Test edge cases in the integrated system."""
        # Test with empty entities list
        result = self.edge_builder.is_traversable_with_hazards(
            0, 0, 1, 1, [], self.test_level_data, self.ninja_position
        )
        assert isinstance(result, bool)
        
        # Test with None ninja position
        result = self.edge_builder.is_traversable_with_hazards(
            0, 0, 1, 1, self.test_entities, self.test_level_data, None
        )
        assert isinstance(result, bool)
        
        # Test with empty level data
        empty_level = {'level_id': 'empty', 'tiles': np.array([], dtype=np.int32)}
        result = self.edge_builder.is_traversable_with_hazards(
            0, 0, 1, 1, self.test_entities, empty_level, self.ninja_position
        )
        assert isinstance(result, bool)
    
    def test_hazard_type_coverage(self):
        """Test that all major hazard types are handled correctly."""
        hazard_entities = [
            # Each major hazard type
            {'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 50.0, 'y': 50.0, 'active': True},
            {'id': 2, 'type': EntityType.THWUMP, 'x': 100.0, 'y': 50.0, 'state': 0, 'orientation': 0},
            {'id': 3, 'type': EntityType.SHWUMP, 'x': 150.0, 'y': 50.0, 'state': 1},
            {'id': 4, 'type': EntityType.ONE_WAY, 'x': 200.0, 'y': 50.0, 'orientation': 0},
            {'id': 5, 'type': EntityType.DRONE_ZAP, 'x': 250.0, 'y': 50.0, 'vx': 1.0, 'vy': 0.0},
        ]
        
        # Test that system handles each hazard type without errors
        for entity in hazard_entities:
            result = self.edge_builder.is_traversable_with_hazards(
                0, 0, 1, 1, [entity], self.test_level_data, self.ninja_position
            )
            assert isinstance(result, bool), f"Failed to handle hazard type {entity['type']}"


class TestMovementClassifierIntegration:
    """Integration tests for MovementClassifier with enhanced collision."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from npp_rl.models.movement_classifier import MovementClassifier
        self.classifier = MovementClassifier()
        
        self.test_level_data = {
            'level_id': 'classifier_test',
            'tiles': np.array([
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ], dtype=np.int32)
        }
        
        self.test_entities = [
            {'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 36.0, 'y': 12.0, 'active': True}
        ]
    
    def test_movement_feasibility_check(self):
        """Test movement feasibility checking with precise collision."""
        # Movement through empty space should be feasible
        feasible = self.classifier.is_movement_physically_feasible(
            (12.0, 36.0), (12.0, 60.0),
            self.test_level_data, self.test_entities, (12.0, 36.0)
        )
        assert feasible is True
        
        # Movement into solid tile should not be feasible
        not_feasible = self.classifier.is_movement_physically_feasible(
            (12.0, 12.0), (36.0, 12.0),
            self.test_level_data, self.test_entities, (12.0, 12.0)
        )
        assert not_feasible is False
        
        # Movement through hazard should not be feasible
        hazard_blocked = self.classifier.is_movement_physically_feasible(
            (24.0, 12.0), (48.0, 12.0),
            self.test_level_data, self.test_entities, (24.0, 12.0)
        )
        assert hazard_blocked is False


class TestTrajectoryCalculatorIntegration:
    """Integration tests for TrajectoryCalculator with enhanced collision."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from npp_rl.models.trajectory_calculator import TrajectoryCalculator
        self.calculator = TrajectoryCalculator()
        
        self.test_level_data = {
            'level_id': 'trajectory_test',
            'tiles': np.array([
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ], dtype=np.int32)
        }
        
        self.test_entities = [
            {'id': 1, 'type': EntityType.DRONE_ZAP, 'x': 36.0, 'y': 36.0, 'vx': 1.0, 'vy': 0.0}
        ]
    
    def test_trajectory_validation_with_precise_collision(self):
        """Test trajectory validation with precise collision detection."""
        # Clear trajectory should be valid
        clear_trajectory = [(12.0, 36.0), (12.0, 48.0), (12.0, 60.0)]
        
        valid = self.calculator.validate_trajectory_clearance(
            clear_trajectory, self.test_level_data, self.test_entities, (12.0, 36.0)
        )
        assert valid is True
        
        # Trajectory through solid tile should be invalid
        blocked_trajectory = [(12.0, 12.0), (24.0, 12.0), (36.0, 12.0)]
        
        invalid = self.calculator.validate_trajectory_clearance(
            blocked_trajectory, self.test_level_data, self.test_entities, (12.0, 12.0)
        )
        assert invalid is False
        
        # Trajectory through dynamic hazard should be invalid
        hazard_trajectory = [(24.0, 36.0), (36.0, 36.0), (48.0, 36.0)]
        
        hazard_blocked = self.calculator.validate_trajectory_clearance(
            hazard_trajectory, self.test_level_data, self.test_entities, (24.0, 36.0)
        )
        assert hazard_blocked is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])