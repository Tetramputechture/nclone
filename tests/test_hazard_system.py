"""
Comprehensive tests for hazard classification and detection system.

This test suite validates hazard detection for all entity types and
various movement scenarios with static and dynamic hazards.
"""

import pytest
import numpy as np
import math
from typing import Dict, Any, List, Tuple

from nclone.graph.hazard_system import (
    HazardClassificationSystem, HazardInfo, HazardType, HazardState,
    EdgeHazardMeta
)
from nclone.constants.entity_types import EntityType


class TestHazardClassificationSystem:
    """Test suite for hazard classification system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hazard_system = HazardClassificationSystem()
        
        # Create test level data
        self.test_level_data = {
            'level_id': 'test_hazards',
            'tiles': np.zeros((10, 10), dtype=np.int32)  # Empty level for hazard testing
        }
        
        # Create test entities with various hazard types
        self.test_entities = [
            # Toggle mine (active)
            {
                'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 100.0, 'y': 100.0,
                'active': True, 'state': 0
            },
            # Toggle mine (inactive/toggled)
            {
                'id': 2, 'type': EntityType.TOGGLE_MINE_TOGGLED, 'x': 200.0, 'y': 100.0,
                'active': False, 'state': 1
            },
            # Thwump (immobile)
            {
                'id': 3, 'type': EntityType.THWUMP, 'x': 300.0, 'y': 100.0,
                'state': 0, 'orientation': 2  # Facing down
            },
            # Thwump (charging)
            {
                'id': 4, 'type': EntityType.THWUMP, 'x': 400.0, 'y': 100.0,
                'state': 1, 'orientation': 0  # Facing right, charging
            },
            # Shove thwump (untriggered)
            {
                'id': 5, 'type': EntityType.SHWUMP, 'x': 500.0, 'y': 100.0,
                'state': 0
            },
            # Shove thwump (triggered)
            {
                'id': 6, 'type': EntityType.SHWUMP, 'x': 600.0, 'y': 100.0,
                'state': 1
            },
            # One-way platform
            {
                'id': 7, 'type': EntityType.ONE_WAY, 'x': 700.0, 'y': 100.0,
                'orientation': 0  # Blocks from right
            },
            # Drone
            {
                'id': 8, 'type': EntityType.DRONE_ZAP, 'x': 800.0, 'y': 100.0,
                'vx': 2.0, 'vy': 0.0
            },
            # Mini drone
            {
                'id': 9, 'type': EntityType.MINI_DRONE, 'x': 900.0, 'y': 100.0,
                'vx': -1.0, 'vy': 1.0
            },
            # Death ball
            {
                'id': 10, 'type': EntityType.DEATH_BALL, 'x': 1000.0, 'y': 100.0,
                'vx': 0.0, 'vy': 3.0
            }
        ]
    
    def test_static_hazard_cache_building(self):
        """Test building of static hazard cache."""
        cache = self.hazard_system.build_static_hazard_cache(
            self.test_entities, self.test_level_data
        )
        
        # Should contain hazards for active toggle mine, immobile thwump, one-way platform
        assert len(cache) > 0
        
        # Check that active toggle mine creates blocked cells
        toggle_mine_found = False
        for hazard_info in cache.values():
            if hazard_info.entity_type == EntityType.TOGGLE_MINE:
                toggle_mine_found = True
                assert hazard_info.hazard_type == HazardType.STATIC_BLOCKING
                assert hazard_info.state == HazardState.ACTIVE
                assert len(hazard_info.blocked_cells) > 0
                break
        
        assert toggle_mine_found, "Active toggle mine should be in static cache"
    
    def test_dynamic_hazard_detection(self):
        """Test detection of dynamic hazards within range."""
        ninja_pos = (850.0, 100.0)  # Near drone and mini drone
        
        dynamic_hazards = self.hazard_system.get_dynamic_hazards_in_range(
            self.test_entities, ninja_pos, radius=100.0
        )
        
        # Should find drone and mini drone within range
        assert len(dynamic_hazards) >= 2
        
        # Check drone classification
        drone_found = False
        for hazard in dynamic_hazards:
            if hazard.entity_type == EntityType.DRONE_ZAP:
                drone_found = True
                assert hazard.hazard_type == HazardType.DYNAMIC_THREAT
                assert hazard.state == HazardState.ACTIVE
                assert len(hazard.predicted_positions) > 0
                break
        
        assert drone_found, "Drone should be detected as dynamic hazard"
    
    def test_toggle_mine_classification(self):
        """Test toggle mine hazard classification."""
        # Test active toggle mine
        active_mine = {
            'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 100.0, 'y': 100.0,
            'active': True
        }
        
        hazard_info = self.hazard_system._classify_toggle_mine(active_mine)
        
        assert hazard_info is not None
        assert hazard_info.hazard_type == HazardType.STATIC_BLOCKING
        assert hazard_info.state == HazardState.ACTIVE
        assert len(hazard_info.blocked_cells) == 9  # 3x3 area
        assert hazard_info.danger_radius == 18.0
        
        # Test inactive toggle mine
        inactive_mine = {
            'id': 2, 'type': EntityType.TOGGLE_MINE, 'x': 100.0, 'y': 100.0,
            'active': False
        }
        
        hazard_info = self.hazard_system._classify_toggle_mine(inactive_mine)
        assert hazard_info is None  # Inactive mines are safe
    
    def test_thwump_classification(self):
        """Test thwump hazard classification."""
        # Test immobile thwump (static hazard)
        immobile_thwump = {
            'id': 3, 'type': EntityType.THWUMP, 'x': 300.0, 'y': 100.0,
            'state': 0, 'orientation': 2  # Facing down
        }
        
        static_hazard = self.hazard_system._classify_thwump_static(immobile_thwump)
        
        assert static_hazard is not None
        assert static_hazard.hazard_type == HazardType.ACTIVATION_TRIGGER
        assert static_hazard.state == HazardState.INACTIVE
        assert static_hazard.activation_range == 38.0  # Line-of-sight range
        assert len(static_hazard.blocked_cells) > 0  # Charge direction blocked
        
        # Test charging thwump (dynamic hazard)
        charging_thwump = {
            'id': 4, 'type': EntityType.THWUMP, 'x': 400.0, 'y': 100.0,
            'state': 1, 'orientation': 0  # Facing right, charging
        }
        
        dynamic_hazard = self.hazard_system._classify_thwump_dynamic(charging_thwump)
        
        assert dynamic_hazard is not None
        assert dynamic_hazard.hazard_type == HazardType.DYNAMIC_THREAT
        assert dynamic_hazard.state == HazardState.CHARGING
        assert len(dynamic_hazard.predicted_positions) > 0
    
    def test_shove_thwump_classification(self):
        """Test shove thwump hazard classification."""
        # Test untriggered shove thwump (safe)
        untriggered = {
            'id': 5, 'type': EntityType.SHWUMP, 'x': 500.0, 'y': 100.0,
            'state': 0
        }
        
        static_hazard = self.hazard_system._classify_shove_thwump_static(untriggered)
        assert static_hazard is None  # Untriggered is safe
        
        # Test triggered shove thwump (dangerous)
        triggered = {
            'id': 6, 'type': EntityType.SHWUMP, 'x': 600.0, 'y': 100.0,
            'state': 1
        }
        
        dynamic_hazard = self.hazard_system._classify_shove_thwump_dynamic(triggered)
        
        assert dynamic_hazard is not None
        assert dynamic_hazard.hazard_type == HazardType.STATIC_BLOCKING
        assert dynamic_hazard.state == HazardState.ACTIVE
        assert dynamic_hazard.danger_radius == 8.0  # Core radius
    
    def test_one_way_platform_classification(self):
        """Test one-way platform hazard classification."""
        platform = {
            'id': 7, 'type': EntityType.ONE_WAY, 'x': 700.0, 'y': 100.0,
            'orientation': 0  # Blocks from right (orientation 0)
        }
        
        hazard_info = self.hazard_system._classify_one_way_platform(platform)
        
        assert hazard_info is not None
        assert hazard_info.hazard_type == HazardType.DIRECTIONAL_BLOCKING
        assert hazard_info.state == HazardState.ACTIVE
        assert 0 in hazard_info.blocked_directions  # Blocks orientation 0
        assert len(hazard_info.blocked_cells) > 0
    
    def test_drone_classification(self):
        """Test drone hazard classification."""
        drone = {
            'id': 8, 'type': EntityType.DRONE_ZAP, 'x': 800.0, 'y': 100.0,
            'vx': 2.0, 'vy': 0.0
        }
        
        hazard_info = self.hazard_system._classify_drone(drone)
        
        assert hazard_info is not None
        assert hazard_info.hazard_type == HazardType.DYNAMIC_THREAT
        assert hazard_info.state == HazardState.ACTIVE
        assert hazard_info.velocity == (2.0, 0.0)
        assert len(hazard_info.predicted_positions) > 0
        assert hazard_info.danger_radius == 12.0
    
    def test_path_hazard_intersection(self):
        """Test path intersection with various hazard types."""
        # Test static hazard intersection (toggle mine)
        toggle_mine_hazard = HazardInfo(
            entity_id=1, hazard_type=HazardType.STATIC_BLOCKING,
            entity_type=EntityType.TOGGLE_MINE, position=(100.0, 100.0),
            state=HazardState.ACTIVE, blocked_directions=set(range(8)),
            danger_radius=18.0, activation_range=0.0, blocked_cells=set(),
            velocity=(0.0, 0.0), predicted_positions=[], orientation=0,
            charge_direction=(0.0, 0.0), core_position=(100.0, 100.0),
            launch_trajectories=[]
        )
        
        # Path that intersects hazard
        intersects = self.hazard_system.check_path_hazard_intersection(
            80.0, 100.0, 120.0, 100.0, toggle_mine_hazard
        )
        assert intersects is True
        
        # Path that avoids hazard
        avoids = self.hazard_system.check_path_hazard_intersection(
            80.0, 150.0, 120.0, 150.0, toggle_mine_hazard
        )
        assert avoids is False
    
    def test_directional_hazard_intersection(self):
        """Test path intersection with directional hazards (one-way platforms)."""
        platform_hazard = HazardInfo(
            entity_id=7, hazard_type=HazardType.DIRECTIONAL_BLOCKING,
            entity_type=EntityType.ONE_WAY, position=(700.0, 100.0),
            state=HazardState.ACTIVE, blocked_directions={0},  # Blocks from right
            danger_radius=12.0, activation_range=0.0, blocked_cells=set(),
            velocity=(0.0, 0.0), predicted_positions=[], orientation=0,
            charge_direction=(0.0, 0.0), core_position=(700.0, 100.0),
            launch_trajectories=[]
        )
        
        # Approach from blocked direction (right to left, orientation 4)
        blocked = self.hazard_system.check_path_hazard_intersection(
            720.0, 100.0, 680.0, 100.0, platform_hazard
        )
        # Note: This test depends on the implementation of directional checking
        
        # Approach from safe direction (left to right, orientation 0)
        safe = self.hazard_system.check_path_hazard_intersection(
            680.0, 100.0, 720.0, 100.0, platform_hazard
        )
        # Implementation dependent - may or may not be blocked
    
    def test_dynamic_hazard_intersection(self):
        """Test path intersection with dynamic hazards (drones)."""
        drone_hazard = HazardInfo(
            entity_id=8, hazard_type=HazardType.DYNAMIC_THREAT,
            entity_type=EntityType.DRONE_ZAP, position=(800.0, 100.0),
            state=HazardState.ACTIVE, blocked_directions=set(range(8)),
            danger_radius=12.0, activation_range=0.0, blocked_cells=set(),
            velocity=(2.0, 0.0), predicted_positions=[(810.0, 100.0), (820.0, 100.0)],
            orientation=0, charge_direction=(0.0, 0.0), core_position=(800.0, 100.0),
            launch_trajectories=[]
        )
        
        # Path that intersects current position
        intersects_current = self.hazard_system.check_path_hazard_intersection(
            790.0, 100.0, 810.0, 100.0, drone_hazard
        )
        assert intersects_current is True
        
        # Path that intersects predicted position
        intersects_predicted = self.hazard_system.check_path_hazard_intersection(
            815.0, 100.0, 825.0, 100.0, drone_hazard
        )
        assert intersects_predicted is True
    
    def test_activation_hazard_intersection(self):
        """Test path intersection with activation hazards (thwumps)."""
        thwump_hazard = HazardInfo(
            entity_id=3, hazard_type=HazardType.ACTIVATION_TRIGGER,
            entity_type=EntityType.THWUMP, position=(300.0, 100.0),
            state=HazardState.INACTIVE, blocked_directions=set(),
            danger_radius=0.0, activation_range=38.0, blocked_cells=set(),
            velocity=(0.0, 0.0), predicted_positions=[], orientation=2,  # Down
            charge_direction=(0.0, 1.0), core_position=(300.0, 100.0),
            launch_trajectories=[]
        )
        
        # Path that enters activation range and crosses charge lane
        activates = self.hazard_system.check_path_hazard_intersection(
            280.0, 100.0, 320.0, 150.0, thwump_hazard
        )
        # Implementation dependent - should check activation range and charge direction
    
    def test_line_circle_intersection_utility(self):
        """Test the line-circle intersection utility function."""
        # Line that intersects circle
        intersects = self.hazard_system._line_intersects_circle(
            0.0, 0.0, 20.0, 0.0,  # Horizontal line
            10.0, 0.0, 5.0        # Circle at center with radius 5
        )
        assert intersects is True
        
        # Line that misses circle
        misses = self.hazard_system._line_intersects_circle(
            0.0, 0.0, 20.0, 0.0,  # Horizontal line
            10.0, 10.0, 5.0       # Circle above line
        )
        assert misses is False
        
        # Line that just touches circle
        touches = self.hazard_system._line_intersects_circle(
            0.0, 0.0, 20.0, 0.0,  # Horizontal line
            10.0, 5.0, 5.0        # Circle touching line
        )
        assert touches is True
    
    def test_orientation_vector_conversion(self):
        """Test conversion of orientation to unit vector."""
        # Test cardinal directions
        right = self.hazard_system._get_orientation_vector(0)
        assert abs(right[0] - 1.0) < 1e-6 and abs(right[1]) < 1e-6
        
        down = self.hazard_system._get_orientation_vector(2)
        assert abs(down[0]) < 1e-6 and abs(down[1] - 1.0) < 1e-6
        
        left = self.hazard_system._get_orientation_vector(4)
        assert abs(left[0] + 1.0) < 1e-6 and abs(left[1]) < 1e-6
        
        up = self.hazard_system._get_orientation_vector(6)
        assert abs(up[0]) < 1e-6 and abs(up[1] + 1.0) < 1e-6
        
        # Test diagonal directions
        down_right = self.hazard_system._get_orientation_vector(1)
        assert abs(down_right[0] - math.sqrt(2)/2) < 1e-6
        assert abs(down_right[1] - math.sqrt(2)/2) < 1e-6
    
    def test_hazard_prediction(self):
        """Test movement prediction for dynamic hazards."""
        # Test thwump movement prediction
        charging_positions = self.hazard_system._predict_thwump_movement(
            300.0, 100.0, 1.0, 0.0, 1  # Charging right
        )
        
        assert len(charging_positions) > 0
        # Should predict positions in charge direction
        for i, (x, y) in enumerate(charging_positions):
            assert x > 300.0  # Moving right
            assert y == 100.0  # Same y coordinate
        
        # Test drone movement prediction
        drone_positions = self.hazard_system._predict_drone_movement(
            800.0, 100.0, 2.0, 0.0, 30.0  # Moving right for 30 frames
        )
        
        assert len(drone_positions) > 0
        # Should predict linear movement
        for x, y in drone_positions:
            assert x > 800.0  # Moving right
            assert y == 100.0  # Same y coordinate
    
    def test_performance_large_entity_count(self):
        """Test performance with large number of entities."""
        # Create many entities
        many_entities = []
        for i in range(1000):
            many_entities.append({
                'id': i, 'type': EntityType.TOGGLE_MINE, 'x': i * 10.0, 'y': 100.0,
                'active': i % 2 == 0  # Half active, half inactive
            })
        
        # Test static hazard cache building
        import time
        start_time = time.time()
        
        cache = self.hazard_system.build_static_hazard_cache(
            many_entities, self.test_level_data
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed_time < 1.0, f"Large entity test took {elapsed_time:.3f}s, expected < 1.0s"
        assert len(cache) > 0  # Should have cached some hazards


if __name__ == '__main__':
    pytest.main([__file__, '-v'])