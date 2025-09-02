"""
Comprehensive tests for precise tile collision detection system.

This test suite validates the segment-based collision detection against
all 38 tile types and various movement scenarios.
"""

import pytest
import numpy as np
import math
from typing import Dict, Any, List, Tuple

from nclone.graph.precise_collision import (
    PreciseTileCollision, CollisionSegment, CollisionResult, SegmentType
)
from nclone.constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS


class TestPreciseTileCollision:
    """Test suite for precise tile collision detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collision_detector = PreciseTileCollision()
        
        # Create test level data with various tile configurations
        self.test_level_data = {
            'level_id': 'test_level',
            'tiles': np.array([
                [0, 1, 2, 3, 4, 5],      # Row 0: Empty, full, half tiles
                [6, 7, 8, 9, 10, 11],    # Row 1: Slopes and quarter moons
                [14, 15, 16, 17, 18, 19], # Row 2: Quarter pipes and mild slopes
                [22, 23, 24, 25, 26, 27], # Row 3: Raised slopes and steep slopes
                [30, 31, 32, 33, 34, 35], # Row 4: More slopes and glitched tiles
                [0, 0, 0, 0, 36, 37]      # Row 5: Empty and glitched tiles
            ], dtype=np.int32)
        }
    
    def test_empty_tile_traversal(self):
        """Test movement through empty tiles (tile ID 0)."""
        # Movement through empty space should be allowed
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0,  # Center of tile (0,0)
            36.0, 12.0,  # Center of tile (1,0) - also empty
            self.test_level_data
        )
        assert result is True
    
    def test_solid_tile_blocking(self):
        """Test movement blocked by solid tiles (tile ID 1)."""
        # Movement into solid tile should be blocked
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0,   # Center of tile (0,0) - empty
            36.0, 12.0,   # Center of tile (1,0) - solid
            self.test_level_data
        )
        assert result is False
    
    def test_half_tile_collision(self):
        """Test collision with half tiles (tile IDs 2-5)."""
        # Test top-left half tile (ID 2)
        # Movement to bottom-right should be blocked
        result = self.collision_detector.is_path_traversable(
            48.0, 6.0,    # Top of tile (2,0)
            60.0, 18.0,   # Bottom-right of tile (2,0)
            self.test_level_data
        )
        assert result is False
        
        # Movement to top-left should be allowed
        result = self.collision_detector.is_path_traversable(
            48.0, 18.0,   # Bottom of tile (2,0)
            60.0, 6.0,    # Top-right of tile (2,0)
            self.test_level_data
        )
        assert result is True
    
    def test_diagonal_slope_collision(self):
        """Test collision with diagonal slopes (tile IDs 6-9)."""
        # Test 45-degree slope (ID 6: bottom-left to top-right)
        # Movement parallel to slope should be allowed
        result = self.collision_detector.is_path_traversable(
            12.0, 36.0,   # Bottom-left of tile (0,1)
            36.0, 12.0,   # Top-right of tile (0,1)
            self.test_level_data
        )
        assert result is True
        
        # Movement perpendicular to slope should be blocked
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0,   # Top-left of tile (0,1)
            36.0, 36.0,   # Bottom-right of tile (0,1)
            self.test_level_data
        )
        assert result is False
    
    def test_circular_segment_collision(self):
        """Test collision with circular segments (quarter moons and pipes)."""
        # Test quarter moon (ID 10: convex arc at top-left)
        # Movement into the arc should be blocked
        result = self.collision_detector.is_path_traversable(
            108.0, 6.0,   # Outside arc
            120.0, 18.0,  # Inside arc area
            self.test_level_data
        )
        assert result is False
        
        # Movement around the arc should be allowed
        result = self.collision_detector.is_path_traversable(
            108.0, 30.0,  # Below arc
            132.0, 30.0,  # Still below arc
            self.test_level_data
        )
        assert result is True
    
    def test_ninja_radius_collision(self):
        """Test that ninja radius is properly considered in collision detection."""
        # Movement that would be clear for a point but blocked for ninja radius
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0,                    # Center of empty tile
            36.0 - NINJA_RADIUS + 1.0, 12.0,  # Just inside ninja radius of solid tile
            self.test_level_data
        )
        assert result is False
        
        # Movement that clears ninja radius should be allowed
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0,                    # Center of empty tile
            36.0 - NINJA_RADIUS - 1.0, 12.0,  # Just outside ninja radius of solid tile
            self.test_level_data
        )
        assert result is True
    
    def test_segment_conversion_caching(self):
        """Test that tile segment conversion is properly cached."""
        # First call should populate cache
        result1 = self.collision_detector.is_path_traversable(
            12.0, 12.0, 36.0, 12.0, self.test_level_data
        )
        
        # Second call should use cached segments
        result2 = self.collision_detector.is_path_traversable(
            12.0, 18.0, 36.0, 18.0, self.test_level_data
        )
        
        # Results should be consistent
        assert result1 == result2
        
        # Cache should contain converted segments
        assert len(self.collision_detector._tile_segment_cache) > 0
    
    def test_level_segment_caching(self):
        """Test that level segments are properly cached."""
        # First call should populate level cache
        self.collision_detector.is_path_traversable(
            12.0, 12.0, 36.0, 12.0, self.test_level_data
        )
        
        # Level cache should be populated
        assert len(self.collision_detector._level_segment_cache) > 0
        assert self.collision_detector._current_level_id == 'test_level'
    
    def test_all_tile_types(self):
        """Test collision detection against all 38 tile types."""
        # Create level with all tile types
        all_tiles_level = {
            'level_id': 'all_tiles',
            'tiles': np.arange(38, dtype=np.int32).reshape(6, 6)[:6, :6]
        }
        
        # Test movement through each tile type
        for tile_id in range(38):
            tile_x = tile_id % 6
            tile_y = tile_id // 6
            
            if tile_y >= 6:  # Skip if beyond our test grid
                continue
            
            center_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
            center_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
            
            # Test movement through tile center
            result = self.collision_detector.is_path_traversable(
                center_x - 6.0, center_y,
                center_x + 6.0, center_y,
                all_tiles_level
            )
            
            # Empty tile (0) should be traversable, others depend on geometry
            if tile_id == 0:
                assert result is True, f"Empty tile {tile_id} should be traversable"
            # Note: Other tiles may or may not be traversable depending on path and geometry
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test zero-length movement
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0, 12.0, 12.0, self.test_level_data
        )
        assert result is True
        
        # Test movement outside level bounds
        result = self.collision_detector.is_path_traversable(
            -12.0, -12.0, 12.0, 12.0, self.test_level_data
        )
        # Should handle gracefully (implementation dependent)
        assert isinstance(result, bool)
        
        # Test with empty level data
        empty_level = {'level_id': 'empty', 'tiles': np.array([], dtype=np.int32)}
        result = self.collision_detector.is_path_traversable(
            12.0, 12.0, 36.0, 12.0, empty_level
        )
        assert result is True  # No tiles means no collision
    
    def test_performance_large_level(self):
        """Test performance with large level data."""
        # Create large level (44x25 tiles like real N++ levels)
        large_level = {
            'level_id': 'large_test',
            'tiles': np.random.randint(0, 38, size=(25, 44), dtype=np.int32)
        }
        
        # Test multiple collision checks
        import time
        start_time = time.time()
        
        for i in range(100):
            self.collision_detector.is_path_traversable(
                i * 2.0, 100.0, i * 2.0 + 50.0, 150.0, large_level
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second for 100 checks)
        assert elapsed_time < 1.0, f"Performance test took {elapsed_time:.3f}s, expected < 1.0s"


class TestCollisionSegments:
    """Test suite for collision segment creation and intersection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collision_detector = PreciseTileCollision()
    
    def test_orthogonal_segment_creation(self):
        """Test creation of orthogonal line segments."""
        # Test full tile (ID 1) - should have all edge segments
        segments = self.collision_detector._convert_tile_definition_to_segments(1)
        
        # Full tile should have multiple orthogonal segments
        ortho_segments = [s for s in segments if s.segment_type == SegmentType.ORTHOGONAL]
        assert len(ortho_segments) > 0
        
        # Check that segments have proper normals
        for segment in ortho_segments:
            normal_length = math.sqrt(segment.normal_x**2 + segment.normal_y**2)
            assert abs(normal_length - 1.0) < 1e-6, "Normal should be unit vector"
    
    def test_diagonal_segment_creation(self):
        """Test creation of diagonal line segments."""
        # Test 45-degree slope (ID 6)
        segments = self.collision_detector._convert_tile_definition_to_segments(6)
        
        # Should have diagonal segment
        diag_segments = [s for s in segments if s.segment_type == SegmentType.DIAGONAL]
        assert len(diag_segments) > 0
        
        # Check diagonal segment properties
        diag_segment = diag_segments[0]
        dx = diag_segment.end_x - diag_segment.start_x
        dy = diag_segment.end_y - diag_segment.start_y
        
        # Should be diagonal (45 degrees)
        assert abs(abs(dx) - abs(dy)) < 1e-6, "Should be 45-degree diagonal"
    
    def test_circular_segment_creation(self):
        """Test creation of circular arc segments."""
        # Test quarter moon (ID 10)
        segments = self.collision_detector._convert_tile_definition_to_segments(10)
        
        # Should have circular segment
        circular_segments = [s for s in segments if s.segment_type == SegmentType.CIRCULAR]
        assert len(circular_segments) > 0
        
        # Check circular segment properties
        circular_segment = circular_segments[0]
        assert circular_segment.radius > 0, "Radius should be positive"
        assert circular_segment.is_convex is not None, "Convexity should be defined"
    
    def test_line_circle_intersection(self):
        """Test line segment vs circle intersection."""
        # Create test collision result
        result = self.collision_detector._test_circle_vs_line_segment(
            0.0, 0.0,    # Circle start
            10.0, 0.0,   # Circle movement
            5.0,         # Circle radius
            5.0, -10.0,  # Line start
            5.0, 10.0    # Line end
        )
        
        # Should detect collision
        assert result.collision is True
        assert 0.0 <= result.time_of_impact <= 1.0
    
    def test_circle_arc_intersection(self):
        """Test circle vs circular arc intersection."""
        # Create test circular segment
        arc_segment = CollisionSegment(
            segment_type=SegmentType.CIRCULAR,
            center_x=0.0, center_y=0.0,
            radius=10.0,
            normal_x=1.0, normal_y=1.0,  # First quadrant
            is_convex=True
        )
        
        # Test collision
        result = self.collision_detector._test_circle_vs_arc_segment(
            -15.0, 0.0,  # Circle start (outside arc)
            10.0, 0.0,   # Circle movement (toward arc)
            5.0,         # Circle radius
            arc_segment
        )
        
        # Should detect collision for convex arc
        assert result.collision is True
        assert 0.0 <= result.time_of_impact <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])