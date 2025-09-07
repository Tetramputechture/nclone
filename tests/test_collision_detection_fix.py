#!/usr/bin/env python3
"""
Unit tests for the collision detection fix that prevents walkable edges in solid tiles.
"""

import unittest
import numpy as np
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants import TILE_PIXEL_SIZE
from nclone.constants.physics_constants import NINJA_RADIUS


class TestCollisionDetectionFix(unittest.TestCase):
    """Test cases for the collision detection fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collision_detector = PreciseTileCollision()
        
        # Create a simple test level with solid borders
        self.tiles = np.zeros((5, 5), dtype=int)
        self.tiles[0, :] = 1  # Top border
        self.tiles[-1, :] = 1  # Bottom border
        self.tiles[:, 0] = 1  # Left border
        self.tiles[:, -1] = 1  # Right border
    
    def test_position_in_center_of_empty_tile_is_traversable(self):
        """Test that positions in the center of empty tiles are traversable."""
        # Center of tile (2, 2) - should be far from any solid tiles
        center_x = 2.5 * TILE_PIXEL_SIZE
        center_y = 2.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            center_x, center_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertTrue(result, "Center of empty tile should be traversable")
    
    def test_position_too_close_to_solid_tile_is_not_traversable(self):
        """Test that positions too close to solid tiles are not traversable."""
        # Position just inside the left border - too close to solid tile at x=0
        too_close_x = TILE_PIXEL_SIZE + NINJA_RADIUS - 1  # 33 pixels (9 pixels from border)
        center_y = 2.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            too_close_x, center_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertFalse(result, "Position too close to solid tile should not be traversable")
    
    def test_position_exactly_at_ninja_radius_from_solid_tile_is_traversable(self):
        """Test that positions exactly at ninja radius from solid tiles are traversable."""
        # Position exactly ninja_radius away from left border
        exact_distance_x = TILE_PIXEL_SIZE + NINJA_RADIUS  # 34 pixels (10 pixels from border)
        center_y = 2.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            exact_distance_x, center_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertTrue(result, "Position exactly at ninja radius should be traversable")
    
    def test_position_in_solid_tile_is_not_traversable(self):
        """Test that positions inside solid tiles are not traversable."""
        # Position in the center of a solid tile (border tile)
        solid_center_x = 0.5 * TILE_PIXEL_SIZE  # Center of tile (0, 0)
        solid_center_y = 0.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            solid_center_x, solid_center_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertFalse(result, "Position inside solid tile should not be traversable")
    
    def test_path_between_valid_positions_is_traversable(self):
        """Test that paths between valid positions are traversable."""
        # Two positions in the center of empty tiles
        src_x, src_y = 2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE
        tgt_x, tgt_y = 2.5 * TILE_PIXEL_SIZE, 1.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertTrue(result, "Path between valid positions should be traversable")
    
    def test_path_from_invalid_position_is_not_traversable(self):
        """Test that paths from invalid positions are not traversable."""
        # Source position too close to solid tile
        src_x = TILE_PIXEL_SIZE + NINJA_RADIUS - 1  # Too close to left border
        src_y = 2.5 * TILE_PIXEL_SIZE
        
        # Target position is valid
        tgt_x, tgt_y = 2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertFalse(result, "Path from invalid position should not be traversable")
    
    def test_path_to_invalid_position_is_not_traversable(self):
        """Test that paths to invalid positions are not traversable."""
        # Source position is valid
        src_x, src_y = 2.5 * TILE_PIXEL_SIZE, 2.5 * TILE_PIXEL_SIZE
        
        # Target position too close to solid tile
        tgt_x = TILE_PIXEL_SIZE + NINJA_RADIUS - 1  # Too close to left border
        tgt_y = 2.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, self.tiles, NINJA_RADIUS
        )
        
        self.assertFalse(result, "Path to invalid position should not be traversable")
    
    def test_ninja_radius_clearance_requirement(self):
        """Test that the ninja radius clearance requirement is enforced correctly."""
        # Test positions at various distances from solid tile
        test_distances = [
            (NINJA_RADIUS - 2, False),  # Too close
            (NINJA_RADIUS - 1, False),  # Still too close
            (NINJA_RADIUS, True),       # Exactly at limit
            (NINJA_RADIUS + 1, True),   # Safe distance
            (NINJA_RADIUS + 5, True),   # Well clear
        ]
        
        for distance, expected_traversable in test_distances:
            with self.subTest(distance=distance):
                test_x = TILE_PIXEL_SIZE + distance
                test_y = 2.5 * TILE_PIXEL_SIZE
                
                result = self.collision_detector._is_position_traversable(
                    test_x, test_y, self.tiles, NINJA_RADIUS
                )
                
                self.assertEqual(
                    result, expected_traversable,
                    f"Position at distance {distance} from solid tile should be "
                    f"{'traversable' if expected_traversable else 'not traversable'}"
                )


class TestCollisionDetectionEdgeCases(unittest.TestCase):
    """Test edge cases for collision detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collision_detector = PreciseTileCollision()
    
    def test_empty_level_all_positions_traversable(self):
        """Test that all positions in an empty level are traversable."""
        # Create completely empty level
        empty_tiles = np.zeros((3, 3), dtype=int)
        
        # Test center position
        center_x = 1.5 * TILE_PIXEL_SIZE
        center_y = 1.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            center_x, center_y, empty_tiles, NINJA_RADIUS
        )
        
        self.assertTrue(result, "All positions in empty level should be traversable")
    
    def test_completely_solid_level_no_positions_traversable(self):
        """Test that no positions in a completely solid level are traversable."""
        # Create completely solid level
        solid_tiles = np.ones((3, 3), dtype=int)
        
        # Test center position
        center_x = 1.5 * TILE_PIXEL_SIZE
        center_y = 1.5 * TILE_PIXEL_SIZE
        
        result = self.collision_detector._is_position_traversable(
            center_x, center_y, solid_tiles, NINJA_RADIUS
        )
        
        self.assertFalse(result, "No positions in completely solid level should be traversable")
    
    def test_out_of_bounds_positions_handled_correctly(self):
        """Test that out-of-bounds positions are handled correctly."""
        tiles = np.zeros((3, 3), dtype=int)
        
        # Test position outside the level bounds
        out_of_bounds_x = -10.0
        out_of_bounds_y = -10.0
        
        # Should not crash and should return a reasonable result
        result = self.collision_detector._is_position_traversable(
            out_of_bounds_x, out_of_bounds_y, tiles, NINJA_RADIUS
        )
        
        # The exact result doesn't matter as much as not crashing
        self.assertIsInstance(result, bool, "Out of bounds check should return boolean")


if __name__ == '__main__':
    unittest.main()