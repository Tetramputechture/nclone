"""Unit tests for path analysis utilities.

Tests path monotonicity detection, direction calculation, and other
path analysis functions used for action masking.
"""

import unittest
from .path_analysis import (
    is_path_monotonic_horizontal,
    is_path_monotonic_vertical,
    calculate_horizontal_offset_from_path,
    get_immediate_path_direction,
    calculate_path_length,
    find_closest_waypoint_index,
)


class TestPathMonotonicity(unittest.TestCase):
    """Test path monotonicity detection."""

    def test_empty_path(self):
        """Empty path should return False, None."""
        is_mono, direction = is_path_monotonic_horizontal([])
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_single_point_path(self):
        """Single point path should return False, None."""
        is_mono, direction = is_path_monotonic_horizontal([(0, 0)])
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_monotonic_right(self):
        """Path moving consistently right should be monotonic."""
        path = [(0, 0), (10, 5), (20, 10), (30, 5)]
        is_mono, direction = is_path_monotonic_horizontal(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Right

    def test_monotonic_left(self):
        """Path moving consistently left should be monotonic."""
        path = [(30, 0), (20, 5), (10, 10), (0, 5)]
        is_mono, direction = is_path_monotonic_horizontal(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, -1)  # Left

    def test_non_monotonic(self):
        """Path with direction reversal should not be monotonic."""
        path = [(0, 0), (10, 5), (5, 10), (15, 5)]  # Goes right, then left, then right
        is_mono, direction = is_path_monotonic_horizontal(path)
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_vertical_path(self):
        """Pure vertical path should return False, None."""
        path = [(10, 0), (10, 10), (10, 20)]
        is_mono, direction = is_path_monotonic_horizontal(path)
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_monotonic_with_vertical_segments(self):
        """Path with vertical segments but monotonic horizontal movement."""
        path = [(0, 0), (10, 10), (10, 20), (20, 20)]
        is_mono, direction = is_path_monotonic_horizontal(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Right


class TestImmediatePathDirection(unittest.TestCase):
    """Test immediate path direction calculation."""

    def test_empty_path(self):
        """Empty path should return None."""
        direction = get_immediate_path_direction((0, 0), [])
        self.assertIsNone(direction)

    def test_direction_to_first_waypoint(self):
        """Should return direction to first waypoint."""
        ninja_pos = (0.0, 0.0)
        path = [(10, 10)]
        direction = get_immediate_path_direction(ninja_pos, path)
        self.assertEqual(direction, (1, 1))  # Right and up

    def test_direction_already_at_waypoint(self):
        """When at waypoint, should look ahead to next."""
        ninja_pos = (10.0, 10.0)
        path = [(10, 10), (20, 20)]
        direction = get_immediate_path_direction(ninja_pos, path)
        self.assertEqual(direction, (1, 1))  # Direction to next waypoint

    def test_direction_at_final_waypoint(self):
        """At final waypoint, should return (0, 0)."""
        ninja_pos = (10.0, 10.0)
        path = [(10, 10)]
        direction = get_immediate_path_direction(ninja_pos, path)
        self.assertEqual(direction, (0, 0))

    def test_lookahead(self):
        """Lookahead should skip intermediate waypoints."""
        ninja_pos = (0.0, 0.0)
        path = [(5, 5), (10, 10), (20, 20)]
        direction = get_immediate_path_direction(ninja_pos, path, lookahead=2)
        # Should look at path[2] = (20, 20)
        self.assertEqual(direction, (1, 1))


class TestPathLength(unittest.TestCase):
    """Test path length calculation."""

    def test_empty_path(self):
        """Empty path has zero length."""
        length = calculate_path_length([])
        self.assertEqual(length, 0.0)

    def test_single_point(self):
        """Single point path has zero length."""
        length = calculate_path_length([(0, 0)])
        self.assertEqual(length, 0.0)

    def test_straight_horizontal(self):
        """Straight horizontal path."""
        path = [(0, 0), (10, 0), (20, 0)]
        length = calculate_path_length(path)
        self.assertAlmostEqual(length, 20.0, places=5)

    def test_straight_vertical(self):
        """Straight vertical path."""
        path = [(0, 0), (0, 10), (0, 20)]
        length = calculate_path_length(path)
        self.assertAlmostEqual(length, 20.0, places=5)

    def test_diagonal(self):
        """Diagonal path."""
        path = [(0, 0), (10, 10)]
        length = calculate_path_length(path)
        expected = 14.142135  # sqrt(100 + 100)
        self.assertAlmostEqual(length, expected, places=5)


class TestClosestWaypoint(unittest.TestCase):
    """Test closest waypoint finding."""

    def test_empty_path(self):
        """Empty path returns -1."""
        idx = find_closest_waypoint_index((0, 0), [])
        self.assertEqual(idx, -1)

    def test_single_waypoint(self):
        """Single waypoint returns index 0."""
        idx = find_closest_waypoint_index((5, 5), [(10, 10)])
        self.assertEqual(idx, 0)

    def test_closest_of_multiple(self):
        """Should find closest waypoint."""
        position = (12.0, 12.0)
        path = [(0, 0), (10, 10), (20, 20), (30, 30)]
        idx = find_closest_waypoint_index(position, path)
        self.assertEqual(idx, 1)  # (10, 10) is closest

    def test_at_waypoint(self):
        """When at waypoint, should return that index."""
        position = (20.0, 20.0)
        path = [(0, 0), (10, 10), (20, 20), (30, 30)]
        idx = find_closest_waypoint_index(position, path)
        self.assertEqual(idx, 2)


class TestVerticalMonotonicity(unittest.TestCase):
    """Test vertical path monotonicity detection."""

    def test_empty_path(self):
        """Empty path should return False, None, 0.0."""
        is_mono, direction, max_dev = is_path_monotonic_vertical([])
        self.assertFalse(is_mono)
        self.assertIsNone(direction)
        self.assertEqual(max_dev, 0.0)

    def test_single_point_path(self):
        """Single point path should return False, None, 0.0."""
        is_mono, direction, max_dev = is_path_monotonic_vertical([(0, 0)])
        self.assertFalse(is_mono)
        self.assertIsNone(direction)
        self.assertEqual(max_dev, 0.0)

    def test_monotonic_down(self):
        """Path moving consistently down should be monotonic."""
        path = [(100, 0), (100, 10), (100, 20), (100, 30)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Down
        self.assertAlmostEqual(max_dev, 0.0)

    def test_monotonic_up(self):
        """Path moving consistently up should be monotonic."""
        path = [(100, 30), (100, 20), (100, 10), (100, 0)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, -1)  # Up
        self.assertAlmostEqual(max_dev, 0.0)

    def test_non_monotonic_vertical(self):
        """Path with vertical direction reversal should not be monotonic."""
        path = [(100, 0), (100, 10), (100, 5), (100, 15)]  # Down, up, down
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_horizontal_path(self):
        """Pure horizontal path should return False, None."""
        path = [(0, 100), (10, 100), (20, 100)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertFalse(is_mono)
        self.assertIsNone(direction)

    def test_vertical_with_horizontal_deviation(self):
        """Vertical path with horizontal deviation should track max deviation."""
        path = [(100, 0), (110, 10), (105, 20), (100, 30)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Down
        self.assertAlmostEqual(max_dev, 10.0)  # Max deviation is 10 pixels

    def test_nearly_vertical_path(self):
        """Nearly vertical path (small horizontal wobble) should still be monotonic."""
        path = [(100, 0), (102, 50), (98, 100), (100, 150)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Down
        self.assertLessEqual(max_dev, 5.0)  # Small deviation


class TestHorizontalOffsetCalculation(unittest.TestCase):
    """Test horizontal offset calculation from vertical path."""

    def test_empty_path(self):
        """Empty path should return 0.0."""
        offset = calculate_horizontal_offset_from_path((100.0, 100.0), [])
        self.assertEqual(offset, 0.0)

    def test_aligned_with_path(self):
        """Ninja aligned with vertical path should have zero offset."""
        ninja_pos = (100.0, 50.0)
        path = [(100, 0), (100, 100), (100, 200)]
        offset = calculate_horizontal_offset_from_path(ninja_pos, path)
        self.assertAlmostEqual(offset, 0.0)

    def test_offset_right_of_path(self):
        """Ninja to right of path should have positive offset."""
        ninja_pos = (150.0, 50.0)
        path = [(100, 0), (100, 100), (100, 200)]
        offset = calculate_horizontal_offset_from_path(ninja_pos, path)
        self.assertAlmostEqual(offset, 50.0)

    def test_offset_left_of_path(self):
        """Ninja to left of path should have positive offset (abs value)."""
        ninja_pos = (50.0, 50.0)
        path = [(100, 0), (100, 100), (100, 200)]
        offset = calculate_horizontal_offset_from_path(ninja_pos, path)
        self.assertAlmostEqual(offset, 50.0)

    def test_uses_first_waypoint(self):
        """Should use first waypoint as reference, not other waypoints."""
        ninja_pos = (110.0, 50.0)
        path = [(100, 0), (120, 100), (100, 200)]  # Path has horizontal movement
        offset = calculate_horizontal_offset_from_path(ninja_pos, path)
        self.assertAlmostEqual(offset, 10.0)  # Relative to first waypoint at x=100


class TestVerticalMasking(unittest.TestCase):
    """Test vertical path masking behaviors."""

    def test_jumps_masked_when_goal_below(self):
        """JUMP actions should be masked when path goes monotonically down."""
        path = [(200, 100), (200, 150), (200, 200), (200, 250)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        
        # Verify path is monotonically downward
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Down
        self.assertLess(max_dev, 48.0)  # Nearly vertical

    def test_jumps_not_masked_when_goal_above(self):
        """JUMP actions should NOT be masked when path goes monotonically up."""
        path = [(200, 250), (200, 200), (200, 150), (200, 100)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        
        # Verify path is monotonically upward
        self.assertTrue(is_mono)
        self.assertEqual(direction, -1)  # Up
        self.assertLess(max_dev, 48.0)  # Nearly vertical

    def test_noop_masked_when_falling_with_goal_above(self):
        """NOOP should be masked when falling with goal monotonically above."""
        path = [(200, 250), (200, 200), (200, 150), (200, 100)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        
        # Verify path is monotonically upward
        self.assertTrue(is_mono)
        self.assertEqual(direction, -1)  # Up
        self.assertLess(max_dev, 48.0)  # Nearly vertical

    def test_path_with_horizontal_deviation(self):
        """Paths with significant horizontal deviation should not trigger masking."""
        path = [(100, 100), (150, 150), (200, 200), (250, 250)]
        is_mono, direction, max_dev = is_path_monotonic_vertical(path)
        
        # Should be monotonic vertically but with horizontal deviation
        self.assertTrue(is_mono)
        self.assertEqual(direction, 1)  # Down
        # Maximum horizontal deviation should be significant
        self.assertGreater(max_dev, 48.0)


if __name__ == "__main__":
    unittest.main()

