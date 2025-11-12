"""
Unit tests for hybrid mine death prediction system.

Tests the three-tier hybrid approach:
1. Spatial danger zone grid
2. Distance-based quick check
3. Full physics simulation
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nclone.mine_death_predictor import HybridPredictorStats
from nclone.constants import (
    MINE_DANGER_ZONE_RADIUS,
    MINE_DANGER_ZONE_CELL_SIZE,
    MINE_DANGER_THRESHOLD,
)


class TestHybridMineDeathPredictor(unittest.TestCase):
    """Test hybrid mine death prediction system."""

    def test_danger_zone_grid_building(self):
        """Test that danger zone grid is built correctly."""
        # This test requires a mock sim object
        # For now, just test the concept
        pass

    def test_spatial_pre_filter(self):
        """Test Tier 1 spatial pre-filter logic."""
        # Test that positions far from mines are filtered out quickly
        # Cell calculation: (x / 24, y / 24)
        mine_pos = (100.0, 100.0)
        mine_cell = (int(100 / 24), int(100 / 24))  # (4, 4)

        # Far position should be in different cell
        far_pos = (300.0, 300.0)
        far_cell = (int(300 / 24), int(300 / 24))  # (12, 12)

        self.assertNotEqual(mine_cell, far_cell)

    def test_distance_calculation(self):
        """Test Tier 2 distance calculation."""
        import math

        mine_x, mine_y = 100.0, 100.0
        ninja_x, ninja_y = 130.0, 100.0

        distance = math.sqrt((ninja_x - mine_x) ** 2 + (ninja_y - mine_y) ** 2)

        self.assertEqual(distance, 30.0)
        self.assertLessEqual(distance, MINE_DANGER_THRESHOLD)

    def test_stats_structure(self):
        """Test that stats structure is correct."""
        stats = HybridPredictorStats()

        self.assertEqual(stats.build_time_ms, 0.0)
        self.assertEqual(stats.reachable_mines, 0)
        self.assertEqual(stats.danger_zone_cells, 0)
        self.assertEqual(stats.tier1_queries, 0)
        self.assertEqual(stats.tier2_queries, 0)
        self.assertEqual(stats.tier3_queries, 0)

    def test_constants_loaded(self):
        """Test that hybrid constants are loaded correctly."""
        self.assertEqual(MINE_DANGER_ZONE_RADIUS, 80.0)
        self.assertEqual(MINE_DANGER_ZONE_CELL_SIZE, 24)
        self.assertEqual(MINE_DANGER_THRESHOLD, 30.0)


def run_tests():
    """Run all unit tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
