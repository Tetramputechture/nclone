"""Forward distance curriculum weighting (Workstream C)."""

import math
import unittest

from nclone.levelgen.distance_curriculum import (
    LevelStat,
    frontier_distance,
    frontier_weights,
    learnability,
)


class TestLearnability(unittest.TestCase):
    def test_peaks_at_half(self):
        self.assertAlmostEqual(learnability(0.5), 0.25)
        self.assertEqual(learnability(0.0), 0.0)
        self.assertEqual(learnability(1.0), 0.0)
        self.assertEqual(learnability(2.0), 0.0)  # clamped


class TestFrontierWeights(unittest.TestCase):
    def test_longer_route_preferred_among_equal_learnability(self):
        # two equally-learnable levels (p=0.5); the longer one should weigh more
        levels = [
            LevelStat(1, distance=200.0, solve_rate=0.5),
            LevelStat(2, distance=800.0, solve_rate=0.5),
        ]
        w = frontier_weights(levels, normalize=False)
        self.assertGreater(w[2], w[1])

    def test_mastered_levels_drop_out(self):
        # p=1.0 -> learnability 0 -> weight 0 even if far
        levels = [
            LevelStat(1, distance=900.0, solve_rate=1.0),
            LevelStat(2, distance=300.0, solve_rate=0.5),
        ]
        w = frontier_weights(levels)
        self.assertEqual(w[1], 0.0)
        self.assertGreater(w[2], 0.0)

    def test_unrouteable_levels_zero_weight(self):
        levels = [
            LevelStat(1, distance=float("inf"), solve_rate=0.5),
            LevelStat(2, distance=300.0, solve_rate=0.5),
        ]
        w = frontier_weights(levels)
        self.assertEqual(w[1], 0.0)
        self.assertGreater(w[2], 0.0)

    def test_unseen_levels_get_explored(self):
        # solve_rate None -> treated as frontier (unseen_p=0.5) so it gets weight
        levels = [
            LevelStat(1, distance=400.0, solve_rate=None),
            LevelStat(2, distance=400.0, solve_rate=1.0),
        ]
        w = frontier_weights(levels)
        self.assertGreater(w[1], 0.0)
        self.assertEqual(w[2], 0.0)

    def test_weights_normalize_to_one(self):
        levels = [
            LevelStat(1, distance=200.0, solve_rate=0.5),
            LevelStat(2, distance=800.0, solve_rate=0.4),
        ]
        w = frontier_weights(levels, normalize=True)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_distance_pref_zero_is_pure_learnability(self):
        levels = [
            LevelStat(1, distance=200.0, solve_rate=0.5),
            LevelStat(2, distance=800.0, solve_rate=0.5),
        ]
        w = frontier_weights(levels, distance_pref=0.0, normalize=False)
        self.assertAlmostEqual(w[1], w[2])  # equal learnability, no distance bias


class TestFrontierDistance(unittest.TestCase):
    def test_returns_longest_mastered(self):
        levels = [
            LevelStat(1, distance=300.0, solve_rate=0.9),
            LevelStat(2, distance=700.0, solve_rate=0.85),
            LevelStat(3, distance=1200.0, solve_rate=0.4),  # not mastered
        ]
        self.assertEqual(frontier_distance(levels), 700.0)

    def test_zero_when_nothing_mastered(self):
        levels = [LevelStat(1, distance=300.0, solve_rate=0.3)]
        self.assertEqual(frontier_distance(levels), 0.0)


if __name__ == "__main__":
    unittest.main()
