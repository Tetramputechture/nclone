"""Regression: path_distance_calculator's OOB bounds must track the CURRENT
adjacency, never a previous level's graph (the stale-cache bug).

Bug: ``build_level_cache`` only refreshed ``_adjacency_bounds`` inside its
``if rebuilt:`` branch. When the underlying ``build_cache`` decided it did NOT
need to rebuild (its rejection key was LevelData-equality + mine/switch
signatures — which collide for e.g. two different levels that both have 0 mines
and one un-activated switch), the calculator kept a PREVIOUS level's bounds. The
next level's ninja was then compared against that stale box and read as "far
outside graph bounds" → ``inf`` distance every step → a log flood (thousands of
identical warnings for one small level whose real graph sat elsewhere).

Fix: bounds are derived from the ``adjacency`` object actually being searched,
cached by its identity (``_bounds_for_adjacency``), so they can never desync
from the graph in use.

Run: ../nclone/.venv/bin/python -m pytest nclone/graph/reachability/test_path_distance_stale_bounds.py
"""
import unittest

from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


def _adj(nodes):
    """Minimal adjacency dict: node -> [] (bounds only need the keys)."""
    return {n: [] for n in nodes}


class TestStaleBounds(unittest.TestCase):
    def test_bounds_track_current_adjacency(self):
        calc = CachedPathDistanceCalculator()

        # Level A: graph sits on the right of the map.
        adj_a = _adj([(600, 300), (900, 500)])
        self.assertEqual(calc._bounds_for_adjacency(adj_a), (600, 900, 300, 500))

        # Level B: a DIFFERENT graph on the left. The bounds MUST switch to B's
        # box — not stay on A's (that was the stale-cache bug).
        adj_b = _adj([(220, 270), (260, 330)])
        self.assertEqual(
            calc._bounds_for_adjacency(adj_b),
            (220, 260, 270, 330),
            "bounds are stale — still reflect the previous level's adjacency",
        )

    def test_same_adjacency_object_is_cached(self):
        calc = CachedPathDistanceCalculator()
        adj = _adj([(100, 100), (200, 200)])
        b1 = calc._bounds_for_adjacency(adj)
        b2 = calc._bounds_for_adjacency(adj)  # same object -> cache hit
        self.assertEqual(b1, b2)
        self.assertIs(calc._adjacency_bounds_adj, adj)

    def test_empty_adjacency_yields_none(self):
        calc = CachedPathDistanceCalculator()
        self.assertIsNone(calc._bounds_for_adjacency({}))

    def test_ninja_in_new_level_not_flagged_oob(self):
        """End-to-end: a ninja inside level B's graph is not rejected just because
        the calculator previously processed level A's far-away graph."""
        calc = CachedPathDistanceCalculator()
        # Prime with level A's far-right graph.
        calc._bounds_for_adjacency(_adj([(600, 300), (900, 500)]))
        # Level B's graph around (240, 300); ninja sits inside it.
        adj_b = _adj([(216, 276), (264, 324), (240, 300)])
        bounds = calc._bounds_for_adjacency(adj_b)
        bx0, bx1, by0, by1 = bounds
        # ninja world pos ~ (264, 324) -> tile space (minus the 24px border).
        from nclone.graph.reachability.pathfinding_utils import (
            NODE_WORLD_COORD_OFFSET,
        )
        nx = 264 - NODE_WORLD_COORD_OFFSET
        ny = 324 - NODE_WORLD_COORD_OFFSET
        margin = 200
        in_bounds = (bx0 - margin <= nx <= bx1 + margin) and (
            by0 - margin <= ny <= by1 + margin
        )
        self.assertTrue(in_bounds, "ninja wrongly flagged OOB against stale bounds")


if __name__ == "__main__":
    unittest.main()
