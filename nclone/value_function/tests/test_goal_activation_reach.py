"""Focused tests for the surface-lattice goal seeding (find_goal_nodes).

Confirms the lattice d* treats a switch/exit as reached once the ninja CENTER is
within the activation reach (NINJA_RADIUS + entity_radius) of the true entity
center -- matching overlap_circle_vs_circle -- with no left/right snap bias.
"""

import unittest

from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
)
from nclone.value_function.surface_graph import (
    build_graph,
    node_id_to_parts,
    find_goal_nodes,
)
from nclone.value_function.surface_extraction import SurfacePoint, SURFACE_FLOOR


def _floor_point(x, y, seg):
    return SurfacePoint(x=x, y=y, normal_x=0.0, normal_y=-1.0,
                        surface_type=SURFACE_FLOOR, segment_id=seg, is_edge=False)


class TestFindGoalNodesActivationReach(unittest.TestCase):
    # Switch center on the 6px grid; reach = 10 + 6 = 16px.
    CENTER = (12.0, 100.0)

    def _included_point_indices(self, goal_type):
        # Points at increasing |dx| from the center; symmetric pairs verify there
        # is no left/right bias when distances are equal.
        pts = [
            _floor_point(12.0, 100.0, 0),   # 0: d=0
            _floor_point(0.0, 100.0, 1),    # 1: d=12 (left)
            _floor_point(24.0, 100.0, 2),   # 2: d=12 (right) -> mirror of pt 1
            _floor_point(26.0, 100.0, 3),   # 3: d=14 (inside switch reach)
            _floor_point(30.0, 100.0, 4),   # 4: d=18 (outside switch, inside exit)
            _floor_point(-6.0, 100.0, 5),   # 5: d=18 (mirror of pt 4)
            _floor_point(36.0, 100.0, 6),   # 6: d=24 (outside exit reach)
        ]
        g = build_graph(pts)
        nodes = find_goal_nodes(g, self.CENTER[0], self.CENTER[1], goal_type)
        return {node_id_to_parts(n, g.max_vel_bins)[0] for n in nodes}

    def test_switch_reach_includes_only_within_16px(self):
        included = self._included_point_indices("switch")
        self.assertEqual(included, {0, 1, 2, 3})

    def test_exit_reach_includes_only_within_22px(self):
        included = self._included_point_indices("exit")
        self.assertEqual(included, {0, 1, 2, 3, 4, 5})

    def test_no_left_right_snap_bias(self):
        # Equidistant points on either side (12px) are BOTH seeded: the seeding
        # uses the true center distance, not a snapped node, so neither side wins.
        included = self._included_point_indices("switch")
        self.assertIn(1, included)
        self.assertIn(2, included)

    def test_reach_matches_constants(self):
        # Guard the activation reach against accidental constant drift.
        self.assertEqual(NINJA_RADIUS + EXIT_SWITCH_RADIUS, 16)
        self.assertEqual(NINJA_RADIUS + EXIT_DOOR_RADIUS, 22)

    def test_unknown_goal_type_raises(self):
        g = build_graph([_floor_point(12.0, 100.0, 0)])
        with self.assertRaises(ValueError):
            find_goal_nodes(g, 12.0, 100.0, "gold")


if __name__ == "__main__":
    unittest.main()
