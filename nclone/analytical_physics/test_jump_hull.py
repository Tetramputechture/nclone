"""Validation tests for jump hull precomputation.

Verifies that the hull simulation produces correct trajectories by:
1. Comparing hull cell positions against an independent frame-by-frame simulator
2. Checking hulls for flat ground, all 3 slope angles, and max-speed velocity bins
3. Verifying that the hull contains ALL cells traced by the simulator

Usage:
    python -m nclone.analytical_physics.test_jump_hull
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nclone.constants.physics_constants import (
    AIR_ACCEL,
    DRAG_REGULAR,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    JUMP_FLOOR_Y,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
)
from nclone.analytical_physics.jump_hull import (
    JumpHullCollection,
    classify_surface,
    compute_slope_impulse,
    get_or_build_hulls,
    surface_normal,
    _simulate_trajectory,
    _build_hull,
    SURFACE_FLAT,
    SURFACE_SLOPE_MILD_L,
    SURFACE_SLOPE_MILD_R,
    SURFACE_SLOPE_45_L,
    SURFACE_SLOPE_45_R,
    SURFACE_SLOPE_STEEP_L,
    SURFACE_SLOPE_STEEP_R,
    NUM_SURFACE_TYPES,
    GRID_CELL,
    _FLOOR_BIN_CENTERS,
    NUM_VELOCITY_BINS_FLOOR,
    D,
)


class TestClassifySurface(unittest.TestCase):
    """Test surface type classification from normals."""

    def test_flat_surface(self):
        self.assertEqual(classify_surface(0.0, -1.0), SURFACE_FLAT)
        self.assertEqual(classify_surface(0.005, -1.0), SURFACE_FLAT)

    def test_mild_slopes(self):
        self.assertEqual(classify_surface(-0.4472, -0.8944), SURFACE_SLOPE_MILD_L)
        self.assertEqual(classify_surface(0.4472, -0.8944), SURFACE_SLOPE_MILD_R)

    def test_45_slopes(self):
        self.assertEqual(classify_surface(-0.7071, -0.7071), SURFACE_SLOPE_45_L)
        self.assertEqual(classify_surface(0.7071, -0.7071), SURFACE_SLOPE_45_R)

    def test_steep_slopes(self):
        self.assertEqual(classify_surface(-0.8944, -0.4472), SURFACE_SLOPE_STEEP_L)
        self.assertEqual(classify_surface(0.8944, -0.4472), SURFACE_SLOPE_STEEP_R)


class TestSlopeImpulse(unittest.TestCase):
    """Test slope impulse matches ninja.py floor_jump logic."""

    def test_flat_ground_impulse(self):
        jx, jy, vx_after, vy_after = compute_slope_impulse(0.0, -1.0, 1.5, 1)
        self.assertAlmostEqual(jx, 0.0)
        self.assertAlmostEqual(jy, JUMP_FLOOR_Y)
        self.assertAlmostEqual(vx_after, 1.5)
        # Flat ground: vy_pre = 0, so vy_after = jy
        self.assertAlmostEqual(vy_after, JUMP_FLOOR_Y)

    def test_slope_downhill_with_input(self):
        """Moving downhill + input same direction -> slope impulse."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)  # nx > 0
        # vx > 0, dx > 0 -> downhill; h1 > 0 -> same direction as vx
        jx, jy, vx_after, vy_after = compute_slope_impulse(nx, ny, 1.0, 1)
        self.assertAlmostEqual(jx, 2.0 / 3.0 * nx)
        self.assertAlmostEqual(jy, 2.0 * ny)
        self.assertAlmostEqual(vx_after, 1.0 + jx)
        # Downhill: vy_pre > 0, clamped to 0 -> vy_after = jy
        self.assertAlmostEqual(vy_after, jy)

    def test_slope_downhill_against_input(self):
        """Moving downhill + input opposite -> default uphill impulse."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        # vx > 0, dx > 0 -> downhill; h1 < 0 -> opposite
        jx, jy, vx_after, vy_after = compute_slope_impulse(nx, ny, 1.0, -1)
        self.assertAlmostEqual(jx, 0.0)
        self.assertAlmostEqual(jy, -1.4)
        self.assertAlmostEqual(vx_after, 1.0)
        # Downhill: vy_pre > 0, clamped to 0 -> vy_after = jy
        self.assertAlmostEqual(vy_after, jy)

    def test_slope_uphill_forward(self):
        """Moving uphill + input same direction as vx -> default uphill."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        # vx < 0, dx > 0 -> uphill; h1 < 0 -> same direction as vx
        jx, jy, vx_after, vy_after = compute_slope_impulse(nx, ny, -1.0, -1)
        self.assertAlmostEqual(jx, 0.0)
        self.assertAlmostEqual(jy, -1.4)
        self.assertAlmostEqual(vx_after, -1.0)
        # Uphill on 45R: vy_pre = vx * (-nx/ny) = -1.0 * 1.0 = -1.0 (preserved)
        self.assertAlmostEqual(vy_after, -1.0 + jy)  # -1.0 + (-1.4) = -2.4

    def test_slope_uphill_perpendicular(self):
        """Moving uphill + no matching input -> perp jump (xspeed zeroed)."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        # vx < 0, dx > 0 -> uphill; h1 = 0 -> xspeed * h1 = 0, not > 0
        jx, jy, vx_after, vy_after = compute_slope_impulse(nx, ny, -1.0, 0)
        expected_jx = 2.0 / 3.0 * nx
        expected_jy = 2.0 * ny
        self.assertAlmostEqual(jx, expected_jx)
        self.assertAlmostEqual(jy, expected_jy)
        # xspeed should have been zeroed before adding jx
        self.assertAlmostEqual(vx_after, 0.0 + expected_jx)
        # Uphill: vy_pre uses ORIGINAL vx (-1.0), not zeroed xspeed
        # vy_pre = -1.0 * (-nx/ny) = -1.0 * 1.0 = -1.0 (preserved)
        self.assertAlmostEqual(vy_after, -1.0 + expected_jy)


class TestIndependentSimulation(unittest.TestCase):
    """Compare hull simulation against an independent frame-by-frame sim.

    This independently reimplements the N++ physics loop to verify
    the hull simulation is correct.
    """

    @staticmethod
    def independent_simulate(vx0, vy0, px0, py0, j, h1, h2, max_frames=500):
        """Independent frame-by-frame simulation (not from jump_hull.py)."""
        from nclone.constants.physics_constants import MAX_SURVIVABLE_IMPACT

        vx, vy = vx0, vy0
        px, py = px0, py0
        cells = {}

        for frame in range(1, max_frames + 1):
            # integrate()
            vx *= DRAG_REGULAR
            vy *= DRAG_REGULAR
            gravity = GRAVITY_JUMP if frame <= j else GRAVITY_FALL
            vy += gravity
            px += vx
            py += vy

            # think() - horizontal acceleration with speed cap
            h = h1 if frame <= j else h2
            vx_new = vx + h * AIR_ACCEL
            if abs(vx_new) < MAX_HOR_SPEED:
                vx = vx_new

            # Stop if fall velocity exceeds survivable impact
            if vy > MAX_SURVIVABLE_IMPACT:
                break

            # Record grid cell
            cx = round(px / GRID_CELL)
            cy = round(py / GRID_CELL)
            if (cx, cy) not in cells or frame < cells[(cx, cy)][0]:
                cells[(cx, cy)] = (frame, vx)

        return cells

    def _check_hull_contains_independent(
        self, surface_type, velocity_bin, description=""
    ):
        """Verify that hull contains all cells from independent simulation."""
        nx, ny = surface_normal(surface_type)
        vx_center = float(_FLOOR_BIN_CENTERS[velocity_bin])

        hull = _build_hull(nx, ny, vx_center)

        # Run independent simulation for all (j, h1, h2) combos
        independent_cells = {}
        for j in range(0, MAX_JUMP_DURATION + 1):
            for h1 in [-1, 0, 1]:
                jx, jy, vx_after, vy_after = compute_slope_impulse(
                    nx, ny, vx_center, h1
                )
                for h2 in [-1, 0, 1]:
                    ind = self.independent_simulate(
                        vx_after,
                        vy_after,
                        jx,
                        jy,
                        j,
                        h1,
                        h2,
                    )
                    for (cx, cy), (frame, vx) in ind.items():
                        if (
                            cx,
                            cy,
                        ) not in independent_cells or frame < independent_cells[
                            (cx, cy)
                        ][0]:
                            independent_cells[(cx, cy)] = (frame, vx)

        # Check that every independently-traced cell is in the hull
        missing = []
        for (cx, cy), (frame, vx) in independent_cells.items():
            if (cx, cy) not in hull:
                missing.append(((cx, cy), frame))

        self.assertEqual(
            len(missing),
            0,
            f"Hull for {description} (surface={surface_type}, vbin={velocity_bin}) "
            f"is missing {len(missing)} cells: {missing[:10]}",
        )

        # Check min_frames consistency
        for (cx, cy), cell in hull.items():
            if (cx, cy) in independent_cells:
                ind_frame = independent_cells[(cx, cy)][0]
                self.assertEqual(
                    cell.min_frames,
                    ind_frame,
                    f"Hull {description} cell ({cx},{cy}): "
                    f"hull.min_frames={cell.min_frames} vs independent={ind_frame}",
                )

    def test_flat_ground_v0(self):
        """Flat ground, zero velocity."""
        self._check_hull_contains_independent(
            SURFACE_FLAT, NUM_VELOCITY_BINS_FLOOR // 2, "flat/v=0"
        )

    def test_flat_ground_max_speed(self):
        """Flat ground, maximum positive speed."""
        self._check_hull_contains_independent(
            SURFACE_FLAT, NUM_VELOCITY_BINS_FLOOR - 1, "flat/v=max"
        )

    def test_flat_ground_neg_max_speed(self):
        """Flat ground, maximum negative speed."""
        self._check_hull_contains_independent(SURFACE_FLAT, 0, "flat/v=-max")

    def test_mild_slope_left_v0(self):
        """Mild left slope, zero velocity."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_MILD_L, NUM_VELOCITY_BINS_FLOOR // 2, "mild-L/v=0"
        )

    def test_mild_slope_right_max(self):
        """Mild right slope, max speed."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_MILD_R, NUM_VELOCITY_BINS_FLOOR - 1, "mild-R/v=max"
        )

    def test_45_slope_left_v0(self):
        """45-degree left slope, zero velocity."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_45_L, NUM_VELOCITY_BINS_FLOOR // 2, "45-L/v=0"
        )

    def test_45_slope_right_max(self):
        """45-degree right slope, max speed."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_45_R, NUM_VELOCITY_BINS_FLOOR - 1, "45-R/v=max"
        )

    def test_steep_slope_left_v0(self):
        """Steep left slope, zero velocity."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_STEEP_L, NUM_VELOCITY_BINS_FLOOR // 2, "steep-L/v=0"
        )

    def test_steep_slope_right_max(self):
        """Steep right slope, max speed."""
        self._check_hull_contains_independent(
            SURFACE_SLOPE_STEEP_R, NUM_VELOCITY_BINS_FLOOR - 1, "steep-R/v=max"
        )


class TestHullCollection(unittest.TestCase):
    """Test the hull collection singleton."""

    def test_build_count(self):
        collection = get_or_build_hulls()
        expected = NUM_SURFACE_TYPES * NUM_VELOCITY_BINS_FLOOR
        self.assertEqual(collection.num_hulls, expected)

    def test_flat_v0_nonempty(self):
        collection = get_or_build_hulls()
        hull = collection.get_hull(SURFACE_FLAT, NUM_VELOCITY_BINS_FLOOR // 2)
        self.assertGreater(len(hull), 0)

    def test_hull_cell_origin(self):
        """Hull should contain (0, 0) since the ninja starts there."""
        collection = get_or_build_hulls()
        hull = collection.get_hull(SURFACE_FLAT, NUM_VELOCITY_BINS_FLOOR // 2)
        self.assertIn((0, 0), hull)

    def test_hull_symmetry_flat(self):
        """For flat ground v=0, left/right should be symmetric."""
        collection = get_or_build_hulls()
        hull = collection.get_hull(SURFACE_FLAT, NUM_VELOCITY_BINS_FLOOR // 2)
        # Check that for every cell (cx, cy) there's a (-cx, cy) cell
        # (approximately, since frame counts may differ slightly)
        asymmetric = 0
        for cx, cy in hull:
            if (-cx, cy) not in hull:
                asymmetric += 1
        # Allow a small number of asymmetric cells due to rounding
        self.assertLess(
            asymmetric,
            5,
            f"Flat v=0 hull has {asymmetric} asymmetric cells",
        )


class TestSlopePreJumpVelocity(unittest.TestCase):
    """Verify that pre-jump slope velocity is correctly included in vy_after."""

    def test_uphill_45_preserves_vy(self):
        """Ascending a 45-deg slope adds pre-jump upward vy to impulse."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        # Ascending at max speed: vx=-3.33, slope_ratio = -nx/ny = 1.0
        vx = float(_FLOOR_BIN_CENTERS[0])  # -3.33
        _, jy, _, vy_after = compute_slope_impulse(nx, ny, vx, 1)
        vy_pre = vx * (-nx / ny)  # -3.33 * 1.0 = -3.33
        self.assertAlmostEqual(vy_after, vy_pre + jy)
        self.assertLess(vy_after, jy, "Uphill vy should make launch more negative")

    def test_uphill_steep_preserves_vy(self):
        """Ascending a steep slope has even larger pre-jump vy (ratio = 2.0)."""
        nx, ny = surface_normal(SURFACE_SLOPE_STEEP_R)
        vx = float(_FLOOR_BIN_CENTERS[0])  # -3.33
        _, jy, _, vy_after = compute_slope_impulse(nx, ny, vx, 1)
        vy_pre = vx * (-nx / ny)  # -3.33 * 2.0 = -6.66
        self.assertAlmostEqual(vy_after, vy_pre + jy)
        self.assertLess(
            vy_after, -7.0, "Steep uphill should produce very large launch vy"
        )

    def test_downhill_clamps_vy(self):
        """Descending a slope clamps positive vy_pre to 0."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        # Descending: vx=+3.33, slope_ratio=1.0, vy_pre=+3.33 -> clamped to 0
        vx = float(_FLOOR_BIN_CENTERS[-1])  # +3.33
        _, jy, _, vy_after = compute_slope_impulse(nx, ny, vx, 1)
        # vy_after should be jy (clamped vy_pre contributes nothing)
        self.assertAlmostEqual(vy_after, jy)

    def test_flat_ground_no_slope_vy(self):
        """Flat ground has no slope velocity contribution."""
        _, jy, _, vy_after = compute_slope_impulse(0.0, -1.0, 3.33, 1)
        self.assertAlmostEqual(vy_after, jy)

    def test_perp_jump_uses_original_vx_for_vy(self):
        """Perpendicular jump zeroes xspeed but vy_pre uses original vx."""
        nx, ny = surface_normal(SURFACE_SLOPE_45_R)
        vx = -2.0  # Ascending
        jx, jy, vx_after, vy_after = compute_slope_impulse(nx, ny, vx, 0)
        # xspeed was zeroed: vx_after = 0 + jx
        self.assertAlmostEqual(vx_after, jx)
        # But vy_pre should use original vx (-2.0), not zeroed value
        vy_pre = vx * (-nx / ny)  # -2.0 * 1.0 = -2.0
        self.assertAlmostEqual(vy_after, vy_pre + jy)

    def test_mild_slope_smaller_ratio(self):
        """Mild slope has ratio 0.5, producing smaller vy contribution."""
        nx, ny = surface_normal(SURFACE_SLOPE_MILD_R)
        vx = float(_FLOOR_BIN_CENTERS[0])  # -3.33
        _, jy, _, vy_after = compute_slope_impulse(nx, ny, vx, 1)
        vy_pre = vx * (-nx / ny)  # -3.33 * 0.5 = -1.665
        self.assertAlmostEqual(vy_after, vy_pre + jy)


class TestSpeedCapHandling(unittest.TestCase):
    """Verify correct speed cap enforcement."""

    def test_speed_cap_not_exceeded(self):
        """Hull simulation should never produce |vx| > MAX_HOR_SPEED."""
        nx, ny = surface_normal(SURFACE_FLAT)
        vx_max = float(_FLOOR_BIN_CENTERS[-1])

        # Run simulation for a trajectory that pushes against the cap
        points = _simulate_trajectory(
            vx0=vx_max,
            vy0=JUMP_FLOOR_Y,
            px0=0.0,
            py0=JUMP_FLOOR_Y,
            j=MAX_JUMP_DURATION,
            h1=1,
            h2=1,
        )

        for frame, dx, dy, vx, vy in points:
            self.assertLess(
                abs(vx),
                MAX_HOR_SPEED + 0.01,
                f"Frame {frame}: |vx|={abs(vx):.4f} exceeds cap {MAX_HOR_SPEED}",
            )

    def test_speed_cap_rejection_model(self):
        """Speed cap uses rejection (not clamping) -- vx oscillates near cap."""
        # At max speed with h1=1, some frames should have acceleration rejected
        # resulting in vx < MAX_HOR_SPEED after drag
        nx, ny = surface_normal(SURFACE_FLAT)
        vx_max = float(_FLOOR_BIN_CENTERS[-1])

        points = _simulate_trajectory(
            vx0=vx_max,
            vy0=JUMP_FLOOR_Y,
            px0=0.0,
            py0=JUMP_FLOOR_Y,
            j=MAX_JUMP_DURATION,
            h1=1,
            h2=1,
        )

        # After drag, vx should be slightly below cap, and then
        # acceleration would push it over -- rejection model means
        # it stays at post-drag value
        below_cap_count = sum(
            1 for _, _, _, vx, _ in points if abs(vx) < MAX_HOR_SPEED - 0.01
        )
        self.assertGreater(
            below_cap_count,
            0,
            "With rejection model, some frames should have vx below cap after drag",
        )


if __name__ == "__main__":
    unittest.main()
