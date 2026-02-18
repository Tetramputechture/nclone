"""
Verification tests for analytical trajectory equations.

Compares closed-form predictions against frame-by-frame simulation
using the Python simulator. All tests must pass with max error < 1e-8
before proceeding with trajectory catalog generation.

Usage:
    python -m nclone.analytical_physics.test_analytical_trajectory
"""

import math
import sys
import os

# Ensure nclone is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nclone.analytical_physics.analytical_trajectory import (
    S1,
    S2,
    Sx2,
    D,
    D_MINUS_1,
    velocity_at,
    position_y_at,
    position_x_at,
    jump_trajectory,
    solve_landing_frame,
    fall_trajectory,
)
from nclone.constants import (
    DRAG_REGULAR,
    GRAVITY_JUMP,
    GRAVITY_FALL,
    AIR_ACCEL,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    JUMP_FLOOR_Y,
)


def test_s1_s2_basic():
    """Test S1 and S2 helper functions against manual computation."""
    print("Testing S1/S2 basic properties...")

    # S1(0) = 0
    assert S1(0) == 0.0, f"S1(0) = {S1(0)}, expected 0"
    assert S2(0) == 0.0, f"S2(0) = {S2(0)}, expected 0"

    # S1(1) = d
    assert abs(S1(1) - D) < 1e-15, f"S1(1) = {S1(1)}, expected {D}"

    # S2(1) = (d - 1)/(d - 1) = 1
    assert abs(S2(1) - 1.0) < 1e-15, f"S2(1) = {S2(1)}, expected 1.0"

    # Sx2(1) = S2(1) - 1 = 0
    assert abs(Sx2(1) - 0.0) < 1e-15, f"Sx2(1) = {Sx2(1)}, expected 0.0"

    # S1(n) should equal sum of d^k for k=1..n
    for n in [1, 5, 10, 45, 100]:
        manual_s1 = sum(D**k for k in range(1, n + 1))
        assert abs(S1(n) - manual_s1) < 1e-10, (
            f"S1({n}) = {S1(n)}, manual = {manual_s1}, diff = {abs(S1(n) - manual_s1)}"
        )

    print("  PASSED")


def test_velocity_recurrence():
    """Test velocity_at against iterative recurrence."""
    print("Testing velocity recurrence...")

    test_cases = [
        ("free fall", 0.0, GRAVITY_FALL, 200),
        ("jump hold", JUMP_FLOOR_Y, GRAVITY_JUMP, 100),
        ("horizontal accel", 0.0, AIR_ACCEL, 100),
        ("horizontal decel", 3.0, -AIR_ACCEL, 100),
        ("mixed initial vy", -1.5, GRAVITY_FALL, 150),
    ]

    for name, v0, c, n_frames in test_cases:
        # Iterate the recurrence: v(n) = d * v(n-1) + c
        v_iter = v0
        for _ in range(n_frames):
            v_iter = D * v_iter + c

        v_formula = velocity_at(v0, c, n_frames)
        error = abs(v_formula - v_iter)
        assert error < 1e-8, (
            f"  {name}: formula={v_formula:.12f}, iter={v_iter:.12f}, error={error:.2e}"
        )

    print("  PASSED")


def test_vertical_position_against_iterate():
    """Test position_y_at against iterative simulation of integrate()."""
    print("Testing vertical position (gravity in integrate)...")

    test_cases = [
        ("free fall from rest", 100.0, 0.0, GRAVITY_FALL, 200),
        ("jump hold", 300.0, JUMP_FLOOR_Y, GRAVITY_JUMP, 45),
        ("fall with initial vy", 200.0, 3.0, GRAVITY_FALL, 100),
        ("jump with upward vy", 400.0, -1.5, GRAVITY_JUMP, 30),
    ]

    for name, p0, v0, g, n_frames in test_cases:
        # Simulate integrate() for vertical axis:
        # v_y *= d; v_y += g; p_y += v_y
        py = p0
        vy = v0
        for _ in range(n_frames):
            vy *= D
            vy += g
            py += vy

        py_formula = position_y_at(p0, v0, g, n_frames)
        error = abs(py_formula - py)
        assert error < 1e-6, (
            f"  {name}: formula={py_formula:.10f}, sim={py:.10f}, error={error:.2e}"
        )

    print("  PASSED")


def test_horizontal_position_against_iterate():
    """Test position_x_at against iterative simulation of integrate()+think().

    The horizontal axis has a different integration order:
      integrate: v_x *= d; p_x += v_x
      think:     v_x += a
    """
    print("Testing horizontal position (accel in think)...")

    test_cases = [
        ("accel from rest", 100.0, 0.0, AIR_ACCEL, 150),
        ("decel from max", 100.0, 3.0, -AIR_ACCEL, 100),
        ("no accel", 100.0, 2.0, 0.0, 100),
        ("accel with initial vx", 100.0, 1.5, AIR_ACCEL, 80),
        ("negative accel", 100.0, -1.0, -AIR_ACCEL, 100),
    ]

    for name, p0, v0, a, n_frames in test_cases:
        # Simulate the actual frame order:
        # Frame start: vx has accel from previous think()
        # integrate(): vx *= d; px += vx
        # think(): vx += a
        #
        # Note: at frame 0, vx = v0 (already includes any previous accel).
        # First frame: integrate applies drag and updates position, then think adds accel.
        px = p0
        vx = v0
        for _ in range(n_frames):
            # integrate()
            vx *= D
            px += vx
            # think() -- no speed cap for this test
            vx += a

        px_formula = position_x_at(p0, v0, a, n_frames)
        error = abs(px_formula - px)
        assert error < 1e-6, (
            f"  {name}: formula={px_formula:.10f}, sim={px:.10f}, error={error:.2e}"
        )

    print("  PASSED")


def test_horizontal_position_NOT_matching_vertical_formula():
    """Verify that using S2 (instead of Sx2) for horizontal gives wrong results.

    This confirms the integration order correction is necessary.
    """
    print("Testing that vertical formula is WRONG for horizontal axis...")

    p0, v0, a, n = 100.0, 0.0, AIR_ACCEL, 100

    # Simulate actual
    px = p0
    vx = v0
    for _ in range(n):
        vx *= D
        px += vx
        vx += a

    # Corrected formula (should match)
    px_corrected = position_x_at(p0, v0, a, n)
    error_corrected = abs(px_corrected - px)

    # Wrong formula using S2 instead of Sx2
    px_wrong = p0 + v0 * S1(n) + a * S2(n)
    error_wrong = abs(px_wrong - px)

    assert error_corrected < 1e-6, (
        f"  Corrected formula has error {error_corrected:.2e} (should be < 1e-6)"
    )
    assert error_wrong > 1.0, (
        f"  Wrong formula error = {error_wrong:.4f} (should be >> 0, "
        f"expected ~{a * n:.1f})"
    )

    print(f"  Corrected error: {error_corrected:.2e}")
    print(f"  Wrong (S2) error: {error_wrong:.4f} (expected ~{a * n:.1f})")
    print("  PASSED")


def test_two_phase_jump_trajectory():
    """Test jump_trajectory against iterative two-phase simulation."""
    print("Testing two-phase jump trajectory...")

    test_cases = [
        # (name, vx0, vy0, h1, j, h2, k)
        ("full hold no input", 0.0, JUMP_FLOOR_Y, 0, 45, 0, 60),
        ("full hold right", 0.0, JUMP_FLOOR_Y, 1, 45, 0, 60),
        ("partial hold left", 1.0, JUMP_FLOOR_Y, -1, 20, 1, 80),
        ("no hold fall", 0.0, JUMP_FLOOR_Y, 0, 0, 1, 100),
        ("full hold switch dir", 2.0, JUMP_FLOOR_Y, 1, 45, -1, 50),
        ("short hold", 0.0, JUMP_FLOOR_Y, 0, 5, 0, 120),
    ]

    px0, py0 = 500.0, 300.0

    for name, vx0, vy0, h1, j, h2, k in test_cases:
        # Simulate frame by frame
        px, py = px0, py0
        vx, vy = vx0, vy0
        a1 = h1 * AIR_ACCEL
        a2 = h2 * AIR_ACCEL

        for frame in range(j + k):
            # Determine which phase
            if frame < j:
                g = GRAVITY_JUMP
                a = a1
            else:
                g = GRAVITY_FALL
                a = a2

            # integrate()
            vx *= D
            vy *= D
            vy += g
            px += vx
            py += vy

            # think()
            vx += a

        # Formula
        px_f, py_f, vx_f, vy_f = jump_trajectory(px0, py0, vx0, vy0, h1, j, h2, k)

        px_err = abs(px_f - px)
        py_err = abs(py_f - py)
        vx_err = abs(vx_f - vx)
        vy_err = abs(vy_f - vy)
        max_err = max(px_err, py_err, vx_err, vy_err)

        assert max_err < 1e-6, (
            f"  {name}: max_err={max_err:.2e} "
            f"(px={px_err:.2e}, py={py_err:.2e}, vx={vx_err:.2e}, vy={vy_err:.2e})"
        )

    print("  PASSED")


def test_landing_frame_solver():
    """Test that solve_landing_frame finds the correct landing frame."""
    print("Testing landing frame solver...")

    test_cases = [
        ("full hold", JUMP_FLOOR_Y, 45),
        ("half hold", JUMP_FLOOR_Y, 22),
        ("short hold", JUMP_FLOOR_Y, 5),
        ("no hold", JUMP_FLOOR_Y, 0),
        ("weak jump", -1.4, 30),
    ]

    for name, vy0, j in test_cases:
        k_land = solve_landing_frame(vy0, j)
        assert k_land is not None, f"  {name}: no landing found"

        # Verify: position should be >= 0 at k_land but < 0 at k_land - 1
        vy_j = velocity_at(vy0, GRAVITY_JUMP, j)
        py_j = position_y_at(0.0, vy0, GRAVITY_JUMP, j)

        py_at_land = py_j + vy_j * S1(k_land) + GRAVITY_FALL * S2(k_land)
        assert py_at_land >= -1.0, (
            f"  {name}: py at k_land={k_land} is {py_at_land:.4f} (should be >= ~0)"
        )

        if k_land > 1:
            py_before = py_j + vy_j * S1(k_land - 1) + GRAVITY_FALL * S2(k_land - 1)
            assert py_before < 1.0, (
                f"  {name}: py at k_land-1 is {py_before:.4f} (should be < ~0)"
            )

    print("  PASSED")


def test_affine_property():
    """Verify the affine decomposition: landing_x = C * vx0 + Bx (constant).

    For a fixed control sequence (j, h1, h2, k), the x-position should be
    exactly affine in vx0. This is the key property enabling O(1) evaluation.
    """
    print("Testing affine velocity decomposition...")

    j, h1, h2 = 30, 1, 0
    k = solve_landing_frame(JUMP_FLOOR_Y, j)
    assert k is not None

    px0, py0 = 0.0, 0.0

    # Evaluate at several vx0 values
    vx0_values = [-3.0, -1.5, 0.0, 1.5, 3.0]
    results_x = []
    for vx0 in vx0_values:
        px, _, _, _ = jump_trajectory(px0, py0, vx0, JUMP_FLOOR_Y, h1, j, h2, k)
        results_x.append(px)

    # If affine, px = C * vx0 + B
    # Use two points to determine C and B
    C = (results_x[4] - results_x[2]) / (vx0_values[4] - vx0_values[2])
    B = results_x[2]  # at vx0=0

    # Verify all other points match
    for vx0, px_actual in zip(vx0_values, results_x):
        px_predicted = C * vx0 + B
        error = abs(px_predicted - px_actual)
        assert error < 1e-8, (
            f"  Affine mismatch at vx0={vx0}: predicted={px_predicted:.10f}, "
            f"actual={px_actual:.10f}, error={error:.2e}"
        )

    print(f"  C = {C:.10f}, B = {B:.10f}")
    print("  PASSED")


def test_affine_y_property():
    """Verify the affine decomposition for the y-axis: landing_y = Cy * vy0 + By."""
    print("Testing affine y-axis decomposition...")

    j, h1, h2 = 30, 0, 0

    # For y-axis affine testing, we need a fixed k (not dependent on vy0)
    # Use a fixed k from the default jump
    k = solve_landing_frame(JUMP_FLOOR_Y, j)
    assert k is not None

    px0, py0 = 0.0, 0.0

    # Evaluate at several vy0 values (different jump impulses)
    vy0_values = [-2.5, -2.0, -1.5, -1.0, -0.5]
    results_y = []
    for vy0 in vy0_values:
        _, py, _, _ = jump_trajectory(px0, py0, 0.0, vy0, h1, j, h2, k)
        results_y.append(py)

    # If affine, py = Cy * vy0 + By
    Cy = (results_y[4] - results_y[0]) / (vy0_values[4] - vy0_values[0])
    By = results_y[1] - Cy * vy0_values[1]

    for vy0, py_actual in zip(vy0_values, results_y):
        py_predicted = Cy * vy0 + By
        error = abs(py_predicted - py_actual)
        assert error < 1e-8, f"  Affine y mismatch at vy0={vy0}: error={error:.2e}"

    print(f"  Cy = {Cy:.10f}, By = {By:.10f}")
    print("  PASSED")


def test_cx_equals_cy():
    """Verify that Cx = Cy (same velocity coefficient for both axes)."""
    print("Testing Cx == Cy...")

    j = 30
    k = solve_landing_frame(JUMP_FLOOR_Y, j)
    assert k is not None

    # Cx from x-axis: (px(vx0=1) - px(vx0=0))
    px1, _, _, _ = jump_trajectory(0, 0, 1.0, JUMP_FLOOR_Y, 0, j, 0, k)
    px0, _, _, _ = jump_trajectory(0, 0, 0.0, JUMP_FLOOR_Y, 0, j, 0, k)
    Cx = px1 - px0

    # Cy from y-axis: (py(vy0=vy+1) - py(vy0=vy)) with same j,k
    _, py1, _, _ = jump_trajectory(0, 0, 0, JUMP_FLOOR_Y + 1.0, 0, j, 0, k)
    _, py0_val, _, _ = jump_trajectory(0, 0, 0, JUMP_FLOOR_Y, 0, j, 0, k)
    Cy = py1 - py0_val

    # Analytical: C = S1(j) + D^j * S1(k)
    from nclone.analytical_physics.analytical_trajectory import _dn

    C_analytical = S1(j) + _dn(j) * S1(k)

    error_cx = abs(Cx - C_analytical)
    error_cy = abs(Cy - C_analytical)
    error_cxcy = abs(Cx - Cy)

    assert error_cxcy < 1e-8, f"  Cx={Cx:.10f}, Cy={Cy:.10f}, diff={error_cxcy:.2e}"
    assert error_cx < 1e-8, f"  Cx error vs analytical: {error_cx:.2e}"
    assert error_cy < 1e-8, f"  Cy error vs analytical: {error_cy:.2e}"

    print(f"  C_analytical = {C_analytical:.10f}")
    print(f"  Cx = {Cx:.10f}, Cy = {Cy:.10f}")
    print("  PASSED")


def test_mid_arc_points_match_simulation():
    """Test that precomputed mid-arc ArcPoint coefficients match simulation.

    For a set of trajectories and intermediate frames, verify that the
    affine formula  px = C_n * vx0 + Bx_n  matches jump_trajectory()
    evaluated at the same frame.
    """
    print("Testing mid-arc point affine coefficients vs simulation...")

    from nclone.analytical_physics.trajectory_catalog import (
        _compute_arc_points_floor,
        ARC_SAMPLE_INTERVAL,
    )

    test_cases = [
        # (j, h1, h2)
        (45, 1, 0),
        (45, 0, 0),
        (45, -1, 1),
        (30, 1, -1),
        (10, 0, 1),
        (0, 0, 0),
    ]

    for j, h1, h2 in test_cases:
        k_land = solve_landing_frame(JUMP_FLOOR_Y, j)
        if k_land is None:
            continue

        arc_points = _compute_arc_points_floor(
            j,
            k_land,
            h1,
            h2,
            JUMP_FLOOR_Y,
        )

        if not arc_points:
            continue

        # Verify each arc point against jump_trajectory() at multiple vx0
        for vx0 in [-3.0, 0.0, 2.5]:
            for ap in arc_points:
                frame = ap.frame

                # Compute via affine formula
                px_affine = ap.C * vx0 + ap.Bx
                py_affine = ap.C * JUMP_FLOOR_Y + ap.By

                # Compute via jump_trajectory()
                if frame <= j:
                    px_sim, py_sim, _, _ = jump_trajectory(
                        0.0,
                        0.0,
                        vx0,
                        JUMP_FLOOR_Y,
                        h1,
                        frame,
                        h2,
                        0,
                    )
                else:
                    k2 = frame - j
                    px_sim, py_sim, _, _ = jump_trajectory(
                        0.0,
                        0.0,
                        vx0,
                        JUMP_FLOOR_Y,
                        h1,
                        j,
                        h2,
                        k2,
                    )

                px_err = abs(px_affine - px_sim)
                py_err = abs(py_affine - py_sim)
                max_err = max(px_err, py_err)

                assert max_err < 1e-6, (
                    f"  j={j}, h1={h1}, h2={h2}, frame={frame}, vx0={vx0}: "
                    f"px_err={px_err:.2e}, py_err={py_err:.2e}"
                )

    print("  PASSED")


def test_mid_arc_points_above_launch():
    """Verify that all precomputed arc points are above launch height.

    The _compute_arc_points_floor function should only return points
    where py < 0 relative to launch (ninja is above launch position).
    """
    print("Testing mid-arc points are all above launch height...")

    from nclone.analytical_physics.trajectory_catalog import (
        _compute_arc_points_floor,
    )

    test_cases = [(45, 1, 0), (45, 0, 0), (30, -1, 1), (10, 0, 0)]

    for j, h1, h2 in test_cases:
        k_land = solve_landing_frame(JUMP_FLOOR_Y, j)
        if k_land is None:
            continue

        arc_points = _compute_arc_points_floor(
            j,
            k_land,
            h1,
            h2,
            JUMP_FLOOR_Y,
        )

        for ap in arc_points:
            py_relative = ap.C * JUMP_FLOOR_Y + ap.By
            assert py_relative < 0, (
                f"  j={j}, h1={h1}, h2={h2}, frame={ap.frame}: "
                f"py_relative={py_relative:.4f} (should be < 0)"
            )

    print("  PASSED")


def test_catalog_has_arc_points():
    """Verify that the built catalog contains arc_points on entries."""
    print("Testing catalog entries have arc_points...")

    from nclone.analytical_physics.trajectory_catalog import TrajectoryCatalog

    catalog = TrajectoryCatalog.build()

    # Floor entries with j > 0 should generally have arc points
    entries_with_arcs = sum(1 for e in catalog.floor_flat if len(e.arc_points) > 0)
    total_entries = len(catalog.floor_flat)

    assert entries_with_arcs > 0, (
        f"  No floor entries have arc_points (out of {total_entries})"
    )

    print(f"  {entries_with_arcs}/{total_entries} floor entries have arc_points")

    # Wall entries should also have arc_offsets
    wall_with_arcs = 0
    wall_total = 0
    for key, entries in catalog.wall.items():
        for e in entries:
            wall_total += 1
            if len(e.arc_offsets) > 0:
                wall_with_arcs += 1

    assert wall_with_arcs > 0, (
        f"  No wall entries have arc_offsets (out of {wall_total})"
    )

    print(f"  {wall_with_arcs}/{wall_total} wall entries have arc_offsets")
    print("  PASSED")


def test_wall_arc_offsets_match_simulation():
    """Test that precomputed wall arc offsets match jump_trajectory().

    Wall jumps have fixed vx0/vy0, so arc offsets are fully precomputed
    (dx, dy) values. Verify they match the analytical jump_trajectory.
    """
    print("Testing wall arc offsets vs simulation...")

    from nclone.analytical_physics.trajectory_catalog import TrajectoryCatalog

    catalog = TrajectoryCatalog.build()

    checked = 0
    for key, entries in catalog.wall.items():
        for wentry in entries:
            if not wentry.arc_offsets:
                continue

            # Determine vx0, vy0 from key
            normal_sign, jump_type = key
            if jump_type == "regular":
                from nclone.constants import (
                    JUMP_WALL_REGULAR_X_MULTIPLIER,
                    JUMP_WALL_REGULAR_Y,
                )

                vx0 = JUMP_WALL_REGULAR_X_MULTIPLIER * normal_sign
                vy0 = JUMP_WALL_REGULAR_Y
            else:
                from nclone.constants import (
                    JUMP_WALL_SLIDE_X_MULTIPLIER,
                    JUMP_WALL_SLIDE_Y,
                )

                vx0 = JUMP_WALL_SLIDE_X_MULTIPLIER * normal_sign
                vy0 = JUMP_WALL_SLIDE_Y

            for arc_frame, arc_dx, arc_dy in wentry.arc_offsets:
                j = wentry.j
                if arc_frame <= j:
                    px_sim, py_sim, _, _ = jump_trajectory(
                        0.0,
                        0.0,
                        vx0,
                        vy0,
                        wentry.h1,
                        arc_frame,
                        wentry.h2,
                        0,
                    )
                else:
                    k2 = arc_frame - j
                    px_sim, py_sim, _, _ = jump_trajectory(
                        0.0,
                        0.0,
                        vx0,
                        vy0,
                        wentry.h1,
                        j,
                        wentry.h2,
                        k2,
                    )

                dx_err = abs(arc_dx - px_sim)
                dy_err = abs(arc_dy - py_sim)
                max_err = max(dx_err, dy_err)

                assert max_err < 1e-6, (
                    f"  key={key}, j={j}, h1={wentry.h1}, h2={wentry.h2}, "
                    f"frame={arc_frame}: dx_err={dx_err:.2e}, "
                    f"dy_err={dy_err:.2e}"
                )
                checked += 1

    assert checked > 0, "  No wall arc offsets were checked"
    print(f"  Checked {checked} wall arc offset points")
    print("  PASSED")


def test_floor_landing_y_with_jump_impulse():
    """Verify that floor jump landing y = C * JUMP_FLOOR_Y + By.

    This validates the bug fix: the landing y must include the C * vy0
    term, not just the By bias.
    """
    print("Testing floor landing y includes jump impulse...")

    from nclone.analytical_physics.trajectory_catalog import TrajectoryCatalog

    catalog = TrajectoryCatalog.build()

    # For each floor entry, verify the affine formula matches jump_trajectory
    for entry in catalog.floor_flat[:50]:  # Check first 50 for speed
        vx0 = 0.0
        px_sim, py_sim, _, _ = jump_trajectory(
            0.0,
            0.0,
            vx0,
            JUMP_FLOOR_Y,
            entry.h1,
            entry.j,
            entry.h2,
            entry.k_land,
        )

        # Landing y via affine: C * vy0 + By
        land_y_affine = entry.C * JUMP_FLOOR_Y + entry.By

        # Landing x via affine: C * vx0 + Bx = 0 + Bx
        land_x_affine = entry.Bx

        y_err = abs(land_y_affine - py_sim)
        x_err = abs(land_x_affine - px_sim)

        assert y_err < 1e-6, (
            f"  j={entry.j}, h1={entry.h1}, h2={entry.h2}: "
            f"affine_y={land_y_affine:.6f}, sim_y={py_sim:.6f}, "
            f"error={y_err:.2e}"
        )
        assert x_err < 1e-6, (
            f"  j={entry.j}, h1={entry.h1}, h2={entry.h2}: "
            f"affine_x={land_x_affine:.6f}, sim_x={px_sim:.6f}, "
            f"error={x_err:.2e}"
        )

        # Also verify that just using By (the old bug) gives wrong results
        # when j > 0 (any actual jump)
        if entry.j > 0:
            wrong_y = entry.By
            wrong_error = abs(wrong_y - py_sim)
            assert wrong_error > 1.0, (
                f"  j={entry.j}: old formula (By only) error={wrong_error:.4f} "
                f"should be >> 0"
            )

    print("  PASSED")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("ANALYTICAL TRAJECTORY VERIFICATION")
    print("=" * 60)
    print()

    tests = [
        test_s1_s2_basic,
        test_velocity_recurrence,
        test_vertical_position_against_iterate,
        test_horizontal_position_against_iterate,
        test_horizontal_position_NOT_matching_vertical_formula,
        test_two_phase_jump_trajectory,
        test_landing_frame_solver,
        test_affine_property,
        test_affine_y_property,
        test_cx_equals_cy,
        test_mid_arc_points_match_simulation,
        test_mid_arc_points_above_launch,
        test_catalog_has_arc_points,
        test_wall_arc_offsets_match_simulation,
        test_floor_landing_y_with_jump_impulse,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
