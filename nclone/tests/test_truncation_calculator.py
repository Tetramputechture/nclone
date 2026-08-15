"""Truncation-budget calculator: the retune (P1) and the two dead-code fixes.

Context in ``npp-rl/docs/RESEARCH_RESPONSE_generalization.md`` §1.1 and P1. The
measured facts these tests pin down:

* the surface-area term is the ONLY term that binds on any measured cell;
* the mine term has never contributed a frame (the caller passes a literal 0) and
  is deliberately retired rather than plumbed -- reviving it would ADD ~39,750
  frames to a PLATFORMING episode;
* ``MAX_TIME_IN_FRAMES`` never bounded anything despite being documented as a cap;
* the ``cost_to_completion`` floor must keep protecting long-route levels after
  the surface term is cut, since that protection is what
  ``COST_TIME_FACTOR 3.0 -> 3.5`` was bought for.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.constants import MAX_TIME_IN_FRAMES
from nclone.gym_environment.truncation_calculator import (
    BASE_TIME_PER_NODE,
    COST_TIME_FACTOR,
    HARD_TRUNCATION_CEILING,
    MIN_TRUNCATION_FRAMES,
    TIME_PER_MINE,
    TRUNCATION_MULTIPLIER,
    calculate_truncation_limit,
)

# Surface areas measured by npp-rl/scripts/probe_truncation_binding_term.py on
# mechanic_benchmark_v7_official. The limits they produced at the OLD
# BASE_TIME_PER_NODE of 20.0 are in the comments.
WALL_JUMP_T2_AREA = (2771.0 / (20.0 * 10.0)) ** 2       # -> 2771 frames
PLATFORMING_T2_AREA = (17190.0 / (20.0 * 10.0)) ** 2    # -> 17189 frames
PLATFORM_STAIRS_T2_AREA = (24794.0 / (20.0 * 10.0)) ** 2  # -> 24000 (hit ceiling)

# Validated planner win lengths (p90) from scripts/measure_route_budget.py.
WALL_JUMP_T2_WIN_P90 = 68
PLATFORMING_T2_WIN_P90 = 149


def test_mine_term_is_retired():
    """Passing a mine count must not change the budget by a single frame."""
    assert TIME_PER_MINE == 0.0
    a = calculate_truncation_limit(PLATFORMING_T2_AREA, 0)
    b = calculate_truncation_limit(PLATFORMING_T2_AREA, 53)
    assert a == b


def test_max_time_in_frames_is_not_a_cap():
    """The old expression folded MAX_TIME_IN_FRAMES in and it changed nothing.

    Re-asserts the equivalence directly so the constant cannot quietly become
    load-bearing again.
    """
    def old(limit: float) -> int:
        cap = min(HARD_TRUNCATION_CEILING, max(MAX_TIME_IN_FRAMES, limit))
        return int(np.clip(limit, MIN_TRUNCATION_FRAMES, cap))

    def new(limit: float) -> int:
        return int(np.clip(limit, MIN_TRUNCATION_FRAMES, HARD_TRUNCATION_CEILING))

    assert all(old(v) == new(v) for v in range(0, 40001, 7))
    # And a budget well above 4500 is genuinely granted, not clamped to it.
    assert calculate_truncation_limit(PLATFORM_STAIRS_T2_AREA, 0) > MAX_TIME_IN_FRAMES


def test_retune_cut_the_blocking_cells_about_fourfold():
    """PLATFORMING/T2 was 17,189 frames; the retune must land it near 4,300."""
    assert BASE_TIME_PER_NODE == 5.0
    lim = calculate_truncation_limit(PLATFORMING_T2_AREA, 0)
    assert 4000 <= lim <= 4700
    old = 17189
    assert 3.5 <= old / lim <= 4.5


def test_no_measured_cell_is_starved_by_the_retune():
    """Every cell must keep a wide margin over its validated winning route.

    This is the check that makes the retune safe, and it is the one the
    surface-term-vs-cost-floor argument cannot supply on its own.
    """
    for area, win_p90, name in (
        (WALL_JUMP_T2_AREA, WALL_JUMP_T2_WIN_P90, "WALL_JUMP/T2"),
        (PLATFORMING_T2_AREA, PLATFORMING_T2_WIN_P90, "PLATFORMING/T2"),
    ):
        lim = calculate_truncation_limit(area, 0)
        assert lim >= 8 * win_p90, f"{name} starved: {lim} < 8x{win_p90}"


def test_cost_floor_still_protects_a_long_route_level():
    """A small-area level with a long route keeps its route-keyed budget.

    This is what COST_TIME_FACTOR exists for, and cutting the surface term must
    not take it away.
    """
    area = 100.0  # tiny surface -> surface term is only sqrt(100)*5*10 = 500
    long_route_px = 500.0
    lim = calculate_truncation_limit(area, 0, cost_to_completion=long_route_px)
    assert lim == pytest.approx(long_route_px * COST_TIME_FACTOR, rel=1e-6)
    # ...and it genuinely dominates the surface term now, which is the point.
    assert lim > np.sqrt(area) * BASE_TIME_PER_NODE * TRUNCATION_MULTIPLIER


# FAR_OBJECTIVE/T2, measured: the SMALLEST surface area of any cell but the
# LONGEST validated route. It is the cell that proves a surface-area budget
# mis-ranks route length, and the one a surface-only cut would have starved.
FAR_OBJECTIVE_T2_AREA = 1200.0
FAR_OBJECTIVE_T2_CTC = 196.7
FAR_OBJECTIVE_T2_WIN_P90 = 477


def test_far_objective_is_not_starved_by_the_surface_cut():
    """The regression this pairing exists to prevent.

    With COST_TIME_FACTOR left at 3.5 the surface cut leaves FAR_OBJECTIVE/T2 at
    1,732 frames against a 477-frame optimal route -- 3.6x, the tightest cell in
    the benchmark by a factor of three, where it had been mid-pack. The floor
    must take over here.
    """
    lim = calculate_truncation_limit(
        FAR_OBJECTIVE_T2_AREA, 0, cost_to_completion=FAR_OBJECTIVE_T2_CTC)
    assert lim >= 8 * FAR_OBJECTIVE_T2_WIN_P90
    # ...and it is the FLOOR providing that, not the surface term.
    surface = (FAR_OBJECTIVE_T2_AREA ** 0.5) * BASE_TIME_PER_NODE * TRUNCATION_MULTIPLIER
    assert lim > surface


def test_the_cost_floor_does_not_undo_the_cut_on_the_blocking_cells():
    """Raising COST_TIME_FACTOR must not re-lengthen PLATFORMING/PLATFORM_STAIRS.

    Their routes are short and their surface areas huge, so the surface term must
    still dominate -- otherwise the floor repair would give back exactly what the
    surface cut was for.
    """
    for area, ctc, name in (
        (PLATFORMING_T2_AREA, 176.0 / 3.5, "PLATFORMING/T2"),
        (PLATFORM_STAIRS_T2_AREA, 69.0 / 3.5, "PLATFORM_STAIRS/T2"),
    ):
        with_floor = calculate_truncation_limit(area, 0, cost_to_completion=ctc)
        without = calculate_truncation_limit(area, 0)
        assert with_floor == without, f"{name}: the floor changed a surface-bound cell"


def test_floor_and_ceiling_are_respected():
    assert calculate_truncation_limit(1.0, 0) == MIN_TRUNCATION_FRAMES
    assert calculate_truncation_limit(1e9, 0) == HARD_TRUNCATION_CEILING
    assert (
        calculate_truncation_limit(1.0, 0, cost_to_completion=1e9)
        == HARD_TRUNCATION_CEILING
    )


def test_limit_is_monotonic_in_surface_area():
    areas = [10, 100, 1000, 10_000, 100_000]
    lims = [calculate_truncation_limit(a, 0) for a in areas]
    assert lims == sorted(lims)


def test_gamma_horizon_ratio_is_now_sane():
    """The Pardo et al. concern: the time limit must not dwarf the discount horizon.

    At gamma=0.997 the horizon is 1/(1-gamma) ~ 333 frames. The blocking cell was
    17,189 frames = 52x that. The retune must bring it under ~15x.
    """
    horizon = 1.0 / (1.0 - 0.997)
    lim = calculate_truncation_limit(PLATFORMING_T2_AREA, 0)
    assert lim / horizon < 15.0
