"""Validation of the v25 directional mine bearing + closing-speed obs.

The v24 fine mine crop (``ego_crop_mines_fine``) restored sub-tile mine geometry
but is a STILL FRAME: it says where the mines are, never whether the ninja is
closing on one. Measured after v24, mine-death fell to ~2% on the easy procgen
buckets yet stayed ~0.51 on OFFICIAL layouts, with only 0.52 of official episodes
reaching the switch at all — half die before the first gate, mostly on mines.

``mine_proximity`` supplies the missing motion-relative terms. Guards:
  1. No deadly mines / all mines out of range -> all-zero (the no-threat reading).
  2. Bearing is a unit vector pointing at the mine, in ninja-centered coords.
  3. proximity is 1.0 at the 14px kill disc and 0.0 at the range boundary.
  4. closing is positive when moving toward the mine, negative when away, ~0
     when tangential — the term a still frame cannot carry.
  5. ttc fires only while closing; receding or tangential motion reads 0.
  6. Mines are ordered nearest-first and the vector is clipped to [-1, 1] so it
     satisfies its Box.
"""

from __future__ import annotations

import types

import numpy as np

from nclone.gym_environment.constants import (
    MINE_PROXIMITY_DIM,
    MINE_PROXIMITY_FEATURES,
    MINE_PROXIMITY_K,
    MINE_PROXIMITY_KILL_RADIUS_PX,
    MINE_PROXIMITY_RANGE_PX,
    MINE_PROXIMITY_SPEED_NORM_PX,
)
from nclone.gym_environment.npp_environment import NppEnvironment

NINJA = (100.0, 100.0)


def _env(mines, velocity=(0.0, 0.0)):
    """Bind the method under test onto a stub carrying only what it reads."""
    env = types.SimpleNamespace(
        nplay_headless=types.SimpleNamespace(ninja_velocity=lambda: velocity),
        _deadly_xy=np.asarray(mines, dtype=np.float64).reshape(-1, 2),
    )
    env._get_deadly_mine_xy = lambda: env._deadly_xy
    env._compute_mine_proximity = types.MethodType(
        NppEnvironment._compute_mine_proximity, env
    )
    return env


def _slot(vec, i):
    base = i * MINE_PROXIMITY_FEATURES
    return vec[base : base + MINE_PROXIMITY_FEATURES]


def test_no_mines_is_all_zero():
    v = _env([])._compute_mine_proximity(*NINJA)
    assert v.shape == (MINE_PROXIMITY_DIM,)
    assert np.all(v == 0.0)


def test_out_of_range_mine_is_all_zero():
    far = (NINJA[0] + MINE_PROXIMITY_RANGE_PX + 50.0, NINJA[1])
    v = _env([far])._compute_mine_proximity(*NINJA)
    assert np.all(v == 0.0), "a mine past the reaction window is not a threat"


def test_bearing_is_a_unit_vector_toward_the_mine():
    # Mine directly to the right, well inside range.
    v = _env([(NINJA[0] + 40.0, NINJA[1])])._compute_mine_proximity(*NINJA)
    ux, uy = _slot(v, 0)[0], _slot(v, 0)[1]
    assert abs(ux - 1.0) < 1e-5 and abs(uy) < 1e-5
    # Mine directly below (screen coords: +y is down).
    v = _env([(NINJA[0], NINJA[1] + 40.0)])._compute_mine_proximity(*NINJA)
    ux, uy = _slot(v, 0)[0], _slot(v, 0)[1]
    assert abs(ux) < 1e-5 and abs(uy - 1.0) < 1e-5


def test_proximity_saturates_at_the_kill_disc_and_zeroes_at_range():
    at_disc = (NINJA[0] + MINE_PROXIMITY_KILL_RADIUS_PX, NINJA[1])
    v = _env([at_disc])._compute_mine_proximity(*NINJA)
    assert abs(_slot(v, 0)[2] - 1.0) < 1e-5, "contact must read maximum proximity"

    at_range = (NINJA[0] + MINE_PROXIMITY_RANGE_PX, NINJA[1])
    v = _env([at_range])._compute_mine_proximity(*NINJA)
    assert abs(_slot(v, 0)[2]) < 1e-5, "the range boundary must read no proximity"

    # Monotone in between.
    prev = None
    for d in (20.0, 40.0, 60.0, 80.0):
        v = _env([(NINJA[0] + d, NINJA[1])])._compute_mine_proximity(*NINJA)
        p = _slot(v, 0)[2]
        if prev is not None:
            assert p < prev, "proximity must fall as the mine gets further away"
        prev = p


def test_closing_speed_sign_distinguishes_approach_from_drift():
    mine = (NINJA[0] + 40.0, NINJA[1])
    # Moving right, straight at the mine.
    v = _env([mine], velocity=(3.0, 0.0))._compute_mine_proximity(*NINJA)
    closing = _slot(v, 0)[3]
    assert closing > 0.0
    assert abs(closing - 3.0 / MINE_PROXIMITY_SPEED_NORM_PX) < 1e-5
    # Moving left, away from it.
    v = _env([mine], velocity=(-3.0, 0.0))._compute_mine_proximity(*NINJA)
    assert _slot(v, 0)[3] < 0.0
    # Moving straight down: tangential, no closing component.
    v = _env([mine], velocity=(0.0, 3.0))._compute_mine_proximity(*NINJA)
    assert abs(_slot(v, 0)[3]) < 1e-5


def test_ttc_only_fires_while_closing():
    mine = (NINJA[0] + 40.0, NINJA[1])
    approaching = _env([mine], velocity=(3.0, 0.0))._compute_mine_proximity(*NINJA)
    receding = _env([mine], velocity=(-3.0, 0.0))._compute_mine_proximity(*NINJA)
    tangential = _env([mine], velocity=(0.0, 3.0))._compute_mine_proximity(*NINJA)
    assert _slot(approaching, 0)[4] > 0.0
    assert _slot(receding, 0)[4] == 0.0
    assert _slot(tangential, 0)[4] == 0.0
    # Faster approach = more imminent.
    faster = _env([mine], velocity=(6.0, 0.0))._compute_mine_proximity(*NINJA)
    assert _slot(faster, 0)[4] > _slot(approaching, 0)[4]


def test_inside_the_kill_disc_is_max_threat_without_a_nan_bearing():
    v = _env([NINJA], velocity=(1.0, 1.0))._compute_mine_proximity(*NINJA)
    s = _slot(v, 0)
    assert np.all(np.isfinite(v))
    assert abs(s[0]) < 1e-9 and abs(s[1]) < 1e-9, "bearing is undefined at d=0"
    assert abs(s[2] - 1.0) < 1e-9


def test_mines_are_ordered_nearest_first_and_extra_slots_stay_zero():
    mines = [
        (NINJA[0] + 80.0, NINJA[1]),
        (NINJA[0] + 20.0, NINJA[1]),
        (NINJA[0] + 50.0, NINJA[1]),
    ]
    v = _env(mines)._compute_mine_proximity(*NINJA)
    proximities = [_slot(v, i)[2] for i in range(3)]
    assert proximities == sorted(proximities, reverse=True), (
        "nearest mine must occupy slot 0"
    )
    for i in range(3, MINE_PROXIMITY_K):
        assert np.all(_slot(v, i) == 0.0)


def test_more_mines_than_slots_keeps_only_the_nearest_k():
    mines = [(NINJA[0] + 10.0 + 5.0 * i, NINJA[1]) for i in range(MINE_PROXIMITY_K + 4)]
    v = _env(mines, velocity=(3.0, 0.0))._compute_mine_proximity(*NINJA)
    assert np.all(np.abs(v) <= 1.0 + 1e-6), "the vector must satisfy its [-1, 1] Box"
    # Every slot filled, none zeroed out.
    for i in range(MINE_PROXIMITY_K):
        assert _slot(v, i)[2] > 0.0


def test_extreme_velocity_stays_inside_the_box():
    mine = (NINJA[0] + 30.0, NINJA[1])
    v = _env([mine], velocity=(1e4, -1e4))._compute_mine_proximity(*NINJA)
    assert np.all(np.abs(v) <= 1.0 + 1e-6)
    assert np.all(np.isfinite(v))
