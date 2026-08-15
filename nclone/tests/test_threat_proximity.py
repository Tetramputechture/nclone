"""The shooter telegraph (`threat_proximity`) — plan §3.3.

WHY THIS KEY EXISTS, and why the field is not enough. The per-entity observation
audit measured, on maps that isolate each hazard, what fraction of the threat
field departs from its "no threat" default over 150 frames of active play:

    DRONE_ZAP      7.5%      TURRET           0.0%
    MINI_DRONE    12.3%      DRONE_CHAINGUN   0.0%
    DRONE_CHASER   6.2%
    FLOOR_GUARD    1.2%

The zeroes are not a scatter bug. A turret's body is a harmless wall fixture and
its beam is HITSCAN: it exists on the frame it kills. No number of forecast
horizons makes that learnable, because there is nothing to forecast until it is
already lethal.

What is learnable is the WIND-UP, and every shooter has one. These tests pin the
mapping from each entity's own state machine onto `imminence`, using the real
constants (turret charges to 90 then fires for 15; rocket charges 15; shooter
drones aim for 53). A made-up countdown would be worse than none — it would look
learnable and teach the wrong timing — so each value below is checked against the
state machine it claims to read.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.constants import (
    THREAT_CLASS_BEAM,
    THREAT_CLASS_CONTACT,
    THREAT_CLASS_HOMING,
    THREAT_CLASS_PROJECTILE,
    THREAT_CLASS_OF_TYPE,
    THREAT_PROXIMITY_DIM,
    THREAT_PROX_CLASSES,
    THREAT_PROX_FEATURES,
)


class _Ent:
    """Minimal stand-in carrying only what the telegraph reads."""

    def __init__(self, **kw):
        self.active = True
        self.xpos = 0.0
        self.ypos = 0.0
        self.state = 0
        for k, v in kw.items():
            setattr(self, k, v)


def _imminence(env, ent, type_id):
    return env._threat_imminence(ent, type_id)


@pytest.fixture(scope="module")
def env():
    """A bare environment instance; only the pure mapping is exercised."""
    from nclone.gym_environment.npp_environment import NppEnvironment

    return NppEnvironment.__new__(NppEnvironment)


# --- turret: SEARCHING -> TRACKING(charge/90) -> FIRING -> RECOVERING -------


def test_turret_searching_is_silent(env):
    assert _imminence(env, _Ent(state=0, charge=0.0), 19) == 0.0


@pytest.mark.parametrize(
    "charge,expected", [(0.0, 0.0), (45.0, 0.5), (90.0, 1.0), (200.0, 1.0)]
)
def test_turret_tracking_reports_charge_fraction(env, charge, expected):
    """TRACKING is the whole point: the countdown to a shot that is hitscan."""
    e = _Ent(state=1, charge=charge, CHARGE_THRESHOLD=90.0)
    assert _imminence(env, e, 19) == pytest.approx(expected)


def test_turret_firing_is_maximal(env):
    assert _imminence(env, _Ent(state=2, charge=0.0), 19) == 1.0


def test_turret_recovering_reports_a_safe_window(env):
    """RECOVERING must read 0.0, and that is a POSITIVE signal, not an absence.

    The turret has just fired and cannot fire again for 15 frames, so this is
    precisely when crossing its line is free. Reporting anything else would hide
    the one moment the agent can exploit.
    """
    assert _imminence(env, _Ent(state=3, charge=14.0), 19) == 0.0


# --- rocket: IDLE -> CHARGING(15) -> FLYING ---------------------------------


def test_rocket_idle_is_silent(env):
    assert _imminence(env, _Ent(state=0), 18) == 0.0


@pytest.mark.parametrize("charge,expected", [(0.0, 0.0), (7.5, 0.5), (15.0, 1.0)])
def test_rocket_charging_counts_down_the_telegraph(env, charge, expected):
    e = _Ent(state=1, charge=charge, LAUNCH_DELAY=15.0)
    assert _imminence(env, e, 18) == pytest.approx(expected)


def test_rocket_flying_is_maximal(env):
    """Once FLYING the projectile homes every frame — it is already the worst case."""
    assert _imminence(env, _Ent(state=2), 18) == 1.0


# --- shooter drones: IDLE -> AIMING(53) -> FIRING -> COOLDOWN ---------------


@pytest.mark.parametrize("type_id", [12, 13])
@pytest.mark.parametrize("timer,expected", [(0.0, 0.0), (26.5, 0.5), (53.0, 1.0)])
def test_shooter_drone_aiming_counts_the_charge(env, type_id, timer, expected):
    e = _Ent(state=1, timer=timer, CHARGE_DURATION=53.0)
    assert _imminence(env, e, type_id) == pytest.approx(expected)


@pytest.mark.parametrize("type_id", [12, 13])
def test_shooter_drone_cooldown_is_silent(env, type_id):
    assert _imminence(env, _Ent(state=3, timer=10.0), type_id) == 0.0


# --- entities with no wind-up ----------------------------------------------


@pytest.mark.parametrize("type_id", [14, 26, 15, 25, 27, 22, 23])
def test_entities_without_a_telegraph_report_zero(env, type_id):
    """Contact/homing hazards and the dual laser are lethal or not, with no
    wind-up. The FIELD already says which, so imminence must not invent one."""
    assert _imminence(env, _Ent(state=1, charge=50.0, timer=50.0), type_id) == 0.0


# --- class partition -------------------------------------------------------


def test_every_mapped_type_lands_in_exactly_one_class():
    for type_id, cls in THREAT_CLASS_OF_TYPE.items():
        assert 0 <= cls < THREAT_PROX_CLASSES, type_id


def test_the_super_classes_cover_the_shooters_and_the_movers():
    assert THREAT_CLASS_OF_TYPE[19] == THREAT_CLASS_BEAM
    assert THREAT_CLASS_OF_TYPE[13] == THREAT_CLASS_BEAM
    assert THREAT_CLASS_OF_TYPE[12] == THREAT_CLASS_BEAM
    assert THREAT_CLASS_OF_TYPE[18] == THREAT_CLASS_PROJECTILE
    assert THREAT_CLASS_OF_TYPE[15] == THREAT_CLASS_HOMING
    # The evil ninja is homing, not contact: it pursues along a recorded path.
    assert THREAT_CLASS_OF_TYPE[22] == THREAT_CLASS_HOMING
    assert THREAT_CLASS_OF_TYPE[14] == THREAT_CLASS_CONTACT


def test_width_is_fixed_and_independent_of_the_corpus():
    """4x8, whatever entities a level happens to contain.

    The per-entity alternative would be 21 bespoke slots against a coverage curve
    where no single type unlocks more than ~6 points of corpus. What the width is
    matters far less than that it does not depend on the level; this pins the
    invariant, and 32 is the current value (5 dims of geometry + telegraph, 3 of
    threat motion — see constants.py).
    """
    assert THREAT_PROXIMITY_DIM == THREAT_PROX_CLASSES * THREAT_PROX_FEATURES == 32
