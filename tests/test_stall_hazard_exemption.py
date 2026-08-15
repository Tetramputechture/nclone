"""Waiting for a patrol must not be taxed as dithering.

The stall penalty fires when cost-to-go has not improved for
``stall_patience_frames`` while the ninja stays within ``stall_radius_px``.
Holding position until a drone clears produces EXACTLY that signature, and the
un-exempted cost is decisive rather than cosmetic: measured on the gamma-1 DRONE
bucket, patrol drones run at 1.143 px/frame over circuits of 31-855 px, so a
worst-case wait for a safe window is ~600-700 frames -- about -9.3 against
R_EXIT = 10. Past ~2,100 frames of episode length the per-step penalty exceeds
the entire DISCOUNTED completion reward, so the policy's best response becomes
"rush the drone or die", which is the opposite of the skill being taught.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.constants import (
    VISITATION_CELL_PX,
    VISITATION_ORIGIN_PX,
)
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig

H, W = 46, 84


def _calc(**kw):
    cfg = RewardConfig(enable_stall_penalty=True, **kw)
    return RewardCalculator(cfg) if _takes_cfg() else RewardCalculator(config=cfg)


def _takes_cfg():
    import inspect

    return "config" not in inspect.signature(RewardCalculator.__init__).parameters


def _ttl(threat_cells=()):
    """A threat_ttl block; 1.0 everywhere (safe) except the named cells."""
    a = np.ones((4, H, W), dtype=np.float32)
    for cy, cx in threat_cells:
        a[1, cy, cx] = 0.0        # certain-tier time-to-lethal = lethal now
    return a


def _cell_of(px, py):
    return (int((py - VISITATION_ORIGIN_PX) / VISITATION_CELL_PX),
            int((px - VISITATION_ORIGIN_PX) / VISITATION_CELL_PX))


POS = (300.0, 300.0)


def test_a_live_threat_nearby_marks_the_ninja_as_blocked():
    c = _calc()
    cy, cx = _cell_of(*POS)
    assert c._hazard_blocked({"threat_ttl": _ttl([(cy, cx)])}, POS) is True


def test_a_clear_field_does_not_exempt():
    c = _calc()
    assert c._hazard_blocked({"threat_ttl": _ttl()}, POS) is False


def test_a_threat_beyond_the_radius_does_not_exempt():
    """48 px default -> 4 cells. A threat 20 cells away is not pinning anyone."""
    c = _calc()
    cy, cx = _cell_of(*POS)
    assert c._hazard_blocked({"threat_ttl": _ttl([(cy, cx + 20)])}, POS) is False


def test_no_threat_observation_means_no_exemption():
    """Without threat_ttl the penalty must behave EXACTLY as it did before --
    the exemption may not silently disable the term on non-hazard levels."""
    c = _calc()
    assert c._hazard_blocked({}, POS) is False
    assert c._hazard_blocked({"threat_ttl": None}, POS) is False


def test_the_origin_offset_is_applied():
    """threat_ttl is border-stripped; forgetting the 24 px origin shifts the
    lookup two cells and exempts the wrong place."""
    c = _calc()
    cy, cx = _cell_of(*POS)
    naive_cy, naive_cx = int(POS[1] / VISITATION_CELL_PX), int(POS[0] / VISITATION_CELL_PX)
    assert (cy, cx) != (naive_cy, naive_cx), "test is vacuous if they agree"
    # a threat at the NAIVE cell only, far enough that the correct cell's
    # radius does not reach it, must not trigger
    assert c._hazard_blocked({"threat_ttl": _ttl([(naive_cy + 12, naive_cx + 12)])},
                             POS) is False


def test_the_exemption_budget_is_finite():
    """Camping beside a drone must not buy permanent immunity."""
    c = _calc(stall_hazard_exempt_max_frames=100)
    cy, cx = _cell_of(*POS)
    obs = {"threat_ttl": _ttl([(cy, cx)])}
    assert c._hazard_blocked(obs, POS) is True
    c._stall_hazard_exempt_frames = 100
    assert c._hazard_blocked(obs, POS) is False


def test_it_can_be_switched_off():
    c = _calc(stall_hazard_exempt=False)
    cy, cx = _cell_of(*POS)
    assert c._hazard_blocked({"threat_ttl": _ttl([(cy, cx)])}, POS) is False


def test_a_malformed_threat_block_never_raises():
    """Shaping must not be able to take a 50M-step run down."""
    c = _calc()
    for bad in ("nonsense", 5, np.ones((2, 2)), np.ones((2, H, W))):
        assert c._hazard_blocked({"threat_ttl": bad}, POS) is False
