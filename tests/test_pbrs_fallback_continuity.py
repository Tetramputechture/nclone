"""P1-1 regression: PBRS must not punish switch activation on the
geometric-fallback path.

When the env's potential substitution does not commit (free-space source
unavailable, spawn not finite, exception), the raw per-phase A* distances
reach the reward calculator and the potential basis used to flip at switch
activation: Phi_pre = -d_switch/D0 (near 0 at the switch) vs
Phi_post = -d_exit/D0 (large negative) — a net-negative activation step
(repro: -2.40) with D0 covering only the switch leg.

The fix: the env exposes ``_potential_substituted`` in the reward-metadata
obs; when it is explicitly False the calculator composes the pre-switch basis
as ``dist_to_switch + dist(switch -> exit)`` (one continuous
cost-to-completion, D0 = full route). If the switch->exit leg is not
computable, the episode degrades to the HWM ratchet (PBRS off, hwm_alpha
un-forced) so it still has a dense progress signal.

Drives ``RewardCalculator`` directly with injected distances, the same
technique as test_reward_cycle_neutrality.py.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import R_SWITCH


# Fixture route: spawn -> switch leg 200 cost units, switch -> exit leg 150.
D_SPAWN_SWITCH = 200.0
D_SWITCH_EXIT = 150.0


def _obs(**kwargs):
    base = {
        "player_x": 0.0,
        "player_y": 0.0,
        "player_dead": False,
        "player_won": False,
        "player_airborn": False,
        "player_walled": False,
        "switch_activated": False,
        "switch_x": 1.0,
        "switch_y": 1.0,
        "exit_door_x": 2.0,
        "exit_door_y": 2.0,
        "_adjacency_graph": {},
        "level_data": None,
        "_graph_data": None,
        "gold_collected": 0,
        "total_gold": 0,
        "map_name": "fallback_fixture",
        # The env's substitution did NOT commit: raw per-phase distances.
        "_potential_substituted": False,
    }
    base.update(kwargs)
    return base


def _calc():
    cfg = RewardConfig(enable_lattice_pbrs=True)
    calc = RewardCalculator(reward_config=cfg)
    calc.reset()
    return calc


def _step(calc, dist_switch, dist_exit, prev_kw=None, **obs_kw):
    with patch.object(
        RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
    ):
        prev = _obs(**(prev_kw or {}))
        cur = _obs(**obs_kw)
        r = calc.calculate_reward(cur, prev, frames_executed=1)
    return r, calc.last_components


def _run_route(calc):
    """Spawn -> switch approach -> activation, with RAW per-phase distances.

    Pre-switch: dist_switch shrinks 200 -> 10; dist_exit is the (irrelevant
    pre-switch) ninja->exit distance. Activation step: switch_activated flips,
    dist_exit becomes the true switch->exit leg (150).
    Returns the activation step's (reward, components).
    """
    _step(calc, D_SPAWN_SWITCH, 350.0)  # episode init at spawn
    _step(calc, 100.0, 250.0)
    _step(calc, 10.0, 160.0)
    return _step(
        calc,
        5.0,
        D_SWITCH_EXIT,
        switch_activated=True,
        prev_kw={"switch_activated": False},
    )


def test_composed_basis_keeps_activation_step_continuous():
    calc = _calc()
    with patch.object(
        RewardCalculator, "_switch_to_exit_distance", return_value=D_SWITCH_EXIT
    ):
        r, comp = _run_route(calc)

    # D0 covers the FULL route (spawn->switch + switch->exit), not just the
    # switch leg.
    assert calc._pbrs_d0 == pytest.approx(D_SPAWN_SWITCH + D_SWITCH_EXIT)
    assert not calc._pbrs_hwm_fallback

    # The activation step's PBRS term is continuous: composed pre-basis
    # (10 + 150)/350 vs post-basis 150/350 — a small POSITIVE delta, not the
    # legacy basis flip (which paid ~ -w * (150/200 - 10/200) = -2.1).
    assert comp["pbrs_reward"] > -1e-9, (
        f"PBRS spiked negative on the activation step: {comp['pbrs_reward']}"
    )
    assert comp["pbrs_reward"] < 0.5
    # Net step reward includes the +R_SWITCH bonus and stays positive.
    assert r > 0.0
    # Sanity: without the fix the net step reward was ~R_SWITCH - 2.1 < 0.


def test_unfixed_basis_flip_would_be_negative_and_is_warned(caplog):
    """The legacy behavior (flag ABSENT: distances treated as substituted)
    still flips the basis — the new debug guard must warn about it once."""
    calc = _calc()

    def _step_legacy(dist_switch, dist_exit, **obs_kw):
        obs_kw.setdefault("prev_kw", None)
        prev_kw = obs_kw.pop("prev_kw")
        with patch.object(
            RewardCalculator, "_get_distances", return_value=(dist_switch, dist_exit)
        ):
            prev = _obs(**(prev_kw or {}))
            cur = _obs(**obs_kw)
            prev.pop("_potential_substituted")
            cur.pop("_potential_substituted")
            r = calc.calculate_reward(cur, prev, frames_executed=1)
        return r, calc.last_components

    _step_legacy(D_SPAWN_SWITCH, 350.0)
    _step_legacy(10.0, 160.0)
    with caplog.at_level(logging.WARNING):
        r, comp = _step_legacy(
            5.0,
            D_SWITCH_EXIT,
            switch_activated=True,
            prev_kw={"switch_activated": False},
        )
    # Legacy D0 = switch leg only; the flip is large and negative.
    assert calc._pbrs_d0 == pytest.approx(D_SPAWN_SWITCH)
    assert comp["pbrs_reward"] < -1.0
    assert any(
        "potential discontinuity at switch activation" in rec.message
        for rec in caplog.records
    ), "activation-step Phi-jump guard did not warn"


def test_uncomputable_exit_leg_degrades_to_hwm_ratchet():
    calc = _calc()
    # PBRS on but hwm_alpha force-zeroed by config — the degraded episode must
    # re-enable the ratchet explicitly.
    assert calc.config.hwm_alpha == 0.0
    with patch.object(
        RewardCalculator, "_switch_to_exit_distance", return_value=None
    ):
        _step(calc, D_SPAWN_SWITCH, 350.0)  # init: fallback decided here
        assert calc._pbrs_hwm_fallback
        r_prog, comp_prog = _step(calc, 100.0, 250.0)  # forward progress
        r_act, comp_act = _step(
            calc,
            5.0,
            D_SWITCH_EXIT,
            switch_activated=True,
            prev_kw={"switch_activated": False},
        )

    # PBRS term is fully disabled for the episode...
    assert comp_prog["pbrs_reward"] == 0.0
    assert comp_act["pbrs_reward"] == 0.0
    # ...the HWM ratchet is live instead (progress pays)...
    assert comp_prog["hwm_reward"] > 0.0
    # ...and the activation step carries no PBRS spike (switch bonus + HWM).
    assert r_act >= R_SWITCH * 0.5


def test_substituted_distances_keep_legacy_d0():
    """When the env's substitution committed, the pre-switch distance is
    already a cost-to-completion — D0 must NOT double-add the exit leg."""
    calc = _calc()
    with patch.object(
        RewardCalculator, "_switch_to_exit_distance", return_value=D_SWITCH_EXIT
    ):
        _step(calc, 350.0, 150.0, _potential_substituted=True)
    assert calc._pbrs_d0 == pytest.approx(350.0)
    assert calc._pbrs_exit_leg is None
    assert not calc._pbrs_hwm_fallback
