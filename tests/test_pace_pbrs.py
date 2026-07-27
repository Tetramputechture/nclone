"""PACE (split-time) PBRS: dense "ahead of / behind the reference run" shaping.

The completion-time objective is TERMINAL, so the agent learns whether it was
fast only hundreds of frames after the maneuver that decided it. Measured on
SI-000, closing the entire remaining 479->465 gap was worth 1.86% of episode
return (0.13% per frame) -- below what PPO resolves against variance, which is
what pinned every configuration at 479-507.

Pace shaping converts that into a per-frame signal using a table of
cost-to-go -> reference frame, and (being potential-based) telescopes to exactly
the total time saved versus the reference, so it cannot be farmed.
"""

import json

import pytest

from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


def _table(tmp_path):
    """Reference run: moves right one cell per 10 frames, 50 cells / 500 frames."""
    cells = {f"{i},0": float(i * 10) for i in range(51)}
    p = tmp_path / "pace.json"
    p.write_text(json.dumps({"cells": cells, "grid": 12, "total_frames": 500}))
    return str(p)


def _calc(tmp_path, weight=1.0, scale=60.0):
    cfg = RewardConfig(enable_pace_pbrs=True, pace_pbrs_weight=weight,
                       pace_scale=scale, pace_table_path=_table(tmp_path))
    c = RewardCalculator(reward_config=cfg)
    c.reset()
    return c


def test_reference_frame_lookup_by_position(tmp_path):
    c = _calc(tmp_path)
    c._load_pace_table()
    assert c._reference_frame_at_pos(0.0, 0.0) == pytest.approx(0.0)
    assert c._reference_frame_at_pos(25 * 12.0, 0.0) == pytest.approx(250.0)
    assert c._reference_frame_at_pos(50 * 12.0, 0.0) == pytest.approx(500.0)


def test_off_route_gives_no_pace_opinion(tmp_path):
    """A cell the reference never visited must yield NO shaping, rather than a
    fabricated pace that would reward or punish leaving the line arbitrarily."""
    c = _calc(tmp_path)
    c._load_pace_table()
    assert c._reference_frame_at_pos(0.0, 500 * 12.0) is None
    assert c._pace_shaping({"sim_frame": 10.0, "player_x": 0.0,
                            "player_y": 500 * 12.0}, 100.0) == 0.0


def test_near_route_snaps_to_a_neighbouring_cell(tmp_path):
    """Small lateral offsets still get a pace signal, using the EARLIEST
    neighbouring reference so the agent reads as slightly behind rather than
    being over-credited."""
    c = _calc(tmp_path)
    c._load_pace_table()
    v = c._reference_frame_at_pos(25 * 12.0, 12.0)   # cell (25,1); row 1 is empty
    assert v is not None
    assert v == pytest.approx(240.0)                 # min over (24..26, 0)
    assert v <= 250.0


def test_being_ahead_of_pace_pays_and_falling_behind_costs(tmp_path):
    c = _calc(tmp_path)
    c._load_pace_table()
    # Establish a baseline potential at the reference pace.
    on = {"sim_frame": 250.0, "player_x": 25 * 12.0, "player_y": 0.0}
    c._pace_shaping(on, 1.0)                                   # exactly on pace
    ahead = c._pace_shaping(
        {"sim_frame": 251.0, "player_x": 30 * 12.0, "player_y": 0.0}, 1.0)
    c2 = _calc(tmp_path)
    c2._load_pace_table()
    c2._pace_shaping(on, 1.0)
    behind = c2._pace_shaping(
        {"sim_frame": 260.0, "player_x": 25 * 12.0, "player_y": 0.0}, 1.0)
    assert ahead > 0.0 > behind


def test_loitering_is_penalised(tmp_path):
    """Standing still grows elapsed while progress does not."""
    c = _calc(tmp_path)
    c._load_pace_table()
    at = {"player_x": 10 * 12.0, "player_y": 0.0}
    c._pace_shaping({"sim_frame": 100.0, **at}, 1.0)
    r = c._pace_shaping({"sim_frame": 130.0, **at}, 1.0)   # 30 frames, no progress
    assert r < 0.0


def test_potential_telescopes_to_total_time_saved(tmp_path):
    """The dense signal must SUM to the true objective -- that is what makes it
    policy-invariant rather than a hackable bonus."""
    c = _calc(tmp_path, weight=1.0, scale=60.0)
    c._load_pace_table()
    # pbrs_gamma is a read-only property defaulting to LATTICE_PBRS_GAMMA = 1.0,
    # which is exactly the pure-telescoping setting this property depends on.
    assert c.config.pbrs_gamma == 1.0

    # A run that finishes 50 frames ahead of the 500-frame reference.
    # cells 0,12,25,37,50 -> reference frames 0,120,250,370,500
    traj = [(0, 0.0), (12, 100.0), (25, 200.0), (37, 320.0), (50, 450.0)]
    total = 0.0
    for cell, frame in traj:
        total += c._pace_shaping(
            {"sim_frame": frame, "player_x": cell * 12.0, "player_y": 0.0}, 1.0)

    phi_0 = -(traj[0][1] - 0.0) / 60.0
    phi_T = -(traj[-1][1] - 500.0) / 60.0
    assert total == pytest.approx(phi_T - phi_0, abs=1e-6)
    assert total == pytest.approx(50.0 / 60.0, abs=1e-6)   # 50 frames saved


def test_missing_table_is_inert_not_fatal(tmp_path):
    cfg = RewardConfig(enable_pace_pbrs=True, pace_table_path=str(tmp_path / "nope.json"))
    c = RewardCalculator(reward_config=cfg)
    c.reset()
    assert c._pace_shaping(
        {"sim_frame": 10.0, "player_x": 0.0, "player_y": 0.0}, 500.0) == 0.0


def test_disabled_by_default():
    assert RewardConfig().enable_pace_pbrs is False
