"""Unit tests for the v8 slope/curve/impact execution features.

Covers: exact per-tile surface normals (slopes + curves), the game_state dim
growth (53), the look-ahead landing-surface normal, and the impact-survivability
risk formula consumed by the impact PBRS.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from nclone.tile_surface_normals import build_normal_table, tile_surface_normal
from nclone.gym_environment import constants as C
from nclone.gym_environment.npp_environment import NppEnvironment


def test_tile_surface_normals_geometry():
    # empty / full-solid carry no slope/curve info
    assert tile_surface_normal(0) == (0.0, 0.0)
    assert tile_surface_normal(1) == (0.0, 0.0)
    # 45deg diagonal slopes (6-9) are unit normals at +/-45/135 deg
    for t in (6, 7, 8, 9):
        nx, ny = tile_surface_normal(t)
        assert abs((nx * nx + ny * ny) ** 0.5 - 1.0) < 1e-5
        assert abs(abs(nx) - 0.70710678) < 1e-4 and abs(abs(ny) - 0.70710678) < 1e-4
    # curves (10-17) are unit radial normals
    for t in range(10, 18):
        nx, ny = tile_surface_normal(t)
        assert abs((nx * nx + ny * ny) ** 0.5 - 1.0) < 1e-5
    # mild slope (18) is shallower than steep slope (26): larger |ny|
    assert abs(tile_surface_normal(18)[1]) > abs(tile_surface_normal(26)[1])
    table = build_normal_table()
    assert table.shape == (38, 2)
    # exactly the 28 slope+curve tiles (6-33) are nonzero
    assert int((np.abs(table).sum(1) > 0).sum()) == 28


def test_impact_risk_formula():
    # bare instance: _impact_risk only touches self for the optional
    # _impact_risk_safe_fraction override (defaults via getattr)
    _env = NppEnvironment.__new__(NppEnvironment)

    def f(*args):
        return NppEnvironment._impact_risk(_env, *args)
    # slow fall onto flat floor (ln=(0,-1)) -> safe
    assert f(0.0, -1.0, 0.0, 0.0, 1.0) == 0.0
    # fast fall onto an adjacent flat floor -> high risk
    fast = f(0.0, -1.0, 0.0, 0.0, 8.0)
    assert fast > 0.5
    # same fast fall but the surface is far away -> proximity discounts risk
    far = f(0.0, -1.0, 0.9, 0.0, 8.0)
    assert far < fast
    # no surface ahead (zero normal) -> never any risk
    assert f(0.0, 0.0, 1.0, 5.0, 5.0) == 0.0
    # risk stays within [0, 1]
    assert 0.0 <= f(0.0, -1.0, 0.0, 0.0, 100.0) <= 1.0


def test_game_state_layout():
    # contiguous, in-range indices for all v8 additions
    assert C.GAME_STATE_LANDING_NORMAL_IDX == 49
    assert C.GAME_STATE_LANDING_DIST_IDX == 51
    assert C.GAME_STATE_VERTICAL_TERMINAL_IDX == 52
    assert C.GAME_STATE_TIME_TO_CONTACT_IDX == 53
    assert C.GAME_STATE_FRAMES_AIRBORNE_IDX == 54
    assert C.GAME_STATE_ACCEL_IDX == 55
    assert C.GAME_STATE_SCANDOTS_IDX == 57
    assert C.GAME_STATE_SCANDOTS_DIM == 12
    # scandots end where the ghost-arc bundle begins (Audit II), then the v13
    # jump-now arcs + arc-phi deltas close out the vector
    assert (
        C.GAME_STATE_SCANDOTS_IDX + C.GAME_STATE_SCANDOTS_DIM
        == C.GAME_STATE_GHOST_ARC_IDX
    )
    assert (
        C.GAME_STATE_GHOST_ARC_IDX + C.GAME_STATE_GHOST_ARC_DIM
        == C.GAME_STATE_JUMP_ARC_IDX
    )
    assert (
        C.GAME_STATE_JUMP_ARC_IDX + C.GAME_STATE_JUMP_ARC_DIM
        == C.GAME_STATE_ARC_PHI_IDX
    )
    # v14/v16 tail chain: arc-phi -> tap -> wall -> passive-release arc.
    assert (
        C.GAME_STATE_ARC_PHI_IDX + C.GAME_STATE_ARC_PHI_DIM
        == C.GAME_STATE_TAP_ARC_IDX
    )
    assert (
        C.GAME_STATE_TAP_ARC_IDX + C.GAME_STATE_TAP_ARC_DIM
        == C.GAME_STATE_WALL_ARC_IDX
    )
    assert (
        C.GAME_STATE_WALL_ARC_IDX + C.GAME_STATE_WALL_ARC_DIM
        == C.GAME_STATE_PASSIVE_ARC_IDX
    )
    assert (
        C.GAME_STATE_PASSIVE_ARC_IDX + C.GAME_STATE_PASSIVE_ARC_DIM
        == C.GAME_STATE_CHANNELS
    )
    assert C.GAME_STATE_CHANNELS == 261


def test_scandots_helper_shape():
    # the scandots helper returns the declared flat width
    env = NppEnvironment.__new__(NppEnvironment)
    env._cached_tile_grid = None  # no grid -> all-ones fallback
    out = env._compute_scandots(1)
    assert out.shape == (C.GAME_STATE_SCANDOTS_DIM,)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


if __name__ == "__main__":
    test_tile_surface_normals_geometry()
    test_impact_risk_formula()
    test_game_state_layout()
    test_scandots_helper_shape()
    print("PASS slope/impact feature tests")
