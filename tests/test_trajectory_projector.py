"""Ghost-arc TrajectoryProjector (Audit II P0-1) — parity + contract tests.

Verifies the native ballistic projection against the REAL Python simulator
stepped frame-by-frame (contact position within collision resolution, tau
within a frame), the bang-bang envelope ordering of the u = -1/0/+1 arcs,
deadly-mine contact classification, bundle shape/normalization, and the
game_state obs contract (96 dims with the 27-dim bundle appended).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from nclone.constants.physics_constants import MAX_JUMP_DURATION  # noqa: F401
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_GHOST_ARC_DIM,
    GAME_STATE_GHOST_ARC_IDX,
    GAME_STATE_GHOST_ARC_PER_ARC,
    GHOST_ARC_MAX_FRAMES,
)

nclone_native = pytest.importorskip("nclone_native")

if not hasattr(nclone_native, "TrajectoryProjector"):
    pytest.skip(
        "nclone_native build predates TrajectoryProjector", allow_module_level=True
    )


def _empty_map_data() -> list[int]:
    from nclone.map_generation.map import Map

    return [int(v) for v in Map().map_data()]


def _projector(md=None):
    md = md or _empty_map_data()
    proj = nclone_native.TrajectoryProjector.load(md)
    proj.set_deadly_mines([], [])
    return proj


def _sim_fall(md, x0, y0, vx0, vy0, hor_input, max_frames=200):
    """Step the PYTHON sim from a teleported airborne state to first ground
    contact; returns (x, y, frames)."""
    from nclone.nplay_headless import NPlayHeadless

    py = NPlayHeadless(enable_debug_overlay=False)
    py.load_map_from_map_data(list(md))
    ninja = py.sim.ninja
    ninja.xpos, ninja.ypos = x0, y0
    ninja.xspeed, ninja.yspeed = vx0, vy0
    ninja.airborn = True
    ninja.state = 4
    for f in range(max_frames):
        py.tick(hor_input, 0)
        if not ninja.airborn or ninja.state in (0, 1, 2):
            return ninja.xpos, ninja.ypos, f + 1
    return ninja.xpos, ninja.ypos, max_frames


class TestParityVsSim:
    def test_free_fall_parity(self):
        """u=0 arc lands where the real sim lands (within collision resolution)."""
        md = _empty_map_data()
        proj = _projector(md)
        x0, y0, vx0, vy0 = 300.0, 150.0, 1.5, 0.0

        cls, cx, cy, nx, ny, tau, imp, thr, cw, cvx, cvy, mclr, gclr = proj.project(
            x0, y0, vx0, vy0, 0, 0, GHOST_ARC_MAX_FRAMES
        )
        sx, sy, sframes = _sim_fall(md, x0, y0, vx0, vy0, 0)

        assert cls == 0, f"expected floor contact, got class {cls}"
        # Collision resolution: the sim depenetrates + runs 4 substeps; allow a
        # few px and one frame of slack.
        assert abs(cx - sx) < 6.0, (cx, sx)
        assert abs(cy - sy) < 6.0, (cy, sy)
        assert abs(tau - sframes) <= 2.0, (tau, sframes)
        assert ny < -0.7  # flat floor normal points up
        assert imp > 0.0 and thr > 0.0
        # v15: contact velocity is the falling velocity at impact — downward
        # (cvy > 0 in screen coords) on a free fall onto the floor.
        assert cvy > 0.0, cvy
        assert imp == pytest.approx(-(nx * cvx + ny * cvy), abs=1e-6)
        # v15: no mines on this map -> clearance is infinite.
        assert mclr == float("inf"), mclr
        # v16 (feature C): no goal set on this bare projector -> goal clearance
        # is infinite (reads 0 in the bundle).
        assert gclr == float("inf"), gclr

    def test_side_input_parity(self):
        """u=+1 arc tracks the sim holding right."""
        md = _empty_map_data()
        proj = _projector(md)
        x0, y0, vx0, vy0 = 300.0, 150.0, 0.0, 0.0

        cls, cx, cy, *_rest = proj.project(x0, y0, vx0, vy0, 1, 0, GHOST_ARC_MAX_FRAMES)
        sx, sy, _ = _sim_fall(md, x0, y0, vx0, vy0, 1)
        assert cls == 0
        assert abs(cx - sx) < 6.0, (cx, sx)
        assert abs(cy - sy) < 6.0, (cy, sy)

    def test_bang_bang_envelope_ordering(self):
        """u=-1 lands left of u=0 lands left of u=+1 (the reachable interval)."""
        proj = _projector()
        x0, y0 = 400.0, 100.0
        xs = []
        for u in (-1, 0, 1):
            _cls, cx, *_ = proj.project(x0, y0, 0.0, 0.0, u, 0, GHOST_ARC_MAX_FRAMES)
            xs.append(cx)
        assert xs[0] < xs[1] < xs[2], xs
        # A full fall's interval is wide — tens of px even from mid-height.
        assert xs[2] - xs[0] > 50.0, xs


class TestMineAndBundle:
    def test_deadly_mine_contact(self):
        md = _empty_map_data()
        proj = nclone_native.TrajectoryProjector.load(md)
        x0, y0 = 300.0, 150.0
        # Mine directly in the fall path, well above the floor.
        proj.set_deadly_mines([300.0], [300.0])
        cls, cx, cy, nx, ny, tau, imp, thr, cw, cvx, cvy, mclr, gclr = proj.project(
            x0, y0, 0.0, 0.0, 0, 0, GHOST_ARC_MAX_FRAMES
        )
        assert cls == 3, f"expected mine contact, got {cls}"
        assert abs(cy - 300.0) < 20.0
        assert cw == 0
        # v15: mine-contact arc's closest approach is within the kill radius.
        assert mclr <= 14.0 + 1e-6, mclr

    def test_bundle_shape_and_bounds(self):
        proj = _projector()
        bundle = proj.project_bundle(400.0, 100.0, 0.5, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        assert len(bundle) == GAME_STATE_GHOST_ARC_DIM
        arr = np.asarray(bundle)
        assert np.isfinite(arr).all()
        assert (arr >= -1.0).all() and (arr <= 1.0).all()
        # 3 arcs x 15 dims (v16); every arc on the empty map is a floor contact:
        # mine/none flags 0, clear width > 0, ny < 0, surface_class = +1 (floor),
        # mine_proximity = 0 (no mines), goal_approach/goal_hit = 0 (no goal set).
        for a in range(3):
            seg = arr[a * GAME_STATE_GHOST_ARC_PER_ARC:(a + 1) * GAME_STATE_GHOST_ARC_PER_ARC]
            assert seg[4] == 0.0 and seg[5] == 0.0  # not mine, not none
            assert seg[3] < 0.0  # floor normal
            assert seg[6] > 0.0  # clear width
            assert seg[8] > 0.0  # tau
            assert seg[11] == 1.0  # v15 surface_class: floor
            assert seg[12] == 0.0  # v15 mine_proximity: no mines -> 0
            assert seg[13] == 0.0  # v16 goal_approach: no goal set -> 0
            assert seg[14] == 0.0  # v16 goal_hit: no goal set -> 0

    def test_mine_proximity_graded_on_near_miss(self):
        # A mine offset ~25 px to the side of the straight-down u=0 fall path:
        # NOT a contact (kill radius 14), but the arc's closest approach lands in
        # the graded proximity zone -> bundle mine_proximity in (0, 1).
        md = _empty_map_data()
        proj = nclone_native.TrajectoryProjector.load(md)
        x0, y0 = 300.0, 150.0
        proj.set_deadly_mines([325.0], [300.0])  # 25 px lateral from the fall column
        # v16: min_mine_clearance is now index [11] (goal clearance at [12]) — the
        # trailing-splat would grab goal, so index [11] explicitly.
        contact = proj.project(x0, y0, 0.0, 0.0, 0, 0, GHOST_ARC_MAX_FRAMES)
        cls, mclr = contact[0], contact[11]
        assert cls != 3, "near-miss must not be a mine contact"
        assert 14.0 < mclr < 48.0, mclr  # inside the graded threading zone
        bundle = proj.project_bundle(x0, y0, 0.0, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        # u=0 is the middle arc -> index 1 (offset 1*PER_ARC), proximity at +12.
        prox = bundle[1 * GAME_STATE_GHOST_ARC_PER_ARC + 12]
        assert 0.0 < prox < 1.0, prox

    def test_goal_approach_and_hit(self):
        # v16 (feature C): an arc whose flight passes through the active goal
        # reports goal_hit=1 + goal_approach=1; an arc that only comes NEAR the
        # goal reports a graded goal_approach in (0, 1) with goal_hit=0; with no
        # goal set both read 0.
        md = _empty_map_data()
        proj = nclone_native.TrajectoryProjector.load(md)
        proj.set_deadly_mines([], [])
        x0, y0 = 300.0, 150.0
        PA = GAME_STATE_GHOST_ARC_PER_ARC

        # No goal -> goal dims are 0.
        b0 = proj.project_bundle(x0, y0, 0.0, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        for a in range(3):
            assert b0[a * PA + 13] == 0.0 and b0[a * PA + 14] == 0.0

        # Goal ON the straight-down u=0 fall column -> the middle arc hits it.
        proj.set_goal(300.0, 300.0)
        bh = proj.project_bundle(x0, y0, 0.0, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        mid = 1 * PA
        assert bh[mid + 14] == 1.0, "u=0 arc through the goal must set goal_hit"
        assert bh[mid + 13] == 1.0, "hit arc has graded approach saturated to 1"

        # Goal ~40 px to the side of the fall column -> near miss: graded approach
        # in (0, 1), no hit.
        proj.set_goal(340.0, 300.0)
        bn = proj.project_bundle(x0, y0, 0.0, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        approach = bn[mid + 13]
        assert 0.0 < approach < 1.0, approach
        assert bn[mid + 14] == 0.0, "near miss must not set goal_hit"

        # clear_goal restores zeros.
        proj.clear_goal()
        bc = proj.project_bundle(x0, y0, 0.0, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        assert bc[mid + 13] == 0.0 and bc[mid + 14] == 0.0

    def test_clear_width_blocked_by_mines(self):
        md = _empty_map_data()
        proj = nclone_native.TrajectoryProjector.load(md)
        x0, y0 = 400.0, 100.0
        # No mines: wide clear surface on the empty floor. clear_width is at
        # index 8 (not the tuple tail — v15 appended contact vx/vy at 9/10).
        proj.set_deadly_mines([], [])
        cw_open = proj.project(x0, y0, 0.0, 0.0, 0, 0, GHOST_ARC_MAX_FRAMES)[8]
        # Mines flanking the landing point one tile away on each side (floor
        # y is the bottom border): clear width collapses.
        _cls, cx, cy, *_rest = proj.project(x0, y0, 0.0, 0.0, 0, 0, GHOST_ARC_MAX_FRAMES)
        proj.set_deadly_mines([cx - 24.0, cx + 24.0], [cy + 8.0, cy + 8.0])
        cw_mined = proj.project(x0, y0, 0.0, 0.0, 0, 0, GHOST_ARC_MAX_FRAMES)[8]
        assert cw_mined < cw_open, (cw_mined, cw_open)

    def test_projection_is_fast(self):
        proj = _projector()
        proj.project_bundle(400.0, 100.0, 0.5, 0.0, 0, GHOST_ARC_MAX_FRAMES)  # warm
        t0 = time.perf_counter()
        n = 200
        for _ in range(n):
            proj.project_bundle(400.0, 100.0, 0.5, 0.0, 0, GHOST_ARC_MAX_FRAMES)
        per = (time.perf_counter() - t0) / n * 1e6
        # Audit estimate ~1-4 us; generous CI bound.
        assert per < 200.0, f"{per:.1f} us per 3-arc bundle"


class TestObsContract:
    def test_game_state_channels(self):
        # v16: each arc row is 15 dims (v15 13 + goal_approach[13]/goal_hit[14]);
        # committed(45) + jump-now(45) + arc-phi(6) + tap(45) + wall(45) +
        # passive-release arc(6) -> game_state = 261.
        assert GAME_STATE_CHANNELS == 261
        assert GAME_STATE_GHOST_ARC_IDX == 69
        assert GAME_STATE_GHOST_ARC_DIM == 45
        from nclone.gym_environment.constants import (
            GAME_STATE_JUMP_ARC_IDX,
            GAME_STATE_JUMP_ARC_DIM,
            GAME_STATE_ARC_PHI_IDX,
            GAME_STATE_ARC_PHI_DIM,
            GAME_STATE_TAP_ARC_IDX,
            GAME_STATE_WALL_ARC_IDX,
            GAME_STATE_WALL_ARC_DIM,
            GAME_STATE_PASSIVE_ARC_IDX,
            GAME_STATE_PASSIVE_ARC_DIM,
            GAME_STATE_GHOST_ARC_PER_ARC,
        )

        assert GAME_STATE_GHOST_ARC_PER_ARC == 15
        assert GAME_STATE_JUMP_ARC_IDX == 114
        assert GAME_STATE_JUMP_ARC_DIM == 45
        assert GAME_STATE_ARC_PHI_IDX == 159
        assert GAME_STATE_ARC_PHI_DIM == 6
        assert GAME_STATE_TAP_ARC_IDX == 165
        assert GAME_STATE_WALL_ARC_IDX == 210
        assert GAME_STATE_WALL_ARC_DIM == 45
        # v16 batch A: passive arc rides directly after the wall row.
        assert GAME_STATE_PASSIVE_ARC_IDX == 210 + GAME_STATE_WALL_ARC_DIM
        assert GAME_STATE_PASSIVE_ARC_IDX == 255
        assert GAME_STATE_PASSIVE_ARC_DIM == 6
        assert GAME_STATE_PASSIVE_ARC_IDX + GAME_STATE_PASSIVE_ARC_DIM == GAME_STATE_CHANNELS


class TestPassiveArc:
    """v16 batch A: the passive-release arc (jf=0, u=0) — 'if I let go of
    everything right now, where do I land, at what vx/vy, and is it fatal'.
    Computed every airborne frame independent of the current jump-hold."""

    def _env(self):
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("MPLBACKEND", "Agg")
        from nclone.gym_environment.config import EnvironmentConfig, RenderConfig
        from nclone.gym_environment.npp_environment import NppEnvironment

        return NppEnvironment(
            config=EnvironmentConfig(
                enable_logging=False,
                render=RenderConfig(
                    render_mode="grayscale_array",
                    enable_animation=False,
                    enable_debug_overlay=False,
                    panel_width=0,
                ),
                frame_skip=1,
            )
        )

    def test_grounded_passive_arc_is_zeroed(self):
        from nclone.gym_environment.constants import (
            GAME_STATE_PASSIVE_ARC_DIM,
            GAME_STATE_PASSIVE_ARC_IDX,
        )

        env = self._env()
        obs, _ = env.reset()
        gs = np.asarray(obs["game_state"])
        assert gs.shape[0] == GAME_STATE_CHANNELS  # 237
        i = GAME_STATE_PASSIVE_ARC_IDX
        # Fresh spawn: ninja is grounded -> passive block is all zeros.
        assert np.all(gs[i : i + GAME_STATE_PASSIVE_ARC_DIM] == 0.0)

    def test_airborne_passive_arc_filled_and_bounded(self):
        from nclone.gym_environment.constants import (
            GAME_STATE_PASSIVE_ARC_DIM,
            GAME_STATE_PASSIVE_ARC_IDX,
        )

        env = self._env()
        obs, _ = env.reset()
        i = GAME_STATE_PASSIVE_ARC_IDX
        # Jump (action 3) until airborne; the passive arc must populate.
        gs = None
        for _ in range(60):
            obs, _r, term, trunc, _info = env.step(3)
            if getattr(env.nplay_headless.sim.ninja, "airborn", False):
                gs = np.asarray(obs["game_state"])
                break
            if term or trunc:
                obs, _ = env.reset()
        assert gs is not None, "never reached an airborne frame"
        block = gs[i : i + GAME_STATE_PASSIVE_ARC_DIM]
        assert np.isfinite(block).all()
        assert (block >= -1.0).all() and (block <= 1.0).all()
        # [4]=fatal, [5]=none are strict flags.
        assert block[4] in (0.0, 1.0)
        assert block[5] in (0.0, 1.0)
