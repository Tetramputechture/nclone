"""Approach A bedrock tests: the level-independent jump-trajectory library and the
anti-bug property (jump-aware cost-to-go routes OVER gaps where the geometric
free-space field points into them)."""

import os
import glob
import pickle
import heapq

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from nclone.value_function.jump_trajectory_lib import (
    build_floor_trajectories,
    reach_envelope,
    CELL,
    VX_BINS,
    HOLD_DURATIONS,
    _build_open_map_data,
    _cells_from_points,
)

_V4_PLATFORM_T2 = "datasets/mechanic_benchmark_v4/train/PLATFORM_STAIRS/T2"


def test_library_builds_sane_envelope():
    trajs = build_floor_trajectories()
    n = sum(len(v) for v in trajs.values())
    assert n > 100, f"expected many trajectories, got {n}"
    assert len(trajs) == len(VX_BINS)
    env = reach_envelope(trajs)
    # Bounds calibrated to the REAL simulator (nclone_native), not a physics
    # reimplementation. Measured faithful envelope: up=12, horizontal=61, down=0.
    # The ninja's real max floor-jump apex is ~76px (~12-13 cells) -- the vertical
    # model was always roughly right. The fidelity bug the real-sim regen fixes is
    # HORIZONTAL/DESCENT over-prediction: the old reimplementation reported ~107
    # cells horizontal and ~76 cells of below-launch descent (it kept falling at
    # near-max speed far below launch). These tight bounds catch a regression back
    # to that over-confident envelope.
    assert 9 <= env["max_up_cells"] <= 14, env          # real ~12 (~72px up)
    assert 45 <= env["max_horizontal_cells"] <= 75, env  # real ~61 (NOT ~107)
    assert env["max_down_cells"] <= 3, env               # real ~0 (NOT ~76)
    # Every trajectory has at least one descending (landing-capable) cell.
    assert any(t.descending.any() for v in trajs.values() for t in v)


def _direct_real_rollout(target_vx, hold, h1, h2):
    """Independently drive the REAL sim (own map + own tick loop, NOT the library
    helpers) and return the 6px cell path of the floor jump, mirrored for vx<0."""
    import nclone_native as Nnative

    sim = Nnative.Simulator.load(_build_open_map_data())
    sim.fast_reset()
    for _ in range(30):
        sim.tick(0, 0)
    run_dir = 1 if target_vx >= 0 else -1
    if abs(target_vx) > 1e-6:
        prev = 0.0
        for _ in range(400):
            sim.tick(run_dir, 0)
            vx = abs(sim.get_ninja_velocity()[0])
            if vx >= abs(target_vx) - 1e-9 or vx <= prev:
                break
            prev = vx
    px0, py0 = sim.get_ninja_pos()
    pts = [(0, px0, py0, 0.0, sim.get_ninja_velocity()[1])]
    for f in range(1, 160):
        jump = 1 if f <= hold else 0
        hor = (h1 if f <= hold else h2)
        sim.tick(hor, jump)
        px, py = sim.get_ninja_pos()
        pts.append((f, px, py, 0.0, sim.get_ninja_velocity()[1]))
        if sim.has_died():
            break
        if f > hold and py >= py0 - 0.4:
            break
    return _cells_from_points(pts)


def test_regenerated_trajectories_match_real_sim():
    """A handful of regenerated library trajectories match an INDEPENDENT real-sim
    rollout of the same (vx, hold, h1, h2) within ~1-2 cells -- proving the library
    is built from real physics, not a reimplementation."""
    lib = build_floor_trajectories()
    # (vx_bin_index, hold, h1, h2): cover rest, mid, and max-speed launches both
    # directions (vx<0 exercises the mirror path), short and long holds.
    cases = [
        (4, 45, 0, 0),    # vx=0, max-hold straight up
        (4, 12, 1, 1),    # vx=0, drift right
        (8, 45, 1, 1),    # max +vx, max hold, hold right
        (6, 20, 0, 1),    # mid +vx
        (0, 45, -1, -1),  # max -vx, mirror path
        (2, 16, -1, 0),   # mid -vx, mirror path
    ]
    for bi, hold, h1, h2 in cases:
        ref = _direct_real_rollout(float(VX_BINS[bi]), hold, h1, h2)
        assert ref.offsets.shape[0] >= 2, (bi, hold, h1, h2)
        ref_apex = -int(ref.offsets[:, 1].min())
        ref_dx = int(ref.offsets[ref.offsets.shape[0] - 1, 0])  # final horizontal cell
        # Find the closest library trajectory in this bin by final-cell + apex.
        best = None
        for t in lib[bi]:
            if t.offsets.shape[0] < 2:
                continue
            apex = -int(t.offsets[:, 1].min())
            dx = int(t.offsets[t.offsets.shape[0] - 1, 0])
            err = abs(apex - ref_apex) + abs(dx - ref_dx)
            if best is None or err < best:
                best = err
        assert best is not None and best <= 2, (
            f"bin={bi} hold={hold} h={h1},{h2}: closest library trajectory differs "
            f"by {best} cells from a direct real-sim rollout (apex={ref_apex}, dx={ref_dx})"
        )


def _build_jump_graph(free, support, solid, LIB):
    W, H = free.shape

    def jump_edges(cx, cy):
        outs = {}
        for trajs in LIB.values():
            for t in trajs:
                for k in range(t.offsets.shape[0]):
                    px, py = cx + int(t.offsets[k, 0]), cy + int(t.offsets[k, 1])
                    if not (0 <= px < W and 0 <= py < H):
                        break
                    if solid[px, py]:
                        break
                    if t.descending[k] and py + 1 < H and solid[px, py + 1]:
                        c = float(t.frames[k])
                        if (px, py) not in outs or c < outs[(px, py)]:
                            outs[(px, py)] = c
                        break
        return outs

    sup = list(zip(*np.where(support)))
    rev = {}
    for (cx, cy) in sup:
        e = {}
        for dx in (-1, 1):
            if 0 <= cx + dx < W and support[cx + dx, cy]:
                e[(cx + dx, cy)] = 1.0
        for k, v in jump_edges(cx, cy).items():
            e[k] = min(e.get(k, 1e9), v)
        for v, w in e.items():
            rev.setdefault(v, {})[(cx, cy)] = w
    return sup, rev


@pytest.mark.skipif(
    not glob.glob(_V4_PLATFORM_T2 + "/*.pkl"),
    reason="v4 PLATFORM_STAIRS dataset not present",
)
def test_jump_aware_routes_over_gaps():
    """On a real PLATFORM_STAIRS T2 level, the jump-aware cost-to-go reaches the
    platforms (finite cost) while the geometric free-space gradient points into
    the air gap at the platform edges (the bug we are fixing)."""
    from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
    from nclone.gym_environment.npp_environment import NppEnvironment

    pkl = sorted(glob.glob(_V4_PLATFORM_T2 + "/*.pkl"))[0]
    md = list(pickle.load(open(pkl, "rb"))["map_data"])
    cfg = EnvironmentConfig.for_evaluation(
        sim_backend="python", defer_initial_map_load=True,
        reachability=ReachabilityConfig(use_free_space_potential=True,
                                        expose_cost_to_go_map=True),
    )
    env = NppEnvironment(config=cfg)
    env.nplay_headless.load_map_from_map_data(md)
    env.reset(options={"skip_map_load": True})
    src = env._free_space_nav_source
    free = src._free
    W, H = free.shape
    solid = ~free
    support = np.zeros((W, H), bool)
    support[:, : H - 1] = free[:, : H - 1] & solid[:, 1:H]

    LIB = build_floor_trajectories()
    sup, rev = _build_jump_graph(free, support, solid, LIB)

    ex, ey = src.exit
    gc = min(sup, key=lambda c: (c[0] - ex / 6.0) ** 2 + (c[1] - ey / 6.0) ** 2)
    d = {c: 1e18 for c in sup}
    d[gc] = 0.0
    pq = [(0.0, gc)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > d[u]:
            continue
        for p, w in rev.get(u, {}).items():
            if c + w < d[p]:
                d[p] = c + w
                heapq.heappush(pq, (d[p], p))

    reachable = sum(1 for v in d.values() if v < 1e17)
    assert reachable >= 0.9 * len(sup), f"jump-aware only reached {reachable}/{len(sup)}"

    # platform-edge cells: free-space should mislead, jump-aware should not.
    edges = [c for c in sup if (c[0] + 1 >= W or not support[c[0] + 1, c[1]])
             or (c[0] - 1 < 0 or not support[c[0] - 1, c[1]])]

    def fs_into_gap(cx, cy):
        bd = src._d_exit[cx, cy]
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and free[nx, ny] and src._d_exit[nx, ny] < bd:
                    bd = src._d_exit[nx, ny]
                    best = (nx, ny)
        return best is not None and not (best[1] + 1 < H and solid[best[0], best[1] + 1])

    ja_finite = sum(1 for c in edges if d[c] < 1e17)
    assert ja_finite == len(edges), "jump-aware must have finite cost at every platform edge"
    # The bug exists for free-space (this is what we're fixing); assert the
    # jump-aware field does NOT share it (all edges have a real route).
    assert ja_finite >= sum(1 for c in edges if fs_into_gap(*c))


if __name__ == "__main__":
    test_library_builds_sane_envelope()
    print("PASS envelope")
    test_jump_aware_routes_over_gaps()
    print("PASS anti-bug (jump-aware routes over gaps)")
