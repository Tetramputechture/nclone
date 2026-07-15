"""Wall-jump extension to the jump-aware cost-to-go source.

Covers:
  * the level-independent wall-jump trajectory library reach envelope, and
  * a synthetic vertical wall-shaft that is reachable ONLY by wall-jumps: the
    floor-only model reads it as disconnected (spawn cost inf), while the
    wall-jump extension connects it with a cost-to-go that DECREASES up the shaft.
"""

import heapq

import numpy as np
import pytest

from nclone.map_generation.map import Map
from nclone.value_function import jump_distance as jd
from nclone.value_function.jump_trajectory_lib import (
    build_wall_trajectories,
    wall_reach_envelope,
    WALL_VY_BINS,
)


# ---------------------------------------------------------------------------
# Faithful FLOOR-ONLY baseline: the original support-cell graph (WALK + floor
# ballistic edges) with NO wall-cling nodes or wall-jump edges. Used to prove the
# wall-shaft was disconnected before the extension.
# ---------------------------------------------------------------------------
class _FloorOnlySource(jd.JumpDistanceSource):
    def _build_cached(self, map_data) -> None:
        # Bypass the shared on-disk cache: its key is class-agnostic, so a cached
        # floor-only field would otherwise poison the real JumpDistanceSource
        # (and vice-versa) for the same map.
        self._build(map_data)

    def _build_jump_fields(self) -> None:
        free = self._free
        solid = ~free
        W, H = free.shape
        support = np.zeros((W, H), bool)
        support[:, : H - 1] = free[:, : H - 1] & solid[:, 1:H]
        self._support = support

        lib = jd._library()
        sup_cells = list(map(tuple, np.argwhere(support)))
        rev = {}

        def _add(u, v, w):
            d = rev.setdefault(v, {})
            if u not in d or w < d[u]:
                d[u] = w

        for (cx, cy) in sup_cells:
            for dx in (-1, 1):
                nx = cx + dx
                if 0 <= nx < W and support[nx, cy]:
                    _add((cx, cy), (nx, cy), jd._WALK_COST)
            for trajs in lib.values():
                for t in trajs:
                    offs, desc, frs = t.offsets, t.descending, t.frames
                    for k in range(offs.shape[0]):
                        px = cx + int(offs[k, 0])
                        py = cy + int(offs[k, 1])
                        if not (0 <= px < W and 0 <= py < H):
                            break
                        if solid[px, py]:
                            break
                        if desc[k] and py + 1 < H and solid[px, py + 1]:
                            _add((cx, cy), (px, py), float(frs[k]))
                            break

        self._d_switch = self._jump_field(rev, support, self.switch)
        self._d_exit = self._jump_field(rev, support, self.exit)


def _build_wall_shaft():
    """A narrow vertical shaft, spawn at the bottom, switch+exit on a ledge at the
    top reachable only by wall-jumping up between the two side walls."""
    m = Map()
    for x in range(42):
        for y in range(23):
            m.set_tile(x, y, 1)  # fill solid
    for x in (18, 19, 20):  # carve the shaft
        for y in range(4, 21):
            m.set_tile(x, y, 0)
    for x in range(18, 31):  # top room (floor at row 6), opens right
        for y in range(3, 6):
            m.set_tile(x, y, 0)
    m.set_ninja_spawn(19, 20)
    m.add_entity(3, 28, 5, switch_x=25, switch_y=5)  # exit door + switch on ledge
    return m.map_data()


_SPAWN = (19 * 24 + 12, 20 * 24 + 12)


@pytest.fixture(autouse=True)
def _reset_wall_lib():
    # Ensure a clean, real wall library for every test (other tests may empty it).
    jd._WALL_LIB_LEFT = None
    jd._WALL_LIB_RIGHT = None
    yield


def _max_up(trajs):
    return max((-int(t.offsets[:, 1].min()) for t in trajs), default=0)


def test_wall_reach_envelope():
    trajs = build_wall_trajectories()  # {vy_bin: [Trajectory]}
    env = wall_reach_envelope(trajs)
    assert env["count"] >= 3, env
    assert env["max_up_cells"] >= 3, env
    assert env["max_horizontal_cells"] >= 2, env


def test_trick_jumps_reach_much_higher_than_from_rest():
    """A wall-jump that ARRIVES with upward velocity (trick jump) accumulates
    impulse and reaches far higher than a from-rest jump. Bin 1 is vy~0 (rest);
    the last bin is the max-upward vy."""
    lib = build_wall_trajectories()
    rest_bin = int(np.argmin(np.abs(WALL_VY_BINS)))  # vy closest to 0
    trick_bin = int(np.argmin(WALL_VY_BINS))          # most-upward vy
    up_rest = _max_up(lib[rest_bin])
    up_trick = _max_up(lib[trick_bin])
    assert up_rest >= 3, up_rest
    assert up_trick >= 20, up_trick  # trick jumps clear tall gaps
    assert up_trick >= 2 * up_rest, (up_rest, up_trick)


def test_wall_shaft_connectivity_and_gradient():
    md = _build_wall_shaft()

    # Floor-only: the top goal region is disconnected from the spawn.
    floor = _FloorOnlySource(md)
    assert not floor.spawn_finite(*_SPAWN), (
        "floor-only model should read the wall-shaft as disconnected"
    )

    # Wall-jump extension: spawn is now connected to the goal.
    src = jd.JumpDistanceSource(md)
    assert src.available
    assert src.spawn_finite(*_SPAWN), "wall-jump model should connect the shaft"

    # The cost-to-go must DECREASE going up the shaft. Sample the reachable
    # left-wall cling column: cost near the bottom must exceed cost near the top.
    ds = src._d_switch
    free = src._free
    solid = ~free
    W, H = free.shape
    wall_left = np.zeros((W, H), bool)
    wall_left[1:W, :] = free[1:W, :] & solid[0 : W - 1, :]
    # locate the shaft's left cling column (tallest free column band)
    tall = np.where(free.sum(axis=1) > 40)[0]
    lcol = tall.min()  # leftmost free col of the shaft
    col_vals = [(cy, ds[lcol, cy]) for cy in range(H)
                if wall_left[lcol, cy] and np.isfinite(ds[lcol, cy])]
    assert len(col_vals) > 5
    # bottom-to-top ordered costs (largest cy = lowest cell first)
    ordered = np.array([v for _, v in sorted(col_vals, key=lambda cv: -cv[0])])
    n = len(ordered)
    q = max(1, n // 4)
    # Aggregate gradient points UP: the bottom of the shaft costs more than the
    # top. (Trick jumps create vertical shortcuts, so the profile is a downward
    # TREND rather than a strict monotone staircase.)
    assert ordered[:q].mean() > ordered[-q:].mean(), ordered
    corr = np.corrcoef(np.arange(n), ordered)[0, 1]
    assert corr < -0.5, f"cost should trend down going up (corr={corr})"


def _build_tall_trick_shaft():
    """An open room with a HIGH ledge (switch+exit) whose vertical gap from the
    spawn exceeds a single from-rest wall-jump (~8 cells). It is reachable only by
    a high-upward-velocity trick jump off the left wall."""
    m = Map()
    for x in range(42):
        for y in range(23):
            m.set_tile(x, y, 1)
    for x in range(6, 31):  # open room
        for y in range(4, 21):
            m.set_tile(x, y, 0)
    for x in range(20, 31):  # high ledge floor (stand at row 5, ~15 cells up)
        m.set_tile(x, 6, 1)
    m.set_ninja_spawn(6, 20)  # bottom-left, against the left wall (col 5 solid)
    m.add_entity(3, 28, 5, switch_x=24, switch_y=5)
    return m.map_data()


_TALL_SPAWN = (6 * 24 + 12, 20 * 24 + 12)


def _from_rest_only_source(md):
    """A JumpDistanceSource whose wall library keeps ONLY the near-rest vy bins
    (no trick jumps), built without the disk cache."""
    left, right = jd._wall_library()
    rest_bins = [bi for bi, vy in enumerate(WALL_VY_BINS) if vy >= 0.0]
    jd._WALL_LIB_LEFT = {bi: left[bi] for bi in rest_bins}
    jd._WALL_LIB_RIGHT = {bi: right[bi] for bi in rest_bins}
    try:
        src = jd.JumpDistanceSource.__new__(jd.JumpDistanceSource)
        src._build(md)
        return src
    finally:
        jd._WALL_LIB_LEFT = None
        jd._WALL_LIB_RIGHT = None


def test_tall_shaft_requires_trick_jump():
    md = _build_tall_trick_shaft()

    # From-rest-only wall library cannot bridge the tall gap -> disconnected.
    fr = _from_rest_only_source(md)
    assert not fr.spawn_finite(*_TALL_SPAWN), (
        "from-rest-only wall-jumps should NOT reach the high ledge"
    )

    # Full vy-binned library (with trick jumps) connects it.
    jd._WALL_LIB_LEFT = None
    jd._WALL_LIB_RIGHT = None
    src = jd.JumpDistanceSource(md)
    assert src.available
    assert src.spawn_finite(*_TALL_SPAWN), (
        "trick jumps should reach the high ledge and connect the spawn"
    )


def test_floor_levels_unaffected_shape():
    # A plain floor level still builds a valid, available field (sanity: the wall
    # extension must not break the common floor case).
    m = Map()
    for x in range(42):
        m.set_tile(x, 20, 1)
    m.set_ninja_spawn(6, 19)
    m.add_entity(3, 34, 19, switch_x=20, switch_y=19)
    src = jd.JumpDistanceSource(m.map_data())
    assert src.available
    assert src.spawn_finite(6 * 24 + 12, 19 * 24 + 12)
