"""Floor-mounted mines are LETHAL and must never be culled from the agent's view.

This test exists because of a real bug that shipped and had to be reverted.

An audit classified a mine as "unreachable phantom hazard" when the mine's own
tile was full-solid, and culled it from the observation and reward. That is wrong,
and the reason is a coordinate convention:

    switches / exit doors / ninja spawn   cx % 4 == 2 and cy % 4 == 2  (tile CENTRES)
    mines                                 cy % 4 == 0 for EVERY mine   (tile BOUNDARY)

Mines are mounted on FLOOR SURFACES, so a mine's centre lies exactly on the line
between the open tile above and the solid tile below. Integer floor
(``py // 24``) therefore assigns it to the SOLID tile, making it look buried,
while its kill zone protrudes into the walkable space above.

The kill threshold is ``NINJA_RADIUS + TOGGLE_MINE_RADII[0] == 10 + 4 == 14`` px.
A ninja standing on that floor has its centre 10 px above the mine centre, i.e.
INSIDE the kill radius. Measured over the mechanic benchmark: 100% of
MINE_NAVIGATION/T1 mines and 49% of PLATFORMING/T2 mines can kill a standing
ninja. Culling them trained the agent blind to hazards that kill it.

The correct test for "unreachable", if anyone wants to cull again, is geometric:
does the mine's kill disk intersect any reachable ninja position? Some mines
genuinely are unreachable (PLATFORM_STAIRS/T2's are 0% lethal). "Is the centre's
tile solid?" is not that test.
"""

from __future__ import annotations

import math

import pytest

from nclone.constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADII

TILE_START = 184
GRID_W, GRID_H = 42, 23
MAP_LEN = 1235
SOLID = frozenset({1, 2, 3, 4, 5})

KILL_PX = NINJA_RADIUS + TOGGLE_MINE_RADII[0]


def test_kill_threshold_is_fourteen_px():
    """Pins the arithmetic the whole argument rests on."""
    assert TOGGLE_MINE_RADII[0] == 4.0
    assert NINJA_RADIUS == 10
    assert KILL_PX == 14.0


def _blank_map() -> list:
    md = [0] * MAP_LEN
    for y in range(GRID_H):
        for x in range(GRID_W):
            solid = x == 0 or y == 0 or x == GRID_W - 1 or y == GRID_H - 1
            md[TILE_START + y * GRID_W + x] = 1 if solid else 0
    md[1156] = 0
    md[1231] = 20 * 4 + 2
    md[1232] = 11 * 4 + 2
    md[1233] = -1
    return md


def _add_floor_mine(md: list, sim_tile_x: int, sim_tile_y: int) -> tuple:
    """A mine mounted on the SURFACE of sim tile (x, y): y on the tile boundary.

    This is the real corpus convention (cy % 4 == 0), NOT a tile centre.
    """
    cx, cy = sim_tile_x * 4 + 2, sim_tile_y * 4
    md.extend([1, cx, cy, 0, 0])
    return cx * 6.0, cy * 6.0


def test_corpus_mines_sit_on_tile_boundaries_not_centres():
    """The convention itself, asserted so a regenerated corpus can't silently break it."""
    import glob
    import pickle
    from collections import Counter

    root = ("/home/tetra/projects/nclone/datasets/"
            "mechanic_benchmark_v7_official/train/PLATFORMING/T2")
    files = sorted(glob.glob(f"{root}/*.pkl"))[:20]
    if not files:
        pytest.skip("mechanic benchmark not available")

    mine_cy, other_cy = Counter(), Counter()
    for f in files:
        md = [int(v) for v in pickle.load(open(f, "rb"))["map_data"]]
        other_cy[md[1232] % 4] += 1          # ninja spawn
        i = 1235
        while i + 5 <= len(md) and md[i] != 0:
            if md[i] in (1, 21):
                mine_cy[md[i + 2] % 4] += 1
            elif md[i] in (3, 4):
                other_cy[md[i + 2] % 4] += 1
            i += 5

    assert set(mine_cy) == {0}, (
        f"every mine's y must be a 24px multiple (tile boundary), got {dict(mine_cy)}"
    )
    assert set(other_cy) == {2}, (
        f"switch/door/spawn must sit at tile centres, got {dict(other_cy)}"
    )


def test_floor_mounted_mine_kills_a_ninja_standing_on_that_floor():
    """Pure geometry: the reason these mines are not phantom."""
    md = _blank_map()
    floor_sim_y = GRID_H          # inner row GRID_H-1 is the solid bottom border
    mx, my = _add_floor_mine(md, 20, floor_sim_y)

    # A ninja standing on that floor: centre one radius above the surface.
    ninja_y = floor_sim_y * 24.0 - NINJA_RADIUS
    ninja_x = 20 * 24.0 + 12.0
    d = math.dist((ninja_x, ninja_y), (mx, my))
    assert d < KILL_PX, (
        f"a ninja standing on the mine's floor is {d:.1f}px away, which must be "
        f"inside the {KILL_PX}px kill radius"
    )


def test_floor_mounted_mine_is_not_culled_from_the_observation():
    """The regression proper: the env must still SEE a floor-mounted mine."""
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.gym_environment.config import EnvironmentConfig
    from nclone.dev_tool.loaders import load_map_data_into_env

    env = NppEnvironment(config=EnvironmentConfig())
    try:
        md = _blank_map()
        for x in range(6, 20):
            _add_floor_mine(md, x, GRID_H)
        load_map_data_into_env(env, list(md), map_name="floor_mine_regression")

        toggle, toggled = env.nplay_headless.get_mine_entities()
        all_mines = [*toggle, *toggled]
        assert len(all_mines) == 14, f"sim should load 14 mines, got {len(all_mines)}"

        kept = env._filter_relevant_mines(all_mines)
        assert len(kept) == len(all_mines), (
            f"floor-mounted mines must NOT be culled: {len(kept)}/{len(all_mines)} "
            "survived. Culling these blinds the agent to hazards that kill it."
        )
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def test_empty_key_set_still_means_cull_everything():
    """The sentinel fix is INDEPENDENT of the reverted pre-filter and is retained.

    ``_get_relevant_mine_keys`` returns ``None`` for "no filtering applies" and a
    SET for "these are the survivors". An empty set means cull everything. A falsy
    guard (``if not keys``) conflated the two and returned the full list, which
    would silently disable the flood-fill cull on any level where it culls all.
    """
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.gym_environment.config import EnvironmentConfig

    env = NppEnvironment(config=EnvironmentConfig())
    try:
        class _M:
            def __init__(self, x, y):
                self.xpos, self.ypos = x, y

        mines = [_M(100.0, 200.0), _M(300.0, 400.0)]
        env._relevant_mine_keys = set()
        env._relevant_mine_keys_map = getattr(
            env.map_loader, "current_map_name", None)
        assert env._filter_relevant_mines(mines) == [], (
            "an EMPTY key set must cull every mine, not fall back to the full list"
        )
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
