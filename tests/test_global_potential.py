"""Production global-potential contract and topology invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
)
from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map import Map
from nclone.value_function.door_aware_field import DoorAwareFreeSpaceField
from nclone.value_function.free_space_distance import (
    FreeSpaceDistanceSource,
    _CELL,
    _FS_MEM_CACHE,
    _fs_cache_key,
)
from nclone.value_function.global_potential import GlobalPotentialSource


def _level(with_mine: bool = False) -> list[int]:
    m = Map()
    for x in range(2, 42):
        for y in range(4, 18):
            m.set_tile(x, y, 0)
        m.set_tile(x, 18, 1)
    m.set_ninja_spawn(4, 17)
    m.add_entity(3, 38, 17, 0, 0, switch_x=10, switch_y=17)
    if with_mine:
        m.add_entity(1, 20, 14, 0, 0)
    return [int(v) for v in m.map_data()]


def _spawn(md):
    import nclone_native

    return nclone_native.Simulator.load(list(md)).get_ninja_pos()


def test_seedless_global_is_primary_when_finite(tmp_path, monkeypatch):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    md = _level()
    sx, sy = _spawn(md)
    src = GlobalPotentialSource(md, lattice_fallback=True)

    assert src.spawn_finite(sx, sy)
    assert src.source_kind.startswith("global")
    assert src._lattice is None, "finite primary must never build the lattice"
    assert np.isfinite(src.cost_to_completion(sx, sy, False))
    assert src.provenance["one_way"] is True


def test_mine_occupancy_and_route_clearance_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    src = FreeSpaceDistanceSource(
        _level(with_mine=True),
        mine_hard=True,
        physics_cost=True,
        one_way=True,
        mine_split_topology=True,
    )
    assert src.available

    # The true 14px occupancy mask retains survivable cells that the
    # conservative 20px route mask refuses to traverse.
    occupiable_but_not_routable = src._free & ~src._route_free
    assert occupiable_but_not_routable.any()

    # The annulus stays in the occupancy mask, but all transitions that would
    # traverse its conservative envelope are represented as blocked edges.
    saw_blocked = False
    for x, y in zip(*np.nonzero(occupiable_but_not_routable)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < src._free.shape[0]
                and 0 <= ny < src._free.shape[1]
                and src._free[nx, ny]
                and src._step_blocked(x, y, dx, dy)
            ):
                saw_blocked = True
                break
        if saw_blocked:
            break
    assert saw_blocked


def test_legacy_mine_hard_mask_remains_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    src = FreeSpaceDistanceSource(
        _level(with_mine=True),
        mine_hard=True,
        physics_cost=True,
        mine_split_topology=False,
    )
    assert np.array_equal(src._free, src._route_free)
    assert not (src._free & (src._mden > 0)).any()


def test_mine_split_topology_has_distinct_cache_key():
    md = _level(with_mine=True)
    legacy = _fs_cache_key(
        md, True, True, "open", False, True, False
    )
    split = _fs_cache_key(
        md, True, True, "open", False, True, True
    )
    assert legacy != split


def test_door_aware_field_implements_distance_source_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    md = _level()
    sx, sy = _spawn(md)
    src = DoorAwareFreeSpaceField(
        md, mine_hard=True, physics_cost=True, one_way=True
    )

    assert src.spawn_finite(sx, sy)
    assert np.isfinite(src.cost_to_go(sx, sy, False, ()))
    assert np.isfinite(src.cost_to_completion(sx, sy, False, ()))
    assert src.nav_feature(sx, sy, False, open_door_indices=()).shape == (4,)


def test_environment_commits_global_source(tmp_path, monkeypatch):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.use_global_potential = True
    cfg.reachability.global_airborne_projection = False
    cfg.reachability.global_lattice_fallback = False
    env = NppEnvironment(cfg)
    try:
        env.reset(options={"map_data": _level(), "map_name": "global_contract"})
        env.step(0)
        assert env._potential_substituted
        assert env._use_global_this_episode
        assert env._global_potential_source_kind.startswith("global")
        assert np.isfinite(env._cached_switch_distance)
        assert np.isfinite(env._cached_exit_distance)
    finally:
        env.close()


def test_true_lethal_radius_is_still_excluded(tmp_path, monkeypatch):
    monkeypatch.setenv("NCLONE_FREESPACE_CACHE_DIR", str(tmp_path))
    _FS_MEM_CACHE.clear()
    src = FreeSpaceDistanceSource(
        _level(with_mine=True), mine_hard=True, physics_cost=True,
        mine_split_topology=True,
    )
    assert src._deadly
    mx, my = src._deadly[0]
    cx, cy = int(mx / _CELL), int(my / _CELL)
    # The nearest cell centre to a deadly mine is inside ninja+mine radius.
    px, py = (cx + 0.5) * _CELL, (cy + 0.5) * _CELL
    kill = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED
    assert (px - mx) ** 2 + (py - my) ** 2 < kill * kill
    assert not src._free[cx, cy]


def test_lattice_fallback_is_cache_only(monkeypatch):
    from nclone.value_function.tools.lattice_audit import LatticeAudit

    src = GlobalPotentialSource(
        _level(), lattice_fallback=True, lattice_max_nodes=1
    )
    src._lattice = None
    monkeypatch.setattr(LatticeAudit, "_cache_path", lambda _self: "/missing")
    assert src._load_cached_lattice() is None
    assert src.fallback_reason == "global severed; lattice cache miss"


def test_global_airborne_projection_includes_flight_time():
    from nclone.gym_environment.mixins.reachability_mixin import (
        ReachabilityMixin,
    )

    class Projector:
        def project(self, *_args):
            # class, landing x/y, normal, tau, impact, threshold, ...
            return (0, 100.0, 100.0, 0.0, -1.0, 7.0, 1.0, 6.0, 0.0)

    class Source:
        def cost_to_completion(self, *_args):
            return 100.0

        def cost_to_go(self, *_args):
            return 50.0

    class Harness(ReachabilityMixin):
        pass

    h = Harness()
    h.config = SimpleNamespace(
        reachability=SimpleNamespace(
            use_global_potential=True,
            global_airborne_projection=True,
            ballistic_air_potential=False,
            ballistic_air_best_arc=False,
        )
    )
    ninja = SimpleNamespace(
        airborn=True, state=4, jump_input_old=0, jump_duration=0
    )
    h.nplay_headless = SimpleNamespace(
        sim=SimpleNamespace(ninja=ninja),
        ninja_velocity=lambda: (1.0, 1.0),
    )
    h._get_trajectory_projector = lambda: Projector()
    h._cached_switch_distance = 999.0
    h._cached_exit_distance = 999.0

    h._maybe_ballistic_reevaluate(Source(), 20.0, 30.0, False)
    assert h._cached_switch_distance == 107.0
    assert h._cached_exit_distance == 57.0
