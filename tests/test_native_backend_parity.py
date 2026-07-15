"""Python vs native sim-backend parity for the RL environment.

The RL env (``NppEnvironment``) can run on either the pure-Python physics sim
(``NPlayHeadless``) or the Rust-native sim (``NativePlayHeadless``), selected by
``EnvironmentConfig.sim_backend``. Native is a drop-in for the training hot path,
but training only stays correct if both backends produce the *same* observations
and outcomes on the maps we train on.

These tests load identical generated maps into a python-backed and a
native-backed env, replay the same deterministic action tape on both, and assert
the policy-facing observations + outcomes agree frame-by-frame. This is the
verify-first gate before opting training into ``sim_backend="native"``.

Native stubs locked doors (returns ``[]``), so we restrict to the toggle-mine
training categories (no locked-door levels), which is the current training scope.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.generator_registry import GeneratorRegistry

# Training categories used for parity (all toggle-mine scope; no locked doors).
PARITY_CATEGORIES = ["simplest", "simplest_with_mines", "simple", "medium"]

# A fixed pseudo-random action tape, shared by both backends.
def _action_tape(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 5) for _ in range(n)]


def _make_env(backend: str) -> NppEnvironment:
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend=backend)
    return NppEnvironment(cfg)


def _gen_map_data(category: str, seed: int) -> list[int]:
    reg = GeneratorRegistry(random.Random(seed))
    return reg.get_map(category, seed).map_data()


@pytest.mark.parametrize("category", PARITY_CATEGORIES)
def test_observation_parity_python_vs_native(category: str):
    """Same map + same actions → identical observations & outcomes on both backends."""
    map_data = _gen_map_data(category, seed=4242)
    actions = _action_tape(150, seed=99)

    py_env = _make_env("python")
    nat_env = _make_env("native")
    try:
        opts = {"map_data": list(map_data), "map_name": f"parity_{category}"}
        py_obs, _ = py_env.reset(seed=0, options=opts)
        nat_obs, _ = nat_env.reset(seed=0, options=dict(opts))

        # Static, level-scoped observations must match exactly at reset.
        np.testing.assert_array_equal(
            np.asarray(py_obs["tile_grid"]), np.asarray(nat_obs["tile_grid"]),
            err_msg=f"[{category}] tile_grid differs at reset",
        )
        np.testing.assert_allclose(
            np.asarray(py_obs["hazard_map"]), np.asarray(nat_obs["hazard_map"]),
            atol=1e-4, err_msg=f"[{category}] hazard_map differs at reset",
        )

        for t, a in enumerate(actions):
            py_o, py_r, py_term, py_trunc, _ = py_env.step(a)
            nat_o, nat_r, nat_term, nat_trunc, _ = nat_env.step(a)

            # Physics is byte-parity tested at the sim level; the policy-facing
            # ninja state vector should therefore match very tightly.
            np.testing.assert_allclose(
                np.asarray(py_o["game_state"]), np.asarray(nat_o["game_state"]),
                atol=1e-4,
                err_msg=f"[{category}] game_state diverged at step {t} (action {a})",
            )
            np.testing.assert_array_equal(
                np.asarray(py_o["action_mask"]), np.asarray(nat_o["action_mask"]),
                err_msg=f"[{category}] action_mask diverged at step {t}",
            )
            assert py_term == nat_term, f"[{category}] terminated diverged at step {t}"
            assert py_trunc == nat_trunc, f"[{category}] truncated diverged at step {t}"
            assert abs(py_r - nat_r) < 1e-3, (
                f"[{category}] reward diverged at step {t}: {py_r} vs {nat_r}"
            )

            if py_term or py_trunc:
                break
    finally:
        py_env.close()
        nat_env.close()


def _corridor_with_locked_door() -> list[int]:
    """Flat corridor with a reachable locked-door switch + exit (toggle-free)."""
    from nclone.map_generation.map import Map

    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)  # solid floor
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)  # exit + switch
    m.add_entity(6, 20, 11, 0, 0, switch_x=10, switch_y=11)  # locked door (switch reachable)
    return [int(v) for v in m.map_data()]


def _locked_door_fields(headless):
    out = []
    for d in headless.locked_doors():
        seg = getattr(d, "segment", None)
        out.append(
            (
                round(float(d.xpos), 3),
                round(float(d.ypos), 3),
                bool(getattr(d, "active", True)),
                bool(getattr(d, "closed", True)),
                (seg.x1, seg.y1, seg.x2, seg.y2) if seg else None,
            )
        )
    return out


def test_locked_doors_wrapper_parity():
    """NativePlayHeadless.locked_doors() matches NPlayHeadless through open/close."""
    from nclone.nplay_headless import NPlayHeadless
    from nclone.native_headless import NativePlayHeadless

    md = _corridor_with_locked_door()
    py = NPlayHeadless()
    py.load_map_from_map_data(list(md))
    py.reset()
    nat = NativePlayHeadless()
    nat.load_map_from_map_data(list(md))
    nat.reset()

    assert _locked_door_fields(py) == _locked_door_fields(nat), "diverged before opening"
    assert len(py.locked_door_switches()) == len(nat.locked_door_switches())

    for _ in range(120):  # hold right; ninja collects the switch and opens the door
        py.tick(1, 0)
        nat.tick(1, 0)

    assert _locked_door_fields(py) == _locked_door_fields(nat), "diverged after opening"
    assert py.sim.ninja.doors_opened == nat.sim.ninja.doors_opened == 1


def test_env_observation_parity_locked_door():
    """Full env obs + reward parity on a locked-door level (door obs channels +
    door feature cache + R_LOCKED_DOOR_SWITCH must agree across backends)."""
    md = _corridor_with_locked_door()
    actions = _action_tape(150, seed=11)
    py_env = _make_env("python")
    nat_env = _make_env("native")
    try:
        opts = {"map_data": list(md), "map_name": "parity_locked_door"}
        py_env.reset(seed=0, options=opts)
        nat_env.reset(seed=0, options=dict(opts))
        for t, a in enumerate(actions):
            py_o, py_r, py_term, py_trunc, _ = py_env.step(a)
            nat_o, nat_r, nat_term, nat_trunc, _ = nat_env.step(a)
            np.testing.assert_allclose(
                np.asarray(py_o["game_state"]), np.asarray(nat_o["game_state"]),
                atol=1e-4, err_msg=f"game_state diverged at step {t}",
            )
            np.testing.assert_array_equal(
                np.asarray(py_o["entity_overlays"]), np.asarray(nat_o["entity_overlays"]),
                err_msg=f"entity_overlays diverged at step {t}",
            )
            assert abs(py_r - nat_r) < 1e-3, f"reward diverged at step {t}: {py_r} vs {nat_r}"
            assert py_term == nat_term and py_trunc == nat_trunc
            if py_term or py_trunc:
                break
    finally:
        py_env.close()
        nat_env.close()


def _chamber_with_one_ways(one_ways: list[tuple[int, int, int]]) -> list[int]:
    """A closed box (floor/ceiling/walls) with one-way platforms.

    `one_ways` entries are `(grid_x, grid_y, orientation)`. Note the ninja spawn
    carries a +6-unit offset (1.5 grid) so a one-way one grid column to the right
    of the spawn column lands within the ~21px lateral capture zone.
    """
    from nclone.map_generation.map import Map

    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 18, 1)  # floor
        m.set_tile(x, 4, 1)  # ceiling
    for y in range(4, 19):
        m.set_tile(2, y, 1)  # left wall
        m.set_tile(40, y, 1)  # right wall
    m.set_ninja_spawn(20, 16)
    m.add_entity(3, 38, 17, 0, 0, switch_x=36, switch_y=17)  # exit + switch
    for gx, gy, orient in one_ways:
        m.add_entity(11, gx, gy, orient, 0)
    return [int(v) for v in m.map_data()]


def _ninja_state(headless):
    """Full physics-relevant ninja state, rounded for exact comparison."""
    n = headless.sim.ninja
    return (
        round(float(n.xpos), 9),
        round(float(n.ypos), 9),
        round(float(n.xspeed), 9),
        round(float(n.yspeed), 9),
        int(n.state),
        int(getattr(n, "floor_count", 0)),
        bool(getattr(n, "walled", False)),
    )


def _drop_onto_one_way_map() -> list[int]:
    """Ninja spawned high above a one-way floor, with a solid catch-floor below it.

    Spawn column is grid x=10 (carries the +6-unit ninja offset); the one-way at
    grid (11, 10) sits one column right so its lateral capture zone (~21px) covers
    the spawn column. Bottom solid floor at grid y=14 (~350px) is the fall-through
    backstop if (and only if) the one-way fails to block.
    """
    from nclone.map_generation.map import Map

    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 14, 1)  # solid catch-floor
    m.set_ninja_spawn(10, 5)
    m.add_entity(3, 38, 13, 0, 0, switch_x=36, switch_y=13)  # exit + switch
    m.add_entity(11, 11, 10, 6, 0)  # one-way floor (orientation 6, normal up)
    return [int(v) for v in m.map_data()]


def test_one_way_blocks_fall_and_parity():
    """A one-way floor (orientation 6, normal up) stops a ninja dropping onto it
    from the blocking side — and both backends agree byte-for-byte."""
    from nclone.nplay_headless import NPlayHeadless
    from nclone.native_headless import NativePlayHeadless

    md = _drop_onto_one_way_map()

    py = NPlayHeadless()
    py.load_map_from_map_data(list(md))
    py.reset()
    nat = NativePlayHeadless()
    nat.load_map_from_map_data(list(md))
    nat.reset()

    for t in range(200):  # NOOP: free-fall straight down
        py.tick(0, 0)
        nat.tick(0, 0)
        assert _ninja_state(py) == _ninja_state(nat), f"ninja state diverged at step {t}"

    # The ninja came to rest ON the one-way (well above the bottom floor at ~350),
    # proving the platform physically blocked the fall.
    _, ny = nat.ninja_position()
    assert ny < 300.0, f"one-way did not block the fall; ninja at y={ny}"


@pytest.mark.parametrize(
    "name,one_ways",
    [
        ("floor_o6", [(21, 10, 6)]),
        ("ceiling_o2", [(21, 7, 2)]),
        ("wall_right_o0", [(12, 14, 0)]),
        ("wall_left_o4", [(28, 14, 4)]),
        ("diagonals", [(14, 10, 1), (26, 10, 3), (16, 14, 5), (24, 14, 7)]),
        ("mixed", [(12, 10, 6), (28, 8, 2), (10, 14, 0), (30, 14, 4), (20, 9, 5)]),
    ],
)
def test_one_way_orientations_parity(name: str, one_ways):
    """Byte-identical ninja physics across all 8 one-way orientations (incl.
    diagonals) under randomized control, exercising depenetration, the type-11
    velocity projection, floor/ceiling accumulation, and the wall-normal path."""
    from nclone.nplay_headless import NPlayHeadless
    from nclone.native_headless import NativePlayHeadless

    md = _chamber_with_one_ways(one_ways)
    for seed in (1, 7, 42):
        py = NPlayHeadless()
        py.load_map_from_map_data(list(md))
        py.reset()
        nat = NativePlayHeadless()
        nat.load_map_from_map_data(list(md))
        nat.reset()
        rng = random.Random(seed)
        tape = [(rng.randint(-1, 1), rng.randint(0, 1)) for _ in range(250)]
        for t, (h, j) in enumerate(tape):
            py.tick(h, j)
            nat.tick(h, j)
            assert _ninja_state(py) == _ninja_state(nat), (
                f"[{name} seed={seed}] ninja state diverged at step {t} "
                f"(hor={h}, jump={j})"
            )


def test_env_observation_parity_one_way():
    """Cross-backend full env obs parity on a level containing one-way platforms.

    One-way platforms are out of the generalist observation scope (not encoded in
    entity_overlays), but they ARE still simulated, so this exercises native<->python
    physics parity: the 5-channel entity_overlays must stay byte-identical across
    backends through a 120-step rollout."""
    from nclone.gym_environment.entity_overlay_utils import ENTITY_OVERLAY_CHANNELS

    md = _chamber_with_one_ways([(12, 10, 6), (28, 14, 4)])  # floor + left-wall
    actions = _action_tape(120, seed=21)
    py_env = _make_env("python")
    nat_env = _make_env("native")
    try:
        opts = {"map_data": list(md), "map_name": "parity_one_way"}
        py_o0, _ = py_env.reset(seed=0, options=opts)
        nat_o0, _ = nat_env.reset(seed=0, options=dict(opts))

        assert py_o0["entity_overlays"].shape[0] == ENTITY_OVERLAY_CHANNELS == 5
        np.testing.assert_array_equal(
            np.asarray(py_o0["entity_overlays"]),
            np.asarray(nat_o0["entity_overlays"]),
            err_msg="entity_overlays differ at reset",
        )

        for t, a in enumerate(actions):
            py_o, py_r, py_term, py_trunc, _ = py_env.step(a)
            nat_o, nat_r, nat_term, nat_trunc, _ = nat_env.step(a)
            np.testing.assert_array_equal(
                np.asarray(py_o["entity_overlays"]),
                np.asarray(nat_o["entity_overlays"]),
                err_msg=f"one-way entity_overlays diverged at step {t}",
            )
            assert py_term == nat_term and py_trunc == nat_trunc
            if py_term or py_trunc:
                break
    finally:
        py_env.close()
        nat_env.close()


def test_position_parity_across_episodes():
    """Ninja trajectory matches across a full episode + reset on both backends."""
    map_data = _gen_map_data("simple", seed=7)
    actions = _action_tape(120, seed=3)

    py_env = _make_env("python")
    nat_env = _make_env("native")
    try:
        opts = {"map_data": list(map_data), "map_name": "parity_traj"}
        py_env.reset(seed=0, options=opts)
        nat_env.reset(seed=0, options=dict(opts))
        for t, a in enumerate(actions):
            py_env.step(a)
            nat_env.step(a)
            px, py = py_env.nplay_headless.ninja_position()
            nx, ny = nat_env.nplay_headless.ninja_position()
            assert abs(px - nx) < 1e-3 and abs(py - ny) < 1e-3, (
                f"ninja position diverged at step {t}: ({px},{py}) vs ({nx},{ny})"
            )
            if py_env.nplay_headless.ninja_has_died() or py_env.nplay_headless.ninja_has_won():
                break
    finally:
        py_env.close()
        nat_env.close()
