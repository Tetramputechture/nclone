"""A rocket projectile must actually kill the ninja.

THE BUG THIS PINS. The kill lived only in `logical_collision`, which is reached
solely via `gather_entities_from_neighbourhood(sim, ninja.x, ninja.y)` -- the
entities registered in the nine cells around the NINJA. But a rocket's
projectile is a FIELD INSIDE THE LAUNCHER, not an entity of its own, so the
launcher never leaves its own cell. The dispatch therefore only fired when the
ninja walked up to the tube.

Measured before the fix, on a flat room with clear line of sight and a
stationary ninja: the projectile passed within **2.0 px** of the ninja (radius
10) over 600 frames and the ninja did not die, in BOTH backends.

Turret (19) and DualLaser (23) already avoid this -- "their bodies do not
collide, and their beams kill inside think" (spec §6.6). A rocket is the same
shape of hazard: its lethal region is not at its body. So the kill now runs
every frame from `think`, which also needs no change to the spatial grid --
native rebuilds `entity_grid` only at init and restore, never inside `tick`.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from nclone.dev_tool.loaders import load_map_data_into_env
from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map import Map

ROCKET = 18


def _room_with_rocket():
    m = Map()
    for x in range(2, 40):
        m.set_tile(x, 12, 1)
    m.set_ninja_spawn(6, 11)
    m.add_entity(3, 36, 11, switch_x=35, switch_y=11)
    m.add_entity(ROCKET, 30, 10)
    return [int(v) for v in m.map_data()]


@pytest.mark.parametrize("backend", ["python", "native"])
def test_a_rocket_projectile_kills_a_stationary_ninja(backend):
    env = NppEnvironment(EnvironmentConfig(
        sim_backend=backend,
        reachability=ReachabilityConfig(enable_threat_ttl=True)))
    try:
        load_map_data_into_env(env, _room_with_rocket(), f"rocket_{backend}")
        died = False
        for _ in range(400):
            _obs, _r, term, trunc, _i = env.step(0)   # stand still
            if term or trunc:
                died = bool(term)
                break
        assert died, (
            f"[{backend}] the ninja survived 400 frames of a homing rocket with "
            "clear line of sight -- the projectile is not killing")
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["python", "native"])
def test_the_death_is_attributed_to_the_rocket(backend):
    """`death/by_cause/rocket` is how the run attributes hazard skill per class;
    a kill logged under the wrong cause is as bad as no kill."""
    env = NppEnvironment(EnvironmentConfig(
        sim_backend=backend,
        reachability=ReachabilityConfig(enable_threat_ttl=True)))
    try:
        load_map_data_into_env(env, _room_with_rocket(), f"rocket_{backend}")
        for _ in range(400):
            _o, _r, term, trunc, _i = env.step(0)
            if term or trunc:
                break
        cause = env.nplay_headless.sim.ninja.death_cause
        assert cause == "rocket", cause
    finally:
        env.close()


def test_the_kill_runs_from_think_not_only_from_grid_dispatch():
    """The structural guard. `logical_collision` alone is unreachable for a
    flying projectile, so a future refactor that moves the test back there
    would silently restore the bug."""
    import inspect

    from nclone.entity_classes.entity_rocket import EntityRocket

    assert hasattr(EntityRocket, "_projectile_kill_test")
    think_src = inspect.getsource(EntityRocket.think)
    assert "_projectile_kill_test" in think_src, (
        "think() must run the kill test every frame; the grid dispatch cannot "
        "reach a projectile that has left the launcher's cell")


def test_the_player_controlled_rocket_shares_the_fix():
    """Type 29 has zero instances in the phase-1 corpus, so this is latent --
    but 18 and 29 share every helper and a divergence is how the next reader
    gets misled."""
    import inspect

    from nclone.entity_classes.entity_pc_rocket import (
        EntityPlayerControlledRocket,
    )

    assert hasattr(EntityPlayerControlledRocket, "_projectile_kill_test")
    assert "_projectile_kill_test" in inspect.getsource(
        EntityPlayerControlledRocket.think)
