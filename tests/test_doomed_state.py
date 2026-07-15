"""Tests for ``BaseNppEnvironment.is_current_state_doomed``.

The doomed check is the sole criterion Go-Explore now uses to discard a
checkpoint candidate: a state is *doomed* iff, for EVERY discrete input, the
ninja dies within ``lookahead_frames`` physics frames. It replaces the old
"discard while falling" heuristic that emptied the archive on aerial
mine-threading levels. These tests lock the core contract:

* a normal spawn is NOT doomed (so recoverable states are kept),
* the check is exactly state-preserving (position, velocity, tick, terminal
  flags are identical before and after),
* it is deterministic, and
* an already-dead state IS doomed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import nclone
from nclone.map_generation.map import Map
from nclone.gym_environment.config import EnvironmentConfig, PBRSConfig
from nclone.gym_environment.npp_environment import NppEnvironment

# A real, guaranteed-lethal level for the dead-state case. Synthetic toggle mines
# don't reliably arm in the native physics (their deadliness is an entity-state
# detail, not the obs-layer type mapping), so we use a map known to kill.
REAL_LETHAL_MAP = (
    Path(nclone.__file__).resolve().parent / "test_maps" / "mines-extremely-difficult"
)


def _corridor_map_path() -> str:
    """A plain flat corridor: spawn on the left, exit + switch on the right."""
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)  # floor
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, switch_x=36, switch_y=11)  # exit door + switch
    data = bytes(int(v) & 0xFF for v in m.map_data())
    fd, path = tempfile.mkstemp(suffix=".map")
    os.write(fd, data)
    os.close(fd)
    return path


def _make_env(map_path: str) -> NppEnvironment:
    cfg = EnvironmentConfig.for_training(pbrs=PBRSConfig(pbrs_gamma=0.99))
    cfg.custom_map_path = map_path
    cfg.frame_skip = 1
    cfg.sim_backend = "native"
    e = NppEnvironment(config=cfg)
    e.reset()
    return e


@pytest.fixture(scope="module")
def env():
    path = _corridor_map_path()
    e = _make_env(path)
    yield e
    e.close()
    os.unlink(path)


def test_spawn_is_not_doomed(env):
    """A normal grounded spawn must be recoverable (at least one input survives)."""
    assert env.is_current_state_doomed(1) is False


def test_check_is_state_preserving(env):
    """The check must restore the EXACT pre-call state — it branches inputs off a
    snapshot and must leave no trace."""
    env.reset()
    n = env.nplay_headless
    pos0, vel0, tick0 = n.ninja_position(), n.ninja_velocity(), n.current_tick
    died0, won0 = n.ninja_has_died(), n.ninja_has_won()
    env.is_current_state_doomed(2)  # deeper lookahead exercises more ticks
    assert n.ninja_position() == pos0
    assert n.ninja_velocity() == vel0
    assert n.current_tick == tick0
    assert n.ninja_has_died() == died0
    assert n.ninja_has_won() == won0


def test_determinism(env):
    env.reset()
    assert env.is_current_state_doomed(1) == env.is_current_state_doomed(1)


@pytest.mark.skipif(
    not REAL_LETHAL_MAP.exists(), reason="mines-extremely-difficult map not present"
)
def test_dead_state_is_doomed():
    """An already-dead ninja is trivially doomed. Reached by random play on a
    real mine-dense level (deaths are abundant there)."""
    import random

    e = _make_env(str(REAL_LETHAL_MAP))
    try:
        n = e.nplay_headless
        rng = random.Random(0)
        died = False
        for _ in range(5000):
            _o, _r, term, trunc, _i = e.step(rng.randint(0, 5))
            if n.ninja_has_died():
                died = True
                break
            if term or trunc:
                e.reset()
        assert died, "expected at least one death on a 13k-mine level in 5000 steps"
        assert e.is_current_state_doomed(1) is True
    finally:
        e.close()
