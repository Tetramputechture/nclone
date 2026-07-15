"""Smoke tests for the TAS editor (nclone/tas_tool/).

Guards the determinism + editing invariants the frame-by-frame editor relies
on: action<->input round-trips, snapshot-based seek determinism, history edits
propagating to the trajectory, batch perform, truncate, payload shape, the
.replay save/load round-trip, and the controller key/mouse handlers.

The editor itself drives a real (deterministic) simulator, so these double as
an integration check that snapshot/restore + tick replay stay bit-stable.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest

pygame = pytest.importorskip("pygame")

# Headless SDL so a display can be created in CI.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from nclone.replay.gameplay_recorder import map_action_to_input  # noqa: E402
from nclone.tas_tool import controller as C  # noqa: E402
from nclone.tas_tool.engine import TASEngine  # noqa: E402
from nclone.tas_tool.layout import (  # noqa: E402
    ACTION_TO_INPUTS,
    frame_at_pixel,
    inputs_to_action,
)
from nclone.tas_tool.movie import TASMovie, input_byte_to_action  # noqa: E402
from nclone.tas_tool.overlay import TASViewState, build_tas_payload  # noqa: E402

_VALID_CATEGORIES = {
    "simplest",
    "simplest_few_mines",
    "simplest_with_mines",
    "simpler",
    "simple",
    "medium",
    "complex",
    "mine_heavy",
    "exploration",
}


@pytest.fixture
def env():
    """A deterministic visual-testing env with a known-good simple map.

    Aliases any curriculum category the registry doesn't know (this WIP branch
    weights an unregistered 'extremely_simple') down to 'simplest' so env
    construction is robust regardless of the active curriculum config.
    """
    import nclone.gym_environment.generator_registry as gr

    orig = gr.GeneratorRegistry.get_map

    def patched(self, category, seed):
        cat = category if category in _VALID_CATEGORIES else "simplest"
        return orig(self, cat, seed)

    gr.GeneratorRegistry.get_map = patched
    try:
        from nclone.gym_environment.environment_factory import (
            create_visual_testing_env,
        )

        e = create_visual_testing_env()
        e.reset()
        yield e
    finally:
        gr.GeneratorRegistry.get_map = orig


def test_action_input_roundtrip():
    for a in range(6):
        assert input_byte_to_action(map_action_to_input(a)) == a
    for a, (h, j) in ACTION_TO_INPUTS.items():
        assert inputs_to_action(h, j) == a


def test_seek_and_rebuild_determinism(env):
    eng = TASEngine(env)
    eng.perform(2, repeat=12)  # RIGHT x12
    x_tip = eng.current_frame_state().x

    eng.back(5)
    eng.forward(5)
    assert eng.current_frame_state().x == pytest.approx(x_tip)

    # A full rebuild must reproduce the same mid-movie state.
    eng.seek(7)
    x7 = eng.current_frame_state().x
    eng.rebuild()
    eng.seek(7)
    assert eng.current_frame_state().x == pytest.approx(x7)


def test_history_edit_changes_trajectory(env):
    eng = TASEngine(env)
    eng.perform(2, repeat=10)
    before = [(f.x, f.y) for f in eng.frames]
    eng.set_action(3, 1)  # overwrite frame 3 RIGHT -> LEFT
    eng.ensure_fresh()
    after = [(f.x, f.y) for f in eng.frames]
    assert before != after
    assert eng.length == 10  # editing an input does not change movie length


def test_batch_insert_delete_truncate(env):
    eng = TASEngine(env)
    eng.perform(0, repeat=30)
    assert eng.length == 30 and eng.cursor == 30

    eng.insert(5, 2)
    eng.ensure_fresh()
    assert eng.length == 31 and eng.actions[5] == 2

    eng.delete(5)
    eng.ensure_fresh()
    assert eng.length == 30

    eng.truncate(20)
    eng.ensure_fresh()
    assert eng.length == 20 and eng.cursor <= 20


def test_payload_and_render(env):
    eng = TASEngine(env)
    eng.perform(5, repeat=8)
    view = TASViewState()
    view.pending_hor = 1
    payload = build_tas_payload(eng, view)
    for key in ("trajectory", "cursor", "length", "actions", "status"):
        assert key in payload
    assert len(payload["trajectory"]) == eng.length + 1

    # The overlay must composite without error (covers the draw path).
    env.set_tas_overlay(payload, enabled=True)
    assert env.render() is not None


def test_movie_save_load_roundtrip(env):
    eng = TASEngine(env)
    eng.perform(2, repeat=6)
    eng.perform(5, repeat=4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "movie.replay")
        eng.to_movie(level_id="t").save(path)
        loaded = TASMovie.load(path)
    assert loaded.actions == eng.actions


def test_controller_handlers(env):
    eng = TASEngine(env)
    view = TASViewState()
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        out = Path(d)

        def kd(k):
            return SimpleNamespace(type=pygame.KEYDOWN, key=k)

        # Build pending JUMP+RIGHT and commit a batch of 3.
        C._handle_key(eng, view, kd(pygame.K_RIGHT), out, "lvl")
        C._handle_key(eng, view, kd(pygame.K_UP), out, "lvl")
        assert view.pending_action() == 5
        C._handle_key(eng, view, kd(pygame.K_3), out, "lvl")
        C._handle_key(eng, view, kd(pygame.K_SPACE), out, "lvl")
        assert eng.length == 3 and eng.cursor == 3

        # Undo / redo scrub.
        C._handle_key(eng, view, kd(pygame.K_BACKSPACE), out, "lvl")
        assert eng.cursor == 2
        C._handle_key(eng, view, kd(pygame.K_PERIOD), out, "lvl")
        assert eng.cursor == 3

        # Save then load the most recent movie.
        C._handle_key(eng, view, kd(pygame.K_s), out, "lvl")
        assert list(out.glob("*.replay"))
        C._handle_key(eng, view, kd(pygame.K_l), out, "lvl")

        # Reset clears, Q signals quit.
        C._handle_key(eng, view, kd(pygame.K_r), out, "lvl")
        assert eng.length == 0
        assert C._handle_key(eng, view, kd(pygame.K_q), out, "lvl") is False


def _find_lethal_engine():
    """Build an engine on a mine-heavy map driven into a death; return it."""
    import random

    from nclone.gym_environment.environment_factory import create_visual_testing_env
    from nclone.gym_environment.generator_registry import GeneratorRegistry

    reg = GeneratorRegistry(random.Random(0))
    for seed in range(80):
        e = create_visual_testing_env()
        e.nplay_headless.load_map_from_map_data(reg.get_map("mine_heavy", seed).map_data())
        eng = TASEngine(e)
        rng = random.Random(seed * 3 + 1)
        for _ in range(200):
            eng.perform(rng.choice([2, 5, 3, 0, 2]), repeat=1)
            if eng.dead:
                return e, eng
    return None, None


def test_perform_while_dead_is_noop_and_undo_revives(env):
    e, eng = _find_lethal_engine()
    if eng is None:
        pytest.skip("no deterministic death found on mine-heavy maps")
    death_len = eng.length
    assert eng.dead and e.nplay_headless.ninja_has_died()

    # Mashing inputs into a dead ninja must not extend the movie.
    eng.perform(2, repeat=5)
    assert eng.length == death_len, "perform extended the movie past death"

    # A single undo from the death frame revives the live ninja.
    eng.back(1)
    assert not eng.dead
    assert not e.nplay_headless.ninja_has_died()
    assert e.nplay_headless.sim.ninja.state not in (6, 7)


def test_predict_returns_path_and_restores_state(env):
    eng = TASEngine(env)
    eng.perform(5, repeat=6)  # JUMP+RIGHT to get airborne
    before = (eng.cursor, eng.current_frame_state().x, eng.current_frame_state().y)
    live_before = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)

    pts = eng.predict(0, n=30)
    assert isinstance(pts, list) and len(pts) > 0
    assert all(len(p) == 3 for p in pts)

    # Prediction must not disturb the live sim or the cursor.
    after = (eng.cursor, eng.current_frame_state().x, eng.current_frame_state().y)
    live_after = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
    assert before == after
    assert live_before == live_after


def test_mouse_click_seeks(env):
    eng = TASEngine(env)
    eng.perform(0, repeat=20)
    surf = pygame.display.get_surface()
    w, h = surf.get_size()
    # Click the first visible cell on the timeline strip.
    target = frame_at_pixel(15, h - 8 - 12, w, h, eng.cursor, eng.length)
    assert target is not None
    C._handle_click(eng, (15, h - 8 - 12))
    assert eng.cursor == target
