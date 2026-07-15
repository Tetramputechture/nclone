"""Interactive TAS editor loop (pygame), wired into test_environment.py.

Takes over event handling for the already-initialised pygame window and drives
the :class:`~nclone.tas_tool.engine.TASEngine`. The simulator's deterministic
snapshot/replay does all the heavy lifting; this module is just input handling,
the play/pause clock, and overlay refresh.

Controls
--------
  ← / →          set/clear pending horizontal input (toggle)
  ↑              toggle pending jump
  C              clear pending to NOOP
  SPACE / RETURN commit pending for N frames (N = typed batch, else 1)
  , / .          scrub cursor back / forward one frame (no edit)
  BACKSPACE      undo (scrub back one frame)
  HOME / END     jump cursor to start / tip
  O              overwrite frame at cursor with pending (no advance)
  I              insert pending frame at cursor
  DELETE / K     delete frame at cursor
  T              truncate movie at cursor (drop the tail)
  0-9            build a batch count (shown in HUD); ESC cancels it
  P              play / pause auto-advance from the cursor
  S              save movie to .replay
  L              load the most recent .replay from the output dir
  R              reset movie (clear all inputs)
  Q / window-X   quit the editor
  mouse click    on the timeline strip -> seek to that frame
"""

import time
from pathlib import Path
from typing import Optional

import pygame

from .engine import TASEngine
from .movie import TASMovie
from .overlay import TASViewState, build_tas_payload


def run_tas_editor(
    env,
    initial_actions=None,
    output_dir: str = "datasets/tas_movies",
    level_id: Optional[str] = None,
    fps: int = 60,
):
    """Run the interactive TAS editor against ``env`` (current loaded map).

    Returns the final :class:`~nclone.tas_tool.engine.TASEngine` (useful for
    tests / programmatic callers).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(_help_text())
    engine = TASEngine(env, actions=initial_actions)
    view = TASViewState()
    clock = pygame.time.Clock()

    running = True
    last_play_step = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                _handle_click(engine, event.pos)
            elif event.type == pygame.KEYDOWN:
                running = _handle_key(engine, view, event, out_dir, level_id)

        # Auto-play: advance one stored frame per tick until tip / terminal.
        if view.playing:
            now = time.perf_counter()
            if now - last_play_step >= 1.0 / max(1, fps):
                if engine.at_tip or engine.won or engine.dead:
                    view.playing = False
                else:
                    engine.forward(1)
                    last_play_step = now

        engine.ensure_fresh()
        # Project a NOOP fall/coast arc ahead of the cursor (restores live state).
        prediction = (
            engine.predict(0, view.predict_frames) if view.show_prediction else []
        )
        env.set_tas_overlay(
            build_tas_payload(engine, view, prediction=prediction), enabled=True
        )
        env.render()
        clock.tick(fps)

    env.set_tas_overlay(None, enabled=False)
    return engine


# ---- event handlers ------------------------------------------------------


def _handle_click(engine: TASEngine, pos):
    from .layout import frame_at_pixel

    surf = pygame.display.get_surface()
    if surf is None:
        return
    w, h = surf.get_size()
    frame = frame_at_pixel(pos[0], pos[1], w, h, engine.cursor, engine.length)
    if frame is not None:
        engine.ensure_fresh()
        engine.seek(frame)


def _handle_key(engine, view, event, out_dir, level_id) -> bool:
    """Process one KEYDOWN. Returns False to quit the editor."""
    key = event.key

    # Batch-count digits.
    if pygame.K_0 <= key <= pygame.K_9:
        view.push_digit(key - pygame.K_0)
        return True

    if key == pygame.K_q:
        return False
    if key == pygame.K_ESCAPE:
        view.reset_batch()
        view.clear_pending()
        view.notify("cancelled")
        return True

    # Pending-input editing.
    if key == pygame.K_LEFT:
        view.toggle_dir(-1)
    elif key == pygame.K_RIGHT:
        view.toggle_dir(1)
    elif key == pygame.K_UP:
        view.toggle_jump()
    elif key == pygame.K_c:
        view.clear_pending()

    # Commit pending (frame advance, possibly batched).
    elif key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
        engine.ensure_fresh()
        n = view.commit_batch()
        engine.perform(view.pending_action(), repeat=n)
        view.notify(f"+{n} frame(s)")

    # Scrub.
    elif key == pygame.K_PERIOD:
        engine.ensure_fresh()
        engine.forward(1)
    elif key == pygame.K_COMMA or key == pygame.K_BACKSPACE:
        engine.ensure_fresh()
        engine.back(1)
    elif key == pygame.K_HOME:
        engine.ensure_fresh()
        engine.seek(0)
    elif key == pygame.K_END:
        engine.ensure_fresh()
        engine.seek(engine.length)

    # History edits at the cursor.
    elif key == pygame.K_o:
        engine.set_action(engine.cursor, view.pending_action())
        view.notify("overwrote frame")
    elif key == pygame.K_i:
        engine.insert(engine.cursor, view.pending_action())
        view.notify("inserted frame")
    elif key in (pygame.K_DELETE, pygame.K_k):
        engine.delete(engine.cursor)
        view.notify("deleted frame")
    elif key == pygame.K_t:
        engine.truncate(engine.cursor)
        view.notify("truncated tail")

    # Playback.
    elif key == pygame.K_p:
        view.playing = not view.playing
        view.notify("play" if view.playing else "pause")

    # NOOP fall-prediction ghost toggle.
    elif key == pygame.K_g:
        view.show_prediction = not view.show_prediction
        view.notify("predict on" if view.show_prediction else "predict off")

    # File ops.
    elif key == pygame.K_s:
        _save(engine, view, out_dir, level_id)
    elif key == pygame.K_l:
        _load(engine, view, out_dir)
    elif key == pygame.K_r:
        engine.clear()
        view.notify("movie reset")

    return True


# ---- file ops ------------------------------------------------------------


def _timestamp_name() -> str:
    from datetime import datetime

    return datetime.now().strftime("tas_%Y%m%d_%H%M%S")


def _save(engine, view, out_dir, level_id):
    path = out_dir / f"{_timestamp_name()}.replay"
    movie = engine.to_movie(level_id=level_id)
    movie.save(str(path), success=engine.won)
    view.notify(f"saved {path.name} ({engine.length}f)")
    print(f"💾 TAS movie saved: {path}  ({engine.length} frames, won={engine.won})")


def _load(engine, view, out_dir):
    replays = sorted(out_dir.glob("*.replay"))
    if not replays:
        view.notify("no .replay to load")
        return
    path = replays[-1]
    movie = TASMovie.load(str(path))
    engine.load_actions(movie.actions)
    view.notify(f"loaded {path.name} ({engine.length}f)")
    print(f"📂 TAS movie loaded: {path}  ({engine.length} frames)")


def _help_text() -> str:
    return (
        "\n=== TAS EDITOR ===\n"
        "  arrows: set pending input (←/→ dir toggle, ↑ jump toggle), C clear\n"
        "  SPACE/ENTER: commit pending xN   ,/. scrub   BKSP undo   HOME/END ends\n"
        "  O overwrite  I insert  DEL/K delete  T truncate-tail\n"
        "  digits: batch count   P play/pause   G NOOP fall-predict\n"
        "  S save  L load  R reset  Q quit   mouse: click timeline to seek\n"
        "==================\n"
    )
