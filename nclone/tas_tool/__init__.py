"""TAS (Tool-Assisted Speedrun) editor for the nclone test environment.

A frame-by-frame, emulator-style input editor layered onto the deterministic
simulator. Lets you perform/undo single ticks, commit an input for a batch of
frames, scrub and rewrite the input history, and watch the resulting ninja
trajectory drawn over the level.

Entry point: ``run_tas_editor(env, ...)`` (see :mod:`nclone.tas_tool.controller`),
wired into ``test_environment.py`` via the ``--tas`` flag.
"""

from .engine import TASEngine, FrameState
from .movie import TASMovie

__all__ = ["TASEngine", "FrameState", "TASMovie", "run_tas_editor"]


def run_tas_editor(env, **kwargs):
    """Lazily import and launch the TAS editor loop (avoids a hard pygame import)."""
    from .controller import run_tas_editor as _run

    return _run(env, **kwargs)
