"""Live read-out of the ninja's input buffers and contact state.

WHY THIS EXISTS. Jumps in N++ fire on a rising edge, and whether that edge does
anything depends on four countdown buffers and on what the ninja is touching.
None of that is visible on screen: a wall-jump that "should have worked" and one
that was two frames late look identical in the render, and the difference is a
number that lives for five frames. Every timing question asked of this simulator
-- bounce-block pump timing, wall-jump chains, why an action mask says a jump is
useless -- has bottomed out in "what was the buffer state on that frame".

The buffers all follow the same scheme: an event sets the counter to 0, it
increments once per frame, and it is dead once it reaches its window. So the
counter is TIME ELAPSED, and the useful number -- what this shows -- is the
frames REMAINING before the window shuts.

Windows are not uniform: jump / floor / wall are 5 frames, launch pad is 4. That
asymmetry is easy to get wrong from memory, so it is taken from the same
predicate the simulator uses (:func:`nclone.ninja.has_active_jump_buffer`)
rather than restated here.

Pure formatting, no pygame: the dev panel wraps these strings in STATUS rows,
and the tests call them directly.
"""

from __future__ import annotations

from typing import List, Optional

from ..ninja import has_active_jump_buffer

#: 0-5 as set in `Ninja.__init__`. The names were duplicated as comments in five
#: places and defined in none; this is the one table.
STATE_NAMES = {
    0: "Immobile",
    1: "Running",
    2: "Ground slide",
    3: "Jumping",
    4: "Falling",
    5: "Wall slide",
}

#: (label, attribute, window). The window is the exclusive upper bound the
#: simulator tests with `-1 < b < window`, so a buffer is live for exactly
#: `window` frames and has `window - b` frames left.
BUFFERS = (
    ("jump", "jump_buffer", 5),
    ("floor", "floor_buffer", 5),
    ("wall", "wall_buffer", 5),
    ("pad", "launch_pad_buffer", 4),
)


def buffer_remaining(value: int, window: int) -> Optional[int]:
    """Frames left before this buffer shuts, or None when it is not armed.

    ``value`` is frames ELAPSED since the buffer armed (-1 means dormant), so a
    freshly armed buffer (0) has the full window left.
    """
    if not -1 < value < window:
        return None
    return window - value


def buffer_bar(value: int, window: int) -> str:
    """`[####-]` style meter -- one cell per frame, filled = frames remaining."""
    left = buffer_remaining(value, window)
    if left is None:
        return "[" + "-" * window + "]"
    return "[" + "#" * left + "-" * (window - left) + "]"


def _fmt_normal(x: float, y: float) -> str:
    return f"({x:+.2f},{y:+.2f})"


def ninja_state_lines(n) -> List[str]:
    """Panel lines for a ninja-like object (real Ninja or native _NinjaProxy).

    Every field read here exists on BOTH, so the read-out cannot silently become
    a python-backend-only feature.
    """
    if n is None:
        return ["no ninja"]

    state = int(getattr(n, "state", 0))
    lines = [f"state {state} {STATE_NAMES.get(state, '?')}"]

    airborn = bool(getattr(n, "airborn", False))
    walled = bool(getattr(n, "walled", False))
    lines.append(
        f"air {'Y' if airborn else 'n'}  "
        f"wall {'Y' if walled else 'n'}  "
        f"wn {int(getattr(n, 'wall_normal', 0)):+d}"
    )

    # Pad the bar field to the widest window so the frame counts line up. The
    # bar itself stays its true width -- the launch pad's shorter meter is the
    # visible reminder that its window really is 4 and not 5.
    widest = max(w for _, _, w in BUFFERS) + 2
    for label, attr, window in BUFFERS:
        value = int(getattr(n, attr, -1))
        left = buffer_remaining(value, window)
        lines.append(
            f"{label:<5} {buffer_bar(value, window):<{widest}} "
            f"{'-' if left is None else str(left) + 'f'}"
        )

    # The headline: would a fresh jump press do anything at all this frame? Read
    # from the simulator's own predicate so the panel cannot disagree with the
    # action mask or the game_state observation.
    try:
        armed = has_active_jump_buffer(n)
    except AttributeError:
        armed = False
    sources = [
        label
        for label, attr, window in BUFFERS
        if buffer_remaining(int(getattr(n, attr, -1)), window) is not None
    ]
    lines.append(
        f"JUMP ARMED ({','.join(sources)})" if armed else "jump buffers idle"
    )
    if getattr(n, "is_jump_useless", False):
        lines.append("jump press = NO-OP")
    if getattr(n, "last_jump_was_buffered", False):
        lines.append("last jump: buffered")

    lines.append(
        f"contact f{int(getattr(n, 'floor_count', 0))} "
        f"w{int(getattr(n, 'wall_count', 0))} "
        f"c{int(getattr(n, 'ceiling_count', 0))}"
    )
    lines.append(
        f"airborne {int(getattr(n, 'frames_airborne', 0))}f  "
        f"jdur {int(getattr(n, 'jump_duration', 0))}"
    )
    lines.append(
        f"floor n {_fmt_normal(float(getattr(n, 'floor_normalized_x', 0.0)), float(getattr(n, 'floor_normalized_y', -1.0)))}"
    )
    lines.append(
        f"ceil  n {_fmt_normal(float(getattr(n, 'ceiling_normalized_x', 0.0)), float(getattr(n, 'ceiling_normalized_y', 1.0)))}"
    )
    return lines
