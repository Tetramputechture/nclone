"""Shared constants and pure geometry helpers for the TAS editor.

Kept dependency-free (no pygame) so both the controller (hit-testing mouse
clicks) and the renderer (drawing the piano-roll) agree on the exact same
layout without duplicating the math.
"""

from typing import Optional, Tuple

# Discrete action (0-5) -> (horizontal_input, jump_input), matching
# nclone.gym_environment.base_environment action mapping.
ACTION_TO_INPUTS = {
    0: (0, 0),    # NOOP
    1: (-1, 0),   # LEFT
    2: (1, 0),    # RIGHT
    3: (0, 1),    # JUMP
    4: (-1, 1),   # JUMP + LEFT
    5: (1, 1),    # JUMP + RIGHT
}

# Inverse of (horizontal_input, jump_input) -> action.
INPUTS_TO_ACTION = {v: k for k, v in ACTION_TO_INPUTS.items()}

# Compact human-readable glyphs for the piano-roll cells / HUD.
ACTION_GLYPH = {
    0: "·",    # NOOP  (middle dot)
    1: "←",    # LEFT  (left arrow)
    2: "→",    # RIGHT (right arrow)
    3: "↑",    # JUMP  (up arrow)
    4: "↖",    # JUMP+LEFT  (up-left arrow)
    5: "↗",    # JUMP+RIGHT (up-right arrow)
}

# ASCII fallbacks (some pygame default fonts lack the arrow glyphs).
ACTION_LABEL = {
    0: "-",
    1: "L",
    2: "R",
    3: "J",
    4: "JL",
    5: "JR",
}

ACTION_NAME = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
    3: "JUMP",
    4: "JUMP+LEFT",
    5: "JUMP+RIGHT",
}


def inputs_to_action(horizontal: int, jump: int) -> int:
    """Map a (horizontal, jump) pending-input pair to a discrete action."""
    return INPUTS_TO_ACTION.get((horizontal, 1 if jump else 0), 0)


# ---- Piano-roll geometry -------------------------------------------------

# Cells are square-ish; the strip sits along the bottom of the window.
CELL_W = 26
CELL_H = 24
STRIP_MARGIN = 10          # gap from window edges
HUD_HEIGHT = 26            # text row above the strip
STRIP_PAD_BOTTOM = 8


def strip_rect(screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the piano-roll cell strip."""
    x = STRIP_MARGIN
    w = screen_w - 2 * STRIP_MARGIN
    y = screen_h - STRIP_PAD_BOTTOM - CELL_H
    return x, y, w, CELL_H


def visible_window(screen_w: int, cursor: int, length: int) -> Tuple[int, int]:
    """Return (start_frame, visible_count) for the cell strip.

    Centers the cursor when possible, clamping so the window stays within
    [0, length]. ``length`` is the number of frames in the movie; the strip can
    also show the one-past-the-end "tip" slot, so frames range over [0, length].
    """
    _, _, w, _ = strip_rect(screen_w, 0)
    visible = max(1, w // CELL_W)
    total = length + 1  # include the tip slot
    if total <= visible:
        return 0, visible
    start = cursor - visible // 2
    start = max(0, min(start, total - visible))
    return start, visible


def frame_at_pixel(
    mx: int, my: int, screen_w: int, screen_h: int, cursor: int, length: int
) -> Optional[int]:
    """Hit-test a mouse pixel against the strip; return frame index or None."""
    sx, sy, sw, sh = strip_rect(screen_w, screen_h)
    if not (sx <= mx < sx + sw and sy <= my < sy + sh):
        return None
    start, visible = visible_window(screen_w, cursor, length)
    idx = start + (mx - sx) // CELL_W
    if 0 <= idx <= length:
        return int(idx)
    return None
