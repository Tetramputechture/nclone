"""The dev panel's buffer / contact read-out.

The point of these is that the panel cannot quietly start lying. A read-out that
disagrees with the simulator is worse than none: it is the instrument every
timing question gets settled with.
"""

import pytest

from nclone.dev_tool.ninja_state import (
    BUFFERS,
    STATE_NAMES,
    buffer_bar,
    buffer_remaining,
    ninja_state_lines,
)
from nclone.ninja import has_active_jump_buffer


class _FakeNinja:
    """Minimal ninja-like object; every field the read-out reads."""

    def __init__(self, **kw):
        self.state = 0
        self.airborn = False
        self.walled = False
        self.wall_normal = 0
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.is_jump_useless = False
        self.last_jump_was_buffered = False
        self.floor_count = 0
        self.wall_count = 0
        self.ceiling_count = 0
        self.frames_airborne = 0
        self.jump_duration = 0
        self.floor_normalized_x = 0.0
        self.floor_normalized_y = -1.0
        self.ceiling_normalized_x = 0.0
        self.ceiling_normalized_y = 1.0
        for k, v in kw.items():
            setattr(self, k, v)


def test_dormant_buffer_has_no_remaining():
    assert buffer_remaining(-1, 5) is None


def test_freshly_armed_buffer_has_the_whole_window():
    # The counter is time ELAPSED, so 0 means the full window is still ahead.
    assert buffer_remaining(0, 5) == 5


def test_buffer_expires_at_the_window_edge():
    assert buffer_remaining(4, 5) == 1
    assert buffer_remaining(5, 5) is None


def test_launch_pad_window_is_four_not_five():
    # The asymmetry that is easy to get wrong from memory.
    assert buffer_remaining(3, 4) == 1
    assert buffer_remaining(4, 4) is None


def test_bar_width_is_one_cell_per_frame():
    assert buffer_bar(-1, 5) == "[-----]"
    assert buffer_bar(0, 5) == "[#####]"
    assert buffer_bar(3, 5) == "[##---]"
    assert buffer_bar(-1, 4) == "[----]"


@pytest.mark.parametrize("label,attr,window", BUFFERS)
def test_each_buffer_matches_the_simulators_own_predicate(label, attr, window):
    """`buffer_remaining` must agree with `has_active_jump_buffer` frame for frame.

    This is the assertion that keeps the panel honest: both sides are derived
    from the same windows, so a change to one that is not made to the other
    fails here rather than showing a wrong number during play.
    """
    for value in range(-1, 8):
        n = _FakeNinja(**{attr: value})
        armed_here = buffer_remaining(value, window) is not None
        assert has_active_jump_buffer(n) is armed_here, (
            f"{attr}={value}: panel says armed={armed_here}, "
            f"simulator says {has_active_jump_buffer(n)}"
        )


def test_windows_match_the_simulator_source():
    """Guard the window table itself against drift in `has_active_jump_buffer`."""
    for label, attr, window in BUFFERS:
        # one frame inside the window is armed, one frame past it is not
        assert has_active_jump_buffer(_FakeNinja(**{attr: window - 1}))
        assert not has_active_jump_buffer(_FakeNinja(**{attr: window}))


def test_lines_report_armed_with_the_source():
    lines = ninja_state_lines(_FakeNinja(wall_buffer=1))
    assert any("JUMP ARMED" in ln and "wall" in ln for ln in lines)


def test_lines_report_idle_when_no_buffer_is_live():
    lines = ninja_state_lines(_FakeNinja())
    assert any("jump buffers idle" in ln for ln in lines)


def test_lines_flag_a_useless_jump():
    lines = ninja_state_lines(_FakeNinja(is_jump_useless=True))
    assert any("NO-OP" in ln for ln in lines)


def test_lines_name_the_movement_state():
    lines = ninja_state_lines(_FakeNinja(state=5))
    assert any("Wall slide" in ln for ln in lines)
    assert set(STATE_NAMES) == {0, 1, 2, 3, 4, 5}


def test_lines_survive_a_none_ninja():
    assert ninja_state_lines(None) == ["no ninja"]


def test_lines_fit_the_panel_width():
    """~36 monospace chars fit in the 320px panel; longer lines are clipped."""
    n = _FakeNinja(
        state=5, airborn=True, walled=True, wall_normal=-1,
        jump_buffer=0, floor_buffer=1, wall_buffer=2, launch_pad_buffer=1,
        floor_count=9, wall_count=9, ceiling_count=9, frames_airborne=999,
        jump_duration=99, last_jump_was_buffered=True, is_jump_useless=True,
        floor_normalized_x=-0.707, floor_normalized_y=-0.707,
    )
    for line in ninja_state_lines(n):
        assert len(line) <= 36, f"{line!r} is {len(line)} chars"
