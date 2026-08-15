"""Regression: every N++ input-byte decoder in the tree must agree on all 8 bytes.

Pipeline audit P0.1. There are FOUR decoders of the same 3-bit input byte:

  * `replay_executor.decode_input_to_controls`  (byte -> (horizontal, jump))
  * `replay_executor.map_input_to_action`       (byte -> 6-way discrete action)
  * `replay.input_decoder.InputDecoder`         (the ntrace.py tables)
  * `render3d.replay_loader._HOR` / `_JUMP`     (playback)

Three of them resolved bytes 6 (left+right) and 7 (left+right+jump) to LEFT and
LEFT+JUMP, matching `Ninja::think` where the left bit wins outright. The fourth,
`map_input_to_action` -- the one `ReplayExecutor.execute_replay()` actually uses
to regenerate BC/DAPG actions -- called them "invalid" and returned NOOP.

That substitutes "do nothing" for "hold left" on 531/4,505 harvested tapes
(11.8%) and 15/22 first-party attract demos (68.2%) in the local corpora; see
`npp-rl/scripts/measure_replay_input_byte_census.py`. The planner corpora are
unaffected because they are ENCODED from the 6-way action space, which is
precisely why the defect could sit unnoticed.

Pinning agreement rather than one table's contents means any future edit to any
one decoder that reintroduces a divergence fails here.
"""
from __future__ import annotations

import pytest

from nclone.replay.gameplay_recorder import map_action_to_input
from nclone.replay.input_decoder import InputDecoder
from nclone.replay.replay_executor import (
    decode_input_to_controls,
    map_input_to_action,
)

ALL_BYTES = tuple(range(8))

# The 6-way discrete action space, as (horizontal, jump).
ACTION_TO_CONTROLS = {
    0: (0, 0),    # NOOP
    1: (-1, 0),   # LEFT
    2: (1, 0),    # RIGHT
    3: (0, 1),    # JUMP
    4: (-1, 1),   # LEFT+JUMP
    5: (1, 1),    # RIGHT+JUMP
}


@pytest.mark.parametrize("byte", ALL_BYTES)
def test_action_mapper_agrees_with_control_decoder(byte):
    """`map_input_to_action` must name the same (horizontal, jump) that
    `decode_input_to_controls` produces."""
    assert ACTION_TO_CONTROLS[map_input_to_action(byte)] == decode_input_to_controls(
        byte
    ), f"decoders disagree on input byte {byte} ({byte:03b})"


@pytest.mark.parametrize("byte", ALL_BYTES)
def test_ntrace_tables_agree_with_control_decoder(byte):
    """The ntrace.py tables are the reference the sim's replay path was ported
    from; the control decoder must reproduce them exactly."""
    expected = (InputDecoder.HOR_INPUTS_DIC[byte], InputDecoder.JUMP_INPUTS_DIC[byte])
    assert decode_input_to_controls(byte) == expected


@pytest.mark.parametrize("byte", ALL_BYTES)
def test_playback_tables_agree_with_control_decoder(byte):
    from nclone.render3d.replay_loader import _HOR, _JUMP

    assert decode_input_to_controls(byte) == (_HOR[byte], _JUMP[byte])


def test_left_wins_when_both_direction_bits_are_set():
    """The specific decomp behavior, stated once so the intent is not re-litigated:
    `Ninja::think` reads `hor = -1` unconditionally whenever the left bit is set,
    regardless of the right bit."""
    assert decode_input_to_controls(6) == (-1, 0)
    assert decode_input_to_controls(7) == (-1, 1)
    assert map_input_to_action(6) == 1   # LEFT
    assert map_input_to_action(7) == 4   # LEFT+JUMP


@pytest.mark.parametrize("action", sorted(ACTION_TO_CONTROLS))
def test_encode_decode_roundtrip_on_the_canonical_bytes(action):
    """`map_input_to_action` stays the exact inverse of `map_action_to_input` on
    the six bytes the recorder can emit -- the 6/7 fix must not disturb that."""
    assert map_input_to_action(map_action_to_input(action)) == action
