"""Load recorded replays into (map_data, [(hor, jump), ...]) for playback.

Currently supports the N++ attract-mode demo format under
``nclone/example_replays/npp_attract/`` (each file bundles a level + the demo
input sequence that solves it). The decoded input bytes (0-7) are turned into
(horizontal, jump) pairs the same way the sim's own replay path does.
"""

from __future__ import annotations

from pathlib import Path

# input byte (0-7) -> (horizontal, jump); bit0=jump, bit1=right, bit2=left
_HOR = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1}
_JUMP = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}


def load_attract_replay(path: str) -> tuple[list[int], list[tuple[int, int]], str]:
    """Parse an npp_attract demo file.

    Returns (map_data, inputs, level_name) where inputs is a list of
    (horizontal, jump) pairs, one per physics frame.
    """
    from nclone.replay.binary_replay_parser import BinaryReplayParser

    parser = BinaryReplayParser()
    raw_inputs, map_data, _level_id, name = parser.parse_single_replay_file(Path(path))
    inputs = [(_HOR[b], _JUMP[b]) for b in raw_inputs]
    return map_data, inputs, name.strip()
