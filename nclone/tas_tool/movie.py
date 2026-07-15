"""Load/save a TAS movie using the existing compact ``.replay`` V1 format.

A movie is just the level's raw map data plus a per-frame list of discrete
actions (0-5). We reuse :class:`nclone.replay.gameplay_recorder.CompactReplay`
so movies round-trip with the rest of the replay/recording tooling
(``ActionReplayer``, ``gameplay_recorder``, the dataset builders).

The on-disk byte stream stores N++ input bytes (bit0 jump, bit1 right, bit2
left); we convert to/from the editor's discrete action space on the way in/out.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..replay.gameplay_recorder import CompactReplay, map_action_to_input

# Inverse of map_action_to_input(): N++ input byte -> discrete action (0-5).
_INPUT_BYTE_TO_ACTION = {
    0: 0,  # NOOP
    4: 1,  # LEFT
    2: 2,  # RIGHT
    1: 3,  # JUMP
    5: 4,  # LEFT + JUMP
    3: 5,  # RIGHT + JUMP
}


def input_byte_to_action(byte: int) -> int:
    """Map an N++ input byte back to a discrete action (unknown -> NOOP)."""
    return _INPUT_BYTE_TO_ACTION.get(byte & 0x07, 0)


class TASMovie:
    """In-memory TAS movie: map data + discrete action list."""

    def __init__(
        self,
        map_data: bytes,
        actions: Optional[List[int]] = None,
        level_id: Optional[str] = None,
    ):
        self.map_data = bytes(map_data)
        self.actions: List[int] = list(actions or [])
        self.level_id = level_id

    # ---- persistence -----------------------------------------------------

    def save(self, path: str, success: bool = False) -> str:
        """Write the movie to ``path`` as a ``.replay`` V1 file."""
        replay = CompactReplay(
            episode_id=Path(path).stem,
            map_data=self.map_data,
            input_sequence=[map_action_to_input(a) for a in self.actions],
            level_id=self.level_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=success,
        )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(replay.to_binary())
        return str(out)

    @classmethod
    def load(cls, path: str) -> "TASMovie":
        """Load a ``.replay`` file into a movie (decoding inputs -> actions)."""
        with open(path, "rb") as f:
            replay = CompactReplay.from_binary(f.read(), episode_id=Path(path).stem)
        actions = [input_byte_to_action(b) for b in replay.input_sequence]
        return cls(replay.map_data, actions, level_id=replay.level_id)
