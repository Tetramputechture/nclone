"""Layer 1 observation processor.

After the strict Layer 1 cleanup the policy observation dict is exactly four
tensors produced inside :class:`NppEnvironment` (plus control masks):

* ``tile_grid`` — ``Box(0, 33, (42, 23), int8)``
* ``entity_overlays`` — ``Box(0, 1, (5, 42, 23), float32)``
* ``hazard_map`` — ``Box(-1, 1, (1, 168, 92), float32)``
* ``game_state`` — ``Box(-1, 1, (27,), float32)``

and the masks ``action_mask`` / ``duration_mask`` used for valid-action filtering
and FiGAR macro-actions. This processor:

* normalizes the ninja state vector,
* packs entity positions for internal bookkeeping, and
* passes through the Layer 1 tensors plus control masks.

Rendering (``obs["screen"]``), visual frames (``player_frame`` / ``global_view``),
graph tensors, reachability vectors, spatial_context, mine_sdf_features and
legacy placeholders are no longer produced.
"""

import logging
import numpy as np
from typing import Any, Dict

from .constants import GAME_STATE_CHANNELS, LEVEL_WIDTH, LEVEL_HEIGHT

# Entity position array indices
# Format: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
NINJA_POS_IDX = 0
SWITCH_POS_IDX = 2
EXIT_POS_IDX = 4
ENTITY_POSITIONS_SIZE = 6

logger = logging.getLogger(__name__)


class ObservationProcessor:
    """Builds the Layer 1 observation dict from the raw environment output."""

    # Layer 1 observation keys that `NppEnvironment._get_observation` may include.
    # Anything outside this list is dropped from the policy-facing dict.
    _LAYER1_PASSTHROUGH_KEYS = (
        "tile_grid",
        "entity_overlays",
        "hazard_map",
        "action_mask",
        "duration_mask",
    )

    def __init__(self) -> None:
        # Pre-allocated buffer reused across observations to minimize churn.
        self._entity_positions_buffer = np.zeros(
            ENTITY_POSITIONS_SIZE, dtype=np.float32
        )

    def process_game_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize the ninja state vector (GAME_STATE_CHANNELS)."""
        raw = obs["game_state"]
        ninja_state = raw[:GAME_STATE_CHANNELS] if len(raw) >= GAME_STATE_CHANNELS else raw
        return np.asarray(ninja_state, dtype=np.float32)

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Reduce the raw observation dict to the Layer 1 policy contract.

        Returns a dict containing `game_state`, `entity_positions` (internal only)
        plus whichever Layer 1 keys the environment produced.
        """
        # Normalize entity positions to [0, 1] for internal bookkeeping.
        ninja_x_norm = obs["player_x"] / LEVEL_WIDTH
        ninja_y_norm = obs["player_y"] / LEVEL_HEIGHT
        switch_x_norm = obs["switch_x"] / LEVEL_WIDTH
        switch_y_norm = obs["switch_y"] / LEVEL_HEIGHT
        exit_x_norm = obs["exit_door_x"] / LEVEL_WIDTH
        exit_y_norm = obs["exit_door_y"] / LEVEL_HEIGHT

        self._entity_positions_buffer[NINJA_POS_IDX : NINJA_POS_IDX + 2] = [
            ninja_x_norm,
            ninja_y_norm,
        ]
        self._entity_positions_buffer[SWITCH_POS_IDX : SWITCH_POS_IDX + 2] = [
            switch_x_norm,
            switch_y_norm,
        ]
        self._entity_positions_buffer[EXIT_POS_IDX : EXIT_POS_IDX + 2] = [
            exit_x_norm,
            exit_y_norm,
        ]
        entity_positions = np.array(self._entity_positions_buffer, copy=True)

        result: Dict[str, np.ndarray] = {
            "game_state": self.process_game_state(obs),
            "entity_positions": entity_positions,
        }

        for key in self._LAYER1_PASSTHROUGH_KEYS:
            if key in obs:
                result[key] = obs[key]

        return result

    def reset(self) -> None:
        """Reset processor state and clear buffers."""
        self._entity_positions_buffer.fill(0)
