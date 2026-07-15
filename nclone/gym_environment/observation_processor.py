"""Layer 1 observation processor.

The policy observation dict is exactly four tensors produced inside
:class:`NppEnvironment` (plus control masks), scoped to the generalist agent's
mechanics {toggle mines, one exit switch, one exit door, one player spawn, tiles}:

* ``tile_grid`` — ``Box(0, 37, (42, 23), int8)`` — raw tile IDs, axis order (W, H)
* ``entity_overlays`` — ``Box(0, 1, (5, 42, 23), float32)`` — ch0 ninja, ch1 exit
  switch, ch2 exit door (all bilinear sub-tile splats), ch3 deadly-mine density,
  ch4 safe-mine density; axis order (C, W, H)
* ``hazard_map`` — ``Box(-1, 1, (2, 168, 92), float32)`` — ch0 deadly-mine SDF,
  ch1 safe-mine SDF at 6px; axis order (C, W, H) to co-register with the grids
* ``game_state`` — ``Box(-1, 1, (49,), float32)`` — see
  :data:`nclone.gym_environment.constants.GAME_STATE_CHANNELS` (23 ninja physics +
  time_remaining + 3 physics-derived + switch_activated + ninja-relative switch
  vector (2) + ninja-relative exit vector (2) + 8 surface ray-casts (compass
  distance to nearest solid surface) + 4-dim free-space nav hint
  ``[dx_hop, dy_hop, cost_norm, phase]`` toward the current objective +
  nearest-deadly-mine distance (1) + closing speed toward the objective (1)).

and the masks ``action_mask`` / ``duration_mask`` used for valid-action filtering
and FiGAR macro-actions. This processor:

* normalizes the ninja state vector,
* packs entity positions for internal bookkeeping, and
* passes through the Layer 1 tensors plus control masks.

Any other keys the environment still produces (``entities``, ``locked_doors``,
``locked_door_switches``, etc.) are filtered out here so they never reach the
policy. Rendering, visual frames, graph tensors, reachability vectors,
``spatial_context``, ``mine_sdf_features`` and legacy placeholders are not
produced by Layer 1.
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
        "lattice_nav",  # optional d* nav hint (only present when enabled)
        "waypoint_nav",  # optional waypoint-subgoal nav hint (only when enabled)
        "cost_to_go_map",  # optional Phase-2 O2 global cost-to-go field (only when enabled)
        "visitation_map",  # optional v17 per-episode episodic visitation grid (only when enabled)
        "frontier_map",  # optional v18 C1 reachable-and-unvisited frontier grid (only when enabled)
        "ego_crop",  # optional ego-centric high-res tile crop (only when enabled)
        "ego_crop_mines",  # optional mine-density crop co-registered with ego_crop
        "ego_crop_route",  # optional grad(d*) unit-vector crop (v14 B2)
        "ego_crop_route_fine",  # optional 6px un-pooled grad(d*) crop (v22 §3-C)
        "ego_crop_coarse",  # optional +-21-tile coarse window (Audit II P1-6)
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
