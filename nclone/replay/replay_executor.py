"""Deterministic Replay Executor.

Replays stored input sequences against map data to regenerate Layer 1 policy
observations. Uses `NppEnvironment` internally so that the produced observation
dict matches training exactly: `tile_grid`, `entity_overlays`, `hazard_map`,
`game_state`, `action_mask`.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..gym_environment.config import EnvironmentConfig
from ..gym_environment.npp_environment import NppEnvironment


def map_input_to_action(input_byte: int) -> int:
    """Map N++ input byte (0-7) to the 6-way discrete action space.

    Inverse of `map_action_to_input` from `gameplay_recorder.py`.
    """
    input_to_action_map = {
        0: 0,  # 000: NOOP
        1: 3,  # 001: JUMP
        2: 2,  # 010: RIGHT
        3: 5,  # 011: RIGHT+JUMP
        4: 1,  # 100: LEFT
        5: 4,  # 101: LEFT+JUMP
        6: 0,  # 110: invalid -> NOOP
        7: 0,  # 111: invalid -> NOOP
    }
    return input_to_action_map.get(input_byte, 0)


def decode_input_to_controls(input_byte: int) -> tuple:
    """Decode input byte to (horizontal, jump) controls.

    Retained for callers that still drive `nplay_headless.tick()` directly.
    """
    jump = 1 if (input_byte & 0x01) else 0
    right = 1 if (input_byte & 0x02) else 0
    left = 1 if (input_byte & 0x04) else 0

    if left and right:
        horizontal = 0
    elif left:
        horizontal = -1
    elif right:
        horizontal = 1
    else:
        horizontal = 0

    return horizontal, jump


class ReplayExecutor:
    """Executes replays deterministically to regenerate Layer 1 observations."""

    def __init__(
        self,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "rgb_array",
        enable_rendering: bool = False,
        frame_skip: int = 1,
    ):
        """Initialize replay executor.

        Args:
            observation_config: Ignored, kept for API compatibility.
            render_mode: Only used when ``enable_rendering`` is True. Layer 1
                observations are never image-based, so rendering is optional.
            enable_rendering: If True, creates the env in ``human``-style render
                mode so callers can grab raw frames via ``env.render()``.
            frame_skip: Number of physics ticks per emitted observation.
        """
        del observation_config  # unused
        self.frame_skip = max(1, frame_skip)

        from ..gym_environment.config import RenderConfig

        render_kwargs: Dict[str, Any] = {}
        if enable_rendering:
            render_kwargs["render"] = RenderConfig(render_mode=render_mode)

        cfg = EnvironmentConfig.for_training(**render_kwargs)
        self.env = NppEnvironment(cfg)
        # Skip PBRS/A* work during replay ingestion — we only need raw obs.
        self.env.skip_reward_calculation = True

    def execute_replay(
        self,
        map_data: bytes,
        input_sequence: List[int],
    ) -> List[Dict[str, Any]]:
        """Execute a replay and return Layer 1 observations per sampled frame.

        Args:
            map_data: Raw map data bytes.
            input_sequence: One input byte per simulator tick.

        Returns:
            Sampled list of ``{"observation", "action", "frame"}`` dicts.
            Observation keys: ``tile_grid``, ``entity_overlays``, ``hazard_map``,
            ``game_state``, ``action_mask``.
        """
        # Inject the custom map then reset the env without reloading a level.
        self.env.nplay_headless.load_map_from_map_data(list(map_data))
        self.env.reset(options={"skip_map_load": True, "new_level": True})

        observations: List[Dict[str, Any]] = []
        for frame_idx, input_byte in enumerate(input_sequence):
            action = map_input_to_action(input_byte)
            obs, _reward, terminated, truncated, _info = self.env.step(action)

            if (frame_idx + 1) % self.frame_skip == 0:
                observations.append(
                    {
                        "observation": {
                            key: np.asarray(value).copy()
                            if isinstance(value, np.ndarray)
                            else value
                            for key, value in obs.items()
                        },
                        "action": action,
                        "frame": frame_idx,
                    }
                )

            if terminated or truncated:
                break

        return observations

    def close(self) -> None:
        """Release the underlying environment."""
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self.env = None
