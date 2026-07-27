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

    # Reachability obs-exposure fields a caller may set so the replayed BC obs
    # MATCH the training/policy obs exactly. Defaults (all False) reproduce the
    # strict Layer-1 contract, so existing callers are unaffected.
    _OBS_EXPOSURE_FIELDS = (
        "use_jump_potential",
        "expose_cost_to_go_map",
        "expose_visitation_map",
        "expose_frontier_map",
        "expose_waypoint_features",
        "use_free_space_potential",
        "enable_ego_crop",
        "ego_crop_size",
        "enable_ego_crop_coarse",
        "ego_crop_coarse_route",
        "cost_map_norm",
        # v22 §3-B/C: obs-dim-growing exposures — demos MUST re-simulate with these
        # or the cached demo obs (game_state -1 dim / missing fine-route key) crash
        # the first DAPG minibatch against the live 262-dim policy space.
        "expose_blocked_goal",
        "expose_ego_crop_route_fine",
        # Same requirement for the speedrun obs revision: without these the demo
        # re-simulation produces a game_state that is 11 dims short of the live
        # policy space (+1 wall tau, +10 precision block) and the first DAPG
        # minibatch dies on a mat1/mat2 shape mismatch. They also change the
        # cache tag, so stale caches re-simulate instead of being silently reused.
        "expose_wall_contact_tau",
        "expose_precision_state",
        # EXECUTION_PRECISION_PLAN §6: same requirement (+TIME_PHASE_DIM dims).
        # time_phase_norm_frames is non-bool and passed through below -- coercing
        # it to True would normalise the demo clock by 1 frame and saturate the
        # ramp at 1.0 from the first step.
        "expose_time_phase",
        "time_phase_norm_frames",
        # v26 W6c: +COMMITMENT_STATE_DIM (6) on game_state. Omitting it here is
        # the same crash as every entry above -- demos re-simulate at the OLD
        # width while the live policy space is 6 wider, and the first DAPG
        # minibatch dies with "mat1 and mat2 shapes cannot be multiplied
        # (Nx263 and 269x64)" in the hazard encoder's FiLM. Observed for real on
        # the v26 launch at 30k steps.
        "expose_commitment_state",
        # v26 W6a: new obs KEYS rather than game_state dims, so a missing one
        # degrades instead of crashing (the extractor substitutes a neutral
        # prior). Still required: without them every demo teaches the fall
        # streams a constant, which is a silent quality bug rather than a loud
        # one -- worse, not better.
        "expose_fall_lethality_map",
        "expose_ego_crop_fall_fine",
        # v27 (HORIZONTAL_AUGMENTATION_PLAN): reflection-equivariant obs
        # conventions. Same requirement as every entry above -- it changes the
        # VALUES of entity_overlays, the free-space cost field and the scandots,
        # so a demo simulated without it teaches the policy a subtly different
        # distribution in three streams. It also changes the cache tag, so a
        # stale cache re-simulates rather than being silently reused.
        "mirror_equivariant_obs",
    )
    # Non-boolean exposure fields whose value is applied verbatim (NOT bool-coerced)
    # so a demo re-simulates the ego-crop at the SAME size as the live policy space
    # (bool-coercing the size would force it to 1 and desync the obs shape).
    # cost_map_norm (v18 C3) is a mode string: it changes cost_to_go_map VALUES,
    # so demos must re-simulate under the run's normalization. A False/None value
    # (the trainer's "default fixed_k" sentinel) is skipped, keeping the default.
    _OBS_EXPOSURE_NONBOOL = (
        "ego_crop_size", "cost_map_norm", "time_phase_norm_frames",
    )

    def __init__(
        self,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "rgb_array",
        enable_rendering: bool = False,
        frame_skip: int = 1,
        obs_exposure: Optional[Dict[str, bool]] = None,
    ):
        """Initialize replay executor.

        Args:
            observation_config: Ignored, kept for API compatibility.
            render_mode: Only used when ``enable_rendering`` is True. Layer 1
                observations are never image-based, so rendering is optional.
            enable_rendering: If True, creates the env in ``human``-style render
                mode so callers can grab raw frames via ``env.render()``.
            frame_skip: Number of physics ticks per emitted observation.
            obs_exposure: Optional dict of ``ReachabilityConfig`` obs-exposure
                flags to enable so the replayed observation MATCHES the policy obs
                the training run was built with. Recognized keys:
                ``use_jump_potential``, ``expose_cost_to_go_map``,
                ``expose_waypoint_features``, ``use_free_space_potential``. Unknown
                keys are ignored; default (None) keeps the strict Layer-1 contract.
        """
        del observation_config  # unused
        self.frame_skip = max(1, frame_skip)

        from ..gym_environment.config import RenderConfig

        render_kwargs: Dict[str, Any] = {}
        if enable_rendering:
            render_kwargs["render"] = RenderConfig(render_mode=render_mode)

        cfg = EnvironmentConfig.for_training(**render_kwargs)
        # Apply obs-exposure flags onto the reachability config BEFORE the env is
        # built so the produced obs dict carries the same extra keys the policy
        # was trained against (cost_to_go_map / waypoint_nav), and the jump/free-
        # space potential source matches.
        if obs_exposure:
            for k in self._OBS_EXPOSURE_FIELDS:
                if k in obs_exposure:
                    v = obs_exposure[k]
                    if k in self._OBS_EXPOSURE_NONBOOL:
                        # Falsy (0 / False / None) = "keep the config default"
                        # sentinel for value-carrying fields.
                        if v:
                            setattr(cfg.reachability, k, v)
                    else:
                        setattr(cfg.reachability, k, bool(v))
        # Faithful replay: one physics tick per emitted input byte. Without this
        # the env runs at the default frame_skip (4 ticks/step), repeating each
        # input 4x and diverging from the recorded demo.
        cfg.frame_skip = self.frame_skip
        # Skip the throwaway init-time procedural map load: the real level is
        # injected per replay in execute_replay (load_map_from_map_data + reset).
        # Without this the env eagerly generates a map by its default category at
        # construction, which crashes if that category name is stale in the
        # generator registry — silently killing BC demo ingestion.
        cfg.defer_initial_map_load = True
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

        # Actual simulated outcome of THIS tape on the CURRENT sim. Callers
        # (BC/DAPG dataset builders) must gate on this, NOT on the replay
        # file's ``success`` metadata flag — a stale or mis-encoded tape (e.g.
        # the far_objective L/R-swap incident) carries success=True while the
        # re-simulation dies, silently poisoning the demo set.
        self.last_replay_won = bool(self.env.nplay_headless.ninja_has_won())

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
