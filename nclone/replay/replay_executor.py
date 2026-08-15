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


# Semantic version of the tape -> (observation, action) decoding contract.
#
# BUMP THIS whenever a change alters the ACTIONS or OBSERVATIONS produced from an
# UNCHANGED tape. Downstream consumers that cache decoded demos (npp-rl's FiGAR
# BC/DAPG dataset) hash it into their cache key, because a semantic change of this
# kind is invisible to every shape-based staleness check they have: the cached
# arrays are exactly the same size and dtype, only the labels differ. Without a
# version bump a corrected decoder silently keeps training on the old, wrong
# targets for as long as the cache survives.
#
#   1 -> original.
#   2 -> 2026-08-14, pipeline-audit P0.1: input bytes 6 and 7 decode to LEFT and
#        LEFT+JUMP instead of NOOP. Affects 11.8% of harvested tapes and 68.2% of
#        the first-party attract demos, so every cache built before this must be
#        rebuilt.
REPLAY_DECODE_CONTRACT_VERSION = 2


def map_input_to_action(input_byte: int) -> int:
    """Map N++ input byte (0-7) to the 6-way discrete action space.

    Inverse of `map_action_to_input` from `gameplay_recorder.py` on bytes 0-5, and
    agrees with `decode_input_to_controls` (below) on ALL eight.

    Bytes 6 and 7 set both direction bits. They are NOT invalid: `Ninja::think`
    resolves them to LEFT, which is what `decode_input_to_controls` has always
    done. This mapper used to send both to NOOP, so `ReplayExecutor` -- which
    drives BC/DAPG observation and action regeneration -- silently substituted
    "do nothing" for "hold left" on exactly those frames.

    Measured cost of that disagreement (scripts/measure_replay_input_byte_census.py
    over 7,318 local tapes / 5.09M frames): 531/4,505 harvested tapes (11.8%) and
    15/22 first-party attract demos (68.2%) contain at least one such frame. The
    planner-generated corpora are unaffected -- they are ENCODED from this same
    6-way action space, so bytes 6/7 are unreachable there by construction.
    """
    input_to_action_map = {
        0: 0,  # 000: NOOP
        1: 3,  # 001: JUMP
        2: 2,  # 010: RIGHT
        3: 5,  # 011: RIGHT+JUMP
        4: 1,  # 100: LEFT
        5: 4,  # 101: LEFT+JUMP
        6: 1,  # 110: left+right -> LEFT wins
        7: 4,  # 111: left+right+jump -> LEFT+JUMP
    }
    return input_to_action_map.get(input_byte, 0)


def decode_input_to_controls(input_byte: int) -> tuple:
    """Decode input byte to (horizontal, jump) controls.

    Retained for callers that still drive `nplay_headless.tick()` directly.
    """
    jump = 1 if (input_byte & 0x01) else 0

    # LEFT WINS OUTRIGHT when both direction bits are set. `Ninja::think`
    # (`docs/decomp/full/ninja_10515300.c:191-197`) is:
    #
    #     if ((input & 4) == 0)  hor = (input >> 1) & 1;   // the right bit
    #     else                   hor = -1;                 // left, unconditionally
    #
    # This port used to map left+right to neutral. The byte 6 occurs exactly once
    # in the 645-frame tape of 10131738, at frame 437, and getting it wrong put
    # a whole AIR_ACCEL step into xspeed and lost the demo -- see
    # `docs/ENTITY_RE_ROUND8_FINDINGS.md` §3.
    horizontal = -1 if (input_byte & 0x04) else ((input_byte >> 1) & 1)

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
        # Hazard obs KEYS. Same class as the fall maps above: a missing one
        # degrades silently rather than crashing, because the extractor
        # substitutes a neutral prior -- so every demo would teach those streams
        # a constant. drone_hazard_map / mine_proximity were already hashed into
        # the demo-cache tag by the trainer but never APPLIED here, so demos
        # under those flags re-simulated without the key they were tagged for.
        "expose_ego_crop_mines_fine",
        "enable_drone_hazard",
        "expose_mine_proximity",
        # Changes the frontier channel's VALUES at the same shape (binary mask
        # -> density-weighted), so it is the cost_map_norm case: a demo
        # re-simulated without it teaches the frontier stream a different
        # distribution than the live policy sees, and no shape error catches it.
        "frontier_map_weighted",
        # Phase 1 (gamma-1): the time-to-lethal threat field and the solid-entity
        # surface map. threat_ttl is the whole reason the phase-1 corpus is
        # perceivable; a demo re-simulated without it teaches the threat encoder
        # "everywhere is safe, always", which is worse than no demo. solid_map
        # additionally forces the NATIVE backend (see below) -- the python
        # backend cannot produce it and raises.
        "enable_threat_ttl",
        "threat_ttl_max_frames",
        "threat_ttl_stride",
        "enable_solid_map",
    )
    # Non-boolean exposure fields whose value is applied verbatim (NOT bool-coerced)
    # so a demo re-simulates the ego-crop at the SAME size as the live policy space
    # (bool-coercing the size would force it to 1 and desync the obs shape).
    # cost_map_norm (v18 C3) is a mode string: it changes cost_to_go_map VALUES,
    # so demos must re-simulate under the run's normalization. A False/None value
    # (the trainer's "default fixed_k" sentinel) is skipped, keeping the default.
    _OBS_EXPOSURE_NONBOOL = (
        "ego_crop_size", "cost_map_norm", "time_phase_norm_frames",
        # Phase 1: the TTL look-ahead window and tier-C stride set the channel
        # VALUES; bool-coercing them would re-simulate demos with a 1-frame
        # horizon and saturate the field.
        "threat_ttl_max_frames", "threat_ttl_stride",
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
        # solid_map is produced only by the native simulator, and the env raises
        # rather than emitting zeros. EnvironmentConfig defaults to the python
        # backend, so a caller asking for solid_map here would get a ValueError
        # naming a backend it never chose. Switch instead: a run that exposes
        # solid_map is native by construction, so native replay is also the
        # FAITHFUL one -- python replay would re-simulate demos on a different
        # engine than the policy trains on.
        if obs_exposure and obs_exposure.get("enable_solid_map"):
            cfg.sim_backend = "native"
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
