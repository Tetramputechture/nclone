"""SAC-Discrete compatible environment wrapper.

Wraps NppEnvironment to provide:
- Flat observation (game_state|reachability|spatial_context|mine_sdf)
- Flat Discrete(48) action space (6 base actions x 8 frame durations)
- Macro-action execution with gamma-discounting and contact-state interrupts
- Sparse rewards (R_SWITCH, R_EXIT, R_DEATH, R_STEP) optionally augmented
  with dense shaping via DenseRewardCalculator (PBRS + velocity alignment +
  distance milestones + mine hazard avoidance)
- reset_to_state() for RFCL curriculum resets (Phase 2)
- get_tile_grid() / get_agent_pos_norm() / level_id for global level context
"""

import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete

from .constants import (
    GAME_STATE_CHANNELS,
    REACHABILITY_FEATURES_DIM,
    SPATIAL_CONTEXT_DIM,
    MINE_SDF_FEATURES_DIM,
)
from .npp_environment import NppEnvironment
from .config import EnvironmentConfig
from .reward_calculation.sparse_reward_calculator import SparseRewardCalculator
from .reward_calculation.dense_reward_calculator import (
    DenseRewardCalculator,
    DenseRewardConfig,
)

logger = logging.getLogger(__name__)

OBS_DIM = (
    GAME_STATE_CHANNELS
    + REACHABILITY_FEATURES_DIM
    + SPATIAL_CONTEXT_DIM
    + MINE_SDF_FEATURES_DIM
)
N_BASE_ACTIONS = 6
N_DURATIONS = 8
N_ACTIONS = N_BASE_ACTIONS * N_DURATIONS
MACRO_GAMMA = 0.99

# Tile grid dimensions for global level context (playable area, 0-indexed)
_TILE_DATA_START = 184
_TILE_DATA_END = 1150  # 966 = 42 * 23
_GRID_W = 42
_GRID_H = 23
_MAP_PX_W = 1056.0
_MAP_PX_H = 600.0

_GROUND_STATES = frozenset({0, 1, 2})
_WALL_STATES = frozenset({5})

# Index of path_dist_to_current_goal within the 8-dim reachability feature vector.
# Must match dense_reward_calculator._REACH_DIST_GOAL (= 7).
_REACH_DIST_GOAL_IDX = 7


def encode_action(base_action: int, duration: int) -> int:
    """Encode (base_action, duration) into flat action index [0..47].

    base_action: 0=NOOP, 1=Left, 2=Right, 3=Jump, 4=Jump+Left, 5=Jump+Right
    duration: 1..8 frames
    """
    assert 0 <= base_action < N_BASE_ACTIONS, f"base_action must be 0-5, got {base_action}"
    assert 1 <= duration <= N_DURATIONS, f"duration must be 1-8, got {duration}"
    return base_action * N_DURATIONS + (duration - 1)


def decode_action(action_idx: int) -> Tuple[int, int]:
    """Decode flat action index into (base_action, duration)."""
    base_action = action_idx // N_DURATIONS
    duration = (action_idx % N_DURATIONS) + 1
    return base_action, duration


def _contact_category(ninja_state: int) -> int:
    """Map ninja state to contact category: 0=ground, 1=wall, 2=air."""
    if ninja_state in _GROUND_STATES:
        return 0
    if ninja_state in _WALL_STATES:
        return 1
    return 2


@dataclass
class DemoSnapshot:
    """Full ninja state snapshot for RFCL curriculum resets.

    Captured either from run-length-encoded demo transitions or from
    arbitrary simulation state captures during demo re-simulation.
    """

    frame_index: int
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    movement_state: int
    jump_duration_remaining: int
    floor_contact: bool
    wall_contact: bool
    floor_normal_x: float
    floor_normal_y: float
    wall_normal_x: float
    switch_activated: bool
    applied_gravity: float
    applied_drag: float
    applied_friction: float


class SACEnvironment(gymnasium.Wrapper):
    """SAC-Discrete wrapper around NppEnvironment.

    Key changes vs NppEnvironment:
    - Flat Box observation instead of dict
    - Flat Discrete(48) action space for macro-actions
    - Sparse rewards; HWM/PBRS rewards from inner env are discarded
    - Optional dense reward shaping via DenseRewardCalculator
    - Macro-action loop with contact-state interrupt logic
    - reset_to_state() for RFCL curriculum
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        dense_reward_config: Optional[DenseRewardConfig] = None,
    ):
        """Initialize SACEnvironment.

        Args:
            config: Environment configuration. frame_skip is forced to 1.
            dense_reward_config: If provided, enables dense reward shaping
                on top of the sparse reward. Pass None to use sparse-only
                rewards (default / legacy behaviour).
        """
        if config.frame_skip != 1:
            logger.warning(
                "SACEnvironment forces frame_skip=1 (was %d).", config.frame_skip
            )
            config = copy.copy(config)
            config.frame_skip = 1

        inner_env = NppEnvironment(config)
        # Skip the inner HWM/PBRS reward calculation on every primitive frame step.
        # SACEnvironment has its own SparseRewardCalculator (and optionally
        # DenseRewardCalculator) and discards the inner env's reward entirely.
        # Setting this flag avoids expensive A*/path-distance calls that would
        # otherwise run on every frame for no benefit.
        inner_env.skip_reward_calculation = True
        super().__init__(inner_env)

        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._reward_calc = SparseRewardCalculator()
        self._dense_reward_calc: Optional[DenseRewardCalculator] = (
            DenseRewardCalculator(dense_reward_config)
            if dense_reward_config is not None
            else None
        )
        self._prev_switch_activated: bool = False
        self._level_id: Optional[int] = None
        # RFCL phase: 0 = pre-switch (episode ends on switch activation),
        # 1 = post-switch (episode ends on exit door), None = no phase-aware termination.
        self._rfcl_phase: Optional[int] = None
        # Last dict observation (pre-_flatten); tracks state between macro-actions
        # for PBRS prev/curr computation.
        self._last_dict_obs: Optional[Dict] = None
        # Cumulative SAC episode reward (sparse + dense, gamma-discounted within
        # macro-actions).  Tracked here so that info["r"] can be overwritten with
        # the true SAC value at episode end, replacing the inner env's heuristic
        # reward that route visualization and TensorBoard otherwise read.
        self._sac_ep_reward: float = 0.0

        # OOB debug tracking: detect first out-of-bounds frame per episode.
        self._oob_logged: bool = False
        self._last_reset_snapshot: Optional[DemoSnapshot] = None
        self._frames_since_reset: int = 0

    # ------------------------------------------------------------------
    # Global level context helpers
    # ------------------------------------------------------------------

    def get_tile_grid(self) -> np.ndarray:
        """Return the full 42x23 tile grid as a uint8 array.

        Reads raw map_data[184:1150], which is stored in column-major order:
        tile_data[x + y * 42] -> grid[x, y].

        Returns:
            tile_grid: int32 array of shape (42, 23), values 0-37.
        """
        tile_data = np.array(
            self.env.nplay_headless.sim.map_data[_TILE_DATA_START:_TILE_DATA_END],
            dtype=np.int32,
        )
        return tile_data.reshape(_GRID_W, _GRID_H, order='F')

    def get_agent_pos_norm(self) -> np.ndarray:
        """Return normalised agent position [x/1056, y/600] as float32[2]."""
        ninja = self.env.nplay_headless.sim.ninja
        return np.array(
            [ninja.xpos / _MAP_PX_W, ninja.ypos / _MAP_PX_H],
            dtype=np.float32,
        )

    @property
    def level_id(self) -> int:
        """Stable integer ID for the current level, derived from tile grid hash.

        Computed once per level load and cached. Returns -1 before first reset.
        """
        if self._level_id is None:
            self._level_id = self._compute_level_id()
        return self._level_id

    def _compute_level_id(self) -> int:
        """Hash the raw tile bytes into a stable 32-bit signed integer."""
        tile_bytes = bytes(
            self.env.nplay_headless.sim.map_data[_TILE_DATA_START:_TILE_DATA_END]
        )
        digest = hashlib.sha256(tile_bytes).digest()
        # Take first 4 bytes as a big-endian signed int32
        raw = int.from_bytes(digest[:4], byteorder='big')
        return raw if raw < 2**31 else raw - 2**32

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        dict_obs, info = self.env.reset(seed=seed, options=options)
        self._prev_switch_activated = False
        # Invalidate cached level_id in case the map changed on reset
        self._level_id = None
        self._rfcl_phase = None  # Non-RFCL reset: disable phase-aware termination
        self._last_dict_obs = dict_obs
        self._sac_ep_reward = 0.0
        self._oob_logged = False
        self._last_reset_snapshot = None
        self._frames_since_reset = 0
        if self._dense_reward_calc is not None:
            self._dense_reward_calc.reset()
        info["tile_grid"] = self.get_tile_grid()
        info["agent_pos_norm"] = self.get_agent_pos_norm()
        info["level_id"] = self.level_id
        return self._flatten(dict_obs), info

    def step(self, action):
        """Execute a macro-action (up to 8 primitive frames).

        Decodes action_idx into (base_action, duration). Executes up to
        `duration` primitive frames, interrupting early on episode
        termination or contact-state change (floor/air/wall transition).
        Reward is the gamma-discounted sum of per-frame rewards. When dense
        rewards are enabled, each frame's sparse reward is augmented with
        PBRS, velocity-alignment, milestone, and hazard-avoidance components.

        Returns:
            (flat_obs, total_reward, terminated, truncated, info)
        """
        base_action, duration = decode_action(int(action))
        headless = self.env.nplay_headless
        sim = headless.sim

        init_contact = _contact_category(sim.ninja.state)
        total_reward = 0.0
        actual_duration = 0
        terminated = False
        truncated = False
        final_dict_obs = None
        # Track whether the switch was activated during this macro-action so
        # compute_macro_pbrs() can skip the discontinuous frame.
        switch_activated_this_macro = False

        # prev_dict_obs tracks state before each primitive frame; initialised
        # from last episode's final state (or from reset/_reset_to_state).
        prev_dict_obs = self._last_dict_obs

        # Record the path-distance BEFORE the macro-action for PBRS (computed
        # once after the loop to avoid the standing-still bias).
        dist_before_macro: Optional[float] = None
        if self._dense_reward_calc is not None and prev_dict_obs is not None:
            reach = np.asarray(
                prev_dict_obs.get(
                    "reachability_features",
                    np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
                ),
                dtype=np.float32,
            )
            dist_before_macro = float(reach[_REACH_DIST_GOAL_IDX])

        for frame_i in range(duration):
            # Interrupt before executing this frame if contact state changed since
            # the macro-action started.  Checked at the top so the extra frame is
            # never executed -- matching the demo_converter run-length encoding which
            # also stops the run when the previous frame's contact_after differs from
            # the run's initial contact category.
            if frame_i > 0 and _contact_category(sim.ninja.state) != init_contact:
                break

            dict_obs, _ignored_heuristic_reward, terminated, truncated, info = self.env.step(
                base_action
            )
            actual_duration += 1

            switch_now = headless.exit_switch_activated()
            switch_just = switch_now and not self._prev_switch_activated
            self._prev_switch_activated = switch_now
            if switch_just:
                switch_activated_this_macro = True

            # Phase 0 RFCL: reaching the switch is the episode goal.
            # Terminate immediately so the agent gets clean credit for the
            # switch activation reward without accumulating post-goal step penalties.
            if self._rfcl_phase == 0 and switch_just:
                terminated = True

            won = headless.ninja_has_won()
            dead = headless.ninja_has_died()

            # OOB debug: log first frame where ninja leaves level bounds.
            self._frames_since_reset += 1
            if not self._oob_logged and sim.ninja.ypos > 600:
                snap = self._last_reset_snapshot
                logger.error(
                    "OOB DETECTED: ninja=(%.2f,%.2f) vel=(%.4f,%.4f) state=%d "
                    "floor_count=%d frames_since_reset=%d | "
                    "reset_snapshot: pos=(%.2f,%.2f) vel=(%.4f,%.4f) state=%d | "
                    "rfcl_phase=%s terminated=%s dead=%s",
                    sim.ninja.xpos, sim.ninja.ypos,
                    sim.ninja.xspeed, sim.ninja.yspeed,
                    sim.ninja.state,
                    sim.ninja.floor_count,
                    self._frames_since_reset,
                    snap.position_x if snap else -1.0,
                    snap.position_y if snap else -1.0,
                    snap.velocity_x if snap else -1.0,
                    snap.velocity_y if snap else -1.0,
                    snap.movement_state if snap else -1,
                    self._rfcl_phase, terminated, dead,
                )
                self._oob_logged = True

            # OOB → death: the physics engine does not register falling below the
            # level boundary (y > 600) as a death, so the episode would otherwise
            # run until the truncation time-limit and be logged as a timeout.
            # Force termination as a death so the penalty and metrics are correct.
            if not dead and not won and sim.ninja.ypos > 600:
                dead = True
                terminated = True
                info["player_dead"] = True

            frame_reward = self._reward_calc.compute_frame_reward(
                switch_just_activated=switch_just,
                player_won=won,
                player_dead=dead,
            )

            # Per-frame dense components (velocity alignment, milestones, hazard).
            # PBRS is NOT applied here — it is computed once after the loop via
            # compute_macro_pbrs() to avoid the standing-still bias.
            if self._dense_reward_calc is not None and prev_dict_obs is not None:
                dense_r, _breakdown = self._dense_reward_calc.compute_frame_reward(
                    prev_dict_obs=prev_dict_obs,
                    curr_dict_obs=dict_obs,
                    info=info,
                )
                frame_reward += dense_r

            total_reward += (MACRO_GAMMA ** frame_i) * frame_reward
            prev_dict_obs = dict_obs
            final_dict_obs = dict_obs

            if terminated or truncated:
                # Attach per-episode dense reward stats for TensorBoard logging.
                # Also record the cumulative SAC episode reward (accumulated up
                # to and including this macro-action) so the trainer can compute
                # dense fraction without needing a potentially mis-aligned slice
                # of the global episode reward list.
                if self._dense_reward_calc is not None:
                    ep_stats = self._dense_reward_calc.get_episode_stats()
                    ep_stats["episode_total_reward"] = (
                        self._sac_ep_reward + total_reward
                    )
                    info["dense_reward_episode_stats"] = ep_stats
                break

        # ------------------------------------------------------------------
        # Macro-level PBRS: computed once after the frame loop.
        # This avoids the standing-still bias caused by per-frame accumulation.
        # ------------------------------------------------------------------
        if (
            self._dense_reward_calc is not None
            and dist_before_macro is not None
            and final_dict_obs is not None
        ):
            reach_after = np.asarray(
                final_dict_obs.get(
                    "reachability_features",
                    np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
                ),
                dtype=np.float32,
            )
            dist_after_macro = float(reach_after[_REACH_DIST_GOAL_IDX])
            pbrs_r = self._dense_reward_calc.compute_macro_pbrs(
                dist_before=dist_before_macro,
                dist_after=dist_after_macro,
                phase_transition=switch_activated_this_macro,
            )
            # The PBRS reward is added to the macro-action total at weight 1.0
            # (not gamma-discounted) since it represents a per-macro signal.
            total_reward += pbrs_r

            # Patch the episode stats that may have been written on episode end
            # to include the macro-level PBRS that was just computed.
            if "dense_reward_episode_stats" in info:
                info["dense_reward_episode_stats"]["dense_pbrs_total"] = (
                    self._dense_reward_calc._ep_pbrs
                )
                info["dense_reward_episode_stats"]["dense_total"] = (
                    self._dense_reward_calc._ep_pbrs
                    + self._dense_reward_calc._ep_vel
                    + self._dense_reward_calc._ep_milestone
                    + self._dense_reward_calc._ep_hazard
                )
                info["dense_reward_episode_stats"]["episode_total_reward"] = (
                    self._sac_ep_reward + total_reward
                )

        # Keep _last_dict_obs current for the next macro-action's first frame
        if final_dict_obs is not None:
            self._last_dict_obs = final_dict_obs

        # Accumulate SAC episode reward and, on episode end, overwrite info["r"]
        # with the true SAC value (sparse + dense, gamma-discounted within macro-
        # actions).  The inner env's info["r"] reflects its own heuristic reward
        # which is discarded by this wrapper; without this override the route
        # visualisation callback and any downstream consumer would read the wrong
        # reward.
        self._sac_ep_reward += total_reward
        if terminated or truncated:
            info["r"] = self._sac_ep_reward
            self._sac_ep_reward = 0.0

            # When the SAC wrapper terminates the episode (e.g. Phase 0 switch
            # hit) the inner BaseNppEnvironment does not know the episode is
            # ending, so it never adds episode_route to info.  Copy and clear
            # current_route here to ensure the route visualization callback
            # always receives a route, regardless of which layer terminated.
            if "episode_route" not in info and self.env.current_route:
                info["episode_route"] = list(self.env.current_route)
                info["route_length"] = len(self.env.current_route)
                self.env.current_route = []

            # Fix is_success for RFCL phase 0: the inner env only sets
            # is_success=True on exit-door wins (player_won).  For phase 0,
            # reaching the switch IS the goal, so override when the SAC wrapper
            # terminated due to switch activation.  _prev_switch_activated is
            # already updated to True before the early-termination check, so it
            # reliably reflects whether the switch was hit this episode.
            if self._rfcl_phase == 0:
                info["is_success"] = self._prev_switch_activated or info.get(
                    "is_success", False
                )

            # When the SAC wrapper terminates early the inner env also skips
            # the episode_tiles/mines/locked_doors capture that normally happens
            # in _build_episode_info.  Populate them here so the route
            # visualization callback gets the correct level data before the
            # trainer resets the environment.
            if "episode_tiles" not in info:
                info["episode_tiles"] = self.env.get_route_visualization_tile_data()
            if "episode_mines" not in info:
                info["episode_mines"] = self.env.get_route_visualization_mine_data()
            if "episode_locked_doors" not in info:
                info["episode_locked_doors"] = (
                    self.env.get_route_visualization_locked_door_data()
                )

            # Always override episode_next_hop_path with a path computed from
            # the ninja's actual current position, not from episode_route[0]
            # (which is always the spawn for SAC-only runs).
            #
            # Goal selection:
            #   RFCL Phase 0 → "switch"  (even if switch_activated=True because
            #     the agent just touched it; the episode goal was the switch)
            #   RFCL Phase 1 → "exit"
            #   SAC-only     → "exit" if switch already activated, else "switch"
            if self._rfcl_phase is not None:
                goal_id = "switch" if self._rfcl_phase == 0 else "exit"
            else:
                switch_activated = info.get("switch_activated", False)
                goal_id = "exit" if switch_activated else "switch"

            current_pos_path = self._get_current_pos_next_hop_path(goal_id)
            if current_pos_path is not None:
                info["episode_next_hop_path"] = current_pos_path

        flat_obs = self._flatten(final_dict_obs)
        info["actual_duration"] = actual_duration
        info["base_action"] = base_action
        info["agent_pos_norm"] = self.get_agent_pos_norm()
        info["level_id"] = self.level_id
        return flat_obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # RFCL state reset interface
    # ------------------------------------------------------------------

    def reset_to_state(self, snapshot: DemoSnapshot, phase: int):
        """Reset simulation to an arbitrary ninja state for RFCL curriculum.

        Calls fast_reset() on the simulator, then restores the ninja state
        from the snapshot. If phase=1, programmatically activates the exit
        switch so the exit door is collidable.

        Args:
            snapshot: DemoSnapshot with full physics state.
            phase: 0 = pre-switch, 1 = post-switch (exit door accessible).

        Returns:
            (flat_obs, info)
        """
        headless = self.env.nplay_headless
        sim = headless.sim

        headless.fast_reset()

        # Validate and clamp snapshot position to level bounds before applying.
        # Out-of-bounds positions (e.g. from noise applied near the boundary) place
        # the ninja below/outside collision geometry, causing infinite falling with no
        # death detection and meaningless pathfinding distances until truncation.
        _LEVEL_X_MIN, _LEVEL_X_MAX = 24.0, 1032.0
        _LEVEL_Y_MIN, _LEVEL_Y_MAX = 24.0, 576.0
        pos_x = snapshot.position_x
        pos_y = snapshot.position_y
        if not (_LEVEL_X_MIN <= pos_x <= _LEVEL_X_MAX and _LEVEL_Y_MIN <= pos_y <= _LEVEL_Y_MAX):
            clamped_x = max(_LEVEL_X_MIN, min(_LEVEL_X_MAX, pos_x))
            clamped_y = max(_LEVEL_Y_MIN, min(_LEVEL_Y_MAX, pos_y))
            logger.warning(
                "reset_to_state: snapshot position (%.1f, %.1f) is outside level bounds "
                "[%.0f-%.0f, %.0f-%.0f]; clamping to (%.1f, %.1f).",
                pos_x, pos_y,
                _LEVEL_X_MIN, _LEVEL_X_MAX, _LEVEL_Y_MIN, _LEVEL_Y_MAX,
                clamped_x, clamped_y,
            )
            pos_x, pos_y = clamped_x, clamped_y

        ninja = sim.ninja
        _safe_set(ninja, "xpos", pos_x)
        _safe_set(ninja, "ypos", pos_y)
        _safe_set(ninja, "xspeed", snapshot.velocity_x)
        _safe_set(ninja, "yspeed", snapshot.velocity_y)
        _safe_set(ninja, "state", snapshot.movement_state)
        _safe_set(ninja, "jump_duration", snapshot.jump_duration_remaining)
        _safe_set(ninja, "applied_gravity", snapshot.applied_gravity)
        _safe_set(ninja, "applied_drag", snapshot.applied_drag)
        _safe_set(ninja, "applied_friction", snapshot.applied_friction)
        _safe_set(ninja, "floor_count", 1 if snapshot.floor_contact else 0)
        _safe_set(ninja, "wall_count", 1 if snapshot.wall_contact else 0)
        _safe_set(ninja, "floor_normalized_x", snapshot.floor_normal_x)
        _safe_set(ninja, "floor_normalized_y", snapshot.floor_normal_y)
        _safe_set(ninja, "wall_normal", snapshot.wall_normal_x)
        _safe_set(ninja, "airborn", snapshot.movement_state in (3, 4))
        _safe_set(ninja, "walled", snapshot.movement_state == 5)

        # Verify attributes were applied and log for OOB debugging.
        logger.debug(
            "reset_to_state: phase=%d "
            "snapshot=(%.2f,%.2f) vel=(%.4f,%.4f) state=%d | "
            "actual=(%.2f,%.2f) vel=(%.4f,%.4f) state=%d | "
            "xpos_old=%.2f ypos_old=%.2f floor_count=%d",
            phase,
            snapshot.position_x, snapshot.position_y,
            snapshot.velocity_x, snapshot.velocity_y,
            snapshot.movement_state,
            ninja.xpos, ninja.ypos,
            ninja.xspeed, ninja.yspeed,
            ninja.state,
            ninja.xpos_old, ninja.ypos_old,
            ninja.floor_count,
        )

        # Reset OOB debug tracking for this episode.
        self._oob_logged = False
        self._last_reset_snapshot = snapshot
        self._frames_since_reset = 0

        if phase == 1:
            _activate_exit_switch(sim)

        self._prev_switch_activated = headless.exit_switch_activated()
        self._rfcl_phase = phase

        # Invalidate NppEnvironment observation caches and position-dependent caches.
        # Must clear more than just obs/prev_obs: the observation processor entity
        # buffer, mine overlay cache, and path calculator step cache can all hold
        # stale data from the previous episode position after a large RFCL jump.
        self.env._cached_observation = None
        self.env._prev_obs_cache = None

        # Invalidate static entity position caches.  fast_reset() resets entities
        # in-place (including toggle mines moving back from TOGGLE_MINE_TOGGLED to
        # TOGGLE_MINE), so any cached entity lists or position tuples from the
        # previous episode can be stale.  Mirrors BaseNppEnvironment.reset() lines
        # 1774-1779.
        self.env._cached_switch_pos = None
        self.env._cached_exit_pos = None
        self.env._cached_locked_doors = None
        self.env._cached_locked_door_switches = None
        self.env._cached_toggle_mines = None

        # Invalidate level_data / entities caches.  switch_states inside
        # _cached_level_data can be stale after locked door switch collection in
        # the previous episode.  Mirrors BaseNppEnvironment.reset() lines 1771-1772.
        self.env._cached_level_data = None
        self.env._cached_entities = None

        # Reset the inner reward calculator so PBRS accumulators, HWM state, and
        # diagnostic counters (closest_distance, episode_pbrs_rewards, etc.) start
        # clean.  The SAC wrapper discards the inner env's heuristic reward, but
        # stale accumulators would corrupt _build_episode_info diagnostic fields.
        self.env.reward_calculator.reset()

        # Reset the position-warning flag so suppressed warnings from a previous
        # episode don't silence legitimate ninja_position() warnings in this one.
        self.env._warned_about_position = False

        # Reset observation processor entity position buffers
        if hasattr(self.env, "observation_processor"):
            self.env.observation_processor.reset()

        # Clear path calculator step cache (ninja node lookup invalidated by position jump)
        if hasattr(self.env, "_path_calculator") and self.env._path_calculator is not None:
            self.env._path_calculator.clear_step_cache()

        # Clear reachability mixin cell cache.  The cache key is (ninja_cell,
        # switch_activated) at 24px granularity; after an RFCL position jump the
        # new cell may collide with the previous episode's last cell and return
        # stale features with wrong distances and next-hop directions.
        if hasattr(self.env, "_last_reachability_key"):
            self.env._last_reachability_key = None
        if hasattr(self.env, "_cached_reachability_features"):
            self.env._cached_reachability_features = None

        # Reset mine overlay position cache so spatial_context recomputes mine proximity
        try:
            from .spatial_context import reset_mine_overlay_cache
            reset_mine_overlay_cache()
        except ImportError:
            pass

        # Reset per-episode tracking state that BaseNppEnvironment.reset() normally
        # clears but reset_to_state() bypasses entirely.  Mirror each operation here:
        # - current_route: prevents previous episode positions bleeding into route viz
        # - _switch_activation_frame/step: prevents stale switch frame from a prior
        #   successful Phase 0 episode appearing in the next episode's title annotation
        # - current_ep_reward: ensures cumulative reward starts clean for this episode
        self.env._switch_activation_frame = None
        self.env._switch_activation_step = None
        self.env.current_ep_reward = 0
        self.env.current_route = []
        try:
            position = self.env.nplay_headless.ninja_position()
            if position is not None:
                self.env.current_route.append(
                    (float(position[0]), float(position[1]))
                )
        except Exception:
            pass

        dict_obs = self.env._get_observation()
        processed = self.env._process_observation(dict_obs)

        # Update last dict obs and reset episode-level dense reward state
        self._last_dict_obs = processed
        self._sac_ep_reward = 0.0
        if self._dense_reward_calc is not None:
            self._dense_reward_calc.reset()

        info = {
            "agent_pos_norm": self.get_agent_pos_norm(),
            "level_id": self.level_id,
        }
        return self._flatten(processed), info

    def set_max_steps(self, frames: int) -> None:
        """Override the episode truncation limit (in physics frames).

        Used by the RFCL curriculum to shorten episodes during the reverse
        curriculum phase when the agent starts near the goal.

        Args:
            frames: Maximum frames before truncation. Clamped to at least 1.
        """
        limit = max(1, int(frames))
        self.env.truncation_checker.current_truncation_limit = limit

    def capture_state(self) -> DemoSnapshot:
        """Capture current simulation state as a DemoSnapshot.

        Useful for demo re-simulation in the demo converter.
        """
        ninja = self.env.nplay_headless.sim.ninja
        headless = self.env.nplay_headless
        return DemoSnapshot(
            frame_index=headless.sim.frame,
            position_x=ninja.xpos,
            position_y=ninja.ypos,
            velocity_x=ninja.xspeed,
            velocity_y=ninja.yspeed,
            movement_state=ninja.state,
            jump_duration_remaining=getattr(ninja, "jump_duration", 0),
            floor_contact=ninja.floor_count > 0,
            wall_contact=ninja.wall_count > 0,
            floor_normal_x=getattr(ninja, "floor_normalized_x", 0.0),
            floor_normal_y=getattr(ninja, "floor_normalized_y", 0.0),
            wall_normal_x=getattr(ninja, "wall_normal", 0.0),
            switch_activated=headless.exit_switch_activated(),
            applied_gravity=getattr(ninja, "applied_gravity", 0.0),
            applied_drag=getattr(ninja, "applied_drag", 0.0),
            applied_friction=getattr(ninja, "applied_friction", 0.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_pos_next_hop_path(self, goal_id: str):
        """Compute the level-cache next-hop path from the ninja's current position.

        Follows the precomputed `get_next_hop` chain in the level cache from the
        node nearest to the ninja's live position toward `goal_id`.  Works for
        both RFCL and SAC-only episodes.

        Args:
            goal_id: Level-cache goal identifier, e.g. ``"switch"`` or ``"exit"``.

        Returns:
            List of graph-node (x, y) positions from current ninja node to goal,
            or ``None`` if the path cannot be computed.
        """
        try:
            path_calculator = getattr(self.env, "_path_calculator", None)
            if path_calculator is None:
                return None
            level_cache = getattr(path_calculator, "level_cache", None)
            if level_cache is None:
                return None

            graph_data = getattr(self.env, "current_graph_data", None)
            if not graph_data:
                return None
            adjacency = graph_data.get("adjacency")
            if not adjacency:
                return None

            from nclone.graph.reachability.pathfinding_utils import find_ninja_node
            from nclone.constants.physics_constants import NINJA_RADIUS

            ninja = self.env.nplay_headless.sim.ninja
            current_pos = (ninja.xpos, ninja.ypos)

            spatial_hash = graph_data.get("spatial_hash")
            subcell_lookup = graph_data.get("subcell_lookup")

            start_node = find_ninja_node(
                current_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )
            if start_node is None:
                return None

            path = []
            current = start_node
            visited: set = set()
            max_hops = 1000
            while current is not None and len(path) < max_hops:
                if current in visited:
                    break
                path.append(current)
                visited.add(current)
                next_hop = level_cache.get_next_hop(current, goal_id)
                if next_hop is None or next_hop == current:
                    break
                current = next_hop

            return path if len(path) >= 2 else None

        except Exception as exc:
            logger.info(
                "_get_current_pos_next_hop_path failed (goal_id=%s): %s",
                goal_id,
                exc,
            )
            return None

    def _flatten(self, dict_obs: Dict[str, Any]) -> np.ndarray:
        """Flatten dict obs to flat float32 array."""
        parts = [
            np.asarray(
                dict_obs.get("game_state", np.zeros(GAME_STATE_CHANNELS, dtype=np.float32)),
                dtype=np.float32,
            ),
            np.asarray(
                dict_obs.get(
                    "reachability_features",
                    np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
                ),
                dtype=np.float32,
            ),
            np.asarray(
                dict_obs.get(
                    "spatial_context", np.zeros(SPATIAL_CONTEXT_DIM, dtype=np.float32)
                ),
                dtype=np.float32,
            ),
            np.asarray(
                dict_obs.get(
                    "mine_sdf_features",
                    np.zeros(MINE_SDF_FEATURES_DIM, dtype=np.float32),
                ),
                dtype=np.float32,
            ),
        ]
        obs = np.concatenate(parts)
        if obs.shape[0] != OBS_DIM:
            if obs.shape[0] < OBS_DIM:
                obs = np.pad(obs, (0, OBS_DIM - obs.shape[0]))
            else:
                obs = obs[:OBS_DIM]
        return obs


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_set(obj: Any, attr: str, value: Any) -> None:
    """Set attribute on obj if it exists."""
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _activate_exit_switch(sim: Any) -> None:
    """Programmatically activate exit switch (makes exit door collidable).

    Mirrors EntityExitSwitch.logical_collision():
    - Deactivates switch entity (active=False)
    - Sets switch_hit=True on parent exit door
    - Adds exit door to entity grid

    entity_dic[3] may contain both EntityExit and EntityExitSwitch objects
    depending on the loader. We check both buckets 3 and 4.
    """
    for bucket_key in (3, 4):
        for entity in sim.entity_dic.get(bucket_key, []):
            if type(entity).__name__ == "EntityExitSwitch":
                entity.active = False
                parent = entity.parent
                parent.switch_hit = True
                door_cell = parent.cell
                if parent not in sim.grid_entity.get(door_cell, []):
                    sim.grid_entity[door_cell].append(parent)
                return
