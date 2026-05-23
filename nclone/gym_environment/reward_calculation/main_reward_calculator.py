"""Unified reward calculator for N++ Layer 1 training.

This is the single reward source used by the environment. It produces a
step-level reward composed of sparse terminals, A* HWM shaping, time penalty,
and a backtracking penalty. BSRS (``(1 - beta) * eta * (gamma * V(s') - V(s))``)
is applied externally by the trainer (``MaskedPPO`` / ``FiGAR-PPO``); the
environment never consumes a precomputed potential at runtime.

Reward formula (per env step, which may span multiple physics frames):

    r = r_sparse                                     # terminal events only
      + beta * alpha * hwm_progress                  # A* HWM shaping
      + time_penalty_per_step * frames_executed      # per-frame idle cost
      + BACKTRACK_PENALTY * frames_executed          # only when moving away

Sparse terminal contributions are added on top when the corresponding event
fires in the step:

    r_sparse = R_SWITCH    (on activation)
             + R_EXIT      (on win)
             + R_DEATH     (+ R_DEATH_MINE if mine)
             + TRUNCATION_PENALTY (on timeout)

Component glossary:

- ``hwm_progress``: normalized A* distance progress toward the current
  objective (two-phase: switch -> exit). Non-negative by construction.
- ``beta``: HWM anneal factor from ``RewardConfig.beta_blend()``. When
  ``enable_bsrs_blend`` is true, beta decays 1 -> 0 over
  ``bsrs_blend_horizon_fraction * total_timesteps`` and the trainer adds the
  complementary BSRS term so the combined dense signal stays at unit scale.
- ``TIME_PENALTY_PER_STEP``: small per-frame cost that discourages aimless
  wandering.
- ``BACKTRACK_PENALTY``: extra negative reward when A* distance to the
  current goal strictly increases step-over-step (HWM alone gives 0 on
  backtracking, which was not a strong enough directional signal).

RND intrinsic rewards are not computed here; they are added on the trainer
side (``npp-rl``) with their own value head.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from .reward_config import RewardConfig
from .hwm_shaping import EpisodeRewardState
from .reward_constants import GLOBAL_REWARD_SCALE, BACKTRACK_PENALTY

from ...constants.physics_constants import (
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    NINJA_RADIUS,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Single runtime reward source for the N++ environment.

    Emits ``r_sparse + beta * alpha * hwm_progress + time_penalty + backtrack_penalty``
    per env step. All diagnostic fields are populated on ``last_components``
    and ``get_episode_reward_metrics()``; no PBRS potential is consumed here.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        pbrs_gamma: float = 0.99,  # Retained for API compat; not used here.
        shared_level_cache: Optional[Any] = None,
        shared_level_caches_by_stage: Optional[Dict[int, Any]] = None,
        goal_curriculum_manager: Optional[Any] = None,
    ):
        self.config = reward_config or RewardConfig()

        self.goal_curriculum_manager = goal_curriculum_manager

        self.path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            shared_level_cache=shared_level_cache,
            shared_level_caches_by_stage=shared_level_caches_by_stage,
            goal_curriculum_manager=goal_curriculum_manager,
        )

        self.hwm_state = EpisodeRewardState()
        self.steps_taken = 0

        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        # Episode-level tracking for TensorBoard logging.
        self.episode_hwm_rewards: List[float] = []
        self.episode_time_penalties: List[float] = []
        self.episode_terminal_reward = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0

        self.episode_path_length = 0.0
        self.optimal_path_length: Optional[float] = None
        self.prev_player_pos: Optional[Tuple[float, float]] = None

        self.spawn_position: Optional[Tuple[float, float]] = None

        # Per-step diagnostic snapshot populated by ``calculate_reward``.
        self.last_components: Dict[str, Any] = {}

        # Episode initialization flag (distances needed for HWM normalization).
        self._episode_initialized = False
        self._initial_dist_switch = 0.0
        self._initial_dist_exit = 0.0

        # Backtracking detection: previous A* distance to the current goal.
        self._prev_goal_dist: Optional[float] = None

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
        frames_executed: int = 1,
        curriculum_manager: Optional[Any] = None,
        truncated: bool = False,
    ) -> float:
        """Compute the unified step-level extrinsic reward.

        Args:
            obs: Current observation.
            prev_obs: Previous observation.
            action: Action taken (optional, unused here).
            frames_executed: Number of physics frames collapsed into this step.
            curriculum_manager: Optional curriculum manager.
            truncated: True when the env step terminated via timeout.

        Returns:
            float: ``reward * GLOBAL_REWARD_SCALE``.
        """
        self.steps_taken += frames_executed

        if self.spawn_position is None:
            self.spawn_position = (prev_obs["player_x"], prev_obs["player_y"])

        switch_activated = obs.get("switch_activated", False)
        switch_just_activated = switch_activated and not prev_obs.get(
            "switch_activated", False
        )

        # L1.5: refresh level cache when mine or switch state changes (e.g. toggle
        # mines). ``build_level_cache`` fast-rejects when signatures match.
        level_data = obs.get("level_data")
        adjacency = obs.get("_adjacency_graph")
        graph_data = obs.get("_graph_data")
        if level_data is not None and adjacency is not None:
            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )
            self.path_calculator.build_level_cache(
                level_data, adjacency, base_adjacency, graph_data
            )

        dist_switch, dist_exit = self._get_distances(obs, curriculum_manager)

        if not self._episode_initialized:
            self._initial_dist_switch = (
                dist_switch if dist_switch < float("inf") else 1.0
            )
            self._initial_dist_exit = dist_exit if dist_exit < float("inf") else 1.0
            self.hwm_state.reset(
                initial_dist_switch=self._initial_dist_switch,
                max_dist_switch=self._initial_dist_switch,
                initial_dist_exit=self._initial_dist_exit,
                max_dist_exit=self._initial_dist_exit,
            )
            self.optimal_path_length = (
                self._initial_dist_switch + self._initial_dist_exit
            )
            self._episode_initialized = True

        # Terminal branches.
        if obs.get("player_dead", False):
            terminal_reward = self.config.death_penalty
            if obs.get("death_cause") == "mine":
                terminal_reward += self.config.mine_death_penalty
            if switch_just_activated:
                terminal_reward += self.config.switch_activation_reward
            self.episode_terminal_reward = terminal_reward

            scaled = terminal_reward * GLOBAL_REWARD_SCALE
            self.last_components = {
                "terminal_reward": terminal_reward,
                "hwm_reward": 0.0,
                "total_reward": terminal_reward,
                "scaled_reward": scaled,
                "is_terminal": True,
                "terminal_type": "death",
                "death_cause": obs.get("death_cause", "unknown"),
            }
            return scaled

        if obs.get("player_won", False):
            terminal_reward = self.config.level_completion_reward
            if switch_just_activated:
                terminal_reward += self.config.switch_activation_reward
            self.episode_terminal_reward = terminal_reward

            scaled = terminal_reward * GLOBAL_REWARD_SCALE
            self.last_components = {
                "terminal_reward": terminal_reward,
                "hwm_reward": 0.0,
                "total_reward": terminal_reward,
                "scaled_reward": scaled,
                "is_terminal": True,
                "terminal_type": "win",
            }
            return scaled

        # Non-terminal: HWM shaping + sparse switch + time + backtrack.
        reward = 0.0

        if switch_just_activated:
            reward += self.config.switch_activation_reward

        hwm_progress = self.hwm_state.step(
            dist_switch=dist_switch,
            dist_exit=dist_exit,
            switch_activated=switch_activated,
        )
        beta_hwm = self.config.beta_blend()
        hwm_reward = self.config.hwm_alpha * hwm_progress * beta_hwm
        reward += hwm_reward

        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward += time_penalty

        current_goal_dist = dist_switch if not switch_activated else dist_exit
        backtrack_penalty = 0.0
        if (
            self._prev_goal_dist is not None
            and current_goal_dist < float("inf")
            and self._prev_goal_dist < float("inf")
            and current_goal_dist > self._prev_goal_dist
        ):
            backtrack_penalty = BACKTRACK_PENALTY * frames_executed
            reward += backtrack_penalty
            self.episode_backtrack_steps += 1
        elif hwm_progress > 0:
            self.episode_forward_steps += 1
        self._prev_goal_dist = current_goal_dist

        self.episode_hwm_rewards.append(hwm_reward)
        self.episode_time_penalties.append(time_penalty)

        current_pos = (obs["player_x"], obs["player_y"])
        movement = 0.0
        if self.prev_player_pos is not None:
            dx = current_pos[0] - self.prev_player_pos[0]
            dy = current_pos[1] - self.prev_player_pos[1]
            movement = (dx * dx + dy * dy) ** 0.5
            self.episode_path_length += movement
        self.prev_player_pos = current_pos

        if current_goal_dist < self.closest_distance_this_episode:
            self.closest_distance_this_episode = current_goal_dist

        from ..util.util import calculate_distance

        eucl_switch = calculate_distance(
            obs["player_x"], obs["player_y"], obs["switch_x"], obs["switch_y"]
        )
        if eucl_switch < self.closest_distance_to_switch:
            self.closest_distance_to_switch = eucl_switch
        if switch_activated:
            eucl_exit = calculate_distance(
                obs["player_x"],
                obs["player_y"],
                obs["exit_door_x"],
                obs["exit_door_y"],
            )
            if eucl_exit < self.closest_distance_to_exit:
                self.closest_distance_to_exit = eucl_exit

        if truncated:
            reward += self.config.truncation_penalty

        scaled = reward * GLOBAL_REWARD_SCALE
        self.last_components = {
            "hwm_reward": hwm_reward,
            "hwm_progress": hwm_progress,
            "beta_blend_hwm": beta_hwm,
            "time_penalty": time_penalty,
            "backtrack_penalty": backtrack_penalty,
            "total_reward": reward,
            "scaled_reward": scaled,
            "is_terminal": truncated,
            "distance_to_switch": dist_switch,
            "distance_to_exit": dist_exit,
            "watermark_switch": self.hwm_state.watermark_switch,
            "watermark_exit": self.hwm_state.watermark_exit,
            "current_phase": self.hwm_state.current_phase,
            "closest_distance_episode": self.closest_distance_this_episode,
            "movement_distance": movement,
            "truncated": truncated,
        }
        return scaled

    def _get_distances(
        self,
        obs: Dict[str, Any],
        curriculum_manager: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """Get A* path distances to switch and exit.

        Prefer distances from the observation dict when present (``NppEnvironment``
        computes them in ``_compute_reachability`` using the env's path
        calculator). The reward calculator holds a *separate*
        ``CachedPathDistanceCalculator``; skipping duplicate ``get_path_cost``
        calls removes two hot-path A* lookups per step.
        """
        cs = obs.get("_cached_switch_distance")
        ce = obs.get("_cached_exit_distance")
        if cs is not None and ce is not None:
            dist_switch = float(cs)
            dist_exit = float(ce)
            obs["_pbrs_last_distance_to_goal"] = (
                dist_switch if not obs.get("switch_activated", False) else dist_exit
            )
            return dist_switch, dist_exit

        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

        player_pos = (int(obs["player_x"]), int(obs["player_y"]))
        switch_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
        exit_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))

        if adjacency is None:
            return float("inf"), float("inf")

        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        dist_switch = self.path_calculator.get_path_cost(
            player_pos,
            switch_pos,
            adjacency,
            base_adjacency,
            cache_key="switch",
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=EXIT_SWITCH_RADIUS,
            ninja_radius=NINJA_RADIUS,
        )

        dist_exit = self.path_calculator.get_path_cost(
            player_pos,
            exit_pos,
            adjacency,
            base_adjacency,
            cache_key="exit",
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
            ninja_radius=NINJA_RADIUS,
        )

        obs["_pbrs_last_distance_to_goal"] = (
            dist_switch if not obs.get("switch_activated", False) else dist_exit
        )

        return dist_switch, dist_exit

    def reset(self):
        """Reset episode-level reward state."""
        self.steps_taken = 0
        self.hwm_state = EpisodeRewardState()
        self._episode_initialized = False

        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        self.episode_hwm_rewards = []
        self.episode_time_penalties = []
        self.episode_terminal_reward = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0

        self.episode_path_length = 0.0
        self.optimal_path_length = None
        self.prev_player_pos = None
        self.spawn_position = None
        self._prev_goal_dist = None

        self.last_components = {}

        if self.path_calculator is not None:
            self.path_calculator.clear_step_cache()

    def update_config(self, timesteps: int, success_rate: float) -> None:
        """Update reward configuration based on training progress."""
        old_phase = self.config.training_phase
        self.config.update(timesteps, success_rate)
        if self.config.training_phase != old_phase:
            logger.warning(
                f"REWARD PHASE TRANSITION: {old_phase} -> {self.config.training_phase} "
                f"(timesteps={timesteps:,}, success={success_rate:.1%})"
            )

    def get_config_state(self) -> Dict[str, Any]:
        return self.config.get_active_components()

    def get_episode_reward_metrics(self) -> Dict[str, Any]:
        """Return aggregate metrics for TensorBoard logging."""
        import numpy as np

        hwm_total = (
            sum(self.episode_hwm_rewards) if self.episode_hwm_rewards else 0.0
        )
        hwm_mean = (
            float(np.mean(self.episode_hwm_rewards)) if self.episode_hwm_rewards else 0.0
        )
        hwm_std = (
            float(np.std(self.episode_hwm_rewards)) if self.episode_hwm_rewards else 0.0
        )
        time_penalty_total = (
            sum(self.episode_time_penalties) if self.episode_time_penalties else 0.0
        )

        path_optimality = 0.0
        if (
            self.episode_path_length > 0
            and self.optimal_path_length is not None
            and self.optimal_path_length > 0
        ):
            path_optimality = self.optimal_path_length / self.episode_path_length

        total_steps = self.steps_taken
        forward_pct = (
            (self.episode_forward_steps / total_steps * 100) if total_steps > 0 else 0.0
        )

        return {
            "hwm_total": hwm_total,
            "hwm_mean": hwm_mean,
            "hwm_std": hwm_std,
            "time_penalty_total": time_penalty_total,
            "terminal_reward": self.episode_terminal_reward,
            "forward_steps": self.episode_forward_steps,
            "backtrack_steps": self.episode_backtrack_steps,
            "forward_progress_pct": forward_pct,
            "backtracking_pct": 100.0 - forward_pct if total_steps > 0 else 0.0,
            "episode_path_length": self.episode_path_length,
            "optimal_path_length": self.optimal_path_length or 0.0,
            "path_optimality": path_optimality,
            "watermark_switch": self.hwm_state.best_switch_distance,
            "watermark_exit": self.hwm_state.best_exit_distance,
            "closest_distance_to_switch": self.closest_distance_to_switch,
            "closest_distance_to_exit": self.closest_distance_to_exit,
        }

    def validate_level_solvability(self, obs: Dict[str, Any]) -> bool:
        """Validate that goals are reachable from spawn."""
        try:
            adjacency = obs.get("_adjacency_graph")
            level_data = obs.get("level_data")
            graph_data = obs.get("_graph_data")

            if not adjacency or not level_data:
                return None

            player_pos = (int(obs["player_x"]), int(obs["player_y"]))
            switch_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
            exit_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))

            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )

            switch_dist = self.path_calculator.get_path_cost(
                player_pos,
                switch_pos,
                adjacency,
                base_adjacency,
                cache_key="validation_switch",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
            exit_dist = self.path_calculator.get_path_cost(
                switch_pos,
                exit_pos,
                adjacency,
                base_adjacency,
                cache_key="validation_exit",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )

            if switch_dist == float("inf"):
                logger.error(
                    f"LEVEL UNSOLVABLE: Switch unreachable from spawn ({player_pos} -> {switch_pos})"
                )
                return False
            if exit_dist == float("inf"):
                logger.error(
                    f"LEVEL UNSOLVABLE: Exit unreachable from switch ({switch_pos} -> {exit_pos})"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate level solvability: {e}")
            return None
