"""HWM A* reward calculator for N++ RL training.

Implements High-Water-Mark (HWM) A* distance shaping:
  r_extrinsic = r_sparse + alpha * hwm_progress

Where:
- r_sparse: Fixed terminal events (R_SWITCH, R_EXIT, R_DEATH)
- hwm_progress: Normalized progress toward current objective (non-negative)
- alpha: Constant weight (HWM_ALPHA = 1.0)

HWM properties:
- Non-negative shaping: backtracking yields 0 reward (not negative)
- Bounded cumulative: total HWM <= 1.0 per phase (normalized by initial distance)
- Non-exploitable: cycling distance can't generate reward
- Two-phase: switch objective then exit objective

RND intrinsic rewards are NOT computed here. They are handled on the
training side (npp-rl) via dual value heads and separate GAE streams.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from .reward_config import RewardConfig
from .hwm_shaping import EpisodeRewardState
from .pbrs_potentials import PBRSCalculator
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
    """HWM A* reward calculator.

    Components:
    - Sparse terminal rewards: R_SWITCH, R_EXIT, R_DEATH (fixed)
    - HWM shaping: non-negative progress toward current objective

    Returns extrinsic reward only. Intrinsic RND reward computed on training side.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        pbrs_gamma: float = 0.99,
        shared_level_cache: Optional[Any] = None,
        shared_level_caches_by_stage: Optional[Dict[int, Any]] = None,
        goal_curriculum_manager: Optional[Any] = None,
        simulation_phi_dir: Optional[str] = None,
    ):
        self.config = reward_config or RewardConfig()

        self.goal_curriculum_manager = goal_curriculum_manager

        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            shared_level_cache=shared_level_cache,
            shared_level_caches_by_stage=shared_level_caches_by_stage,
            goal_curriculum_manager=goal_curriculum_manager,
            simulation_phi_dir=simulation_phi_dir,
        )
        # Keep pbrs_calculator for backward compat with base_environment
        # (it accesses path_calculator and surface area through this)
        self.pbrs_calculator = PBRSCalculator(
            path_calculator=path_calculator, reward_config=self.config
        )

        self.hwm_state = EpisodeRewardState()
        self.steps_taken = 0

        # Diagnostic tracking
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        # Episode-level tracking for TensorBoard logging
        self.episode_hwm_rewards: List[float] = []
        self.episode_terminal_reward = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self.current_potential = None
        self.prev_potential = None

        # Backward compat aliases
        self.episode_pbrs_rewards: List[float] = []
        self.episode_time_penalties: List[float] = []

        # Path optimality tracking
        self.episode_path_length = 0.0
        self.optimal_path_length: Optional[float] = None
        self.prev_player_pos: Optional[Tuple[float, float]] = None

        self.spawn_position: Optional[Tuple[float, float]] = None

        self.last_pbrs_components: Dict[str, Any] = {}

        # Visualization data
        self.velocity_history: List[Tuple[float, float]] = []
        self.alignment_history: List[float] = []
        self.path_gradient_history: List[Optional[Tuple[float, float]]] = []
        self.potential_history: List[float] = []
        self._last_next_hop_world: Optional[Tuple[float, float]] = None
        self._last_next_hop_goal_id: Optional[str] = None

        # Cached paths for visualization
        self._cached_switch_path: Optional[List[Tuple[int, int]]] = None
        self._cached_exit_path: Optional[List[Tuple[int, int]]] = None
        self._cached_spawn_node: Optional[Tuple[int, int]] = None
        self._cached_switch_node: Optional[Tuple[int, int]] = None
        self._cached_exit_node: Optional[Tuple[int, int]] = None

        # Episode initialization flag (distances needed for HWM normalization)
        self._episode_initialized = False
        self._initial_dist_switch = 0.0
        self._initial_dist_exit = 0.0

        # Backtracking penalty: track previous A* distance to current goal
        # to detect steps where the agent moves away from the objective.
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
        """Calculate extrinsic reward using HWM A* shaping.

        Args:
            obs: Current observation.
            prev_obs: Previous observation.
            action: Action taken (optional).
            frames_executed: Number of physics frames in this step.
            curriculum_manager: Optional curriculum manager.
            truncated: True if episode ended by timeout (adds TRUNCATION_PENALTY).

        Returns:
            float: Extrinsic reward (sparse + HWM shaping) * GLOBAL_REWARD_SCALE
        """
        self.steps_taken += frames_executed

        if self.spawn_position is None:
            self.spawn_position = (prev_obs["player_x"], prev_obs["player_y"])

        switch_activated = obs.get("switch_activated", False)
        switch_just_activated = switch_activated and not prev_obs.get(
            "switch_activated", False
        )

        # L1.5: refresh level cache when mine or switch state changes (e.g. toggle mines).
        # build_level_cache fast-rejects when signatures match; ~100ms on actual rebuild.
        level_data = obs.get("level_data")
        adjacency = obs.get("_adjacency_graph")
        graph_data = obs.get("_graph_data")
        if level_data is not None and adjacency is not None:
            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )
            self.pbrs_calculator.path_calculator.build_level_cache(
                level_data, adjacency, base_adjacency, graph_data
            )

        # === Compute A* distances ===
        dist_switch, dist_exit = self._get_distances(obs, curriculum_manager)

        # === Initialize HWM on first step ===
        if not self._episode_initialized:
            self._initial_dist_switch = dist_switch if dist_switch < float("inf") else 1.0
            self._initial_dist_exit = dist_exit if dist_exit < float("inf") else 1.0
            self.hwm_state.reset(
                initial_dist_switch=self._initial_dist_switch,
                max_dist_switch=self._initial_dist_switch,
                initial_dist_exit=self._initial_dist_exit,
                max_dist_exit=self._initial_dist_exit,
            )
            self.optimal_path_length = self._initial_dist_switch + self._initial_dist_exit
            self._episode_initialized = True

        # === TERMINAL REWARDS ===
        if obs.get("player_dead", False):
            terminal_reward = self.config.death_penalty
            if obs.get("death_cause") == "mine":
                terminal_reward += self.config.mine_death_penalty
            if switch_just_activated:
                terminal_reward += self.config.switch_activation_reward
            self.episode_terminal_reward = terminal_reward

            scaled = terminal_reward * GLOBAL_REWARD_SCALE
            self.potential_history.append(0.0)
            self.last_pbrs_components = {
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
            self.potential_history.append(1.0)
            self.last_pbrs_components = {
                "terminal_reward": terminal_reward,
                "hwm_reward": 0.0,
                "total_reward": terminal_reward,
                "scaled_reward": scaled,
                "is_terminal": True,
                "terminal_type": "win",
            }
            return scaled

        # === NON-TERMINAL: HWM shaping + sparse switch reward + time penalty ===
        reward = 0.0

        if switch_just_activated:
            reward += self.config.switch_activation_reward

        hwm_progress = self.hwm_state.step(
            dist_switch=dist_switch,
            dist_exit=dist_exit,
            switch_activated=switch_activated,
        )
        # NPP-14: when BSRS blend is on, scale HWM by beta so total dense signal is
        # beta * HWM (env) + (1-beta) * eta * (gamma V' - V) (trainer, MaskedPPO).
        beta_hwm = self.config.beta_blend()
        hwm_reward = self.config.hwm_alpha * hwm_progress * beta_hwm
        reward += hwm_reward

        # Time penalty: small per-step cost to discourage aimless backtracking.
        # Applied ONLY on non-terminal steps (terminal rewards already carry signal).
        # Value -0.001/step means a 500-step episode costs -0.5 (same as R_DEATH).
        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward += time_penalty

        # Backtracking penalty: applied when agent moves away from current A* goal.
        # HWM shaping gives 0 reward for backtracking (non-negative property), but this
        # makes backtracking actively negative. Run-221-2 showed 80-93% backtracking
        # across all stages -- this penalty provides an additional directional nudge.
        # Skipped on the first step of an episode (no previous distance to compare).
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
        self.episode_pbrs_rewards.append(hwm_reward)
        self.episode_time_penalties.append(time_penalty)

        # === Track movement for diagnostics ===
        current_pos = (obs["player_x"], obs["player_y"])
        movement = 0.0
        if self.prev_player_pos is not None:
            dx = current_pos[0] - self.prev_player_pos[0]
            dy = current_pos[1] - self.prev_player_pos[1]
            movement = (dx * dx + dy * dy) ** 0.5
            self.episode_path_length += movement
            self.velocity_history.append((dx, dy))
        else:
            self.velocity_history.append((0.0, 0.0))
        self.prev_player_pos = current_pos

        # Track closest distances
        current_goal_dist = dist_switch if not switch_activated else dist_exit
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
                obs["player_x"], obs["player_y"], obs["exit_door_x"], obs["exit_door_y"]
            )
            if eucl_exit < self.closest_distance_to_exit:
                self.closest_distance_to_exit = eucl_exit

        self.potential_history.append(hwm_progress)

        if truncated:
            reward += self.config.truncation_penalty

        scaled = reward * GLOBAL_REWARD_SCALE
        self.last_pbrs_components = {
            "hwm_reward": hwm_reward,
            "hwm_progress": hwm_progress,
            "beta_blend_hwm": beta_hwm,
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

        Prefer distances from the observation dict when present (NppEnvironment
        computes them in ``_compute_reachability`` using the env's path calculator).
        The reward calculator holds a *separate* ``CachedPathDistanceCalculator``;
        skipping duplicate ``get_path_cost`` calls removes two hot-path A* lookups per step.
        """
        # Fast path: obs/reward sharing (see NppEnvironment._get_observation)
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

        dist_switch = self.pbrs_calculator.path_calculator.get_path_cost(
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

        dist_exit = self.pbrs_calculator.path_calculator.get_path_cost(
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

        # Store in obs for downstream consumers
        obs["_pbrs_last_distance_to_goal"] = (
            dist_switch if not obs.get("switch_activated", False) else dist_exit
        )

        return dist_switch, dist_exit

    def reset(self):
        """Reset episode state."""
        self.steps_taken = 0
        self.hwm_state = EpisodeRewardState()
        self._episode_initialized = False

        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        self.episode_hwm_rewards = []
        self.episode_terminal_reward = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self.current_potential = None
        self.prev_potential = None

        self.episode_pbrs_rewards = []
        self.episode_time_penalties = []

        self.episode_path_length = 0.0
        self.optimal_path_length = None
        self.prev_player_pos = None
        self.spawn_position = None
        self._prev_goal_dist = None

        self.last_pbrs_components = {}

        self.velocity_history = []
        self.alignment_history = []
        self.path_gradient_history = []
        self.potential_history = []
        self._last_next_hop_world = None
        self._last_next_hop_goal_id = None

        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()

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

    def get_pbrs_metrics(self) -> Dict[str, Any]:
        """Backward-compatible metrics accessor."""
        return {
            "prev_potential": self.prev_potential,
            "pbrs_gamma": 0.99,
            "closest_distance_to_switch": self.closest_distance_to_switch,
            "closest_distance_to_exit": self.closest_distance_to_exit,
        }

    def get_cached_paths(self) -> Dict[str, Any]:
        return {
            "switch_path": self._cached_switch_path,
            "exit_path": self._cached_exit_path,
            "spawn_node": self._cached_spawn_node,
            "switch_node": self._cached_switch_node,
            "exit_node": self._cached_exit_node,
        }

    def get_pbrs_diagnostic_data(self) -> Dict[str, Any]:
        return {
            "velocity_history": self.velocity_history,
            "alignment_history": self.alignment_history,
            "path_gradient_history": self.path_gradient_history,
            "potential_history": self.potential_history,
            "last_next_hop_world": self._last_next_hop_world,
            "last_next_hop_goal_id": self._last_next_hop_goal_id,
        }

    def get_episode_reward_metrics(self) -> Dict[str, Any]:
        import numpy as np

        hwm_total = sum(self.episode_hwm_rewards) if self.episode_hwm_rewards else 0.0
        hwm_mean = float(np.mean(self.episode_hwm_rewards)) if self.episode_hwm_rewards else 0.0
        hwm_std = float(np.std(self.episode_hwm_rewards)) if self.episode_hwm_rewards else 0.0

        path_optimality = 0.0
        if (
            self.episode_path_length > 0
            and self.optimal_path_length is not None
            and self.optimal_path_length > 0
        ):
            path_optimality = self.optimal_path_length / self.episode_path_length

        total_steps = self.steps_taken
        forward_pct = (self.episode_forward_steps / total_steps * 100) if total_steps > 0 else 0.0

        return {
            "pbrs_total": hwm_total,
            "hwm_total": hwm_total,
            "time_penalty_total": 0.0,
            "terminal_reward": self.episode_terminal_reward,
            "pbrs_to_penalty_ratio": 0.0,
            "forward_steps": self.episode_forward_steps,
            "backtrack_steps": self.episode_backtrack_steps,
            "pbrs_mean": hwm_mean,
            "pbrs_std": hwm_std,
            "hwm_mean": hwm_mean,
            "hwm_std": hwm_std,
            "current_potential": self.current_potential,
            "prev_potential": self.prev_potential,
            "path_optimality": path_optimality,
            "forward_progress_pct": forward_pct,
            "backtracking_pct": 100.0 - forward_pct if total_steps > 0 else 0.0,
            "episode_path_length": self.episode_path_length,
            "optimal_path_length": self.optimal_path_length or 0.0,
            "watermark_switch": self.hwm_state.best_switch_distance,
            "watermark_exit": self.hwm_state.best_exit_distance,
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

            switch_dist = self.pbrs_calculator.path_calculator.get_path_cost(
                player_pos, switch_pos, adjacency, base_adjacency,
                cache_key="validation_switch", level_data=level_data,
                graph_data=graph_data, entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
            exit_dist = self.pbrs_calculator.path_calculator.get_path_cost(
                switch_pos, exit_pos, adjacency, base_adjacency,
                cache_key="validation_exit", level_data=level_data,
                graph_data=graph_data, entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )

            if switch_dist == float("inf"):
                logger.error(f"LEVEL UNSOLVABLE: Switch unreachable from spawn ({player_pos} -> {switch_pos})")
                return False
            if exit_dist == float("inf"):
                logger.error(f"LEVEL UNSOLVABLE: Exit unreachable from switch ({switch_pos} -> {exit_pos})")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate level solvability: {e}")
            return None
