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
from ..constants import GAME_STATE_NEAREST_MINE_IDX

from ...constants.physics_constants import (
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    NINJA_RADIUS,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)

logger = logging.getLogger(__name__)

# Reward-boundary guard: a non-finite distance (the reachability graph returns inf
# for graph-OOB query positions) must NEVER reach the HWM/PBRS terms — `0 * inf`
# yields NaN and corrupts the whole policy. The env already substitutes a free-space
# fallback at the source (ReachabilityMixin._sanitize_cached_distances); this is the
# belt-and-suspenders clamp at the reward boundary for any path that bypasses it
# (e.g. the slow direct-graph fallback when no cached distance is in the obs).
_FAR_FINITE_DISTANCE = 4000.0  # px; larger than any real level path, finite


def _finite_distance(d: float) -> float:
    """Clamp a possibly-inf/NaN path distance to a large finite value."""
    if d != d or d == float("inf") or d == float("-inf"):
        return _FAR_FINITE_DISTANCE
    return d


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
    ):
        self.config = reward_config or RewardConfig()

        self.path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            shared_level_cache=shared_level_cache,
        )

        self.hwm_state = EpisodeRewardState()
        self.steps_taken = 0

        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        # Episode-level tracking for TensorBoard logging.
        self.episode_hwm_rewards: List[float] = []
        self.episode_pbrs_rewards: List[float] = []
        self.episode_time_penalties: List[float] = []
        self.episode_terminal_reward = 0.0
        # Exact reward accounting (observability/reconciliation only): the TRUE
        # running sum of every scaled reward this calculator returned, plus the
        # two components not captured by the lists above (truncation + backtrack).
        # Together they let a consumer verify components_sum == episode return ==
        # the VecMonitor return (the residual is the proof the displayed reward is
        # the actual episode reward). None of these feed the returned reward.
        self.episode_total_scaled = 0.0
        self.episode_truncation_penalty = 0.0
        self.episode_backtrack_penalty_total = 0.0
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

        # Goal-distance reverse curriculum: spawn cost-to-completion (the basis for
        # the "near goal" threshold) and whether the intermediate goal was reached
        # this episode. None spawn = no valid distance -> curriculum off this episode.
        self._spawn_goal_dist: Optional[float] = None
        self._reached_intermediate_goal: bool = False
        self._initial_dist_switch = 0.0
        self._initial_dist_exit = 0.0

        # Backtracking detection: previous A* distance to the current goal.
        self._prev_goal_dist: Optional[float] = None

        # Lattice/free-space PBRS (gated by config.enable_lattice_pbrs): the
        # previous-step potential Phi(s) and the spawn cost-to-completion D0 used
        # to normalize it. Phi(s) = -clamp(completion_cost / D0, 0, 1).
        self._prev_potential: Optional[float] = None
        self._pbrs_d0: Optional[float] = None
        # Normalized cost-to-completion of the CURRENT step (= -Phi, in [0,1]),
        # exposed so the trainer can use it as a cost-to-go auxiliary-task target.
        # -1.0 = no valid signal this step (mask it out of the aux loss).
        self._last_cost_to_go_norm: float = -1.0

        # Per-leg waypoint PBRS (gated by config.enable_waypoint_pbrs): previous-step
        # potential toward the CURRENT local waypoint and the waypoint index it was
        # measured against. The potential is rebased (no spurious reward) whenever the
        # index advances. See RewardConfig.enable_waypoint_pbrs.
        self._prev_waypoint_potential: Optional[float] = None
        self._prev_waypoint_index: Optional[int] = None
        self.episode_waypoint_reach_reward: float = 0.0

        # Frontier watermark bonus (gated by config.enable_frontier_bonus): the
        # best (minimum) free-space cost-to-go to the CURRENT live objective seen
        # this episode, and the objective phase (switch=False, exit=True) it was
        # measured against. Pays a small bonus only on a strictly-new minimum; the
        # watermark is REBASED (no spurious bonus) when the phase flips switch->exit
        # so the basis change is not paid as progress. Reuses ``current_goal_dist``
        # (the same cost-to-go the PBRS potential is built from). See
        # RewardConfig.enable_frontier_bonus.
        self._frontier_best_dist: Optional[float] = None
        self._frontier_phase: Optional[bool] = None

        # Dense mine-hazard PBRS (gated by config.enable_hazard_pbrs): previous-step
        # normalized nearest-deadly-mine distance potential (Phi_haz = d in [0,1]).
        # See RewardConfig.enable_hazard_pbrs.
        self._prev_hazard_potential: Optional[float] = None
        self.episode_hazard_pbrs_reward: float = 0.0

        # Gold collection (gated by config.enable_gold_rewards): track the
        # running collected count so we can reward the per-step delta, plus a
        # one-shot flag for the all-gold bonus.
        self._prev_gold_collected: Optional[int] = None
        self._all_gold_bonus_paid = False

        # Locked-door switches (gated by config.enable_locked_door_rewards):
        # reward each newly-opened NECESSARY door. obs["necessary_doors_open"] is
        # a monotonic non-decreasing count provided by the env's door-dependency
        # plan; we reward its positive delta.
        self._prev_necessary_doors_open: Optional[int] = None

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
        frames_executed: int = 1,
        truncated: bool = False,
    ) -> float:
        """Compute the unified step-level extrinsic reward.

        Args:
            obs: Current observation.
            prev_obs: Previous observation.
            action: Action taken (optional, unused here).
            frames_executed: Number of physics frames collapsed into this step.
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

        # Gold pickup reward (delta + all-gold bonus). Computed once per step,
        # before terminal branches, so gold collected on the same step the ninja
        # wins/dies still counts. No-op unless config.enable_gold_rewards.
        gold_reward = self._compute_gold_reward(obs, prev_obs)

        # Necessary-locked-door-switch reward (delta of opened necessary doors).
        # Computed once per step, before terminal branches, so a door opened on
        # the same step the ninja wins/dies still counts. Also yields the current
        # sub-goal phase for HWM watermark resets.
        door_reward, subgoal_phase = self._compute_locked_door_reward(obs)

        # Per-episode component accumulation (metrics only; computed before the
        # terminal branches so gold/door/switch earned on the winning/dying step
        # still counts). Does NOT affect the reward returned below.
        self.episode_gold_reward_total += gold_reward
        self.episode_locked_door_reward_total += door_reward
        if switch_just_activated:
            self.episode_sparse_switch_reward += self.config.switch_activation_reward

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

        dist_switch, dist_exit = self._get_distances(obs)

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
            # PBRS normalization: D0 = spawn cost-to-completion. Under the lattice/
            # free-space substitution the pre-switch distance is ALREADY composed to
            # cost-to-completion, so the spawn value is the phase-0 distance itself
            # (not switch+exit, which would double-count the exit tail).
            #
            # A FIXED reference (config._pbrs_d0_ref_override) overrides the
            # per-episode spawn D0 so the per-step shaping magnitude is route-length-
            # invariant across a geometry-morphing curriculum (otherwise it scales
            # 1/D0 and fades on the long, tight rungs). See RewardConfig.
            d0_ref = getattr(self.config, "_pbrs_d0_ref_override", None)
            self._pbrs_d0 = (
                max(float(d0_ref), 1e-6) if d0_ref is not None
                else max(self._initial_dist_switch, 1e-6)
            )
            self._prev_potential = self._potential(
                dist_switch if not switch_activated else dist_exit
            )
            # Goal-distance reverse curriculum basis: spawn cost-to-completion.
            # Only valid when the distance is finite (a reachable path exists).
            self._spawn_goal_dist = (
                float(dist_switch) if dist_switch < float("inf") else None
            )
            self._reached_intermediate_goal = False
            self._episode_initialized = True

        # Cost-to-completion to the current sub-goal (phase-aware). Under the
        # lattice/free-space substitution this is one continuous potential basis
        # across switch activation; under the geometric fallback it is the legacy
        # per-phase distance. Used by both the backtrack check and the PBRS term.
        current_goal_dist = dist_switch if not switch_activated else dist_exit

        # Goal-distance reverse curriculum: the episode SUCCEEDS once the agent has
        # covered a curriculum-controlled fraction of the path to the true goal,
        # i.e. its cost-to-completion drops to <= C_spawn*(1 - f). Treated exactly
        # like a real win below (pays the exit reward, terminates). f=1.0 disables
        # it (threshold 0 -> only the true goal). Off when no valid spawn distance.
        if (
            self.config.enable_goal_distance_curriculum
            and self.config.goal_distance_fraction < 1.0
            and self._spawn_goal_dist is not None
            and not self._reached_intermediate_goal
            and not obs.get("player_dead", False)
        ):
            threshold = self._spawn_goal_dist * (
                1.0 - self.config.goal_distance_fraction
            )
            if current_goal_dist <= threshold:
                self._reached_intermediate_goal = True

        # A "win" is the real exit OR the reverse-curriculum intermediate goal.
        won = bool(obs.get("player_won", False)) or self._reached_intermediate_goal

        # Terminal branches.
        if obs.get("player_dead", False):
            terminal_reward = self.config.death_penalty
            if obs.get("death_cause") == "mine":
                terminal_reward += self.config.mine_death_penalty
            if switch_just_activated:
                terminal_reward += self.config.switch_activation_reward
            terminal_reward += gold_reward
            terminal_reward += door_reward
            # PBRS terminal on DEATH, scaled by config.pbrs_death_clawback:
            #   clawback=1.0 -> pay -w*Phi(s_last): telescopes to the same constant
            #     +w as a win (strict policy-invariant, but removes the approach
            #     gradient -> fast-death collapse on hard exploration levels).
            #   clawback=0.0 (DEFAULT) -> pay nothing: the dense approach reward
            #     w*(Phi(s_last)+1) the agent earned EN ROUTE stays banked, so dying
            #     deeper in the level beats dying at spawn -> exploration gradient.
            # See RewardConfig.pbrs_death_clawback. Win is unaffected (full terminal).
            pbrs_reward = 0.0
            if self.config.enable_lattice_pbrs and self._prev_potential is not None:
                pbrs_reward = (
                    -self.config.pbrs_weight
                    * self.config.pbrs_death_clawback
                    * self._prev_potential
                )
                terminal_reward += pbrs_reward
                self.episode_pbrs_rewards.append(pbrs_reward)
            self.episode_terminal_reward = terminal_reward
            self.episode_death_penalty_reward = self.config.death_penalty
            if obs.get("death_cause") == "mine":
                self.episode_death_penalty_reward += self.config.mine_death_penalty

            scaled = terminal_reward * GLOBAL_REWARD_SCALE
            self.last_components = {
                "terminal_reward": terminal_reward,
                "hwm_reward": 0.0,
                "pbrs_reward": pbrs_reward,
                "gold_reward": gold_reward,
                "locked_door_reward": door_reward,
                "total_reward": terminal_reward,
                "scaled_reward": scaled,
                "is_terminal": True,
                "terminal_type": "death",
                "death_cause": obs.get("death_cause", "unknown"),
            }
            self.episode_total_scaled += scaled
            return scaled

        if won:
            terminal_reward = self.config.level_completion_reward
            if switch_just_activated:
                terminal_reward += self.config.switch_activation_reward
            terminal_reward += gold_reward
            terminal_reward += door_reward
            # PBRS terminal: Phi(terminal) = 0, so F = w*(gamma*0 - Phi(s)) =
            # -w*Phi(s) >= 0. This completes the telescoping sum to
            # w*(Phi_T - Phi_0) and pays a clean completion bonus (up to pbrs_weight).
            pbrs_reward = 0.0
            if self.config.enable_lattice_pbrs and self._prev_potential is not None:
                pbrs_reward = -self.config.pbrs_weight * self._prev_potential
                terminal_reward += pbrs_reward
                self.episode_pbrs_rewards.append(pbrs_reward)
            self.episode_terminal_reward = terminal_reward
            self.episode_sparse_exit_reward = self.config.level_completion_reward

            scaled = terminal_reward * GLOBAL_REWARD_SCALE
            self.last_components = {
                "terminal_reward": terminal_reward,
                "hwm_reward": 0.0,
                "pbrs_reward": pbrs_reward,
                "gold_reward": gold_reward,
                "locked_door_reward": door_reward,
                "total_reward": terminal_reward,
                "scaled_reward": scaled,
                "is_terminal": True,
                "terminal_type": "win",
            }
            self.episode_total_scaled += scaled
            return scaled

        # Non-terminal: HWM shaping + sparse switch + gold + time + backtrack.
        reward = 0.0

        if switch_just_activated:
            reward += self.config.switch_activation_reward

        reward += gold_reward
        reward += door_reward

        hwm_progress = self.hwm_state.step(
            dist_switch=dist_switch,
            dist_exit=dist_exit,
            switch_activated=switch_activated,
            subgoal_phase=subgoal_phase,
        )
        beta_hwm = self.config.beta_blend()
        hwm_reward = self.config.hwm_alpha * hwm_progress * beta_hwm
        reward += hwm_reward

        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward += time_penalty

        # Lattice/free-space PBRS: F = w*(gamma*Phi(s') - Phi(s)). Symmetric dense
        # per-frame signal that REPLACES the HWM ratchet + backtrack penalty when
        # enabled (hwm_alpha is forced to 0 in the config and the backtrack penalty
        # below is skipped). Telescopes to w*(Phi_T - Phi_0) over the episode.
        pbrs_reward = 0.0
        if self.config.enable_lattice_pbrs:
            phi_curr = self._potential(current_goal_dist)
            # Expose the normalized cost-to-go (-Phi in [0,1]) as the aux-task
            # target; mask (-1) when the cost is non-finite (no usable signal).
            self._last_cost_to_go_norm = (
                float(min(max(-phi_curr, 0.0), 1.0))
                if current_goal_dist < float("inf")
                else -1.0
            )
            prev_phi = (
                self._prev_potential if self._prev_potential is not None else phi_curr
            )
            pbrs_reward = self.config.pbrs_weight * (
                self.config.pbrs_gamma * phi_curr - prev_phi
            )
            self._prev_potential = phi_curr
            reward += pbrs_reward
            self.episode_pbrs_rewards.append(pbrs_reward)

        # Per-leg waypoint PBRS: F = w*(gamma*Phi(s') - Phi(s)) with Phi toward the
        # CURRENT local waypoint, REBASED on each waypoint advance so every leg is
        # short and the per-step gradient is route-length-invariant. Decoupled from the
        # planner: reads waypoint_leg_distance + waypoint_index that the env's
        # WaypointTracker writes into obs (all-zeros/absent -> safe no-op).
        if self.config.enable_waypoint_pbrs:
            leg_d = obs.get("waypoint_leg_distance")
            wp_idx = obs.get("waypoint_index")
            if leg_d is not None and wp_idx is not None and leg_d < float("inf"):
                leg_scale = max(float(self.config.waypoint_leg_scale), 1e-6)
                phi_wp = -min(max(float(leg_d) / leg_scale, 0.0), 1.0)
                # Explicit per-waypoint REACH BONUS on a FORWARD cursor advance: a
                # concrete dense progress signal for the long return leg (the
                # per-leg PBRS alone is policy-invariant shaping). Forward-only, so
                # it rewards real route progress and cannot be farmed by oscillating.
                reach_w = float(getattr(self.config, "waypoint_reach_bonus", 0.0))
                if (
                    reach_w > 0.0
                    and self._prev_waypoint_index is not None
                    and wp_idx > self._prev_waypoint_index
                ):
                    reach = reach_w * float(wp_idx - self._prev_waypoint_index)
                    reward += reach
                    self.episode_waypoint_reach_reward += reach
                if wp_idx != self._prev_waypoint_index:
                    # New leg (or first step): hand-off pays 0; rebase the basis.
                    self._prev_waypoint_index = wp_idx
                    self._prev_waypoint_potential = phi_wp
                prev_phi_wp = (
                    self._prev_waypoint_potential
                    if self._prev_waypoint_potential is not None
                    else phi_wp
                )
                wp_pbrs = self.config.pbrs_weight * (
                    self.config.pbrs_gamma * phi_wp - prev_phi_wp
                )
                self._prev_waypoint_potential = phi_wp
                pbrs_reward += wp_pbrs
                reward += wp_pbrs
                self.episode_pbrs_rewards.append(wp_pbrs)

        # Dense mine-hazard avoidance PBRS: Phi_haz = normalized nearest-deadly-mine
        # distance d in [0,1] (1=safe/none near, 0=touching), read from game_state.
        # F = w*(Phi' - Phi) (gamma=1, pure telescoping) gives a per-step gradient
        # that REWARDS increasing mine distance and PENALIZES decreasing it -- the
        # dense "steer around the mine" signal the terminal death penalty alone
        # cannot provide. gamma=1 makes F exactly 0 wherever d is constant (e.g. no
        # mine within range -> d=1 every step), so it never adds survival drift and
        # is a no-op on mine-free levels. Policy-invariant within the episode.
        if self.config.enable_hazard_pbrs:
            gs = obs.get("game_state")
            d_mine = None
            if gs is not None:
                try:
                    if len(gs) > GAME_STATE_NEAREST_MINE_IDX:
                        d_mine = float(gs[GAME_STATE_NEAREST_MINE_IDX])
                except (TypeError, IndexError):
                    d_mine = None
            if d_mine is not None and d_mine == d_mine:  # finite, not NaN
                phi_haz = min(max(d_mine, 0.0), 1.0)
                prev_phi_haz = (
                    self._prev_hazard_potential
                    if self._prev_hazard_potential is not None
                    else phi_haz
                )
                haz_pbrs = float(self.config.hazard_pbrs_weight) * (
                    phi_haz - prev_phi_haz
                )
                self._prev_hazard_potential = phi_haz
                reward += haz_pbrs
                pbrs_reward += haz_pbrs
                self.episode_hazard_pbrs_reward += haz_pbrs
                self.episode_pbrs_rewards.append(haz_pbrs)

        # Frontier watermark bonus: pay a small bonus ONLY on a strictly-new
        # closest-approach (new minimum cost-to-go) to the current live objective
        # this episode -- rewarding monotonic progress around a long loop, not
        # basin-wandering. Reuses ``current_goal_dist`` (the SAME phase-aware
        # cost-to-go the PBRS potential is built from). The watermark is REBASED
        # (no bonus) when the live objective flips switch->exit so the basis change
        # is not mistaken for a huge improvement. Robust to inf/None cost-to-go
        # (no bonus, watermark untouched). Off unless config.enable_frontier_bonus.
        frontier_bonus = 0.0
        if self.config.enable_frontier_bonus:
            phase = bool(switch_activated)
            if current_goal_dist < float("inf"):
                if (
                    self._frontier_best_dist is None
                    or phase != self._frontier_phase
                ):
                    # First valid measurement or objective change: rebase (no bonus).
                    self._frontier_best_dist = current_goal_dist
                    self._frontier_phase = phase
                elif current_goal_dist < self._frontier_best_dist - 1e-6:
                    # NORMALIZE the improvement by D0 (the PBRS potential's
                    # denominator) so the bonus is in the SAME units as the per-step
                    # PBRS term, as its docstring intends. Without this the bonus is
                    # in raw cost-to-go units (~D0 over a full route) and banks
                    # ~frontier_bonus_weight*D0 just for APPROACHING — on a long loop
                    # that rivals/exceeds R_EXIT, so the agent is paid to approach,
                    # not complete (completion-preference dilution). Normalized, the
                    # whole-route bonus is bounded by ~frontier_bonus_weight (a small
                    # commitment carrot that can never out-weigh the win).
                    d0 = max(float(getattr(self, "_pbrs_d0", 1.0) or 1.0), 1e-6)
                    frontier_bonus = self.config.frontier_bonus_weight * (
                        (self._frontier_best_dist - current_goal_dist) / d0
                    )
                    self._frontier_best_dist = current_goal_dist
                    reward += frontier_bonus
                    self.episode_frontier_bonus_total += frontier_bonus

        # Forward/backtrack bookkeeping. The backtrack PENALTY applies only in the
        # HWM pipeline — PBRS already penalizes regress symmetrically — but the
        # step counters stay populated for observability in both modes.
        backtrack_penalty = 0.0
        regressed = (
            self._prev_goal_dist is not None
            and current_goal_dist < float("inf")
            and self._prev_goal_dist < float("inf")
            and current_goal_dist > self._prev_goal_dist
        )
        if regressed:
            if not (self.config.enable_lattice_pbrs or self.config.enable_waypoint_pbrs):
                backtrack_penalty = BACKTRACK_PENALTY * frames_executed
                reward += backtrack_penalty
                self.episode_backtrack_penalty_total += backtrack_penalty
            self.episode_backtrack_steps += 1
        elif (
            hwm_progress > 0
            or self.config.enable_lattice_pbrs
            or self.config.enable_waypoint_pbrs
        ):
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
            self.episode_truncation_penalty += self.config.truncation_penalty

        scaled = reward * GLOBAL_REWARD_SCALE
        self.last_components = {
            "hwm_reward": hwm_reward,
            "hwm_progress": hwm_progress,
            "pbrs_reward": pbrs_reward,
            "potential": self._prev_potential,
            "completion_cost": current_goal_dist,
            "gold_reward": gold_reward,
            "locked_door_reward": door_reward,
            "subgoal_phase": subgoal_phase,
            "beta_blend_hwm": beta_hwm,
            "time_penalty": time_penalty,
            "backtrack_penalty": backtrack_penalty,
            "frontier_bonus": frontier_bonus,
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
        self.episode_total_scaled += scaled
        return scaled

    def _potential(self, completion_cost: float) -> float:
        """Normalized PBRS potential ``Phi(s) = -clamp(cost / D0, 0, 1)`` in [-1, 0].

        D0 is the spawn cost-to-completion, so ``Phi(spawn) = -1`` and
        ``Phi(complete) = 0``. A non-finite cost (off-route / transiently
        unreachable) clamps to the worst potential (-1) so transitions in and out
        of severance are bounded rather than infinite.
        """
        d0 = self._pbrs_d0
        if d0 is None or not (completion_cost < float("inf")):
            return -1.0
        return -min(max(completion_cost / d0, 0.0), 1.0)

    def _get_distances(
        self,
        obs: Dict[str, Any],
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
            dist_switch = _finite_distance(float(cs))
            dist_exit = _finite_distance(float(ce))
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
            return _FAR_FINITE_DISTANCE, _FAR_FINITE_DISTANCE

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

        dist_switch = _finite_distance(dist_switch)
        dist_exit = _finite_distance(dist_exit)
        obs["_pbrs_last_distance_to_goal"] = (
            dist_switch if not obs.get("switch_activated", False) else dist_exit
        )

        return dist_switch, dist_exit

    def _compute_gold_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Per-step gold reward: pickup delta + one-shot all-gold bonus.

        Order-agnostic and non-negative: rewards the increase in collected gold
        regardless of which piece or order, so opportunistic cluster grabs are
        never penalized. Soft objective — does not gate completion. Returns 0
        when ``enable_gold_rewards`` is off. Must be called exactly once per step
        (it advances ``_prev_gold_collected``).
        """
        if not self.config.enable_gold_rewards:
            return 0.0
        gold_now = int(obs.get("gold_collected", 0) or 0)
        total_gold = int(obs.get("total_gold", 0) or 0)
        if self._prev_gold_collected is None:
            self._prev_gold_collected = int(prev_obs.get("gold_collected", 0) or 0)
        newly = gold_now - self._prev_gold_collected
        self._prev_gold_collected = gold_now

        gold_reward = 0.0
        if newly > 0:
            gold_reward += newly * self.config.gold_reward_per_piece
        if total_gold > 0 and gold_now >= total_gold and not self._all_gold_bonus_paid:
            gold_reward += self.config.all_gold_bonus
            self._all_gold_bonus_paid = True
        return gold_reward

    def _compute_locked_door_reward(self, obs: Dict[str, Any]) -> Tuple[float, int]:
        """Reward for opening NECESSARY locked doors this step + the sub-goal phase.

        ``obs["necessary_doors_open"]`` is a monotonic non-decreasing count of the
        necessary locked doors whose switch has been collected (computed by the
        env from the planner's door-dependency plan). We reward its positive delta
        so each newly-opened necessary door pays a one-shot bonus; non-necessary
        doors are excluded from the count and pay nothing. Returns
        ``(reward, subgoal_phase)`` where ``subgoal_phase`` (the same count) drives
        the HWM watermark reset. Must be called exactly once per step (it advances
        ``_prev_necessary_doors_open``). Reward is 0 when the feature is disabled.
        """
        n_open = int(obs.get("necessary_doors_open", 0) or 0)
        if not self.config.enable_locked_door_rewards:
            return 0.0, n_open
        if self._prev_necessary_doors_open is None:
            self._prev_necessary_doors_open = n_open
        newly = n_open - self._prev_necessary_doors_open
        self._prev_necessary_doors_open = n_open
        reward = newly * self.config.locked_door_switch_reward if newly > 0 else 0.0
        return reward, n_open

    def reset(self):
        """Reset episode-level reward state."""
        self.steps_taken = 0
        self.hwm_state = EpisodeRewardState()
        self._episode_initialized = False

        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.closest_distance_this_episode = float("inf")

        self.episode_hwm_rewards = []
        self.episode_pbrs_rewards = []
        self.episode_time_penalties = []
        self.episode_terminal_reward = 0.0
        self.episode_total_scaled = 0.0
        self.episode_truncation_penalty = 0.0
        self.episode_backtrack_penalty_total = 0.0
        # Per-episode reward-component breakdown (metrics/observability only;
        # these never feed the returned reward). Lets TensorBoard split the
        # lumped ``terminal_reward`` into sparse switch / sparse exit / death and
        # surface gold/door contributions separately.
        self.episode_sparse_switch_reward = 0.0
        self.episode_sparse_exit_reward = 0.0
        self.episode_death_penalty_reward = 0.0
        self.episode_gold_reward_total = 0.0
        self.episode_locked_door_reward_total = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self.episode_frontier_bonus_total = 0.0

        self.episode_path_length = 0.0
        self.optimal_path_length = None
        self.prev_player_pos = None
        self.spawn_position = None
        self._prev_goal_dist = None
        self._prev_potential = None
        self._pbrs_d0 = None
        self._prev_waypoint_potential = None
        self._prev_waypoint_index = None
        self.episode_waypoint_reach_reward = 0.0
        self._frontier_best_dist = None
        self._frontier_phase = None
        self._prev_hazard_potential = None
        self.episode_hazard_pbrs_reward = 0.0
        self._prev_gold_collected = None
        self._all_gold_bonus_paid = False
        self._prev_necessary_doors_open = None

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
        pbrs_total = (
            sum(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
        )
        pbrs_mean = (
            float(np.mean(self.episode_pbrs_rewards))
            if self.episode_pbrs_rewards
            else 0.0
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

        # Exact decomposition: these MUTUALLY-EXCLUSIVE component sums plus the
        # shaping lists add up to the returned episode reward. sparse_total folds
        # every non-shaping, non-time contribution (switch/exit/death/gold/door);
        # the terminal PBRS bonus lives in pbrs_total (not double-counted here).
        sparse_total = (
            self.episode_sparse_switch_reward
            + self.episode_sparse_exit_reward
            + self.episode_death_penalty_reward
            + self.episode_gold_reward_total
            + self.episode_locked_door_reward_total
        )
        components_sum = (
            pbrs_total
            + hwm_total
            + time_penalty_total
            + sparse_total
            + self.episode_truncation_penalty
            + self.episode_backtrack_penalty_total
            + self.episode_frontier_bonus_total
        ) * GLOBAL_REWARD_SCALE
        # residual ~ 0 proves the decomposition is complete (no unaccounted reward).
        reconciliation_residual = components_sum - self.episode_total_scaled

        return {
            "hwm_total": hwm_total,
            "hwm_mean": hwm_mean,
            "hwm_std": hwm_std,
            "pbrs_total": pbrs_total,
            "pbrs_mean": pbrs_mean,
            "time_penalty_total": time_penalty_total,
            "sparse_total": sparse_total,
            "truncation_penalty_total": self.episode_truncation_penalty,
            "backtrack_penalty_total": self.episode_backtrack_penalty_total,
            "frontier_bonus_total": self.episode_frontier_bonus_total,
            # The TRUE sum of every scaled reward returned this episode. Equals the
            # VecMonitor episode return; components_sum equals this too (residual~0).
            "episode_return_internal": self.episode_total_scaled,
            "components_sum": components_sum,
            "reconciliation_residual": reconciliation_residual,
            "terminal_reward": self.episode_terminal_reward,
            "sparse_switch_reward": self.episode_sparse_switch_reward,
            "sparse_exit_reward": self.episode_sparse_exit_reward,
            "death_penalty_reward": self.episode_death_penalty_reward,
            "gold_reward_total": self.episode_gold_reward_total,
            "locked_door_reward_total": self.episode_locked_door_reward_total,
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
