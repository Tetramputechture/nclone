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
from ..constants import (
    GAME_STATE_NEAREST_MINE_IDX,
    NINJA_STATE_DIM,
    VISITATION_CELL_PX,
    VISITATION_ORIGIN_PX,
)

from ...constants.physics_constants import (
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    MAX_HOR_SPEED,
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
        # Per-episode component breakdown accumulators (metrics only). Defined
        # here as well as in reset() so calculate_reward() is safe even before the
        # first reset() (they are read/incremented on every step, incl. terminals).
        self.episode_sparse_switch_reward = 0.0
        self.episode_sparse_exit_reward = 0.0
        self.episode_death_penalty_reward = 0.0
        self.episode_gold_reward_total = 0.0
        # Gold reward actually PAID into the returned rewards this episode
        # (differs from episode_gold_reward_total under banking or a death
        # clawback). This is the term that makes the reconciliation exact.
        self.episode_gold_paid_total = 0.0
        self.episode_locked_door_reward_total = 0.0
        self.episode_frontier_bonus_total = 0.0

        self.episode_path_length = 0.0
        self.optimal_path_length: Optional[float] = None
        # Fastest winning frame count seen on the CURRENT level, for the
        # record-relative completion-speed bonus. Deliberately survives
        # ``reset()`` (it is a cross-episode personal best) and is cleared only
        # when a new level is loaded.
        self._best_completion_frames: Optional[float] = None
        self.episode_new_record_reward: float = 0.0
        # Momentum-PBRS state: previous goal distance (to derive closing speed)
        # and the previous momentum potential. Both are per-EPISODE.
        self._prev_goal_dist: Optional[float] = None
        self._prev_momentum_potential: Optional[float] = None
        self.episode_momentum_pbrs_reward: float = 0.0
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
        # Severance diagnostics -- see `_potential`. `_pbrs_steps` counts
        # potential evaluations so "severed on the first one" is detectable.
        self._pbrs_steps: int = 0
        self._pbrs_severed_steps: int = 0
        self._pbrs_severed_from_spawn: bool = False
        # Geometric-fallback continuity (per episode). When PBRS is enabled but
        # the env's potential substitution did NOT commit (obs
        # "_potential_substituted" is False), the raw per-phase A* distances
        # would flip the potential basis at switch activation (net-negative
        # activation step). Fix: compose the pre-switch basis to
        # dist_to_switch + dist(switch -> exit) so Phi is continuous.
        # ``_pbrs_exit_leg`` holds that constant exit tail; when it cannot be
        # computed, ``_pbrs_hwm_fallback`` disables the PBRS term for the
        # episode and re-enables the HWM ratchet (+ backtrack penalty) instead.
        self._pbrs_exit_leg: Optional[float] = None
        self._pbrs_hwm_fallback: bool = False
        # One-shot per-episode warning latch for the switch-activation
        # potential-continuity guard.
        self._phi_flip_warned: bool = False
        # Normalized cost-to-completion of the CURRENT step (= -Phi, in [0,1]),
        # exposed so the trainer can use it as a cost-to-go auxiliary-task target.
        # -1.0 = no valid signal this step (mask it out of the aux loss).
        self._last_cost_to_go_norm: float = -1.0
        # Raw (un-normalized) cost-to-go, so the pace table built offline and the
        # runtime potential are keyed on the SAME quantity.
        self._last_cost_to_go_raw: float = float("inf")
        self._pace_cells = None      # (gx, gy) -> earliest reference frame
        self._pace_grid = 12.0
        self._pace_loaded = False
        self._prev_pace_potential: Optional[float] = None
        self.episode_pace_pbrs_reward: float = 0.0

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

        # Grounded-landing high-water-mark (gated by config.enable_landing_pbrs):
        # the closest-ever *grounded* cost-to-go to the current live objective this
        # episode, plus the objective phase it was measured against (rebased on the
        # switch->exit flip, like the frontier bonus). See RewardConfig.
        self._landing_best_dist: Optional[float] = None
        self._landing_phase: Optional[bool] = None
        self.episode_landing_bonus_total: float = 0.0
        # Previous-step airborne flag: the landing bonus fires ONLY on the
        # airborne->contact transition (an actual landing / wall-catch), never on
        # continuous ground contact, so flat running earns nothing here.
        self._prev_airborne: Optional[bool] = None
        # Progress-stall (dither) penalty state (gated by config.enable_stall_penalty).
        # _stall_best_cost = best (min) cost-to-go to the live objective seen this
        # episode; _stall_phase = objective phase it was measured against (rebased on
        # switch->exit flip); _stall_no_improve_frames = sim frames since the last
        # strict improvement; _stall_anchor_pos = position at the last improvement
        # (the low-variance reference).
        self._stall_best_cost: Optional[float] = None
        self._stall_phase: Optional[bool] = None
        self._stall_no_improve_frames: int = 0
        self._stall_anchor_pos: Optional[tuple] = None
        # Frames the hazard exemption has absorbed this episode. Bounded by
        # config.stall_hazard_exempt_max_frames so parking next to a drone
        # is not a way to buy permanent immunity from the dither penalty.
        self._stall_hazard_exempt_frames: int = 0
        self.episode_stall_penalty_total: float = 0.0
        # Per-episode route-length normalizer for the frontier/landing HWM bonuses
        # (spawn cost-to-completion). SEPARATE from the PBRS _pbrs_d0, which may be a
        # fixed cross-level reference: a fixed reference below a long maze's cost-to-go
        # would let the whole-route bonus balloon to weight * route/ref instead of
        # staying bounded by ~weight. None until the first finite cost-to-completion.
        self._shaping_d0: Optional[float] = None

        # Dense mine-hazard PBRS (gated by config.enable_hazard_pbrs): previous-step
        # normalized nearest-deadly-mine distance potential (Phi_haz = d in [0,1]).
        # See RewardConfig.enable_hazard_pbrs.
        self._prev_hazard_potential: Optional[float] = None
        self.episode_hazard_pbrs_reward: float = 0.0
        # Dense impact-survivability PBRS (gated by config.enable_impact_pbrs).
        self._prev_impact_potential: Optional[float] = None
        self.episode_impact_pbrs_reward: float = 0.0
        # Doomed-arc risk PBRS (gated by config.enable_arc_risk_pbrs).
        self._prev_arc_risk_potential: Optional[float] = None
        self.episode_arc_risk_pbrs_reward: float = 0.0
        self.episode_speed_bonus_reward: float = 0.0

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

    def _hazard_blocked(self, obs: Dict[str, Any], pos) -> bool:
        """Is a live threat near enough that holding position is correct?

        Reads ``threat_ttl``'s time-to-lethal channels (1 = certain tier,
        3 = reactive tier) on their native 12 px observation grid. Polarity is
        1.0 = safe, so anything below 1.0 is "lethal within the forecast
        window". Returns False whenever the key is absent (no threat obs -> no
        exemption -> the penalty behaves exactly as it did before), or once the
        per-episode exemption budget is spent.

        The CERTAIN tier is what carries this: patrol drones are forecastable
        from frame 0 and do not need the ninja to move. The reactive tier is
        included for completeness but a stationary ninja often will not have
        triggered acquisition -- see the turret finding in the gamma-1 audit
        (§9.3), which is why this must never be the only thing keeping a level
        winnable.
        """
        if not getattr(self.config, "stall_hazard_exempt", False):
            return False
        budget = int(getattr(self.config, "stall_hazard_exempt_max_frames", 900))
        if self._stall_hazard_exempt_frames >= budget:
            return False
        ttl = obs.get("threat_ttl")
        if ttl is None:
            return False
        try:
            import numpy as _np

            arr = _np.asarray(ttl)
            if arr.ndim != 3 or arr.shape[0] < 4:
                return False
            _c, h, w = arr.shape
            # threat_ttl is emitted co-registered with cost_to_go/visitation:
            # 12 px cells on the BORDER-STRIPPED playable frame, so the origin
            # offset is not optional -- dropping it shifts the lookup two cells
            # and silently exempts the wrong place.
            cell = float(VISITATION_CELL_PX)
            r = max(1, int(round(
                float(getattr(self.config, "stall_hazard_exempt_radius_px", 48.0))
                / cell)))
            cx = int((pos[0] - VISITATION_ORIGIN_PX) / cell)
            cy = int((pos[1] - VISITATION_ORIGIN_PX) / cell)
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            if x0 >= x1 or y0 >= y1:
                return False
            window = arr[(1, 3), y0:y1, x0:x1]
            return bool(window.min() < 1.0)
        except Exception:  # noqa: BLE001 - shaping never takes the run down
            return False

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
        # Completion-primary banking: under `gold_bank_until_completion` the
        # pickup pays NOTHING as it happens; the whole episode's gold is paid in
        # one lump on the winning step (and forfeited on death/timeout). Every
        # non-win site therefore adds `gold_payout`, not `gold_reward`.
        gold_banked = self.config.gold_bank_until_completion
        gold_payout = 0.0 if gold_banked else gold_reward

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
            # Geometric-fallback continuity: when PBRS is on but the env's
            # potential substitution did NOT commit (the obs flag is explicitly
            # False; absent = legacy/substituted behavior), the raw pre-switch
            # distance is switch-leg-only and Phi would flip basis at
            # activation. Compose a continuous cost-to-completion basis by
            # adding the constant switch->exit tail; if that tail is not
            # computable, degrade this episode to the HWM ratchet instead.
            self._pbrs_exit_leg = None
            self._pbrs_hwm_fallback = False
            self._phi_flip_warned = False
            if (
                self.config.enable_lattice_pbrs
                and obs.get("_potential_substituted") is False
            ):
                exit_leg = self._switch_to_exit_distance(obs)
                if exit_leg is not None:
                    self._pbrs_exit_leg = float(exit_leg)
                else:
                    self._pbrs_hwm_fallback = True
                    logger.debug(
                        "PBRS potential unsubstituted and switch->exit leg "
                        "uncomputable (level=%s); degrading to HWM ratchet "
                        "this episode.",
                        obs.get("map_name"),
                    )
            # PBRS normalization: D0 = spawn cost-to-completion. Under the lattice/
            # free-space substitution the pre-switch distance is ALREADY composed to
            # cost-to-completion, so the spawn value is the phase-0 distance itself
            # (not switch+exit, which would double-count the exit tail). Under the
            # composed geometric fallback the exit tail is added so D0 covers the
            # full spawn -> switch -> exit route.
            #
            # A FIXED reference (config._pbrs_d0_ref_override) overrides the
            # per-episode spawn D0 so the per-step shaping magnitude is route-length-
            # invariant across a geometry-morphing curriculum (otherwise it scales
            # 1/D0 and fades on the long, tight rungs). See RewardConfig.
            d0_ref = getattr(self.config, "_pbrs_d0_ref_override", None)
            spawn_completion = self._initial_dist_switch + (
                self._pbrs_exit_leg or 0.0
            )
            self._pbrs_d0 = (
                max(float(d0_ref), 1e-6) if d0_ref is not None
                else max(spawn_completion, 1e-6)
            )
            # SHAPING normalizer for the frontier/landing HWM bonuses: ALWAYS the
            # per-episode spawn cost-to-completion (route length), independent of
            # the PBRS _pbrs_d0 fixed-reference override. This keeps the whole-route
            # bonus bounded by ~weight regardless of level size. None when the spawn
            # cost was non-finite (cache not ready); re-established below off the
            # first finite cost-to-completion.
            self._shaping_d0 = (
                max(spawn_completion, 1e-6)
                if dist_switch < float("inf")
                else None
            )
            if self._pbrs_hwm_fallback:
                # PBRS disabled for this episode (see the shaping block below);
                # leaving the potential None also skips the terminal PBRS terms.
                self._prev_potential = None
            else:
                self._prev_potential = self._potential(
                    self._completion_cost(dist_switch, dist_exit, switch_activated)
                )
            # Goal-distance reverse curriculum basis: spawn cost-to-completion.
            # Only valid when the distance is finite (a reachable path exists).
            self._spawn_goal_dist = (
                float(spawn_completion) if dist_switch < float("inf") else None
            )
            self._reached_intermediate_goal = False
            self._episode_initialized = True

        # Cost-to-completion to the current sub-goal (phase-aware). Under the
        # lattice/free-space substitution this is one continuous potential basis
        # across switch activation; under the composed geometric fallback the
        # constant switch->exit tail is added pre-switch so it stays continuous
        # too. Used by the backtrack check, frontier/landing HWMs, the reverse
        # curriculum, and the PBRS term.
        current_goal_dist = self._completion_cost(
            dist_switch, dist_exit, switch_activated
        )

        # Establish the SHAPING normalizer (per-episode spawn cost-to-completion =
        # route length) from the first finite cost-to-completion when the spawn
        # value was non-finite at step 0. This is deliberately SEPARATE from the
        # PBRS _pbrs_d0 (which may be a fixed cross-level reference chosen for
        # dense-gradient scale-invariance): the frontier/landing HWM bonuses must
        # normalize by the ACTUAL route length so their whole-route total stays
        # bounded by ~weight on EVERY level. A fixed reference far below a long
        # maze's cost-to-go (e.g. ref=1000 vs a ~1e5 maze route) would otherwise
        # let them balloon to weight * route/ref. Rebase the HWMs so no delta pays.
        if self._shaping_d0 is None and current_goal_dist < float("inf"):
            self._shaping_d0 = max(current_goal_dist, 1e-6)
            self._frontier_best_dist = None
            self._landing_best_dist = None
            self._stall_best_cost = None
            self._stall_anchor_pos = None
            self._stall_no_improve_frames = 0
            self._stall_hazard_exempt_frames = 0

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
            terminal_reward += gold_payout
            terminal_reward += door_reward
            # Gold clawback on DEATH (config.gold_death_clawback in [0, 1]): a death
            # (especially a mine death) should erase the value of gold collected en
            # route so the policy stops treating hazardous gold detours as cheap.
            # ``episode_gold_reward_total`` is the episode-cumulative banked gold
            # reward — accumulated in this method BEFORE the terminal branches (so it
            # already includes any gold picked up on this dying step) and reset in
            # reset(). At clawback=1.0 the full episode gold is subtracted, so the
            # net gold contribution to the episode return is 0. Applied ONLY here on
            # death — WIN and truncation keep the full gold. No-op when the flag is 0.
            # Also a no-op under `gold_bank_until_completion`, which already paid
            # nothing: clawing back on top would net a spurious gold PENALTY.
            gold_clawback = 0.0
            if not gold_banked and self.config.gold_death_clawback > 0.0:
                gold_clawback = (
                    -self.config.gold_death_clawback * self.episode_gold_reward_total
                )
                terminal_reward += gold_clawback
            self.episode_gold_paid_total += gold_payout + gold_clawback
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
                "gold_reward": gold_payout,
                "gold_pickup_reward": gold_reward,
                "gold_clawback": gold_clawback,
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
            # Under banking the win is the ONLY payout point, so it pays the whole
            # episode's banked gold (episode_gold_reward_total already includes any
            # gold picked up on this winning step). Otherwise it pays only this
            # step's pickup, exactly as before.
            gold_win_payout = (
                self.episode_gold_reward_total if gold_banked else gold_reward
            )
            terminal_reward += gold_win_payout
            self.episode_gold_paid_total += gold_win_payout
            terminal_reward += door_reward
            # Completion SPEED bonus: pay extra for a faster, more direct route.
            # Terminal-only, so it never touches the per-step PBRS telescoping.
            # Kept < R_EXIT so a slow win still beats dying. Two forms, selected by
            # completion_speed_par_norm:
            #   par form (v15): efficiency ratio vs the per-level optimal route
            #     (par_frames = optimal_path_length / par_speed_px) -> learnable on
            #     EVERY level, not just long ones where the budget is tight.
            #   legacy form: normalized time_remaining vs the truncation budget
            #     (game_state[NINJA_STATE_DIM-1] in [0,1]); saturates on short levels.
            # The par form falls back to the legacy form when the optimal route
            # length is unavailable, so the bonus always has a signal to pay on.
            speed_bonus = 0.0
            # Cleared unconditionally each win, like speed_bonus: assigning it
            # only when a record is beaten makes it STICKY across episodes and
            # mis-reports the per-episode telemetry.
            self.episode_new_record_reward = 0.0
            if self.config.completion_speed_bonus > 0.0:
                frac = None  # in [0, 1]; scaled by completion_speed_bonus
                # Cap applied to `frac`. The par/budget forms clamp at 1.0; the
                # record form allows >1.0 so beating your own best still pays.
                frac_cap = 1.0
                if getattr(self.config, "completion_speed_record_norm", False):
                    frames_used = obs.get("sim_frame")
                    if frames_used is not None and float(frames_used) > 0.0:
                        frames_used = float(frames_used)
                        frac_cap = max(
                            1.0,
                            float(
                                getattr(
                                    self.config, "completion_speed_record_cap", 1.5
                                )
                            ),
                        )
                        if self._best_completion_frames is None:
                            # First win seeds the record and pays the 1.0x rate.
                            frac = 1.0
                        else:
                            frac = self._best_completion_frames / frames_used
                        # Only a genuine spawn-to-exit run may SET the record. A
                        # Go-Explore/demo restore starts mid-trajectory, so its
                        # completion time is a demo prefix + an agent suffix --
                        # crediting it would report a time the agent cannot
                        # reproduce from spawn, i.e. a false result against the
                        # run's own target. Such episodes still EARN the bonus
                        # (that is what teaches the fast continuation); they just
                        # cannot move the high-water mark.
                        if not bool(obs.get("from_checkpoint", False)):
                            _prev_best = self._best_completion_frames
                            self._best_completion_frames = min(
                                frames_used,
                                self._best_completion_frames
                                if self._best_completion_frames is not None
                                else frames_used,
                            )
                            # Discrete NEW-RECORD bonus. The ratio form above is
                            # nearly flat near the record, so shaving frames off
                            # an already-good route pays almost nothing; this
                            # makes each genuine improvement clearly visible to
                            # the optimizer. Ratcheting => unfarmable.
                            _rb = float(
                                getattr(
                                    self.config, "completion_speed_record_bonus", 0.0
                                )
                            )
                            if _rb > 0.0 and _prev_best is not None and frames_used < _prev_best:
                                speed_bonus_extra = _rb
                                terminal_reward += speed_bonus_extra
                                self.episode_new_record_reward = speed_bonus_extra
                if frac is None and self.config.completion_speed_par_norm and self.optimal_path_length:
                    par_frames = float(self.optimal_path_length) / max(
                        self.config.completion_speed_par_speed_px, 1e-6
                    )
                    frames_used = obs.get("sim_frame")
                    if (
                        par_frames > 0.0
                        and frames_used is not None
                        and float(frames_used) > 0.0
                    ):
                        # Efficiency ratio in (0, 1]: 1.0 at/under par, decaying
                        # as the win takes more frames. Clamps at 1.0 -> beating
                        # par cannot inflate the bonus past completion_speed_bonus.
                        frac = par_frames / max(float(frames_used), par_frames)
                if frac is None:
                    # Legacy / fallback: normalized time_remaining vs the budget.
                    gs = obs.get("game_state")
                    if gs is not None and len(gs) >= NINJA_STATE_DIM:
                        t_rem = float(gs[NINJA_STATE_DIM - 1])
                        if t_rem == t_rem:  # not NaN
                            frac = t_rem
                if frac is not None:
                    speed_bonus = self.config.completion_speed_bonus * min(
                        max(frac, 0.0), frac_cap
                    )
                    terminal_reward += speed_bonus
            self.episode_speed_bonus_reward = speed_bonus
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
                "gold_reward": gold_win_payout,
                "gold_pickup_reward": gold_reward,
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

        reward += gold_payout
        self.episode_gold_paid_total += gold_payout
        reward += door_reward

        hwm_progress = self.hwm_state.step(
            dist_switch=dist_switch,
            dist_exit=dist_exit,
            switch_activated=switch_activated,
            subgoal_phase=subgoal_phase,
        )
        beta_hwm = self.config.beta_blend()
        # Degraded-episode fallback: with PBRS disabled for this episode
        # (unsubstituted potential + uncomputable exit leg) the HWM ratchet is
        # explicitly re-enabled — otherwise the episode would have NO dense
        # progress signal at all (hwm_alpha is force-zeroed under PBRS).
        hwm_alpha = (
            self.config.hwm_fallback_alpha
            if self._pbrs_hwm_fallback
            else self.config.hwm_alpha
        )
        hwm_reward = hwm_alpha * hwm_progress * beta_hwm
        reward += hwm_reward

        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward += time_penalty

        # Lattice/free-space PBRS: F = w*(gamma*Phi(s') - Phi(s)). Symmetric dense
        # per-frame signal that REPLACES the HWM ratchet + backtrack penalty when
        # enabled (hwm_alpha is forced to 0 in the config and the backtrack penalty
        # below is skipped). Telescopes to w*(Phi_T - Phi_0) over the episode.
        # The RAW cost-to-completion is a plain readout of this step's state, not a
        # property of the lattice-PBRS branch. It used to be assigned only INSIDE
        # that branch, so on every configuration that takes the free-space/jump
        # potential path instead (which is all of them -- see the hoist note below)
        # it stayed at its `inf` initial value forever, and `_momentum_shaping`
        # read `inf` on every step. Assign it unconditionally.
        self._last_cost_to_go_raw = float(current_goal_dist)

        pbrs_reward = 0.0
        if self.config.enable_lattice_pbrs and not self._pbrs_hwm_fallback:
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
            # Continuity guard: switch activation must NOT flip the potential
            # basis (composed cost-to-completion is continuous there). A large
            # jump means an unsubstituted/inconsistent basis slipped through —
            # warn once per episode with the level name so it is debuggable.
            if (
                switch_just_activated
                and not self._phi_flip_warned
                and abs(phi_curr - prev_phi) > 0.25
            ):
                self._phi_flip_warned = True
                logger.warning(
                    "PBRS potential discontinuity at switch activation: "
                    "|Phi_post - Phi_pre| = %.3f (level=%s, dist_switch=%.1f, "
                    "dist_exit=%.1f, substituted=%s)",
                    abs(phi_curr - prev_phi),
                    obs.get("map_name"),
                    dist_switch,
                    dist_exit,
                    obs.get("_potential_substituted"),
                )
            pbrs_reward = self.config.pbrs_weight * (
                self.config.pbrs_gamma * phi_curr - prev_phi
            )
            self._prev_potential = phi_curr
            reward += pbrs_reward
            self.episode_pbrs_rewards.append(pbrs_reward)
        elif self._pbrs_hwm_fallback:
            # Degraded episode: no PBRS potential -> mask the aux-task target.
            self._last_cost_to_go_norm = -1.0

        # MOMENTUM + PACE shaping, applied UNCONDITIONALLY.
        #
        # These were originally nested inside the `enable_lattice_pbrs` branch
        # above, which is NEVER TAKEN in the speedrun configuration
        # (enable_lattice_pbrs is falsy there; the run uses the free-space /
        # jump potential path instead). Both terms were therefore completely
        # inert in every run that "enabled" them -- the trainer logged
        # "PACE PBRS: ON", the table loaded, and reward/pace_pbrs_total was
        # 0.0000 forever. Each guards itself on its own config flag, so hoisting
        # them here is safe for every potential path.
        if self.config.enable_momentum_pbrs:
            _mom = self._momentum_shaping(obs, frames_executed)
            reward += _mom
            self.episode_momentum_pbrs_reward += _mom
        if self.config.enable_pace_pbrs:
            _pace = self._pace_shaping(obs, self._last_cost_to_go_raw)
            reward += _pace
            self.episode_pace_pbrs_reward += _pace

        # Per-leg waypoint PBRS: F = w*(gamma*Phi(s') - Phi(s)) with Phi toward the
        # CURRENT local waypoint, REBASED on each waypoint advance so every leg is
        # short and the per-step gradient is route-length-invariant. Decoupled from the
        # planner: reads waypoint_leg_distance + waypoint_index that the env's
        # WaypointTracker writes into obs (all-zeros/absent -> safe no-op).
        # v19 R3: per-step waypoint reach bonus, surfaced to the env for telemetry
        # (waypoint/reach_bonus_mean). Reset each step; set in the reach branch.
        self.step_waypoint_reach_bonus = 0.0
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
                    self.step_waypoint_reach_bonus = float(reach)
                if wp_idx != self._prev_waypoint_index:
                    # New leg (or first step): hand-off pays 0; rebase the basis.
                    self._prev_waypoint_index = wp_idx
                    self._prev_waypoint_potential = phi_wp
                prev_phi_wp = (
                    self._prev_waypoint_potential
                    if self._prev_waypoint_potential is not None
                    else phi_wp
                )
                # Decoupled leg weight (Audit II P0-5c): reusing pbrs_weight (2.0)
                # banks ~+19.5 per 1000 px of route — ~2x R_EXIT, drowning every
                # precision-sized dense term. None = legacy (reuse pbrs_weight).
                wp_weight = getattr(self.config, "waypoint_pbrs_weight", None)
                if wp_weight is None:
                    wp_weight = self.config.pbrs_weight
                wp_pbrs = float(wp_weight) * (
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
                # Short-range rescale: recover raw px (obs was divided by
                # obs_norm_px) and re-normalize to range_px so the potential
                # saturates at 1.0 beyond range_px -> the gradient is LOCAL.
                range_px = float(getattr(self.config, "hazard_pbrs_range_px", 168.0))
                obs_norm = float(getattr(self.config, "hazard_pbrs_obs_norm_px", 168.0))
                raw_px = min(max(d_mine, 0.0), 1.0) * obs_norm
                phi_haz = min(raw_px / range_px, 1.0) if range_px > 0 else 1.0
                prev_phi_haz = (
                    self._prev_hazard_potential
                    if self._prev_hazard_potential is not None
                    else phi_haz
                )
                d_phi = phi_haz - prev_phi_haz
                # One-sided: keep only the approach penalty (dPhi < 0). Dropping the
                # retreat reward removes the incentive to back away from mines.
                if getattr(self.config, "hazard_pbrs_one_sided", False):
                    d_phi = min(0.0, d_phi)
                # Progress gate (Audit II P0-5b): a NECESSARY mine-adjacent
                # approach — one that strictly decreases the goal cost-to-go this
                # step — pays no penalty; only unproductive drift toward a mine is
                # taxed. ``_prev_goal_dist`` is updated at the end of this method,
                # so here it still holds the previous step's value.
                if (
                    d_phi < 0.0
                    and getattr(self.config, "hazard_pbrs_progress_gated", False)
                    and self._prev_goal_dist is not None
                    and self._prev_goal_dist < float("inf")
                    and current_goal_dist < float("inf")
                    and current_goal_dist < self._prev_goal_dist - 1e-6
                ):
                    d_phi = 0.0
                haz_pbrs = float(self.config.hazard_pbrs_weight) * d_phi
                self._prev_hazard_potential = phi_haz
                reward += haz_pbrs
                pbrs_reward += haz_pbrs
                self.episode_hazard_pbrs_reward += haz_pbrs
                self.episode_pbrs_rewards.append(haz_pbrs)

        # Impact-survivability PBRS: shape the SAFETY potential Phi = 1 - impact_risk
        # (impact_risk in [0,1] = speed into the look-ahead landing surface vs its
        # survivable threshold; computed in npp_environment). Rewards reducing impact
        # risk, penalizes building it; gamma=1 telescoping -> policy-invariant. No-op
        # at safe speeds / no surface (risk==0 -> Phi==1 constant).
        if self.config.enable_impact_pbrs:
            risk = obs.get("_impact_risk")
            if risk is not None and risk == risk:  # finite, not NaN
                phi_imp = 1.0 - min(max(float(risk), 0.0), 1.0)
                prev_phi_imp = (
                    self._prev_impact_potential
                    if self._prev_impact_potential is not None
                    else phi_imp
                )
                imp_pbrs = float(self.config.impact_pbrs_weight) * (
                    phi_imp - prev_phi_imp
                )
                self._prev_impact_potential = phi_imp
                reward += imp_pbrs
                pbrs_reward += imp_pbrs
                self.episode_impact_pbrs_reward += imp_pbrs
                self.episode_pbrs_rewards.append(imp_pbrs)

        # Doomed-arc risk PBRS (v13): shape the binary flight-recoverability
        # potential Phi = 0 if obs["_arc_doomed"] (airborne with ALL THREE
        # steering arcs ending in a deadly mine / lethal impact) else 1. The
        # penalty telescopes onto the STEP THE FLIGHT BECAME UNRECOVERABLE —
        # the bad takeoff/steer frame, 30-60 frames before the death the
        # terminal penalty would otherwise be the first signal of (the
        # credit-assignment fix for landing-miss mine deaths). Recovering
        # (an arc opening back up, e.g. via a hold change re-projecting the
        # bundle) pays the penalty back; death while doomed keeps it — the
        # standard PBRS terminal asymmetry. Policy-invariant (gamma=1
        # telescoping of a pure state potential).
        if getattr(self.config, "enable_arc_risk_pbrs", False):
            doomed = obs.get("_arc_doomed")
            if doomed is not None:
                phi_arc = 0.0 if doomed else 1.0
                prev_phi_arc = (
                    self._prev_arc_risk_potential
                    if self._prev_arc_risk_potential is not None
                    else phi_arc
                )
                arc_pbrs = float(
                    getattr(self.config, "arc_risk_pbrs_weight", 0.5)
                ) * (phi_arc - prev_phi_arc)
                self._prev_arc_risk_potential = phi_arc
                reward += arc_pbrs
                pbrs_reward += arc_pbrs
                self.episode_arc_risk_pbrs_reward += arc_pbrs
                self.episode_pbrs_rewards.append(arc_pbrs)

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
                    # Normalize by the per-episode ROUTE LENGTH (_shaping_d0), NOT
                    # the PBRS fixed reference, so the whole-route bonus is bounded
                    # by ~frontier_bonus_weight on every level size. Clamp the
                    # per-step improvement to [0, 1] as a hard safety net.
                    d0 = max(float(self._shaping_d0 or self._pbrs_d0 or 1.0), 1e-6)
                    frontier_bonus = self.config.frontier_bonus_weight * min(
                        1.0, (self._frontier_best_dist - current_goal_dist) / d0
                    )
                    self._frontier_best_dist = current_goal_dist
                    reward += frontier_bonus
                    self.episode_frontier_bonus_total += frontier_bonus

        # Grounded-landing high-water-mark bonus: the SAME monotone-progress idea as
        # the frontier bonus, but gated on the ninja being on SOLID GROUND -- it pays
        # only for a new closest-ever SAFE LANDING, not mid-air proximity. The
        # per-step free-space/jump PBRS already rewards airborne approach (weighting a
        # drift toward a gap the same as a completed jump); this term specifically
        # up-weights STICKING the landing on precision buckets (single-tile platforms,
        # mines under every gap, zero recovery). Because landing on a mine is death
        # (not a surface contact), it rewards only safe landings. Monotone HWM (never
        # re-paid) so it cannot be farmed by oscillating (jump forward, fall back,
        # jump forward); D0-normalized so it is in PBRS units and bounded by
        # landing_pbrs_weight over the whole episode; rebased on the switch->exit flip
        # so the basis change is not paid. Off unless config.enable_landing_pbrs.
        #
        # A "safe progress contact" is EITHER grounded (a landing) OR a WALL contact
        # while moving toward the goal along the geometric path -- the wall branch
        # extends the bonus to WALL-JUMPS (the ninja clings/slides a wall and pushes
        # off toward the goal, so a wall touch at a new closest-approach is real
        # progress worth reinforcing). "Velocity in the direction of the goal
        # geometric path" is evaluated ON the cost-to-go field itself (a strict
        # cost-to-go DECREASE this step) rather than a straight-line velocity·(goal-
        # pos) dot, which is WRONG on wall-jump geometry: the goal typically sits
        # BEYOND the wall, so straight-line points into it while the productive
        # wall-jump motion is up-and-away. The cost-to-go field is jump- and
        # wall-jump-aware under --use-jump-potential, so its descent is exactly "the
        # geometric path toward the goal". The shared HWM keeps the wall branch
        # non-farmable too (only a strictly-new closest wall/ground contact pays).
        #
        # TRANSITION-GATED: the bonus fires ONLY on the airborne->contact edge (an
        # actual landing or wall-catch this step), never while the ninja is already
        # grounded. This restricts it to precision jumps that STICK a landing closer
        # to the goal -- flat/continuous ground running is not a landing and earns
        # nothing here -- which is the only situation the term was introduced for.
        landing_bonus = 0.0
        if self.config.enable_landing_pbrs:
            airborne = bool(obs.get("player_airborn", True))
            grounded = not airborne
            walled = bool(obs.get("player_walled", False))
            moving_toward_goal = (
                self._prev_goal_dist is not None
                and current_goal_dist < float("inf")
                and current_goal_dist < self._prev_goal_dist - 1e-6
            )
            wall_progress = walled and moving_toward_goal
            # Only the airborne->contact transition qualifies (a genuine landing /
            # wall-catch), so continuous ground contact never pays.
            just_contacted = (self._prev_airborne is True) and (grounded or walled)
            if (
                just_contacted
                and (grounded or wall_progress)
                and current_goal_dist < float("inf")
            ):
                phase = bool(switch_activated)
                if self._landing_best_dist is None or phase != self._landing_phase:
                    # First safe contact or objective change: rebase (no bonus).
                    self._landing_best_dist = current_goal_dist
                    self._landing_phase = phase
                elif current_goal_dist < self._landing_best_dist - 1e-6:
                    # Landed strictly closer to the goal than any prior safe contact
                    # (the "resulting position is closer to the goal" gate). Normalize
                    # by the per-episode ROUTE LENGTH (_shaping_d0), NOT the PBRS fixed
                    # reference, so the whole-route landing total is bounded by ~weight
                    # on every level size; clamp per landing to [0, 1] as a safety net.
                    d0 = max(float(self._shaping_d0 or self._pbrs_d0 or 1.0), 1e-6)
                    landing_bonus = self.config.landing_pbrs_weight * min(
                        1.0, (self._landing_best_dist - current_goal_dist) / d0
                    )
                    self._landing_best_dist = current_goal_dist
                    reward += landing_bonus
                    self.episode_landing_bonus_total += landing_bonus
            self._prev_airborne = airborne

        # Forward/backtrack bookkeeping. The backtrack PENALTY applies only in the
        # HWM pipeline — PBRS already penalizes regress symmetrically — but the
        # step counters stay populated for observability in both modes.
        backtrack_penalty = 0.0
        # The lattice/free-space PBRS term counts as active only when it is not
        # degraded to the HWM ratchet this episode (the ratchet pipeline keeps
        # its backtrack penalty).
        lattice_pbrs_active = (
            self.config.enable_lattice_pbrs and not self._pbrs_hwm_fallback
        )
        regressed = (
            self._prev_goal_dist is not None
            and current_goal_dist < float("inf")
            and self._prev_goal_dist < float("inf")
            and current_goal_dist > self._prev_goal_dist
        )
        if regressed:
            if not (lattice_pbrs_active or self.config.enable_waypoint_pbrs):
                backtrack_penalty = BACKTRACK_PENALTY * frames_executed
                reward += backtrack_penalty
                self.episode_backtrack_penalty_total += backtrack_penalty
            self.episode_backtrack_steps += 1
        elif (
            hwm_progress > 0
            or lattice_pbrs_active
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

        # Progress-stall (dither) penalty [v22, §3-A]. See RewardConfig.enable_stall_penalty.
        if self.config.enable_stall_penalty and current_goal_dist < float("inf"):
            phase = bool(switch_activated)
            if self._stall_best_cost is None or phase != self._stall_phase:
                # First finite cost, or the objective flipped (switch->exit): rebase,
                # never counting the basis change as a stall.
                self._stall_best_cost = current_goal_dist
                self._stall_phase = phase
                self._stall_no_improve_frames = 0
                self._stall_anchor_pos = current_pos
            elif current_goal_dist < self._stall_best_cost - 1e-6:
                # Strict cost-to-go improvement: reset the stall clock and anchor.
                self._stall_best_cost = current_goal_dist
                self._stall_no_improve_frames = 0
                self._stall_anchor_pos = current_pos
            else:
                # No improvement this step. Only accrue the stall clock while the
                # ninja stays within stall_radius_px of the last-improvement anchor
                # (low position variance): a real jump / traverse moves far and
                # resets the anchor, so it never triggers -- only tight oscillation
                # in a pocket does. Drift beyond the radius without improving (e.g.
                # slowly wandering away) also resets the anchor so we measure a
                # fresh dwell rather than penalizing transit.
                anchor = self._stall_anchor_pos or current_pos
                adx = current_pos[0] - anchor[0]
                ady = current_pos[1] - anchor[1]
                within = (adx * adx + ady * ady) ** 0.5 <= self.config.stall_radius_px
                # Hazard exemption. Holding position until a patrol clears is
                # the CORRECT play, and the test above cannot tell it from
                # dithering: cost-to-go does not improve and the ninja does not
                # move. Three-way rather than two, because the distinction
                # matters -- FREEZE while pinned (do not reset), so a genuine
                # ditherer parked beside a hazard resumes accruing the instant
                # the threat clears rather than being handed a clean slate.
                if within and self._hazard_blocked(obs, current_pos):
                    self._stall_hazard_exempt_frames += frames_executed
                elif within:
                    self._stall_no_improve_frames += frames_executed
                    over = self._stall_no_improve_frames - self.config.stall_patience_frames
                    if over > 0:
                        # Escalating with dwell time past patience, normalized by the
                        # patience window and per-step frames, capped.
                        mag = self.config.stall_penalty_weight * (
                            over / max(1, self.config.stall_patience_frames)
                        ) * frames_executed
                        stall_penalty = -min(mag, self.config.stall_penalty_cap)
                        reward += stall_penalty
                        self.episode_stall_penalty_total += stall_penalty
                else:
                    self._stall_no_improve_frames = 0
                    self._stall_anchor_pos = current_pos

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
            "gold_reward": gold_payout,
            "gold_pickup_reward": gold_reward,
            "locked_door_reward": door_reward,
            "subgoal_phase": subgoal_phase,
            "beta_blend_hwm": beta_hwm,
            "time_penalty": time_penalty,
            "backtrack_penalty": backtrack_penalty,
            "frontier_bonus": frontier_bonus,
            "landing_bonus": landing_bonus,
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
        # Counts evaluations, not env steps: the potential is read for both s and
        # s' on a transition. The RATIO is what is reported, so the factor is
        # immaterial, and it makes "severed on the very first evaluation" -- i.e.
        # at the spawn -- exactly detectable.
        self._pbrs_steps += 1
        d0 = self._pbrs_d0
        if d0 is None or not (completion_cost < float("inf")):
            # SEVERANCE. The clamp is bounded on purpose, but -1.0 is also
            # Phi(spawn), so a route the graph cannot cost produces a potential
            # that is FLAT at the worst value for as long as it stays severed --
            # zero dense gradient, and indistinguishable from standing on the
            # spawn. That is the failure mode an unmodelled traversal entity
            # (bounce block, thwump, launch pad) produces today: not a dropped
            # level, which would at least be visible in a level count, but a
            # silently dead shaping signal on exactly the levels that need it.
            #
            # Counted so it is observable. `severed_from_spawn` separates the two
            # causes that matter: severed on the FIRST step means the route was
            # never costable -- an unmodelled entity or a genuinely unsolvable
            # level -- whereas severance appearing later means the agent walked
            # off the costed region, which is normal and self-correcting.
            self._pbrs_severed_steps += 1
            if self._pbrs_steps == 1:
                self._pbrs_severed_from_spawn = True
            return -1.0
        return -min(max(completion_cost / d0, 0.0), 1.0)

    def _completion_cost(
        self, dist_switch: float, dist_exit: float, switch_activated: bool
    ) -> float:
        """Phase-aware cost-to-completion for the potential basis.

        Post-switch this is the exit distance. Pre-switch it is the switch
        distance, PLUS the constant switch->exit tail when the composed
        geometric fallback is active (``_pbrs_exit_leg`` set at episode init) —
        under the env's substitution the switch distance is already composed
        and the tail is None, so this is a pass-through there.
        """
        if switch_activated:
            return dist_exit
        if self._pbrs_exit_leg is not None and dist_switch < float("inf"):
            return dist_switch + self._pbrs_exit_leg
        return dist_switch

    def _switch_to_exit_distance(self, obs: Dict[str, Any]) -> Optional[float]:
        """A* path distance from the exit switch to the exit door.

        The constant exit-leg tail used to compose the raw per-phase fallback
        distances into one continuous cost-to-completion basis (computed once
        per episode at PBRS init). Returns None when not computable (missing
        adjacency, unreachable, or any error) — the caller then degrades the
        episode to the HWM ratchet.
        """
        try:
            adjacency = obs.get("_adjacency_graph")
            if not adjacency:
                return None
            level_data = obs.get("level_data")
            graph_data = obs.get("_graph_data")
            switch_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
            exit_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))
            base_adjacency = (
                graph_data.get("base_adjacency", adjacency)
                if graph_data
                else adjacency
            )
            d = self.path_calculator.get_path_cost(
                switch_pos,
                exit_pos,
                adjacency,
                base_adjacency,
                cache_key="switch_exit_leg",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
            if d is None or d != d or d == float("inf"):
                return None
            return float(d)
        except Exception:
            return None

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
        if newly <= 0:
            return 0.0, n_open
        # Spread a bounded TOTAL over the level's necessary doors rather than
        # paying a flat per-door bonus. Doors are irreversible, so on a level
        # with many of them a flat bonus makes "open everything, then run out
        # the clock" pay more than winning: measured over the official corpus,
        # 9 levels exceed R_EXIT and the worst pays 35.5 against 10.0. See
        # R_LOCKED_DOOR_TOTAL_CAP. k is static per map, so the per-door value is
        # constant within an episode -- this rescales the signal, it does not
        # make it non-stationary. Levels needing <= 4 doors (225 of the 294 that
        # need any) are unaffected.
        per_door = self.config.locked_door_switch_reward
        cap = self.config.locked_door_total_cap
        n_total = int(obs.get("necessary_doors_total", 0) or 0)
        if cap > 0 and n_total > 0:
            per_door = min(per_door, cap / n_total)
        return newly * per_door, n_open

    def _load_pace_table(self) -> None:
        """Load the cost-to-go -> reference-frame table once per calculator."""
        self._pace_loaded = True
        path = getattr(self.config, "pace_table_path", None)
        if not path:
            return
        try:
            import json as _json

            data = _json.loads(open(path).read())
            cells = data.get("cells") or {}
            parsed = {}
            for k, v in cells.items():
                gx, gy = k.split(",")
                parsed[(int(gx), int(gy))] = float(v)
            if len(parsed) >= 2:
                self._pace_cells = parsed
                self._pace_grid = float(data.get("grid", 12) or 12)
                logger.info(
                    "Pace table loaded: %d cells, grid %s, reference total %s frames",
                    len(parsed), self._pace_grid, data.get("total_frames"),
                )
            else:
                logger.error("Pace table at %s is malformed; pace PBRS inert.", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load pace table %s: %s", path, exc)

    def _reference_frame_at_pos(self, x: float, y: float) -> Optional[float]:
        """Frame at which the reference run first reached this position cell.

        Returns None when the agent is on a cell the reference never visited --
        i.e. genuinely off-route. The caller then emits NO shaping for that step
        rather than inventing a pace, so wandering off the reference line is
        neither rewarded nor punished by this term (the time penalty still
        applies). A 1-cell neighbourhood search absorbs small lateral offsets,
        taking the EARLIEST neighbouring reference: that reads as slightly
        "behind pace", which biases toward pushing rather than coasting, and
        never over-credits a state the reference reached later.
        """
        cells = self._pace_cells
        if not cells:
            return None
        g = max(self._pace_grid, 1e-6)
        gx, gy = int(x // g), int(y // g)
        best = cells.get((gx, gy))
        if best is not None:
            return best
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                v = cells.get((gx + dx, gy + dy))
                if v is not None and (best is None or v < best):
                    best = v
        return best

    def _momentum_shaping(self, obs, frames_executed: int = 1) -> float:
        """Closing-speed PBRS: F = w * (gamma*Phi(s') - Phi(s)), Phi = closing speed.

        Uses the closing speed on the goal, NOT raw |v|: it is the component of
        velocity that actually shortens the route, which is what a momentum
        conversion (walljump) maximizes. Raw |v| would PENALISE a walljump, which
        momentarily reduces |vx| (3.19 -> 1.08).
        """
        cur = self._last_cost_to_go_raw
        _frames = max(int(frames_executed or 1), 1)
        if self._prev_goal_dist is not None and cur < float("inf"):
            closing = (self._prev_goal_dist - cur) / _frames
            phi_mom = min(max(closing / MAX_HOR_SPEED, -1.0), 1.0)
        else:
            phi_mom = 0.0
        prev = (
            self._prev_momentum_potential
            if self._prev_momentum_potential is not None
            else phi_mom
        )
        r = float(self.config.momentum_pbrs_weight) * (
            self.config.pbrs_gamma * phi_mom - prev
        )
        self._prev_momentum_potential = phi_mom
        # `_prev_goal_dist` is owned by `calculate_reward`, which advances it ONCE
        # at the end of the step. Advancing it here too made this method's own
        # `closing` read zero on the next call and, worse, blinded every LATER
        # consumer in the same step (the hazard-PBRS progress gate, the landing
        # bonus's `moving_toward_goal`, and the backtrack counter) to the actual
        # previous step. Read-only here.
        return r

    def _pace_shaping(self, obs, current_goal_dist) -> float:
        """F = w * (gamma * Phi(s') - Phi(s)) with Phi = -(elapsed - reference)."""
        if not self._pace_loaded:
            self._load_pace_table()
        if not self._pace_cells:
            return 0.0
        elapsed = obs.get("sim_frame")
        px, py = obs.get("player_x"), obs.get("player_y")
        if elapsed is None or px is None or py is None:
            return 0.0
        ref = self._reference_frame_at_pos(float(px), float(py))
        if ref is None:
            return 0.0  # off-route: no pace opinion, rather than a fabricated one
        scale = max(float(getattr(self.config, "pace_scale", 60.0)), 1e-6)
        # NEGATIVE when behind the reference, so being ahead is higher potential.
        phi = -(float(elapsed) - ref) / scale
        prev = self._prev_pace_potential
        if prev is None:
            prev = phi
        r = float(self.config.pace_pbrs_weight) * (
            self.config.pbrs_gamma * phi - prev
        )
        self._prev_pace_potential = phi
        return r

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
        self._prev_goal_dist = None
        self._prev_momentum_potential = None
        self.episode_momentum_pbrs_reward = 0.0
        self.episode_new_record_reward = 0.0
        self._prev_pace_potential = None
        self.episode_pace_pbrs_reward = 0.0
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
        # Gold reward actually PAID into the returned rewards this episode
        # (differs from episode_gold_reward_total under banking or a death
        # clawback). This is the term that makes the reconciliation exact.
        self.episode_gold_paid_total = 0.0
        self.episode_locked_door_reward_total = 0.0
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self.episode_frontier_bonus_total = 0.0
        self.episode_landing_bonus_total = 0.0

        self.episode_path_length = 0.0
        self.optimal_path_length = None
        self.prev_player_pos = None
        self.spawn_position = None
        self._prev_goal_dist = None
        self._prev_potential = None
        self._pbrs_d0 = None
        self._pbrs_steps = 0
        self._pbrs_severed_steps = 0
        self._pbrs_severed_from_spawn = False
        self._shaping_d0 = None
        self._pbrs_exit_leg = None
        self._pbrs_hwm_fallback = False
        self._phi_flip_warned = False
        self._last_cost_to_go_norm = -1.0
        self._prev_waypoint_potential = None
        self._prev_waypoint_index = None
        self.episode_waypoint_reach_reward = 0.0
        self._frontier_best_dist = None
        self._frontier_phase = None
        self._landing_best_dist = None
        self._landing_phase = None
        self._prev_airborne = None
        self._stall_best_cost = None
        self._stall_phase = None
        self._stall_no_improve_frames = 0
        self._stall_anchor_pos = None
        self._stall_hazard_exempt_frames = 0
        self.episode_stall_penalty_total = 0.0
        self._prev_hazard_potential = None
        self.episode_hazard_pbrs_reward = 0.0
        self._prev_impact_potential = None
        self.episode_impact_pbrs_reward = 0.0
        self._prev_arc_risk_potential = None
        self.episode_arc_risk_pbrs_reward = 0.0
        self.episode_speed_bonus_reward = 0.0
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

        # PBRS severance (see `_potential`). `pbrs_severed_fraction` is the share
        # of the episode with NO dense gradient at all; `pbrs_severed_from_spawn`
        # flags the case where the route was never costable in the first place,
        # which on a level carrying a traversal entity the graph does not model
        # is the signature of that gap rather than of agent behaviour.
        severed_fraction = (
            self._pbrs_severed_steps / self._pbrs_steps
            if self._pbrs_steps
            else 0.0
        )

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
        # NOTE: this uses gold PAID, not gold earned. They differ whenever gold is
        # banked-until-completion or clawed back on death; using the earned total
        # would leave a residual equal to the unpaid gold.
        sparse_total = (
            self.episode_sparse_switch_reward
            + self.episode_sparse_exit_reward
            + self.episode_death_penalty_reward
            + self.episode_gold_paid_total
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
            + self.episode_landing_bonus_total
            + self.episode_stall_penalty_total
            + self.episode_speed_bonus_reward
            + self.episode_waypoint_reach_reward
            # These three were returned in `reward`/`terminal_reward` but omitted
            # here, so `reconciliation_residual` was non-zero -- by exactly the
            # omitted amount -- whenever any of them was enabled. That silently
            # invalidated the residual as a completeness proof, which is the one
            # thing it exists to be.
            + self.episode_momentum_pbrs_reward
            + self.episode_pace_pbrs_reward
            + self.episode_new_record_reward
        ) * GLOBAL_REWARD_SCALE
        # residual ~ 0 proves the decomposition is complete (no unaccounted reward).
        reconciliation_residual = components_sum - self.episode_total_scaled

        return {
            "hwm_total": hwm_total,
            "hwm_mean": hwm_mean,
            "hwm_std": hwm_std,
            "pbrs_total": pbrs_total,
            "pbrs_mean": pbrs_mean,
            "pbrs_severed_fraction": severed_fraction,
            "pbrs_severed_from_spawn": float(self._pbrs_severed_from_spawn),
            "time_penalty_total": time_penalty_total,
            "sparse_total": sparse_total,
            "truncation_penalty_total": self.episode_truncation_penalty,
            "backtrack_penalty_total": self.episode_backtrack_penalty_total,
            "frontier_bonus_total": self.episode_frontier_bonus_total,
            "landing_bonus_total": self.episode_landing_bonus_total,
            "stall_penalty_total": self.episode_stall_penalty_total,
            "stall_hazard_exempt_frames": self._stall_hazard_exempt_frames,
            "speed_bonus_total": self.episode_speed_bonus_reward,
            "momentum_pbrs_total": self.episode_momentum_pbrs_reward,
            "new_record_bonus": self.episode_new_record_reward,
            "pace_pbrs_total": self.episode_pace_pbrs_reward,
            # Fastest WIN recorded on this level so far, in sim frames, measured
            # from spawn. This is the single number a single-level speedrun is
            # trying to drive down. Distinct from the episode-length metrics,
            # which mix spawn-started and Go-Explore checkpoint-restored episodes
            # (the latter begin mid-level and so look artificially short).
            # 0.0 until the first win.
            "best_completion_frames": float(self._best_completion_frames or 0.0),
            "waypoint_reach_bonus_total": self.episode_waypoint_reach_reward,
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
            "gold_reward_paid_total": self.episode_gold_paid_total,
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
