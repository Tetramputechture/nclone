"""Policy-invariant reward calculator using unified γ=0.99 PBRS.

Implements Potential-Based Reward Shaping (Ng et al. 1999) with unified design:
1. Terminal rewards (completion, death) - Define task success/failure
2. PBRS shaping: F(s,s') = 0.99 * Φ(s') - Φ(s) - Dense path guidance + implicit time pressure

CRITICAL: γ_PBRS = γ_PPO = 0.99 for policy invariance guarantee (Ng et al. 1999).

Unified γ=0.99 PBRS provides:
- Dense reward signal at every step (not sparse achievement bonuses)
- Progress signal: Forward = positive, Backward = negative (larger penalty)
- Anti-oscillation: Oscillating A↔B costs -0.01·(Φ(A)+Φ(B)) - GUARANTEED LOSS
- Implicit time pressure: Staying still at Φ=0.5 costs -0.005/step
- Policy invariance: γ_PBRS matches γ_PPO for theoretical guarantees
- Markov property (no episode history dependencies)

Gamma Choice (γ=0.99):
- Forward progress: 0.99·Φ(s') - Φ(s) reward (progress toward goal)
- Backtracking: Negative (loses progress + 1% discount penalty)
- Oscillation A↔B: Net = -0.01·Φ(A) - 0.01·Φ(B) (always negative, prevents exploitation)
- Staying still: -0.01·Φ(current) (small negative, natural time pressure)
- Unified signal: No separate time penalty needed

The 1% discount creates self-tuning urgency that scales with potential magnitude.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from .reward_config import RewardConfig
from .pbrs_potentials import PBRSCalculator
from .reward_constants import (
    PBRS_GAMMA,
    GLOBAL_REWARD_SCALE,
)

# Get distance to goal (use cached value if available from PBRS calculation)
from ...constants.physics_constants import (
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    NINJA_RADIUS,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


logger = logging.getLogger(__name__)


# SIMPLIFIED: PositionTracker removed - revisit penalties and exploration bonuses
# are redundant with PBRS. PBRS already:
# - Penalizes revisiting higher-distance states (backtracking)
# - Gives 0 reward for oscillation (staying at same distance)
# - Provides complete gradient field (no exploration needed)


class RewardCalculator:
    """Policy-invariant reward calculator using unified γ=0.99 PBRS.

    Components:
    - Terminal rewards: Always active (completion, death, switch milestone)
    - PBRS shaping: F(s,s') = 0.99 * Φ(s') - Φ(s) for dense, policy-invariant path guidance

    CRITICAL: γ_PBRS = γ_PPO = 0.99 for policy invariance guarantee.

    PBRS Implementation:
    - Uses potential function Φ(s) based on shortest path distance to objective
    - Applies Ng et al. (1999) formula with γ=0.99 to match PPO discount factor
    - Guarantees policy invariance while providing dense reward gradients
    - Automatically rewards distance decreases and penalizes distance increases
    - Creates natural time pressure through implicit discount term
    - Curriculum-aware weight and normalization scaling

    Unified γ=0.99 Design Benefits:
    - Single coherent signal (no separate time penalty needed)
    - Natural anti-oscillation (guaranteed loss for A↔B movement)
    - Self-tuning urgency (scales with potential magnitude)
    - Maintains policy invariance (γ_PBRS = γ_PPO)

    Removed components (policy-invariant alternative):
    - Discrete achievement bonuses (replaced with dense PBRS)
    - Episode length normalization (violated Markov property)
    - Per-episode caps (violated Markov property)
    - Physics discovery rewards (disabled for clean path focus)
    - Time penalty (replaced with unified γ<1 PBRS)
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        shared_level_cache: Optional[Any] = None,
        shared_level_caches_by_stage: Optional[Dict[int, Any]] = None,
        goal_curriculum_manager: Optional[Any] = None,
    ):
        """Initialize simplified reward calculator.

        Args:
            reward_config: RewardConfig instance managing curriculum-aware component lifecycle
            pbrs_gamma: Discount factor for PBRS (γ in F(s,s') = γ * Φ(s') - Φ(s))
            shared_level_cache: Optional SharedLevelCache for zero-copy multi-worker training
            shared_level_caches_by_stage: Optional multi-stage caches for goal curriculum
            goal_curriculum_manager: Optional IntermediateGoalManager for stage-aware cache selection
        """
        # Single config object manages all curriculum logic
        self.config = reward_config or RewardConfig()

        # OPTIMIZATION: Cache curriculum properties per episode to avoid @property overhead
        # Properties are episode-invariant but called many times per step
        self._cached_pbrs_objective_weight: Optional[float] = None
        self._episode_config_cache_valid: bool = False

        # PBRS configuration (kept for backwards compatibility with path calculator)
        self.pbrs_gamma = pbrs_gamma

        # Store goal curriculum manager for stage-aware cache selection
        self.goal_curriculum_manager = goal_curriculum_manager

        # Create path calculator for PBRS (with optional shared cache(s))
        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            use_astar=True,
            shared_level_cache=shared_level_cache,
            shared_level_caches_by_stage=shared_level_caches_by_stage,
            goal_curriculum_manager=goal_curriculum_manager,
        )
        # Pass reward_config to PBRS calculator for curriculum-adaptive weights (Phase 1.2)
        self.pbrs_calculator = PBRSCalculator(
            path_calculator=path_calculator, reward_config=self.config
        )

        self.steps_taken = 0

        # Track closest distances for diagnostic metrics only
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

        # Track previous potential for PBRS calculation: F(s,s') = γ * Φ(s') - Φ(s)
        self.prev_potential = None

        # Episode-level tracking for TensorBoard logging
        self.episode_pbrs_rewards = []  # List of PBRS rewards per step
        self.episode_time_penalties = []  # List of time penalties per step
        self.episode_forward_steps = 0  # Count of steps with potential increase
        self.episode_backtrack_steps = 0  # Count of steps with potential decrease
        self.episode_terminal_reward = 0.0  # Terminal reward (completion/death)
        self.current_potential = None  # Current potential for logging

        # Path optimality tracking for efficiency metrics
        self.episode_path_length = 0.0  # Actual path taken (accumulated movement)
        self.optimal_path_length = None  # Shortest possible path (spawn→switch→exit)
        self.prev_player_pos = None  # Track previous position for movement calculation

        # Distance tracking for adaptive PBRS gating (Phase 2 enhancement)
        self._prev_distance_to_goal = None

        # Track last next_hop for visualization at failure point (Phase 1 enhancement)
        self._last_next_hop_world: Optional[Tuple[float, float]] = None
        self._last_next_hop_goal_id: Optional[str] = (
            None  # Track distance for improvement calculation
        )

        # Store computed paths for visualization (to ensure debug overlay shows same paths as reward uses)
        self._cached_switch_path: Optional[List[Tuple[int, int]]] = None
        self._cached_exit_path: Optional[List[Tuple[int, int]]] = None
        self._cached_spawn_node: Optional[Tuple[int, int]] = None
        self._cached_switch_node: Optional[Tuple[int, int]] = None
        self._cached_exit_node: Optional[Tuple[int, int]] = None

        # Step-level component tracking for TensorBoard callback
        # Always populated after calculate_reward() for info dict logging
        self.last_pbrs_components = {}

        # SIMPLIFIED: Removed PositionTracker - revisit penalties redundant with PBRS
        # SIMPLIFIED: Removed oscillation detection - PBRS gives 0 for net-zero displacement

        # Progress tracking (diagnostic only, no penalty)
        self.closest_distance_this_episode = float("inf")

        # Spawn position tracking for displacement gate
        # Positive PBRS only applies if agent has moved from spawn
        # Prevents reward hacking via oscillation/jumping in place
        self.spawn_position = None  # Set on first calculate_reward() call

        # Distance-based milestone tracking (replaces pre-computed subgoals)
        # Rewards when shortest path distance decreases by >= milestone_interval
        self._last_milestone_distance_to_switch: Optional[float] = None
        self._last_milestone_distance_to_exit: Optional[float] = None
        self._milestone_interval_px: float = 48.0  # From SUB_GOAL_SPACING constant

        # Track diagnostic data for PBRS visualization (Phase 1 enhancement)
        self.velocity_history: List[Tuple[float, float]] = []
        self.alignment_history: List[float] = []
        self.path_gradient_history: List[Optional[Tuple[float, float]]] = []
        self.potential_history: List[float] = []

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
        frames_executed: int = 1,
        curriculum_manager: Optional[Any] = None,
    ) -> float:
        """Calculate reward using unified γ=0.99 PBRS.

        Components:
        1. Terminal rewards (always active)
        2. PBRS shaping: F(s,s') = 0.99 * Φ(s') - Φ(s) for dense path guidance + implicit time pressure

        CRITICAL: γ_PBRS = γ_PPO = 0.99 for policy invariance guarantee.

        Unified γ=0.99 PBRS ensures:
        - Dense reward signal at every step (not just improvements)
        - Progress signal: positive for forward, negative for backward (with discount penalty)
        - Anti-oscillation: A↔B movement costs -0.01·(Φ(A)+Φ(B)) - cannot break even
        - Implicit time pressure: Staying still costs -0.01·Φ(current) per step
        - Policy invariance: Matches PPO gamma for theoretical guarantees

        Progress Signal Mechanism (weight=40, γ=0.99):
        - Forward 12px (ΔΦ=+0.01): PBRS ≈ +0.39 (positive reward for progress)
        - Stay still (ΔΦ=0): PBRS = -0.01×Φ×weight (implicit time penalty)
        - Backward 12px (ΔΦ=-0.01): PBRS ≈ -0.41 (loses progress + discount)
        - Episode PBRS sum = Φ(final) - Φ(start) (clean telescoping, no bias)

        Goal Transition Handling:
        - When switch is activated, goal changes from switch → exit
        - Potential function Φ changes (switch distance → exit distance)
        - To maintain policy invariance, we reset prev_potential = None
        - This treats the transition like hierarchical RL "option termination"
        - Next step starts fresh with exit-based potential (no PBRS comparison)

        Frame Skip Integration:
        - With frame skip, we calculate reward once per action (not per frame)
        - PBRS telescopes identically: Φ(final) - Φ(initial) regardless of intermediate frames
        - Time penalty scales by frames_executed for correct per-action magnitude
        - This provides 75% computational savings with 4-frame skip

        Args:
            obs: Current game state (must include _adjacency_graph, level_data, physics_cache)
            prev_obs: Previous game state
            action: Action taken (optional, unused)
            frames_executed: Number of frames executed in this action (for time penalty scaling)
            curriculum_manager: Optional IntermediateGoalManager for curriculum-aware PBRS

        Returns:
            float: Total reward for the transition
        """
        self.steps_taken += frames_executed

        # OPTIMIZATION: Clear step-level cache at start of each step
        # This cache avoids duplicate distance calculations within the same step
        # (e.g., switch distance computed for PBRS is reused for diagnostics)
        if hasattr(self.pbrs_calculator, "path_calculator"):
            self.pbrs_calculator.path_calculator.clear_step_cache()

        # Track spawn position for displacement gate (first step only)
        # CRITICAL: Use prev_obs (position before action) not obs (position after action)
        # This ensures checkpoint episodes use the checkpoint position as spawn baseline
        if self.spawn_position is None:
            self.spawn_position = (prev_obs["player_x"], prev_obs["player_y"])

        # Run validation check if needed (once per 10K steps)
        if hasattr(self, "_needs_validation") and self._needs_validation:
            self.validate_level_solvability(obs)
            self._needs_validation = False

        # === MILESTONE REWARD (checked BEFORE terminal to credit even on death) ===
        # CRITICAL: Switch activation must be credited even if agent dies on same step.
        # This ensures routes that achieve the switch are valued over those that don't.
        switch_just_activated = obs.get("switch_activated", False) and not prev_obs.get(
            "switch_activated", False
        )
        # UPDATED 2026-01-21: Use curriculum-scaled switch reward
        milestone_reward = (
            self.config.switch_activation_reward if switch_just_activated else 0.0
        )

        # SIMPLIFIED: Removed hierarchical exploration reset - no longer using position tracker

        # === TERMINAL REWARDS (always active) ===
        # Get level_data for validation and pathfinding
        level_data = obs.get("level_data")

        # Death - simple symbolic penalty (opportunity cost is main penalty)
        # UPDATED 2025-12-17: Curriculum-adaptive death penalty to fix local minima.
        # Penalty scales with success rate: -15 (discovery) -> 0 (advanced).
        # This breaks "progress+death" local optimum in early training while
        # preserving risk-taking incentives once agent demonstrates competence.
        if obs.get("player_dead", False):
            death_cause = obs.get("death_cause", None)  # Log for diagnostics only

            # Curriculum-adaptive death penalty based on success rate
            terminal_reward = self.config.death_penalty  # -8 to 0 based on competence

            # Credit switch activation if it happened (before death)
            terminal_reward += milestone_reward

            # Simple death penalty based on success rate - no additional penalties needed

            self.episode_terminal_reward = terminal_reward
            scaled_reward = terminal_reward * GLOBAL_REWARD_SCALE

            # Store terminal potential for route visualization (use 0.0 for consistency)
            # Terminal states don't have meaningful potentials but we need length match
            self.potential_history.append(0.0)

            # Populate PBRS components for TensorBoard logging (terminal state)
            self.last_pbrs_components = {
                "terminal_reward": terminal_reward,
                "milestone_reward": milestone_reward,
                "pbrs_reward": 0.0,
                "time_penalty": 0.0,
                "total_reward": terminal_reward,
                "scaled_reward": scaled_reward,
                "is_terminal": True,
                "terminal_type": "death",
                "death_cause": death_cause or "unknown",
                "switch_activated_on_death": switch_just_activated,
            }
            return scaled_reward

        # Completion reward with path efficiency bonus
        # UPDATED 2026-01-21: Use curriculum-scaled completion reward
        if obs.get("player_won", False):
            terminal_reward = self.config.level_completion_reward

            # Path efficiency bonus: reward direct paths to encourage optimal navigation
            # efficiency = optimal_path / actual_path (1.0 = perfect, 0.5 = 2x optimal)
            # Bonus = completion_reward × 0.5 × min(1.0, efficiency)
            efficiency_bonus = 0.0
            path_efficiency = 0.0
            if (
                self.optimal_path_length is not None
                and self.optimal_path_length > 0
                and self.episode_path_length > 0
            ):
                path_efficiency = self.optimal_path_length / self.episode_path_length
                # Cap efficiency at 1.0 (can't be better than optimal)
                efficiency_bonus = (
                    self.config.level_completion_reward
                    * 0.5
                    * min(1.0, path_efficiency)
                )
                terminal_reward += efficiency_bonus

            # Include milestone if switch was activated on same step as win (edge case)
            total_terminal = terminal_reward + milestone_reward
            self.episode_terminal_reward = total_terminal

            # Apply global reward scaling for value function stability
            scaled_reward = total_terminal * GLOBAL_REWARD_SCALE

            # Store terminal potential for route visualization (use 1.0 for win - at goal)
            # Terminal states don't affect PBRS but we need length match with positions
            self.potential_history.append(1.0)

            # Populate PBRS components for TensorBoard logging (terminal state)
            # Log UNSCALED values for interpretability
            self.last_pbrs_components = {
                "terminal_reward": terminal_reward,
                "milestone_reward": milestone_reward,
                "pbrs_reward": 0.0,
                "time_penalty": 0.0,
                "total_reward": total_terminal,
                "scaled_reward": scaled_reward,
                "is_terminal": True,
                "terminal_type": "win",
                "path_efficiency": path_efficiency,
                "efficiency_bonus": efficiency_bonus,
            }
            return scaled_reward

        # === TIME PENALTY (curriculum-controlled, scaled by frame skip) ===
        # Config determines if active and magnitude based on training phase
        # Scale by frames_executed to account for frame skip (e.g., 4 frames per action)
        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward = time_penalty

        # Track time penalty for logging (store scaled penalty per action)
        self.episode_time_penalties.append(time_penalty)

        # === MILESTONE REWARD (already computed above, add to running reward) ===
        # Switch activation was checked before terminal rewards to credit even on death.
        # Now add it to the running reward for non-terminal steps.
        # UPDATED 2026-01-21: Use curriculum-scaled switch reward
        if switch_just_activated:
            reward += self.config.switch_activation_reward

        # === DISTANCE-BASED MILESTONE REWARDS (curriculum-aware progress tracking) ===
        switch_activated = obs.get("switch_activated", False)
        # Check for distance milestones
        # Uses cached distances from PBRS calculator (no extra pathfinding overhead)
        distance_to_switch = self.pbrs_calculator.get_last_distance_to_goal("switch")
        distance_to_exit = self.pbrs_calculator.get_last_distance_to_goal("exit")
        milestone_reward = self._check_distance_milestones(
            distance_to_switch, distance_to_exit, switch_activated
        )
        reward += milestone_reward

        # === PBRS OBJECTIVE POTENTIAL (curriculum-scaled, policy-invariant) ===
        # Calculate F(s,s') = γ * Φ(s') - Φ(s) for dense, policy-invariant path guidance
        adjacency = obs.get("_adjacency_graph")
        # level_data already retrieved earlier for terminal rewards
        graph_data = obs.get("_graph_data")

        # === HIERARCHICAL POTENTIAL COMPOSITION (Phase 3.1) ===
        # Compute BOTH switch and exit potentials, blend based on switch_activated
        # This maintains continuity for LSTM temporal credit assignment
        # Prevents discontinuity at switch activation that breaks temporal learning
        #
        # Traditional approach (REMOVED): Reset potential at switch activation
        #   Problem: LSTM sees reward discontinuity, breaks temporal credit assignment
        #
        # Hierarchical approach: Φ_total = Φ_switch × (1 - flag) + Φ_exit × flag
        #   Benefit: Smooth transition, continuous gradient field for LSTM
        #
        # This is still policy-invariant because:
        #   - Both potentials are policy-invariant individually
        #   - Linear combination preserves policy invariance
        #   - F(s,s') = γ * Φ_total(s') - Φ_total(s) remains valid PBRS

        # OPTIMIZATION: Use cached curriculum weight if valid (avoid @property overhead)
        objective_weight = (
            self._cached_pbrs_objective_weight
            if self._episode_config_cache_valid
            else self.config.pbrs_objective_weight
        )

        # DEPRECATED (2026-01-04): Gradient scaling now handled automatically in PBRS calculator
        # The scale_factor is no longer used. Gradient scaling is computed automatically
        # based on actual path length using sqrt scaling in PBRSCalculator.
        # Kept for backward compatibility but always set to 1.0.
        scale_factor = 1.0

        current_potential = self.pbrs_calculator.calculate_combined_potential(
            state=obs,
            adjacency=adjacency,
            level_data=level_data,
            graph_data=graph_data,
            objective_weight=objective_weight,
            scale_factor=scale_factor,
            curriculum_manager=curriculum_manager,
        )

        # Store potential for route visualization
        self.potential_history.append(current_potential)

        # DEBUG: Log every append for first 30 steps to diagnose 2x issue
        # if self.steps_taken <= 30:
        #     logger.warning(
        #         f"[POTENTIAL_APPEND] step={self.steps_taken}, appended={current_potential:.6f}, "
        #         f"total_count={len(self.potential_history)}"
        #     )

        # CRITICAL DEBUG: Log potential calculation to diagnose zero PBRS bug
        # if self.steps_taken % 100 == 0 and self.steps_taken > 0:
        #     prev_pot = self.prev_potential if self.prev_potential is not None else 0.0
        #     combined_path = obs.get("_pbrs_combined_path_distance", "MISSING")
        #     combined_physics = obs.get("_pbrs_combined_physics_cost", "MISSING")

        #     logger.warning(
        #         f"[POTENTIAL_DEBUG] step={self.steps_taken}, "
        #         f"current_potential={current_potential:.6f}, "
        #         f"prev_potential={prev_pot:.6f}, "
        #         f"combined_path={combined_path}, "
        #         f"combined_physics={combined_physics}, "
        #         f"objective_weight={objective_weight:.2f}"
        #     )

        # Track actual path length for efficiency metrics
        current_player_pos = (obs["player_x"], obs["player_y"])
        movement_distance = 0.0
        if self.prev_player_pos is not None:
            # Calculate Euclidean distance moved (approximation of actual path)
            dx = current_player_pos[0] - self.prev_player_pos[0]
            dy = current_player_pos[1] - self.prev_player_pos[1]
            movement_distance = (dx * dx + dy * dy) ** 0.5
            self.episode_path_length += movement_distance

            # SIMPLIFIED: Removed ineffective action penalty - PBRS already gives 0 reward
            # for no movement (potential unchanged = F(s,s') = 0)

        # Track velocity for visualization (Phase 1)
        if self.prev_player_pos is not None and movement_distance > 0:
            dx = current_player_pos[0] - self.prev_player_pos[0]
            dy = current_player_pos[1] - self.prev_player_pos[1]
            self.velocity_history.append((dx, dy))
        else:
            self.velocity_history.append((0.0, 0.0))

        self.prev_player_pos = current_player_pos

        # SIMPLIFIED: Removed oscillation detection - PBRS already gives 0 net reward
        # for oscillation (returning to same position = same potential = zero PBRS)

        # Store optimal path length (combined path distance) on first step
        if self.optimal_path_length is None:
            self.optimal_path_length = obs.get("_pbrs_combined_path_distance", 0.0)

        # Calculate PBRS shaping reward: F(s,s') = γ * Φ(s') - Φ(s)
        # Provides continuous gradient everywhere on the map
        pbrs_reward = 0.0
        actual_potential_change = 0.0

        if self.prev_potential is not None:
            # Apply PBRS formula for policy-invariant shaping
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
            actual_potential_change = current_potential - self.prev_potential

            # Track forward/backtracking steps for logging
            if actual_potential_change > 0.001:  # Forward progress
                self.episode_forward_steps += 1
            elif actual_potential_change < -0.001:  # Backtracking
                self.episode_backtrack_steps += 1

            # Log PBRS components for debugging and verification
            logger.debug(
                f"PBRS: Φ(s)={self.prev_potential:.4f}, "
                f"Φ(s')={current_potential:.4f}, "
                f"ΔΦ={actual_potential_change:.4f}, "
                f"F(s,s')={pbrs_reward:.4f} "
                f"(γ={self.pbrs_gamma})"
            )

            # Warn if potential decreased significantly (backtracking detected)
            if actual_potential_change < -0.05:
                logger.debug(
                    f"Backtracking detected: potential decreased by {-actual_potential_change:.4f} "
                    f"(PBRS penalty: {pbrs_reward:.4f})"
                )
            # Log significant progress
            elif actual_potential_change > 0.05:
                logger.debug(
                    f"Progress detected: potential increased by {actual_potential_change:.4f} "
                    f"(PBRS reward: {pbrs_reward:.4f})"
                )

            # Store current potential for next step
            self.prev_potential = current_potential
        else:
            # First step of episode - initialize potential
            self.prev_potential = current_potential
            logger.debug(f"Initializing PBRS potential: Φ(s')={current_potential:.4f}")

        # Store current potential for logging
        self.current_potential = current_potential

        # Track displacement from spawn for diagnostic logging
        displacement_from_spawn = 0.0
        if self.spawn_position is not None:
            dx = current_player_pos[0] - self.spawn_position[0]
            dy = current_player_pos[1] - self.spawn_position[1]
            displacement_from_spawn = (dx * dx + dy * dy) ** 0.5

        # Store for TensorBoard logging
        obs["_displacement_from_spawn"] = displacement_from_spawn
        obs["_episode_path_length"] = self.episode_path_length

        # # Warn if oscillating (moved a lot but ended up near spawn)
        # # Threshold: 10x means agent moved 10x more than net displacement (severe oscillation)
        # if self.steps_taken > 500 and oscillation_ratio > 10.0:
        #     logger.warning(
        #         f"[OSCILLATION_DIAG] High oscillation ratio: {oscillation_ratio:.1f}. "
        #         f"path_length={self.episode_path_length:.1f}px, "
        #         f"displacement={displacement_from_spawn:.1f}px, "
        #         f"steps={self.steps_taken}, "
        #         f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
        #         f"spawn={self.spawn_position}. "
        #         f"Agent is moving but not making net progress!"
        #     )

        # # Warn if stuck near wall (displacement okay but no PBRS progress)
        # # This catches the case where agent went somewhere but is now stuck
        # if self.steps_taken > 500 and displacement_from_spawn > 100:
        #     # Check if PBRS hasn't changed much recently (stuck at a wall)
        #     if len(self.episode_pbrs_rewards) >= 100:
        #         recent_pbrs = sum(self.episode_pbrs_rewards[-100:])
        #         if abs(recent_pbrs) < 0.5:  # Very little PBRS in last 100 steps
        #             logger.warning(
        #                 f"[WALL_STUCK_DIAG] No PBRS progress despite displacement. "
        #                 f"recent_pbrs_sum={recent_pbrs:.3f}, "
        #                 f"displacement={displacement_from_spawn:.1f}px, "
        #                 f"steps={self.steps_taken}, "
        #                 f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}). "
        #                 f"Agent may be stuck at a wall!"
        #             )

        # Track PBRS reward for logging
        self.episode_pbrs_rewards.append(pbrs_reward)

        reward += pbrs_reward

        # Disabled auxiliary components
        velocity_alignment_bonus = 0.0
        exit_direction_bonus = 0.0
        completion_approach_bonus = (
            0.0  # Will be calculated after distance_to_goal is available
        )

        # SIMPLIFIED: Removed position tracking (revisit penalty) - PBRS already penalizes
        # returning to states with higher distance to goal

        # === CALCULATE DISTANCE TO GOAL (needed for progress tracking) ===
        # OPTIMIZATION: Reuse distance already computed during PBRS calculation
        # This eliminates redundant pathfinding calls (saves ~2-3s per profile)
        distance_to_goal = obs.get("_pbrs_last_distance_to_goal")

        # Fallback: Compute if not available (shouldn't happen in normal flow)
        if distance_to_goal is None:
            if not obs.get("switch_activated", False):
                goal_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
                cache_key = "switch"
            else:
                goal_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))
                cache_key = "exit"

            player_pos = (int(obs["player_x"]), int(obs["player_y"]))

            entity_radius = (
                EXIT_SWITCH_RADIUS if cache_key == "switch" else EXIT_DOOR_RADIUS
            )
            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )
            distance_to_goal = self.pbrs_calculator.path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                base_adjacency,
                cache_key=cache_key,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )
            logger.debug(
                "Distance to goal not cached, computed on demand (fallback path)"
            )
        # Update previous distance tracking for adaptive gating (used before setting current)
        # Store BEFORE distance calculation section above to use in gating logic
        # This is set at the end of this method after all gating decisions

        # === PROGRESS TRACKING (diagnostic only, no penalty) ===
        # Track closest distance for logging - PBRS already handles backtracking
        # via potential decrease, so no explicit penalty is needed.
        if distance_to_goal < self.closest_distance_this_episode:
            self.closest_distance_this_episode = distance_to_goal

        # Track diagnostic metrics (Euclidean distance)
        from ..util.util import calculate_distance

        distance_to_switch = calculate_distance(
            obs["player_x"], obs["player_y"], obs["switch_x"], obs["switch_y"]
        )
        if distance_to_switch < self.closest_distance_to_switch:
            self.closest_distance_to_switch = distance_to_switch

        if obs.get("switch_activated", False):
            distance_to_exit = calculate_distance(
                obs["player_x"], obs["player_y"], obs["exit_door_x"], obs["exit_door_y"]
            )
            if distance_to_exit < self.closest_distance_to_exit:
                self.closest_distance_to_exit = distance_to_exit

        # Extract diagnostic metrics from observation (set by objective_distance_potential)
        normalized_distance = obs.get("_pbrs_normalized_distance", 0.0)
        combined_path_distance = obs.get("_pbrs_combined_path_distance", 0.0)

        self.last_pbrs_components = {
            "pbrs_reward": pbrs_reward,
            "time_penalty": time_penalty,
            "milestone_reward": milestone_reward,
            "exit_direction_bonus": exit_direction_bonus,
            "velocity_alignment_bonus": velocity_alignment_bonus,
            "completion_approach_bonus": completion_approach_bonus,
            "total_reward": reward,
            "is_terminal": False,
            # Potential information for debugging
            "current_potential": current_potential
            if self.current_potential is not None
            else 0.0,
            "prev_potential": self.prev_potential
            if self.prev_potential is not None
            else 0.0,
            "potential_change": actual_potential_change,
            # PBRS verification fields
            "potential_gradient": abs(actual_potential_change),
            "pbrs_gamma": self.pbrs_gamma,
            "theoretical_pbrs": actual_potential_change,
            # Diagnostic fields for monitoring
            "distance_to_goal": distance_to_goal,
            "combined_path_distance": combined_path_distance,
            "normalized_distance": normalized_distance,
            "frames_executed": frames_executed,
            "closest_distance_episode": self.closest_distance_this_episode,
            "displacement_from_spawn": displacement_from_spawn,
            "movement_distance": movement_distance,
            # Cache profiling
            "_pbrs_cache_hit": obs.get("_pbrs_cache_hit", False),
            # Direction diagnostics
            "euclidean_alignment": obs.get("_pbrs_euclidean_alignment", 0.0),
            "path_euclidean_ratio": obs.get("_pbrs_path_euclidean_ratio", 1.0),
        }

        # === REWARD BOUNDS SAFEGUARD ===
        # Clamp reward to prevent extreme values that could cause gradient explosion
        # TensorBoard analysis showed policy_loss exploded to 25 million due to unstable rewards
        # from PBRS bugs (e.g., infinity path distances). Clamping provides defense-in-depth.
        # Bounds [-100, 100] are generous enough for any valid reward composition:
        #   Max positive: completion(50) + switch(30) + PBRS(80) + efficiency(25) = 185
        #   Max negative: death(-40) + time(-0.06) + stagnation(-20) = -60
        # SIMPLIFIED: After removing oscillation/revisit penalties, bounds are even more conservative
        # REWARD_CLAMP_MIN = -100.0
        # REWARD_CLAMP_MAX = 100.0
        # if reward < REWARD_CLAMP_MIN or reward > REWARD_CLAMP_MAX:
        #     logger.warning(
        #         f"Reward {reward:.2f} exceeds bounds, clamping to "
        #         f"[{REWARD_CLAMP_MIN}, {REWARD_CLAMP_MAX}]. "
        #         f"This may indicate a bug in reward calculation."
        #     )
        #     reward = max(REWARD_CLAMP_MIN, min(REWARD_CLAMP_MAX, reward))

        # Apply global reward scaling for value function stability
        # This reduces reward magnitudes by 10x, making value loss more manageable
        # Without changing relative reward proportions or optimal policy
        scaled_reward = reward * GLOBAL_REWARD_SCALE

        # Update last_pbrs_components with scaled reward for logging
        if self.last_pbrs_components is not None:
            self.last_pbrs_components["scaled_reward"] = scaled_reward

        # Update distance tracking for next step's adaptive gating
        self._prev_distance_to_goal = distance_to_goal

        return scaled_reward

    def _check_distance_milestones(
        self,
        current_distance_to_switch: Optional[float],
        current_distance_to_exit: Optional[float],
        switch_activated: bool,
    ) -> float:
        """Check if agent crossed distance milestone and return reward.

        Rewards when shortest path distance decreases by >= milestone_interval.
        Automatically curriculum-aware since distances use curriculum goal positions.

        Args:
            current_distance_to_switch: Current path distance to switch (pixels)
            current_distance_to_exit: Current path distance to exit (pixels)
            switch_activated: Whether switch has been activated

        Returns:
            Milestone reward (30.0 for progress, scaled by reward hierarchy)
        """
        from .reward_constants import SUB_GOAL_REWARD_PROGRESS

        milestone_reward = 0.0

        # Check pre-switch milestones
        if not switch_activated:
            if current_distance_to_switch is None:
                return 0.0

            if self._last_milestone_distance_to_switch is None:
                # Initialize at level start
                self._last_milestone_distance_to_switch = current_distance_to_switch
            elif (
                current_distance_to_switch
                < self._last_milestone_distance_to_switch - self._milestone_interval_px
            ):
                # Crossed threshold - give reward
                milestone_reward = SUB_GOAL_REWARD_PROGRESS
                # Update milestone to current distance (allow multiple milestones)
                self._last_milestone_distance_to_switch = current_distance_to_switch
                logger.debug(
                    f"Milestone reward: {milestone_reward} "
                    f"(distance to switch: {current_distance_to_switch:.1f}px)"
                )

        # Check post-switch milestones
        else:
            if current_distance_to_exit is None:
                return 0.0

            if self._last_milestone_distance_to_exit is None:
                # Initialize when switch activates
                self._last_milestone_distance_to_exit = current_distance_to_exit
            elif (
                current_distance_to_exit
                < self._last_milestone_distance_to_exit - self._milestone_interval_px
            ):
                # Crossed threshold - give reward
                milestone_reward = SUB_GOAL_REWARD_PROGRESS
                self._last_milestone_distance_to_exit = current_distance_to_exit
                logger.debug(
                    f"Milestone reward: {milestone_reward} "
                    f"(distance to exit: {current_distance_to_exit:.1f}px)"
                )

        return milestone_reward

    def validate_level_solvability(self, obs: Dict[str, Any]) -> bool:
        """Validate that goals are reachable from spawn.

        Call at episode start to detect impossible levels.
        With 0% success rate, this confirms whether the issue is reward structure
        or actual level unsolvability.

        Args:
            obs: Game state observation with graph and level data

        Returns:
            True if level is solvable, False if unreachable, None if check failed
        """
        try:
            from ...constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )

            adjacency = obs.get("_adjacency_graph")
            level_data = obs.get("level_data")
            graph_data = obs.get("_graph_data")

            if not adjacency or not level_data:
                logger.warning(
                    "Cannot validate solvability: missing graph or level data"
                )
                return None

            player_pos = (int(obs["player_x"]), int(obs["player_y"]))
            switch_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
            exit_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))

            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )

            # Check spawn→switch path
            switch_dist = self.pbrs_calculator.path_calculator.get_distance(
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

            # Check switch→exit path
            exit_dist = self.pbrs_calculator.path_calculator.get_distance(
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
                    f"❌ LEVEL UNSOLVABLE: Switch unreachable from spawn!\n"
                    f"   Spawn: {player_pos}\n"
                    f"   Switch: {switch_pos}\n"
                    f"   This explains 0% success rate!"
                )
                return False

            if exit_dist == float("inf"):
                logger.error(
                    f"❌ LEVEL UNSOLVABLE: Exit unreachable from switch!\n"
                    f"   Switch: {switch_pos}\n"
                    f"   Exit: {exit_pos}\n"
                    f"   This explains 0% success rate!"
                )
                return False

            # combined_dist = switch_dist + exit_dist
            # logger.warning(
            #     f"✓ Level validated as solvable:\n"
            #     f"   Spawn→Switch: {switch_dist:.0f}px\n"
            #     f"   Switch→Exit: {exit_dist:.0f}px\n"
            #     f"   Combined: {combined_dist:.0f}px\n"
            #     f"   Expected PBRS (weight={self.config.pbrs_objective_weight}): "
            #     f"{self.config.pbrs_objective_weight * (1.0 - 0.0):.1f}"
            # )
            return True

        except Exception as e:
            logger.error(f"Failed to validate level solvability: {e}")
            import traceback

            traceback.print_exc()
            return None

    def reset(self):
        """Reset episode state for new episode."""
        self.steps_taken = 0

        # OPTIMIZATION: Refresh curriculum property cache at episode start
        # These are episode-invariant but called many times per step
        self._cached_pbrs_objective_weight = self.config.pbrs_objective_weight
        self._episode_config_cache_valid = True

        # Reset diagnostic tracking
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

        # Reset PBRS potential tracking for new episode
        # Ensures F(s,s') calculation starts fresh
        self.prev_potential = None
        self.current_potential = None

        # Reset episode-level tracking for TensorBoard logging
        self.episode_pbrs_rewards = []
        self.episode_time_penalties = []
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self.episode_terminal_reward = 0.0

        # Reset path optimality tracking
        self.episode_path_length = 0.0
        self.optimal_path_length = None
        self.prev_player_pos = None

        # Reset step-level component tracking
        self.last_pbrs_components = {}

        # SIMPLIFIED: Removed position tracker and oscillation detection resets

        # Reset PBRS calculator state
        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()

        # Reset progress tracking
        self.closest_distance_this_episode = float("inf")

        # Reset spawn position tracking for displacement gate
        self.spawn_position = None

        # Reset distance tracking for adaptive gating (Phase 2 enhancement)
        self._prev_distance_to_goal = None

        # Reset distance milestone tracking for new episode
        self._last_milestone_distance_to_switch = None
        self._last_milestone_distance_to_exit = None

        # Reset diagnostic tracking for visualization (Phase 1 enhancement)
        # old_potential_count = (
        #     len(self.potential_history) if hasattr(self, "potential_history") else 0
        # )
        self.velocity_history = []
        self.alignment_history = []
        self.path_gradient_history = []
        self.potential_history = []
        self._last_next_hop_world = None
        self._last_next_hop_goal_id = None

        # # DEBUG: Log potential_history reset
        # if old_potential_count > 0:
        #     logger.warning(
        #         f"[POTENTIAL_HISTORY_RESET] Cleared {old_potential_count} potentials during reward_calculator.reset()"
        #     )

        # Validation: Check level is solvable (helpful for debugging 0% success rate)
        # Only run occasionally to avoid performance impact
        if self.steps_taken == 0 or (
            hasattr(self, "_last_validation_step")
            and self.steps_taken - self._last_validation_step > 10000
        ):
            # Will be called with first obs in calculate_reward
            self._needs_validation = True
            self._last_validation_step = self.steps_taken

    def update_config(self, timesteps: int, success_rate: float) -> None:
        """Update reward configuration based on training progress.

        Called by trainer at evaluation checkpoints to trigger curriculum transitions.

        Args:
            timesteps: Total timesteps trained so far
            success_rate: Recent evaluation success rate (0.0-1.0)
        """
        old_phase = self.config.training_phase
        self.config.update(timesteps, success_rate)

        # Log phase transitions
        if self.config.training_phase != old_phase:
            logger.warning(
                f"\n{'=' * 60}\n"
                f"REWARD PHASE TRANSITION: {old_phase} → {self.config.training_phase}\n"
                f"Timesteps: {timesteps:,}\n"
                f"Success Rate: {success_rate:.1%}\n"
                f"Active Components:\n"
                f"  PBRS Weight: {self.config.pbrs_objective_weight:.2f}\n"
                f"  Time Penalty: {self.config.time_penalty_per_step:.4f}/step\n"
                f"  Normalization Scale: {self.config.pbrs_normalization_scale:.2f}\n"
                f"{'=' * 60}\n"
            )

    def get_config_state(self) -> Dict[str, Any]:
        """Get current reward configuration state for logging.

        Returns:
            Dictionary with current configuration values
        """
        return self.config.get_active_components()

    def get_pbrs_metrics(self) -> Dict[str, Any]:
        """Get current PBRS metrics for monitoring and verification.

        Useful for TensorBoard logging and debugging to verify:
        - PBRS formula is being applied correctly
        - Potentials are changing as expected
        - Markov property holds (same state → same potential)

        Returns:
            Dictionary with PBRS state:
            - prev_potential: Previous state's potential Φ(s)
            - pbrs_gamma: Discount factor γ used in F(s,s') formula
            - closest_distance_to_switch: Diagnostic Euclidean distance
            - closest_distance_to_exit: Diagnostic Euclidean distance
        """
        return {
            "prev_potential": self.prev_potential,
            "pbrs_gamma": self.pbrs_gamma,
            "closest_distance_to_switch": self.closest_distance_to_switch,
            "closest_distance_to_exit": self.closest_distance_to_exit,
        }

    def _compute_original_paths(
        self, level_data: Any, graph_data: Dict[str, Any]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Compute paths from original entity positions (fallback when no curriculum).

        Args:
            level_data: LevelData object with tiles and entities
            graph_data: Graph data dict with adjacency, physics_cache

        Returns:
            Tuple of (spawn_to_switch_path, switch_to_exit_path)
        """
        try:
            from ...constants.entity_types import EntityType
            from ...constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )
            from ...graph.reachability.pathfinding_utils import (
                find_ninja_node,
                find_goal_node_closest_to_start,
                find_shortest_path,
            )
            from ...graph.reachability.mine_proximity_cache import (
                MineProximityCostCache,
            )

            # Extract graph components
            adjacency = graph_data.get("adjacency")
            base_adjacency = graph_data.get("base_adjacency", adjacency)
            physics_cache = graph_data.get("node_physics")
            spatial_hash = graph_data.get("spatial_hash")
            subcell_lookup = graph_data.get("subcell_lookup")

            if not adjacency or not physics_cache:
                logger.warning("Cannot compute paths: missing graph data")
                return [], []

            # Build mine proximity cache for physics-aware pathfinding
            mine_proximity_cache = MineProximityCostCache()
            mine_proximity_cache.build_cache(level_data, adjacency)

            # Get spawn position
            spawn_pos = level_data.start_position
            spawn_node = find_ninja_node(
                spawn_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )

            if spawn_node is None:
                logger.warning(f"Cannot find spawn node at {spawn_pos}")
                return [], []

            # Get switch position
            exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
            if not exit_switches:
                logger.warning("No exit switch found in level data")
                return [], []
            switch = exit_switches[0]
            switch_pos = (int(switch.get("x", 0)), int(switch.get("y", 0)))
            switch_node = find_goal_node_closest_to_start(
                switch_pos,
                spawn_node,
                adjacency,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if switch_node is None:
                logger.warning(f"Cannot find switch node at {switch_pos}")
                return [], []

            # Get exit door position
            exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
            if not exit_doors:
                logger.warning("No exit door found in level data")
                return [], []
            exit_door = exit_doors[0]
            exit_pos = (int(exit_door.get("x", 0)), int(exit_door.get("y", 0)))
            exit_node = find_goal_node_closest_to_start(
                exit_pos,
                switch_node,
                adjacency,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if exit_node is None:
                logger.warning(f"Cannot find exit node at {exit_pos}")
                return [], []

            # Calculate spawn→switch path
            spawn_to_switch_path, spawn_to_switch_dist = find_shortest_path(
                spawn_node,
                switch_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data=level_data,
                mine_proximity_cache=mine_proximity_cache,
            )

            if spawn_to_switch_path is None:
                logger.warning("No path from spawn to switch")
                return [], []

            # Calculate switch→exit path
            switch_to_exit_path, switch_to_exit_dist = find_shortest_path(
                switch_node,
                exit_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data=level_data,
                mine_proximity_cache=mine_proximity_cache,
            )

            if switch_to_exit_path is None:
                logger.warning("No path from switch to exit")
                return [], []

            logger.debug(
                f"Computed original paths: spawn→switch={len(spawn_to_switch_path)} nodes, "
                f"switch→exit={len(switch_to_exit_path)} nodes"
            )

            # Cache computed paths for visualization
            self._cached_switch_path = spawn_to_switch_path
            self._cached_exit_path = switch_to_exit_path
            self._cached_spawn_node = spawn_node
            self._cached_switch_node = switch_node
            self._cached_exit_node = exit_node

            return spawn_to_switch_path, switch_to_exit_path

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to compute original paths: {e}\n{traceback.format_exc()}"
            )
            return [], []

    def _compute_curriculum_paths(
        self,
        level_data: Any,
        graph_data: Dict[str, Any],
        curriculum_switch_pos: Tuple[float, float],
        curriculum_exit_pos: Tuple[float, float],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Compute paths to curriculum entity positions (not truncated from original).

        This computes FRESH paths to where the curriculum entities actually are,
        ensuring waypoints are labeled correctly based on actual curriculum positions.

        Args:
            level_data: LevelData object with tiles and entities
            graph_data: Graph data dict with adjacency, physics_cache
            curriculum_switch_pos: Current curriculum switch position
            curriculum_exit_pos: Current curriculum exit position

        Returns:
            Tuple of (spawn_to_curriculum_switch_path, curriculum_switch_to_curriculum_exit_path)
        """
        try:
            from ...constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )
            from ...graph.reachability.pathfinding_utils import (
                find_ninja_node,
                find_closest_node_to_position,
                find_shortest_path,
            )
            from ...graph.reachability.mine_proximity_cache import (
                MineProximityCostCache,
            )

            # Extract graph components
            adjacency = graph_data.get("adjacency")
            base_adjacency = graph_data.get("base_adjacency", adjacency)
            physics_cache = graph_data.get("node_physics")
            spatial_hash = graph_data.get("spatial_hash")
            subcell_lookup = graph_data.get("subcell_lookup")

            if not adjacency or not physics_cache:
                logger.warning("Cannot compute curriculum paths: missing graph data")
                return [], []

            # Build mine proximity cache for physics-aware pathfinding
            mine_proximity_cache = MineProximityCostCache()
            mine_proximity_cache.build_cache(level_data, adjacency)

            # Get spawn position
            spawn_pos = level_data.start_position
            spawn_node = find_ninja_node(
                spawn_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )

            if spawn_node is None:
                logger.warning(f"Cannot find spawn node at {spawn_pos}")
                return [], []

            # Find curriculum switch node
            curriculum_switch_node = find_closest_node_to_position(
                curriculum_switch_pos,
                adjacency,
                threshold=50.0,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if curriculum_switch_node is None:
                logger.warning(
                    f"Cannot find curriculum switch node at {curriculum_switch_pos}"
                )
                return [], []

            # Find curriculum exit node
            curriculum_exit_node = find_closest_node_to_position(
                curriculum_exit_pos,
                adjacency,
                threshold=50.0,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if curriculum_exit_node is None:
                logger.warning(
                    f"Cannot find curriculum exit node at {curriculum_exit_pos}"
                )
                return [], []

            # Calculate spawn→curriculum_switch path
            spawn_to_curriculum_switch, spawn_switch_dist = find_shortest_path(
                spawn_node,
                curriculum_switch_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data=level_data,
                mine_proximity_cache=mine_proximity_cache,
            )

            if spawn_to_curriculum_switch is None:
                logger.warning("No path from spawn to curriculum switch")
                return [], []

            # Calculate curriculum_switch→curriculum_exit path
            curriculum_switch_to_exit, switch_exit_dist = find_shortest_path(
                curriculum_switch_node,
                curriculum_exit_node,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data=level_data,
                mine_proximity_cache=mine_proximity_cache,
            )

            if curriculum_switch_to_exit is None:
                logger.warning("No path from curriculum switch to curriculum exit")
                return [], []

            # logger.warning(
            #     f"[WAYPOINT] Computed FRESH curriculum paths: "
            #     f"spawn→curriculum_switch={len(spawn_to_curriculum_switch)} nodes ({spawn_switch_dist:.0f}px), "
            #     f"curriculum_switch→curriculum_exit={len(curriculum_switch_to_exit)} nodes ({switch_exit_dist:.0f}px)"
            # )

            # Cache computed paths for visualization
            self._cached_switch_path = spawn_to_curriculum_switch
            self._cached_exit_path = curriculum_switch_to_exit
            self._cached_spawn_node = spawn_node
            self._cached_switch_node = curriculum_switch_node
            self._cached_exit_node = curriculum_exit_node

            return spawn_to_curriculum_switch, curriculum_switch_to_exit

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to compute curriculum paths: {e}\n{traceback.format_exc()}"
            )
            return [], []

    def get_cached_paths(self) -> Dict[str, Any]:
        """Get cached paths computed during waypoint extraction.

        This ensures visualization uses the same paths that the reward calculator uses,
        preventing divergence between what's shown and what's being rewarded.

        Returns:
            Dict containing:
                - switch_path: Path from spawn to switch
                - exit_path: Path from switch to exit
                - spawn_node: Spawn node position
                - switch_node: Switch node position
                - exit_node: Exit node position
        """
        return {
            "switch_path": self._cached_switch_path,
            "exit_path": self._cached_exit_path,
            "spawn_node": self._cached_spawn_node,
            "switch_node": self._cached_switch_node,
            "exit_node": self._cached_exit_node,
        }

    def _calculate_path_distance(self, path: List[Tuple[int, int]]) -> float:
        """Calculate geometric distance along a path.

        Args:
            path: List of node positions

        Returns:
            Total distance in pixels
        """
        if not path or len(path) < 2:
            return 0.0

        distance = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            distance += (dx * dx + dy * dy) ** 0.5

        return distance

    def get_pbrs_diagnostic_data(self) -> Dict[str, Any]:
        """Get PBRS diagnostic data for route visualization (Phase 1 enhancement).

        Returns:
            Dictionary with diagnostic data:
            - velocity_history: List of (dx, dy) velocity vectors
            - alignment_history: List of path alignment scores [-1, 1]
            - path_gradient_history: List of (gx, gy) expected direction vectors or None
            - potential_history: List of PBRS potential values (scaled) at each step
            - last_next_hop_world: Last calculated next_hop position in world coords
            - last_next_hop_goal_id: Goal ID for last next_hop ("switch" or "exit")
        """
        return {
            "velocity_history": self.velocity_history,
            "alignment_history": self.alignment_history,
            "path_gradient_history": self.path_gradient_history,
            "potential_history": self.potential_history,
            "last_next_hop_world": self._last_next_hop_world,
            "last_next_hop_goal_id": self._last_next_hop_goal_id,
        }

    def get_episode_reward_metrics(self) -> Dict[str, Any]:
        """Get episode-level reward component metrics for TensorBoard logging.

        Returns:
            Dictionary with episode reward metrics:
            - pbrs_total: Sum of all PBRS rewards in episode
            - time_penalty_total: Sum of all time penalties in episode
            - terminal_reward: Terminal reward (completion/death)
            - pbrs_to_penalty_ratio: |PBRS sum| / |penalty sum| (0 if no penalties)
            - forward_steps: Count of steps with potential increase
            - backtrack_steps: Count of steps with potential decrease
            - pbrs_mean: Mean PBRS reward per step
            - pbrs_std: Standard deviation of PBRS rewards
            - current_potential: Current potential Φ(s')
            - prev_potential: Previous potential Φ(s)
            - waypoint_statistics: Statistics from adaptive waypoint system
        """
        import numpy as np

        pbrs_total = (
            sum(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
        )
        penalty_total = (
            sum(self.episode_time_penalties) if self.episode_time_penalties else 0.0
        )

        # Calculate ratio (avoid division by zero)
        if penalty_total < 0:
            ratio = abs(pbrs_total) / abs(penalty_total)
        else:
            ratio = 0.0

        # Calculate statistics (guard against empty arrays)
        pbrs_mean = (
            np.mean(self.episode_pbrs_rewards)
            if len(self.episode_pbrs_rewards) > 0
            else 0.0
        )
        pbrs_std = (
            np.std(self.episode_pbrs_rewards)
            if len(self.episode_pbrs_rewards) > 0
            else 0.0
        )

        # Calculate path optimality (ratio of optimal to actual path length)
        # Values closer to 1.0 indicate more efficient navigation
        if (
            self.episode_path_length > 0
            and self.optimal_path_length is not None
            and self.optimal_path_length > 0
        ):
            path_optimality = self.optimal_path_length / self.episode_path_length
        else:
            path_optimality = 0.0

        # Calculate forward progress and backtracking percentages
        total_steps = self.episode_forward_steps + self.episode_backtrack_steps
        if total_steps > 0:
            forward_progress_pct = (self.episode_forward_steps / total_steps) * 100
            backtracking_pct = (self.episode_backtrack_steps / total_steps) * 100
        else:
            forward_progress_pct = 0.0
            backtracking_pct = 0.0

        return {
            "pbrs_total": pbrs_total,
            "time_penalty_total": penalty_total,
            "terminal_reward": self.episode_terminal_reward,
            "pbrs_to_penalty_ratio": ratio,
            "forward_steps": self.episode_forward_steps,
            "backtrack_steps": self.episode_backtrack_steps,
            "pbrs_mean": pbrs_mean,
            "pbrs_std": pbrs_std,
            "current_potential": self.current_potential,
            "prev_potential": self.prev_potential,
            # New path efficiency metrics
            "path_optimality": path_optimality,
            "forward_progress_pct": forward_progress_pct,
            "backtracking_pct": backtracking_pct,
            "episode_path_length": self.episode_path_length,
            "optimal_path_length": self.optimal_path_length
            if self.optimal_path_length is not None
            else 0.0,
        }
