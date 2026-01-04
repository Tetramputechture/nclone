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
from .path_waypoint_extractor import PathWaypointExtractor
from .reward_constants import (
    LEVEL_COMPLETION_REWARD,
    SWITCH_ACTIVATION_REWARD,
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

        # Initialize path waypoint extractor for uniform 12px spacing guidance
        # Uses simplified extraction with uniform values along optimal path
        self.path_waypoint_extractor = PathWaypointExtractor(
            progress_spacing=self.config.path_waypoint_progress_spacing,
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

        # Path waypoint collection tracking (for visualization and bonuses)
        # Waypoints are extracted from optimal path and wired to PBRS for routing
        # Collection tracking enables visualization and bonus rewards
        self._collected_path_waypoints: set = set()
        self.current_path_waypoints: List[Any] = []
        self.current_path_waypoints_by_phase: Dict[str, List[Any]] = {}

        # Waypoint sequence tracking for soft sequential guidance
        self._waypoint_sequence: List[Any] = []
        self._next_expected_waypoint_index: int = 0
        self._last_collected_waypoint: Optional[Any] = None
        self._frames_since_last_collection: int = 0

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
        milestone_reward = SWITCH_ACTIVATION_REWARD if switch_just_activated else 0.0

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
        if obs.get("player_won", False):
            terminal_reward = LEVEL_COMPLETION_REWARD

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
                    LEVEL_COMPLETION_REWARD * 0.5 * min(1.0, path_efficiency)
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
        if switch_just_activated:
            reward += SWITCH_ACTIVATION_REWARD

        # === DISTANCE-BASED MILESTONE REWARDS (curriculum-aware progress tracking) ===
        switch_activated = obs.get("switch_activated", False)
        # Check for distance milestones - simpler alternative to pre-computed waypoints
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
        # level_data already retrieved earlier for terminal rewards and waypoints
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

        # # CRITICAL DEBUG: Log potential calculation to diagnose zero PBRS bug
        # if self.steps_taken % 100 == 0 and self.steps_taken > 0:
        #     prev_pot = self.prev_potential if self.prev_potential is not None else 0.0
        #     combined_path = obs.get("_pbrs_combined_path_distance", "MISSING")
        #     combined_physics = obs.get("_pbrs_combined_physics_cost", "MISSING")

        #     logger.error(
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
        pbrs_reward = 0.0

        # === TWO-PHASE PBRS NORMALIZATION (Continuous at Switch Activation) ===
        # Potential is continuous across switch activation using two-phase normalization:
        # - Switch phase: Φ ∈ [0.0, 0.5] normalized by spawn→switch distance
        # - Exit phase: Φ ∈ [0.5, 1.0] normalized by switch→exit distance
        # - At switch activation: both formulas yield Φ = 0.5 (continuous!)
        #
        # This eliminates the discontinuity that would occur with single-phase normalization:
        #   OLD: Φ = 1 - (distance / combined_path_distance)
        #        At switch: Φ_before = 1.0, Φ_after = 0.5 → PBRS penalty of -20!
        #   NEW: Two-phase normalization ensures Φ = 0.5 on both sides
        #        At switch: Φ_before = 0.5, Φ_after = 0.5 → PBRS = 0 (no penalty!)
        #
        # Benefits:
        # - No negative PBRS penalty for achieving switch milestone
        # - Equal per-pixel gradient in both phases
        # - LSTM learns temporal dependencies across full episode without discontinuity
        # - Policy invariance maintained (still valid PBRS)
        # Track potential change for diagnostics (must be computed before updating prev_potential)
        actual_potential_change = 0.0

        if self.prev_potential is not None:
            # Apply PBRS formula for policy-invariant shaping
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential

            # CRITICAL: Save potential change BEFORE updating prev_potential for correct logging
            actual_potential_change = current_potential - self.prev_potential

            # # === DIAGNOSTIC LOGGING FOR SWITCH ACTIVATION ===
            # # Verify two-phase normalization maintains continuity at switch activation
            # if switch_just_activated:
            #     spawn_to_switch = obs.get("_pbrs_spawn_to_switch_distance", 0.0)
            #     switch_to_exit = obs.get("_pbrs_switch_to_exit_distance", 0.0)
            #     combined = obs.get("_pbrs_combined_path_distance", 0.0)
            #     logger.warning(
            #         f"[SWITCH_ACTIVATION_PBRS] Two-phase normalization verification:\n"
            #         f"  Φ(s) = {self.prev_potential:.4f} (switch phase, should be ~0.5)\n"
            #         f"  Φ(s') = {current_potential:.4f} (exit phase, should be ~0.5)\n"
            #         f"  ΔΦ = {actual_potential_change:.4f} (should be ~0.0 for continuity)\n"
            #         f"  PBRS = {pbrs_reward:.4f} (should be ~0.0, not large negative!)\n"
            #         f"  Path metrics: spawn→switch={spawn_to_switch:.1f}px, "
            #         f"switch→exit={switch_to_exit:.1f}px, combined={combined:.1f}px\n"
            #         f"  Switch milestone reward: {milestone_reward:.1f}\n"
            #         f"  Total reward for switch frame: {milestone_reward + pbrs_reward:.1f}"
            #     )

            # Log PBRS components for debugging and verification
            logger.debug(
                f"PBRS: Φ(s)={self.prev_potential:.4f}, "
                f"Φ(s')={current_potential:.4f}, "
                f"ΔΦ={actual_potential_change:.4f}, "
                f"F(s,s')={pbrs_reward:.4f} "
                f"(γ={self.pbrs_gamma})"
            )

            # Track forward/backtracking steps for logging
            if actual_potential_change > 0.001:  # Forward progress
                self.episode_forward_steps += 1
            elif actual_potential_change < -0.001:  # Backtracking
                self.episode_backtrack_steps += 1

            # === PBRS DIRECTION VALIDATION DIAGNOSTIC ===
            # Check if movement direction aligns with expected path direction
            # This catches the bug where PBRS rewards going TOWARD goal coordinates
            # when the actual path requires going AWAY from goal first

            # Initialize diagnostic tracking flags (Phase 1)
            alignment_tracked = False

            if self.prev_player_pos is not None and movement_distance > 0.5:
                # Movement direction (normalized)
                move_dx = (
                    current_player_pos[0] - self.prev_player_pos[0]
                ) / movement_distance
                move_dy = (
                    current_player_pos[1] - self.prev_player_pos[1]
                ) / movement_distance

                # Get current goal position for direction comparison
                if not obs.get("switch_activated", False):
                    diag_goal_x, diag_goal_y = obs["switch_x"], obs["switch_y"]
                else:
                    diag_goal_x, diag_goal_y = obs["exit_door_x"], obs["exit_door_y"]

                # Calculate Euclidean direction to goal for comparison
                goal_dx = diag_goal_x - current_player_pos[0]
                goal_dy = diag_goal_y - current_player_pos[1]
                euclidean_to_goal = (goal_dx * goal_dx + goal_dy * goal_dy) ** 0.5

                if euclidean_to_goal > 1.0:
                    # Normalized direction to goal (Euclidean)
                    goal_dir_x = goal_dx / euclidean_to_goal
                    goal_dir_y = goal_dy / euclidean_to_goal

                    # Dot product: +1 = moving toward goal, -1 = moving away
                    euclidean_alignment = move_dx * goal_dir_x + move_dy * goal_dir_y

                    # Store alignment for TensorBoard logging
                    obs["_pbrs_euclidean_alignment"] = euclidean_alignment
                    obs["_pbrs_potential_change"] = actual_potential_change

                    # === ENHANCED PBRS GRADIENT VERIFICATION ===
                    # Verify PBRS gradient aligns with expected direction
                    # This helps diagnose if PBRS is providing correct guidance

                    # Check for PBRS/movement mismatch (potential bug indicator)
                    moving_toward_goal = euclidean_alignment > 0.3  # Moving toward goal
                    moving_away_goal = (
                        euclidean_alignment < -0.3
                    )  # Moving away from goal
                    potential_increased = actual_potential_change > 0.01
                    potential_decreased = actual_potential_change < -0.01

                    # Expected: moving toward goal → potential increases
                    # Expected: moving away from goal → potential decreases
                    if moving_toward_goal and potential_decreased:
                        logger.warning(
                            f"[PBRS_MISMATCH] Moving toward goal but potential DECREASED! "
                            f"alignment={euclidean_alignment:.3f}, ΔΦ={actual_potential_change:.4f}, "
                            f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                            f"goal=({diag_goal_x:.0f},{diag_goal_y:.0f}). "
                            f"This suggests PBRS path may be incorrect or obstacles blocking."
                        )
                    elif moving_away_goal and potential_increased:
                        logger.warning(
                            f"[PBRS_PATH] Moving away from goal but potential increased "
                            f"(alignment={euclidean_alignment:.3f}, ΔΦ={actual_potential_change:.4f}). "
                            f"This is EXPECTED if optimal path requires detour. Verify path is correct."
                        )

            # === PATH DIRECTION DIAGNOSTICS (NO GATING) ===
            # Track path direction for visualization but DO NOT gate PBRS rewards.
            # The adjacency graph already encodes optimal paths via A* - trust it.
            # Gating breaks policy invariance and blocks counter-intuitive navigation.
            # REMOVED: All direction-based PBRS gating (lines 746-953 in original)
            if not alignment_tracked:
                self.alignment_history.append(0.0)
                self.path_gradient_history.append(None)

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

            # Store current potential for next step (only after successful comparison)
            self.prev_potential = current_potential
        else:
            # First step of episode or after goal transition - initialize potential
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

        # # CRITICAL DEBUG: Log PBRS accumulation every 100 steps to diagnose zero bug
        # if self.steps_taken % 100 == 0 and self.steps_taken > 0:
        #     logger.error(
        #         f"[PBRS_ACCUM] step={self.steps_taken}, "
        #         f"pbrs_this_step={pbrs_reward:.4f}, "
        #         f"pbrs_accumulated={sum(self.episode_pbrs_rewards):.2f} unscaled "
        #         f"({sum(self.episode_pbrs_rewards) * GLOBAL_REWARD_SCALE:.3f} scaled), "
        #         f"list_length={len(self.episode_pbrs_rewards)}, "
        #         f"forward_steps={self.episode_forward_steps}, "
        #         f"backtrack_steps={self.episode_backtrack_steps}"
        #     )

        # === AUXILIARY COMPONENTS DISABLED (2025-12-25) ===
        # All auxiliary shaping components removed to focus learning on core objectives:
        # - Path waypoint bonuses: Redundant with PBRS distance reduction
        # - Velocity alignment: PBRS gradient provides directional signal
        # - Exit direction bonus: Not needed with pure PBRS
        # - Completion approach bonus: PBRS provides sufficient gradient near goal

        switch_activated = obs.get("switch_activated", False)
        waypoint_bonus = 0.0
        velocity_alignment_bonus = 0.0
        exit_direction_bonus = 0.0
        completion_approach_bonus = 0.0

        # Keep waypoint tracking for visualization only (no reward impact)
        self._track_waypoint_collection(
            current_player_pos, self.prev_player_pos, switch_activated
        )

        # Initialize metrics dict for logging compatibility
        waypoint_metrics = {
            "sequence_multiplier": 0.0,
            "collection_type": "none",
            "skip_distance": 0,
            "approach_bonus": 0.0,
        }

        # Track frames since last waypoint collection (for compatibility)
        if hasattr(self, "_frames_since_last_collection"):
            self._frames_since_last_collection += frames_executed

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

        # # === PATH vs EUCLIDEAN DISTANCE DIVERGENCE DIAGNOSTIC ===
        # # NOTE: distance_to_goal is PHYSICS-WEIGHTED (not geometric pixels)
        # # Physics costs are ~0.04 per horizontal pixel, so a 1000px path = ~40 physics cost
        # # We need GEOMETRIC distance for meaningful comparison with Euclidean
        # euclidean_to_current_goal_raw = (
        #     distance_to_switch
        #     if not obs.get("switch_activated", False)
        #     else distance_to_exit
        # )

        # # Get geometric path distance for proper comparison (not physics-weighted)
        # geometric_path_distance = 0.0
        # try:
        #     geometric_path_distance = (
        #         self.pbrs_calculator.path_calculator.get_geometric_distance(
        #             player_pos,
        #             goal_pos,
        #             adjacency,
        #             base_adjacency,
        #             level_data=level_data,
        #             graph_data=graph_data,
        #             entity_radius=entity_radius,
        #             ninja_radius=NINJA_RADIUS,
        #             goal_id="switch"
        #             if not obs.get("switch_activated", False)
        #             else "exit",
        #         )
        #     )
        # except Exception:
        #     geometric_path_distance = 0.0

        # FIX: Apply same radius adjustment to Euclidean for fair comparison
        # geometric_path_distance already has (ninja_radius + entity_radius) subtracted
        # so we must apply the same adjustment to Euclidean distance
        # combined_radius = NINJA_RADIUS + entity_radius
        # euclidean_adjusted = max(0.0, euclidean_to_current_goal_raw - combined_radius)

        # # Only compute ratio for meaningful distances (> 30px to avoid grid-snapping artifacts)
        # # Path distance is computed between 12px grid nodes, so small distances have high error
        # if euclidean_adjusted > 30.0 and geometric_path_distance > 0:
        #     divergence_ratio = geometric_path_distance / euclidean_adjusted

        #     # Store for TensorBoard logging
        #     obs["_pbrs_path_euclidean_ratio"] = divergence_ratio

        #     # Log anomalies: path should be longer than Euclidean for complex levels
        #     # Using 0.70 threshold (30% tolerance) because:
        #     # - Path is computed between 12px grid nodes, not exact positions
        #     # - Node snapping can cause up to ~17px discrepancy (diagonal of 12px cell)
        #     # - This is a diagnostic, not a critical error
        #     if divergence_ratio < 0.70:
        #         # Path is significantly shorter than Euclidean - likely a real bug
        #         logger.warning(
        #             f"[PATH_DIAG] Geometric path < Euclidean distance! "
        #             f"ratio={divergence_ratio:.3f}, "
        #             f"path={geometric_path_distance:.1f}px, euclidean_adj={euclidean_adjusted:.1f}px, "
        #             f"pos=({obs['player_x']:.0f},{obs['player_y']:.0f}), "
        #             f"goal=({goal_pos[0]},{goal_pos[1]}). "
        #             f"Pathfinding may be broken!"
        #         )
        #     elif divergence_ratio > 5.0 and self.steps_taken == 1:
        #         # Very high divergence on first step - level has complex geometry
        #         # This is expected for levels where path goes away from goal first
        #         logger.debug(
        #             f"[PATH_DIAG] High path/Euclidean divergence at spawn. "
        #             f"ratio={divergence_ratio:.2f}, "
        #             f"path={geometric_path_distance:.1f}px, euclidean_adj={euclidean_adjusted:.1f}px, "
        #             f"pos=({obs['player_x']:.0f},{obs['player_y']:.0f}), "
        #             f"goal=({goal_pos[0]},{goal_pos[1]}). "
        #             f"Level geometry requires going away from goal first."
        #         )

        # Populate PBRS components for TensorBoard logging (non-terminal state)
        # SIMPLIFIED: Only core components - terminal, PBRS, time penalty

        # Extract diagnostic metrics from observation (set by objective_distance_potential)
        normalized_distance = obs.get("_pbrs_normalized_distance", 0.0)
        combined_path_distance = obs.get("_pbrs_combined_path_distance", 0.0)

        self.last_pbrs_components = {
            "pbrs_reward": pbrs_reward,
            "time_penalty": time_penalty,
            "milestone_reward": milestone_reward,
            "waypoint_bonus": waypoint_bonus,
            "waypoint_metrics": waypoint_metrics,
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

        # Reset waypoint collection tracking for new episode
        self._collected_path_waypoints = set()
        self._next_expected_waypoint_index = 0
        self._last_collected_waypoint = None
        self._frames_since_last_collection = 0

        # Reset distance milestone tracking for new episode
        self._last_milestone_distance_to_switch = None
        self._last_milestone_distance_to_exit = None

        # Reset diagnostic tracking for visualization (Phase 1 enhancement)
        self.velocity_history = []
        self.alignment_history = []
        self.path_gradient_history = []
        self.potential_history = []
        self._last_next_hop_world = None
        self._last_next_hop_goal_id = None

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

    def extract_path_waypoints_for_level(
        self,
        level_data: Any,
        graph_data: Dict[str, Any],
        map_name: str,
    ) -> int:
        """Extract path-based waypoints from curriculum-aware optimal paths.

        UPDATED 2026-01-03: Uses pre-computed curriculum-aware paths from
        IntermediateGoalManager instead of recomputing from original positions.
        This ensures waypoints are:
        1. Evenly spaced along the curriculum path
        2. Always on the PBRS-optimal path the agent is actually following
        3. Not clustered at the start due to truncation

        Called during episode reset after graph is built. Extracts dense waypoints
        from spawn→curriculum_switch and curriculum_switch→curriculum_exit paths.

        Args:
            level_data: LevelData object with tiles and entities
            graph_data: Graph data dict with adjacency, physics_cache
            map_name: Level name for caching

        Returns:
            Number of waypoints extracted
        """
        # Check if path waypoints are enabled in config
        if not self.config.enable_path_waypoints:
            return 0

        if self.path_waypoint_extractor is None:
            return 0

        try:
            # CRITICAL: Compute paths TO curriculum entity positions, not truncated original paths
            # Waypoints must reflect where the curriculum entities actually are
            if self.goal_curriculum_manager is not None:
                # Get actual curriculum entity positions
                curriculum_switch_pos = (
                    self.goal_curriculum_manager.get_curriculum_switch_position()
                )
                curriculum_exit_pos = (
                    self.goal_curriculum_manager.get_curriculum_exit_position()
                )

                # Compute NEW paths to curriculum positions
                spawn_to_switch_path, switch_to_exit_path = (
                    self._compute_curriculum_paths(
                        level_data,
                        graph_data,
                        curriculum_switch_pos,
                        curriculum_exit_pos,
                    )
                )

                # Calculate distances for logging
                curriculum_switch_dist = self._calculate_path_distance(
                    spawn_to_switch_path
                )
                curriculum_exit_dist = self._calculate_path_distance(
                    switch_to_exit_path
                )

                # Early curriculum stages may have degenerate paths
                if not spawn_to_switch_path and not switch_to_exit_path:
                    logger.error(
                        f"[WAYPOINT] Both curriculum paths empty "
                        f"(stage {self.goal_curriculum_manager.state.unified_stage}) - skipping waypoint extraction"
                    )
                    return 0  # Skip waypoint extraction entirely

                # If one path is empty, log it but continue with the non-empty path
                if not switch_to_exit_path:
                    logger.warning(
                        f"[WAYPOINT] Early curriculum stage {self.goal_curriculum_manager.state.unified_stage}: "
                        f"exit overlaps with switch, extracting waypoints only from spawn→curriculum_switch path "
                        f"({len(spawn_to_switch_path)} nodes, {curriculum_switch_dist:.0f}px)"
                    )
                elif not spawn_to_switch_path:
                    logger.warning(
                        f"Unusual curriculum state at stage {self.goal_curriculum_manager.state.unified_stage}: "
                        f"spawn→curriculum_switch path empty, extracting waypoints only from curriculum_switch→curriculum_exit path "
                        f"({len(switch_to_exit_path)} nodes, {curriculum_exit_dist:.0f}px)"
                    )
            else:
                # No curriculum - compute paths from original positions
                spawn_to_switch_path, switch_to_exit_path = (
                    self._compute_original_paths(level_data, graph_data)
                )

                # Without curriculum, we expect both paths to exist
                if not spawn_to_switch_path or not switch_to_exit_path:
                    logger.warning("Cannot extract waypoints: no valid paths found")
                    return 0

                # Calculate distances for logging
                curriculum_switch_dist = self._calculate_path_distance(
                    spawn_to_switch_path
                )
                curriculum_exit_dist = self._calculate_path_distance(
                    switch_to_exit_path
                )

            # Extract graph components for waypoint extraction
            physics_cache = graph_data.get("node_physics")

            if not physics_cache:
                logger.warning("Cannot extract path waypoints: missing physics cache")
                return 0

            # Extract waypoints from curriculum-aware paths
            # These will be evenly spaced along the actual PBRS-optimal path
            # IMPORTANT: Disable cache when using curriculum (paths change per stage)
            use_waypoint_cache = (
                self.goal_curriculum_manager is None
            )  # Only cache when no curriculum

            # Calculate path distances for diagnostics
            spawn_to_switch_distance = 0.0
            if len(spawn_to_switch_path) > 1:
                for i in range(1, len(spawn_to_switch_path)):
                    dx = spawn_to_switch_path[i][0] - spawn_to_switch_path[i - 1][0]
                    dy = spawn_to_switch_path[i][1] - spawn_to_switch_path[i - 1][1]
                    spawn_to_switch_distance += (dx * dx + dy * dy) ** 0.5

            switch_to_exit_distance = 0.0
            if len(switch_to_exit_path) > 1:
                for i in range(1, len(switch_to_exit_path)):
                    dx = switch_to_exit_path[i][0] - switch_to_exit_path[i - 1][0]
                    dy = switch_to_exit_path[i][1] - switch_to_exit_path[i - 1][1]
                    switch_to_exit_distance += (dx * dx + dy * dy) ** 0.5

            # logger.warning(
            #     f"[WAYPOINT] Extracting waypoints: "
            #     f"spawn_to_switch={len(spawn_to_switch_path)} nodes ({spawn_to_switch_distance:.0f}px), "
            #     f"switch_to_exit={len(switch_to_exit_path)} nodes ({switch_to_exit_distance:.0f}px), "
            #     f"use_cache={use_waypoint_cache}"
            # )

            # Extract waypoints from paths (curriculum-aware or original)
            # When curriculum is active, paths are already computed to curriculum positions
            # so waypoints naturally have correct phase labels
            waypoints_list = self.path_waypoint_extractor.extract_waypoints_from_paths(
                spawn_to_switch_path=spawn_to_switch_path,
                switch_to_exit_path=switch_to_exit_path,
                physics_cache=physics_cache,
                level_id=map_name,
                use_cache=use_waypoint_cache,
            )

            # logger.warning(
            #     f"[WAYPOINT] Extracted {len(waypoints_list)} raw waypoints from paths"
            # )

            # Convert list to dictionary grouped by phase for efficient filtering
            waypoints_by_phase = {
                "pre_switch": [wp for wp in waypoints_list if wp.phase == "pre_switch"],
                "post_switch": [
                    wp for wp in waypoints_list if wp.phase == "post_switch"
                ],
            }

            # Store for reward calculation
            self.current_path_waypoints_by_phase = waypoints_by_phase
            # Also store full list for metrics (waypoint_collection_rate calculation)
            self.current_path_waypoints = waypoints_list

            # DEBUG: Verify waypoint phases match their dict keys
            # for phase_key, wps in waypoints_by_phase.items():
            #     if len(wps) > 0:
            #         first_wp_phase = getattr(wps[0], "phase", "N/A")
            #         logger.debug(
            #             f"[WAYPOINT] Stored {len(wps)} waypoints in dict['{phase_key}'], "
            #             f"first waypoint.phase={first_wp_phase}"
            #         )

            total_waypoints = sum(len(wps) for wps in waypoints_by_phase.values())
            # logger.warning(
            #     f"[WAYPOINT] ✓ Extracted {total_waypoints} curriculum-aware waypoints: "
            #     f"pre_switch={len(waypoints_by_phase.get('pre_switch', []))}, "
            #     f"post_switch={len(waypoints_by_phase.get('post_switch', []))}"
            # )

            if total_waypoints == 0:
                logger.warning(
                    f"[WAYPOINT] ⚠️  No waypoints extracted! Path lengths: "
                    f"spawn→switch={len(spawn_to_switch_path)}, "
                    f"switch→exit={len(switch_to_exit_path)}"
                )

            return total_waypoints

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to extract path waypoints: {e}\n{traceback.format_exc()}"
            )
            return 0

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

            return spawn_to_curriculum_switch, curriculum_switch_to_exit

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to compute curriculum paths: {e}\n{traceback.format_exc()}"
            )
            return [], []

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

    def _get_sequence_multiplier(self, collected_index: int) -> float:
        """Get soft sequential multiplier based on collection order.

        Rewards following optimal path while allowing novel exploration:
        - Perfect sequence (next expected): 1.0x full bonus
        - Skip one waypoint: 0.7x (might be smart shortcut)
        - Skip multiple waypoints: 0.5x (novel path discovery)
        - Backward collection: 0.2x (going back, still rewarded)

        Args:
            collected_index: node_index of waypoint being collected

        Returns:
            Multiplier in range [0.2, 1.0]
        """
        expected_index = self._next_expected_waypoint_index

        if collected_index == expected_index:
            # Perfect sequence - full bonus
            return 1.0
        elif collected_index == expected_index + 1:
            # Skipped exactly one - might be smart shortcut
            return 0.7
        elif collected_index > expected_index:
            # Skipped multiple - novel path discovery
            return 0.5
        else:
            # Collected a waypoint we "passed" - going backward
            # Still reward it (might be intentional) but minimal
            return 0.2

    def _update_sequence_tracking(
        self, collected_index: int, collected_waypoint: Any
    ) -> None:
        """Update sequence tracking after waypoint collection.

        Advances expected index to handle skips gracefully.

        Args:
            collected_index: node_index of waypoint just collected
            collected_waypoint: The PathWaypoint object that was collected
        """
        # Advance expected index to max(current, collected+1)
        # This allows agent to skip waypoints without breaking sequence tracking
        self._next_expected_waypoint_index = max(
            self._next_expected_waypoint_index, collected_index + 1
        )

        # Track last collected waypoint for exit direction bonus
        self._last_collected_waypoint = collected_waypoint
        self._frames_since_last_collection = 0

    def _get_turn_velocity_alignment(
        self,
        obs: Dict[str, Any],
        current_pos: Tuple[float, float],
        switch_activated: bool,
    ) -> float:
        """Calculate velocity alignment bonus with multi-hop path gradient.

        Re-enables velocity alignment globally (not just near turns) using multi-hop
        lookahead direction. This provides continuous directional guidance that helps
        with all navigation, especially sharp turns near hazards where anticipatory
        adjustment is critical.

        The multi-hop gradient shows the "bent" direction accounting for upcoming
        curvature, allowing the agent to start adjusting trajectory well before
        reaching the inflection point.

        BALANCE: Velocity alignment set to 0.2 (20% of PBRS magnitude) to ensure
        agent feels the penalty for wrong direction BEFORE hitting hazard. Over
        10 steps of misalignment, this creates -2.0 accumulated penalty, making
        "continue toward hazard" clearly worse than "follow turn".

        Args:
            obs: Current observation with velocity and graph data
            current_pos: Current player position
            switch_activated: Whether switch is activated

        Returns:
            Velocity alignment bonus (can be negative for misalignment)
        """
        # Use curriculum-adaptive weight from RewardConfig
        # This scales from 2.0 in discovery (strong guidance) to 0.5 in advanced (light guidance)
        alignment_weight = self.config.velocity_alignment_weight
        MIN_SPEED = 0.3  # Only apply when moving meaningfully (px/frame)

        # Get velocity
        vx = obs.get("player_xspeed", 0.0)
        vy = obs.get("player_yspeed", 0.0)
        vel_mag = math.sqrt(vx * vx + vy * vy)

        # Only apply if moving (velocity > MIN_SPEED px/frame)
        if vel_mag < MIN_SPEED:
            return 0.0

        # Normalize velocity
        vel_norm_x = vx / vel_mag
        vel_norm_y = vy / vel_mag

        # Get multi-hop path direction (anticipatory guidance)
        adjacency = obs.get("_adjacency_graph")
        graph_data = obs.get("_graph_data")

        if adjacency is None or graph_data is None:
            return 0.0

        # Get path gradient using multi-hop lookahead
        from .pbrs_potentials import get_path_gradient_from_next_hop
        from ...graph.reachability.pathfinding_utils import (
            extract_spatial_lookups_from_graph_data,
        )

        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        # Determine current goal
        goal_id = "exit" if switch_activated else "switch"

        # Get level cache from path calculator
        if not hasattr(self.pbrs_calculator, "path_calculator"):
            return 0.0

        path_calculator = self.pbrs_calculator.path_calculator
        if not hasattr(path_calculator, "level_cache"):
            return 0.0

        level_cache = path_calculator.level_cache

        path_gradient = get_path_gradient_from_next_hop(
            current_pos,
            adjacency,
            level_cache,
            goal_id,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
            use_multi_hop=True,  # Use multi-hop for anticipatory guidance
        )

        if path_gradient is None:
            return 0.0

        # Calculate alignment with multi-hop path direction
        # +1 = moving along path, 0 = perpendicular, -1 = opposite
        alignment = vel_norm_x * path_gradient[0] + vel_norm_y * path_gradient[1]

        # === DISTANCE-GATED VELOCITY ALIGNMENT ===
        # Apply full weight when far from goal (need path following)
        # Taper to zero when very close (need precision, not rigid guidance)
        # This allows fine-grained control near switch/exit while providing
        # strong directional guidance during path traversal

        # Get distance to current goal
        distance_to_goal = obs.get("_pbrs_last_distance_to_goal", 1000.0)

        # Distance gates:
        # - >200px from goal: full velocity alignment (path following critical)
        # - 50-200px: linear taper (transition zone)
        # - <50px: zero velocity alignment (precision required)
        if distance_to_goal > 200.0:
            distance_scale = 1.0  # Full weight
        elif distance_to_goal > 50.0:
            distance_scale = (distance_to_goal - 50.0) / 150.0  # Linear taper
        else:
            distance_scale = 0.0  # Disable near goal

        # Return scaled alignment bonus using curriculum weight and distance gate
        # Positive for moving along path, negative for moving against it
        # Discovery phase: weight=2.0 provides strong anticipatory guidance for turns
        alignment_bonus = alignment_weight * alignment * distance_scale

        return alignment_bonus

    def _calculate_turn_approach_bonus(
        self,
        current_pos: Tuple[float, float],
        previous_pos: Optional[Tuple[float, float]],
        phase_waypoints: List[Any],
    ) -> float:
        """Calculate continuous approach bonus for turn waypoints.

        Provides anticipatory guidance as agent approaches critical turns (curvature > 60deg),
        not just binary reward on collection. Creates a "pull" toward turn points that scales
        with distance and velocity alignment.

        This solves the "sharp turn near hazard" problem by rewarding the agent for moving
        toward the turn point BEFORE reaching it, giving time to adjust trajectory.

        Args:
            current_pos: Current agent position
            previous_pos: Previous agent position (for velocity calculation)
            phase_waypoints: List of waypoints for current phase

        Returns:
            Continuous approach bonus (0.0+) for approaching turn waypoints
        """
        if previous_pos is None:
            return 0.0

        # Only apply to significant turn waypoints (curvature > 60 degrees)
        MIN_TURN_CURVATURE = 60.0
        APPROACH_RADIUS = 50.0  # Start giving bonus within 50px of turn
        MAX_BONUS_PER_WAYPOINT = 0.15  # Max bonus per waypoint per step

        total_approach_bonus = 0.0

        # Calculate current velocity
        vel_x = current_pos[0] - previous_pos[0]
        vel_y = current_pos[1] - previous_pos[1]
        vel_mag = math.sqrt(vel_x * vel_x + vel_y * vel_y)

        if vel_mag < 0.1:
            # Agent not moving meaningfully
            return 0.0

        # Normalize velocity
        vel_norm_x = vel_x / vel_mag
        vel_norm_y = vel_y / vel_mag

        # Check each uncollected turn waypoint
        for wp in phase_waypoints:
            # Skip if already collected
            if (wp.position, wp.phase) in getattr(
                self, "_collected_path_waypoints", set()
            ):
                continue

            # Only apply to significant turns
            if wp.curvature < MIN_TURN_CURVATURE:
                continue

            # Calculate distance to waypoint
            wp_dx = wp.position[0] - current_pos[0]
            wp_dy = wp.position[1] - current_pos[1]
            distance = math.sqrt(wp_dx * wp_dx + wp_dy * wp_dy)

            # Only apply within approach radius
            if distance > APPROACH_RADIUS:
                continue

            # Calculate approach alignment (velocity toward waypoint)
            # Normalize direction to waypoint
            if distance > 0.1:
                wp_dir_x = wp_dx / distance
                wp_dir_y = wp_dy / distance

                # Dot product: +1 = moving toward waypoint, -1 = moving away
                approach_alignment = vel_norm_x * wp_dir_x + vel_norm_y * wp_dir_y

                # Only reward positive alignment (moving toward turn)
                if approach_alignment > 0:
                    # Bonus scales with:
                    # 1. Proximity (1.0 at waypoint, 0.0 at APPROACH_RADIUS)
                    # 2. Approach alignment (1.0 = directly toward, 0.0 = perpendicular)
                    # 3. Waypoint curvature (sharper turns = higher bonus)
                    proximity_factor = 1.0 - (distance / APPROACH_RADIUS)
                    curvature_factor = min(1.0, wp.curvature / 120.0)  # Cap at 120deg

                    waypoint_bonus = (
                        MAX_BONUS_PER_WAYPOINT
                        * proximity_factor
                        * approach_alignment
                        * curvature_factor
                    )

                    total_approach_bonus += waypoint_bonus

        return total_approach_bonus

    def _get_exit_direction_bonus(
        self,
        current_velocity: Tuple[float, float],
    ) -> float:
        """Calculate bonus for continuing in correct direction after waypoint collection.

        Provides soft guidance to continue along optimal path after collecting waypoint.
        This is GUIDANCE, not enforcement - agent can still go other directions without penalty.

        Only applies for ~10 frames after collection to encourage immediate continuation.

        Args:
            current_velocity: Current agent velocity (vx, vy)

        Returns:
            Small bonus for velocity alignment with exit direction (0.0-0.05)
        """
        if self._last_collected_waypoint is None:
            return 0.0

        # Only apply for limited time after collection (30 frames ~= 7-8 actions with frame_skip=4)
        # INCREASED from 10 to 30 frames to provide stronger guidance through trajectory changes
        if self._frames_since_last_collection > 30:
            return 0.0

        # Check if waypoint has exit direction
        exit_dir = getattr(self._last_collected_waypoint, "exit_direction", None)
        if exit_dir is None:
            return 0.0

        # Calculate velocity magnitude
        vx, vy = current_velocity
        vel_mag = math.sqrt(vx * vx + vy * vy)

        # Only apply if agent is moving (velocity > 0.5 px/frame)
        if vel_mag < 0.5:
            return 0.0

        # Normalize velocity
        vel_norm_x = vx / vel_mag
        vel_norm_y = vy / vel_mag

        # Calculate alignment with exit direction (dot product)
        # 1.0 = same direction, 0 = perpendicular, -1 = opposite
        alignment = vel_norm_x * exit_dir[0] + vel_norm_y * exit_dir[1]

        # Only reward positive alignment (moving in right direction)
        # No penalty for going other directions (allows exploration)
        if alignment > 0:
            # Small bonus: max 0.05 per frame for perfect alignment
            return 0.05 * alignment

        return 0.0

    def _get_path_waypoint_bonus(
        self,
        current_pos: Tuple[float, float],
        previous_pos: Optional[Tuple[float, float]],
        switch_activated: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate bonus for reaching path waypoints with soft sequential guidance.

        NEW: Includes continuous approach bonus for turn waypoints (curvature > 60deg).
        This provides anticipatory guidance as agent approaches critical turns, not just
        binary reward on collection. Helps agent commit to turn direction before overshooting.

        Uses soft sequential multiplier that rewards following optimal path while
        still allowing collection of any waypoint (enables novel path discovery).

        Args:
            current_pos: Current agent position
            previous_pos: Previous agent position
            switch_activated: Whether switch is activated

        Returns:
            Tuple of (waypoint_bonus, metrics_dict) where metrics_dict contains:
            - sequence_multiplier: Multiplier applied (0.2-1.0)
            - collection_type: "sequential", "skip_1", "skip_multiple", "backward", or "none"
            - skip_distance: How many waypoints were skipped (0 for sequential)
            - approach_bonus: Continuous bonus for approaching turn waypoints (0.0+)
        """
        metrics = {
            "sequence_multiplier": 0.0,
            "collection_type": "none",
            "skip_distance": 0,
            "approach_bonus": 0.0,
        }

        if previous_pos is None or not self.current_path_waypoints:
            return 0.0, metrics

        # Define minimum spawn distance to prevent trivial early bonuses
        # REDUCED: 50px → 24px (2 tiles) for less restrictive filtering
        MIN_SPAWN_DISTANCE = 24.0

        # Get spawn position (cached from first calculate_reward call)
        spawn_pos = self.spawn_position
        if spawn_pos is None:
            # Fallback: use previous_pos on first call (will be set properly next call)
            spawn_pos = previous_pos

        # Calculate distance from spawn once per call
        spawn_dx = current_pos[0] - spawn_pos[0]
        spawn_dy = current_pos[1] - spawn_pos[1]
        distance_from_spawn = (spawn_dx * spawn_dx + spawn_dy * spawn_dy) ** 0.5

        # Skip all waypoint checks if too close to spawn
        if distance_from_spawn < MIN_SPAWN_DISTANCE:
            return 0.0, metrics

        # Get waypoints for current phase
        current_phase = "post_switch" if switch_activated else "pre_switch"
        phase_waypoints = self.current_path_waypoints_by_phase.get(current_phase, [])

        if not phase_waypoints:
            logger.warning(
                f"[WAYPOINT] No waypoints for phase={current_phase}. "
                f"Available phases: {list(self.current_path_waypoints_by_phase.keys())}, "
                f"counts: {[(p, len(wps)) for p, wps in self.current_path_waypoints_by_phase.items()]}"
            )
            return 0.0, metrics

        total_bonus = 0.0

        # Import constants for simplified waypoint system
        from .reward_constants import (
            WAYPOINT_COLLECTION_REWARD,
            WAYPOINT_COLLECTION_RADIUS,
        )

        # Check each waypoint to see if we just crossed into its radius
        for wp in phase_waypoints:
            wp_pos = wp.position

            # Check if already collected this episode
            collected_key = (wp_pos, wp.phase)
            if collected_key in getattr(self, "_collected_path_waypoints", set()):
                continue

            # Calculate distances
            prev_dist = math.sqrt(
                (previous_pos[0] - wp_pos[0]) ** 2 + (previous_pos[1] - wp_pos[1]) ** 2
            )
            curr_dist = math.sqrt(
                (current_pos[0] - wp_pos[0]) ** 2 + (current_pos[1] - wp_pos[1]) ** 2
            )

            # IMPROVED: Collect if currently within radius OR just passed through
            # This handles agents that move quickly or spawn near waypoints
            if curr_dist <= WAYPOINT_COLLECTION_RADIUS or (
                prev_dist > WAYPOINT_COLLECTION_RADIUS
                and curr_dist <= WAYPOINT_COLLECTION_RADIUS
            ):
                # Flat reward per waypoint - simple and predictable
                bonus = WAYPOINT_COLLECTION_REWARD
                total_bonus += bonus

                # Mark as collected
                if not hasattr(self, "_collected_path_waypoints"):
                    self._collected_path_waypoints = set()
                self._collected_path_waypoints.add((wp_pos, wp.phase))

                # Update metrics for TensorBoard logging
                metrics["sequence_multiplier"] = (
                    1.0  # No multiplier in simplified system
                )
                metrics["collection_type"] = "collected"
                metrics["skip_distance"] = 0

        return total_bonus, metrics

    def _convert_path_waypoints_to_momentum(
        self,
        path_waypoints: List[Any],  # List[PathWaypoint]
        switch_activated: bool,
        goal_pos: Tuple[float, float],
    ) -> List[Any]:
        """Convert path waypoints to momentum waypoint format for PBRS routing.

        Args:
            path_waypoints: List of PathWaypoint objects
            switch_activated: Whether switch is activated
            goal_pos: Current goal position (switch or exit)

        Returns:
            List of MomentumWaypoint objects
        """
        from collections import namedtuple

        MomentumWaypoint = namedtuple(
            "MomentumWaypoint", ["position", "approach_direction"]
        )

        # Filter by phase
        current_phase = "post_switch" if switch_activated else "pre_switch"
        filtered = [wp for wp in path_waypoints if wp.phase == current_phase]

        momentum_waypoints = []

        for wp in filtered:
            # Compute approach direction from waypoint toward goal
            wp_pos = wp.position
            dx = goal_pos[0] - wp_pos[0]
            dy = goal_pos[1] - wp_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0.001:
                approach_dir = (dx / dist, dy / dist)
            else:
                approach_dir = (0.0, 0.0)

            momentum_waypoints.append(MomentumWaypoint(wp_pos, approach_dir))

        return momentum_waypoints

    def _merge_waypoints(
        self,
        path_waypoints: List[Any],
        demo_waypoints: List[Tuple[Tuple[float, float], float]],
        cluster_radius: float = 30.0,
    ) -> List[Any]:
        """Merge path and demo waypoints, prioritizing demos on overlap.

        When a demo waypoint is within cluster_radius of a path waypoint,
        the demo waypoint's value is used (demos are physically validated).

        Args:
            path_waypoints: List of PathWaypoint objects
            demo_waypoints: List of (position, value) tuples from demos
            cluster_radius: Distance threshold for considering waypoints overlapping

        Returns:
            List of PathWaypoint objects with merged values
        """
        if not demo_waypoints:
            return path_waypoints

        if not path_waypoints:
            # Convert demo waypoints to PathWaypoint format
            # For now, just return path_waypoints since we don't create PathWaypoint objects here
            return path_waypoints

        merged = list(path_waypoints)  # Start with path waypoints

        # For each demo waypoint, check if it overlaps with a path waypoint
        for demo_pos, demo_value in demo_waypoints:
            # Check for overlap with existing path waypoints
            for i, path_wp in enumerate(merged):
                dx = demo_pos[0] - path_wp.position[0]
                dy = demo_pos[1] - path_wp.position[1]
                distance = (dx * dx + dy * dy) ** 0.5

                if distance < cluster_radius:
                    # Overlaps - upgrade the value to demo value (usually higher)
                    # Create a new PathWaypoint with the demo value
                    from collections import namedtuple

                    PathWaypoint = namedtuple(
                        "PathWaypoint", ["position", "waypoint_type", "value", "phase"]
                    )
                    merged[i] = PathWaypoint(
                        position=path_wp.position,
                        waypoint_type="demo_enhanced",
                        value=max(path_wp.value, demo_value),  # Take higher value
                        phase=path_wp.phase,
                    )
                    logger.debug(
                        f"Enhanced path waypoint at ({path_wp.position[0]:.0f}, {path_wp.position[1]:.0f}) "
                        f"with demo value: {path_wp.value:.2f} → {max(path_wp.value, demo_value):.2f}"
                    )
                    break  # Found overlap, move to next demo waypoint

            # If no overlap, this demo waypoint represents a unique strategic position
            # not captured by the path - we could add it, but for simplicity,
            # we'll rely on path waypoints for bonuses and use demos only for PBRS routing

        return merged

    def _track_waypoint_collection(
        self,
        current_pos: Optional[Tuple[float, float]],
        previous_pos: Optional[Tuple[float, float]],
        switch_activated: bool,
    ) -> None:
        """Track waypoint collection for visualization (no bonus rewards).

        Checks if agent passed through any uncollected waypoints during movement.
        Waypoints are considered collected if agent moved within radius.

        This enables waypoint visualization in route rendering without adding
        bonus rewards - waypoints are already rewarded via PBRS routing.

        FIXED: Checks ALL waypoints (both phases) since switch_activated from obs
        may be stale or incorrect. Waypoints are phase-agnostic for visualization.

        Args:
            current_pos: Current agent position
            previous_pos: Previous agent position (for movement check)
            switch_activated: Whether exit switch is activated (NOTE: may be stale, so we check all phases)
        """
        if (
            current_pos is None
            or previous_pos is None
            or not self.current_path_waypoints_by_phase
        ):
            return

        # Check ALL waypoints from both phases for visualization
        # Use the waypoint's own phase attribute, not just the dict key
        all_waypoints = []
        for phase_name, waypoints in self.current_path_waypoints_by_phase.items():
            # Use waypoint's phase attribute if available, otherwise use dict key
            for wp in waypoints:
                wp_phase = getattr(wp, "phase", phase_name)
                all_waypoints.append((wp, wp_phase))

        phase_waypoints = all_waypoints

        # DEBUG: Log total waypoints being checked (only once per episode to reduce spam)
        if len(all_waypoints) > 0 and len(self._collected_path_waypoints) == 0:
            pre_count = sum(1 for wp, p in all_waypoints if p == "pre_switch")
            post_count = sum(1 for wp, p in all_waypoints if p == "post_switch")
            # logger.debug(
            #     f"[WAYPOINT] Episode start: Checking {len(all_waypoints)} total waypoints: "
            #     f"{pre_count} pre_switch, {post_count} post_switch"
            # )

        if not phase_waypoints:
            return

        # Import collection radius constant
        from .reward_constants import WAYPOINT_COLLECTION_RADIUS

        # DEBUG: Track collections in this call
        collections_this_step = 0

        for wp, wp_phase in phase_waypoints:
            wp_pos = wp.position

            # Skip if already collected
            if (wp_pos, wp_phase) in self._collected_path_waypoints:
                continue

            # CRITICAL: Block post-switch waypoints until switch is activated
            # Post-switch waypoints represent the path from switch to exit,
            # and should only be collectible after the switch has been activated
            if wp_phase == "post_switch" and not switch_activated:
                continue

            # Check if agent passed through waypoint radius
            # Use previous distance for "passing through" detection
            prev_dist = math.sqrt(
                (previous_pos[0] - wp_pos[0]) ** 2 + (previous_pos[1] - wp_pos[1]) ** 2
            )
            curr_dist = math.sqrt(
                (current_pos[0] - wp_pos[0]) ** 2 + (current_pos[1] - wp_pos[1]) ** 2
            )

            # Collect if currently within radius OR passed through during movement
            if (
                curr_dist <= WAYPOINT_COLLECTION_RADIUS
                or prev_dist <= WAYPOINT_COLLECTION_RADIUS
            ):
                self._collected_path_waypoints.add((wp_pos, wp_phase))
                collections_this_step += 1

        # DEBUG: Log summary if checking waypoints
        if len(phase_waypoints) > 0 and collections_this_step == 0:
            # Find closest uncollected waypoint
            min_dist = float("inf")
            for wp, wp_phase in phase_waypoints:
                if (wp.position, wp_phase) not in self._collected_path_waypoints:
                    dist = math.sqrt(
                        (current_pos[0] - wp.position[0]) ** 2
                        + (current_pos[1] - wp.position[1]) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist

    def seed_demo_waypoints(
        self,
        demo_waypoints: List[Tuple[Tuple[float, float], float, str, int, str]],
        level_id: str,
    ) -> int:
        """Seed demo waypoints for PBRS routing (simplified from adaptive system).

        Demo waypoints are merged with path waypoints and used for PBRS potential routing only.
        No separate collection bonuses - all bonuses come from unified path waypoint system.

        Args:
            demo_waypoints: List of 5-tuples (position, value, type, discovery_count, phase)
                from DemoCheckpointSeeder.extract_waypoints_from_checkpoints()
                - position: (x, y) world coordinates
                - value: importance score (0.5-1.5 typically)
                - type: "demo" for demo-extracted waypoints
                - discovery_count: initial count (typically 1)
                - phase: "pre_switch" or "post_switch" - when waypoint should be active
            level_id: Level identifier for level-specific waypoint tracking

        Returns:
            Number of waypoints successfully seeded
        """
        if not demo_waypoints:
            logger.warning("No demo waypoints to seed")
            return 0

        # Clear previous demo waypoints for this level
        self.demo_waypoints_by_phase = {
            "pre_switch": [],
            "post_switch": [],
        }

        # Add demo waypoints to storage by phase
        seeded_count = 0
        for wp_data in demo_waypoints:
            if len(wp_data) != 5:
                logger.warning(
                    f"Invalid waypoint format: expected 5-tuple (pos, value, type, count, phase), "
                    f"got {len(wp_data)}-tuple. Skipping waypoint."
                )
                continue

            wp_pos, wp_value, wp_type, wp_count, wp_phase = wp_data

            # Validate phase
            if wp_phase not in ("pre_switch", "post_switch"):
                logger.warning(
                    f"Invalid waypoint phase '{wp_phase}', defaulting to 'pre_switch'"
                )
                wp_phase = "pre_switch"

            # Store as (position, value) tuple for merging with path waypoints
            self.demo_waypoints_by_phase[wp_phase].append((wp_pos, wp_value))
            seeded_count += 1
            logger.debug(
                f"Seeded demo waypoint at ({wp_pos[0]:.0f}, {wp_pos[1]:.0f}), "
                f"value={wp_value:.2f}, phase={wp_phase}"
            )

        # Count waypoints per phase for logging
        pre_switch_count = len(self.demo_waypoints_by_phase["pre_switch"])
        post_switch_count = len(self.demo_waypoints_by_phase["post_switch"])

        logger.warning(
            f"Seeded {seeded_count} demo waypoints for level '{level_id}' "
            f"(pre_switch: {pre_switch_count}, post_switch: {post_switch_count})"
        )

        return seeded_count

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

        # Calculate statistics
        pbrs_mean = (
            np.mean(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
        )
        pbrs_std = (
            np.std(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
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

        # === PBRS EPISODE SUMMARY LOGGING ===
        # Log episode-level PBRS performance for debugging
        # logger.warning(
        #     f"[PBRS_EPISODE] "
        #     f"Forward: {forward_progress_pct:.1f}% ({self.episode_forward_steps} steps), "
        #     f"Backtrack: {backtracking_pct:.1f}% ({self.episode_backtrack_steps} steps), "
        #     f"PBRS total: {pbrs_total:.2f} unscaled ({pbrs_total * GLOBAL_REWARD_SCALE:.3f} scaled), "
        #     f"Mean: {pbrs_mean:.4f}, Std: {pbrs_std:.4f}, "
        #     f"Path efficiency: {path_optimality:.3f}, "
        #     f"Actual path: {self.episode_path_length:.1f}px, Optimal: {self.optimal_path_length:.1f}px"
        # )

        # Count waypoints collected this episode
        waypoints_collected = len(self._collected_path_waypoints)
        waypoints_total_available = (
            len(self.current_path_waypoints) if self.current_path_waypoints else 0
        )
        waypoint_collection_rate = (
            waypoints_collected / waypoints_total_available
            if waypoints_total_available > 0
            else 0.0
        )

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
            # Waypoint collection metrics
            "waypoints_collected": waypoints_collected,
            "waypoints_total": waypoints_total_available,
            "waypoint_collection_rate": waypoint_collection_rate,
        }
