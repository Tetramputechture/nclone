"""Policy-invariant reward calculator using true PBRS for path guidance.

Implements Potential-Based Reward Shaping (Ng et al. 1999) with clear hierarchy:
1. Terminal rewards (completion, death) - Define task success/failure
2. PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s) - Dense, policy-invariant path guidance
3. Time penalty - Efficiency pressure (curriculum-controlled)

PBRS provides:
- Dense reward signal at every step (not sparse achievement bonuses)
- Automatic backtracking penalties (moving away from goal reduces potential)
- Policy invariance guarantee (optimal policy unchanged by shaping)
- Markov property (no episode history dependencies)

Phase 2 Enhancements (Break Policy Invariance):
- Waypoint guidance: Intermediate rewards along optimal path
- Progress preservation: Penalty for regressing past best distance
Trade-off: Lose theoretical guarantees for practical learning at low success rates.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .reward_config import RewardConfig
from .pbrs_potentials import PBRSCalculator
from .reward_constants import (
    LEVEL_COMPLETION_REWARD,
    DEATH_PENALTY,
    IMPACT_DEATH_PENALTY,
    HAZARD_DEATH_PENALTY,
    SWITCH_ACTIVATION_REWARD,
    PBRS_GAMMA,
    PBRS_DISPLACEMENT_THRESHOLD,
    INEFFECTIVE_ACTION_THRESHOLD,
    INEFFECTIVE_ACTION_PENALTY,
    OSCILLATION_WINDOW,
    NET_DISPLACEMENT_THRESHOLD,
    OSCILLATION_PENALTY,
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


class PositionTracker:
    """Position tracking for revisit penalties and exploration bonuses.

    Tracks positions using hierarchical grid system:
    - Revisit penalties: Sliding window to penalize oscillation/looping
    - Exploration bonuses: Episode-wide novelty tracking with hierarchical reset

    UPDATED: 12px grid (matches graph sub-node resolution) for 4× finer granularity.
    Hierarchical reset on switch activation treats Phase 1 and Phase 2 as separate.
    """

    def __init__(self, grid_size: int = 12, window_size: int = 100):
        """Initialize position tracker.

        Args:
            grid_size: Cell size in pixels for position discretization (default: 12px sub-node size)
            window_size: Number of recent positions to track for revisit penalty
        """
        self.grid_size = grid_size
        self.visited_cells: set = (
            set()
        )  # Episode-wide tracking (for stats and exploration)
        self.position_history = deque(
            maxlen=window_size
        )  # Sliding window for revisit penalty
        self.position_counts: Dict[tuple, int] = {}  # Visit counts in current window
        self.cells_visited_this_episode = 0

        # Hierarchical exploration tracking (resets on goal changes)
        self.visited_cells_current_phase: set = set()  # Resets when switch activates
        self.cells_visited_current_phase = 0
        self.hierarchical_resets = 0  # Count of switch activation resets

    def get_position_reward(
        self,
        position: tuple,
        revisit_penalty_weight: float = 0.001,
        exploration_bonus: float = 0.0,
        pbrs_reward: float = 0.0,
    ) -> tuple:
        """Calculate revisit penalty and exploration bonus for current position.

        UPDATED: Now includes hierarchical exploration bonuses that reset on switch activation.
        This ensures consistent exploration incentive in both task phases.

        Uses LOGARITHMIC scaling for revisit penalties with per-step soft cap to prevent
        accumulated penalties from dominating terminal rewards:
        - Logarithmic: -weight × log(1 + visit_count) instead of linear
        - Soft cap: max penalty of -0.5 per step

        Exploration bonus (PBRS-gated, hierarchical):
        - Only awards bonus when pbrs_reward > 0 (productive exploration)
        - Resets on switch activation to treat Phase 2 as fresh exploration
        - Grid: 12px (matches graph sub-node resolution, 4× finer than tile)

        Args:
            position: (x, y) position in pixels
            revisit_penalty_weight: Penalty weight for revisits (default: 0.001)
            exploration_bonus: Bonus value for discovering new cell (if PBRS > 0)
            pbrs_reward: PBRS reward for this step (bonus only if positive)

        Returns:
            Tuple of (total_reward, exploration_reward, revisit_penalty, is_new_cell)
        """
        # Snap to grid (OPTIMIZED: integer math, pre-computed inverse)
        # For 128 envs × 2048 steps per rollout = 262K calls
        # Optimization saves ~0.5ms per rollout
        grid_x = int(position[0]) // self.grid_size * self.grid_size
        grid_y = int(position[1]) // self.grid_size * self.grid_size
        cell = (grid_x, grid_y)

        total_reward = 0.0
        exploration_reward = 0.0
        revisit_penalty = 0.0

        # Check novelty in CURRENT PHASE (hierarchical tracking)
        is_new_in_phase = cell not in self.visited_cells_current_phase
        is_new_in_episode = cell not in self.visited_cells

        # Track visited cells for stats
        if is_new_in_episode:
            self.visited_cells.add(cell)
            self.cells_visited_this_episode += 1

        # Track visited cells for current phase (hierarchical)
        if is_new_in_phase:
            self.visited_cells_current_phase.add(cell)
            self.cells_visited_current_phase += 1

            # Award exploration bonus if PBRS is positive (productive exploration)
            if pbrs_reward > 0.001 and exploration_bonus > 0:
                exploration_reward = exploration_bonus
                total_reward += exploration_reward

        # Revisit penalty (always active, uses sliding window)
        # LOGARITHMIC scaling with soft cap for bounded accumulation
        # log(1 + count) grows slowly: log(11) = 2.4, log(51) = 3.9
        visit_count = self.position_counts.get(cell, 0)
        if visit_count > 0:
            # Logarithmic scaling: diminishing returns for repeated visits
            revisit_penalty = -revisit_penalty_weight * math.log1p(visit_count)
            # Per-step soft cap: prevent worst-case from dominating
            revisit_penalty = max(revisit_penalty, -0.5)
            total_reward += revisit_penalty

        # Save reference to oldest cell before appending
        oldest_cell = None
        if len(self.position_history) == self.position_history.maxlen:
            oldest_cell = self.position_history[0]

        # Update sliding window
        self.position_history.append(cell)
        self.position_counts[cell] = visit_count + 1

        # Decrement count for cell that was expelled from window
        if oldest_cell is not None:
            if oldest_cell in self.position_counts:
                self.position_counts[oldest_cell] -= 1
                if self.position_counts[oldest_cell] == 0:
                    del self.position_counts[oldest_cell]

        return total_reward, exploration_reward, revisit_penalty, is_new_in_episode

    def reset_for_goal_change(self):
        """Reset hierarchical phase tracking when goal changes (e.g., switch activates).

        CRITICAL: This treats Phase 1 (→switch) and Phase 2 (→exit) as separate
        exploration problems. Without this reset, Phase 2 exploration receives no
        bonuses because cells were already visited in Phase 1.

        Clears:
        - visited_cells_current_phase: Phase-specific cell tracking
        - cells_visited_current_phase: Counter for diagnostics

        Keeps:
        - visited_cells: Episode-wide tracking (for stats)
        - position_history: Sliding window for revisit penalty
        - position_counts: Visit counts for current window
        """
        prev_size = len(self.visited_cells_current_phase)
        self.visited_cells_current_phase.clear()
        self.cells_visited_current_phase = 0
        self.hierarchical_resets += 1

        logger.debug(
            f"[POSITION_TRACKER_HIERARCHICAL] Goal change detected, "
            f"reset {prev_size} cells for Phase 2 exploration "
            f"(total resets: {self.hierarchical_resets})"
        )

    def reset(self):
        """Reset tracking for new episode."""
        self.visited_cells.clear()
        self.visited_cells_current_phase.clear()
        self.position_history.clear()
        self.position_counts.clear()
        self.cells_visited_this_episode = 0
        self.cells_visited_current_phase = 0


class RewardCalculator:
    """Policy-invariant reward calculator using true PBRS for path guidance.

    Components:
    - Terminal rewards: Always active (completion, death, switch milestone)
    - PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s) for dense, policy-invariant path guidance
    - Time penalty: Efficiency pressure (curriculum-controlled)

    PBRS Implementation:
    - Uses potential function Φ(s) based on shortest path distance to objective
    - Applies Ng et al. (1999) formula with γ=1.0 for heuristic potentials
    - Guarantees policy invariance while providing dense reward gradients
    - Automatically rewards distance decreases and penalizes distance increases
    - Curriculum-aware weight and normalization scaling

    PBRS Gamma Choice (γ=1.0):
    - Eliminates negative bias from (γ-1)*Σ Φ term
    - For heuristic potentials, γ=1.0 is standard (not tied to MDP discount)
    - Accumulated PBRS = Φ(goal) - Φ(start) exactly (clean telescoping)
    - Policy invariance holds for ANY γ (Ng et al. 1999 Theorem 1)

    Removed components (policy-invariant alternative):
    - Discrete achievement bonuses (replaced with dense PBRS)
    - Episode length normalization (violated Markov property)
    - Per-episode caps (violated Markov property)
    - Physics discovery rewards (disabled for clean path focus)
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        shared_level_cache: Optional[Any] = None,
    ):
        """Initialize simplified reward calculator.

        Args:
            reward_config: RewardConfig instance managing curriculum-aware component lifecycle
            pbrs_gamma: Discount factor for PBRS (γ in F(s,s') = γ * Φ(s') - Φ(s))
            shared_level_cache: Optional SharedLevelCache for zero-copy multi-worker training
        """
        # Single config object manages all curriculum logic
        self.config = reward_config or RewardConfig()

        # OPTIMIZATION: Cache curriculum properties per episode to avoid @property overhead
        # Properties are episode-invariant but called many times per step
        self._cached_velocity_alignment_weight: Optional[float] = None
        self._cached_mine_hazard_multiplier: Optional[float] = None
        self._cached_pbrs_objective_weight: Optional[float] = None
        self._episode_config_cache_valid: bool = False

        # PBRS configuration (kept for backwards compatibility with path calculator)
        self.pbrs_gamma = pbrs_gamma

        # Create path calculator for PBRS (with optional shared cache)
        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            use_astar=True,
            shared_level_cache=shared_level_cache,
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

        # Step-level component tracking for TensorBoard callback
        # Always populated after calculate_reward() for info dict logging
        self.last_pbrs_components = {}

        # Position tracking for revisit penalty and hierarchical exploration
        # UPDATED: 12px grid (matches graph sub-node resolution) with hierarchical reset
        self.position_tracker = PositionTracker(grid_size=12, window_size=100)

        # Oscillation detection: Track recent positions for net displacement check
        # Detects when agent moves but doesn't make net progress (oscillating)
        self.position_history: List[Tuple[float, float]] = []
        self.oscillation_penalty_count = (
            0  # Count of oscillation penalties this episode
        )

        # Progress preservation tracking
        # Tracks closest distance to goal and penalizes regression
        self.closest_distance_this_episode = float("inf")

        # Spawn position tracking for displacement gate
        # Positive PBRS only applies if agent has moved from spawn
        # Prevents reward hacking via oscillation/jumping in place
        self.spawn_position = None  # Set on first calculate_reward() call

        # Path-based waypoint system (always enabled for simplified unified system)
        # Provides immediate dense guidance from first episode
        from .path_waypoint_extractor import PathWaypointExtractor

        self.path_waypoint_extractor = PathWaypointExtractor(
            progress_spacing=self.config.path_waypoint_progress_spacing,
            min_turn_angle=self.config.path_waypoint_min_angle,
            cluster_radius=self.config.path_waypoint_cluster_radius,
            segment_min_length=150.0,  # Minimum length for segment midpoints
        )
        logger.info(
            "Path-based waypoint system enabled (dense guidance from optimal paths)"
        )

        # Storage for current level's path waypoints
        self.current_path_waypoints: List[Any] = []  # List[PathWaypoint]
        self.current_path_waypoints_by_phase: Dict[str, List[Any]] = {
            "pre_switch": [],
            "post_switch": [],
        }

        # Demo waypoint storage (seeded from expert demonstrations)
        # Merged with path waypoints for unified PBRS routing
        self.demo_waypoints_by_phase: Dict[
            str, List[Tuple[Tuple[float, float], float]]
        ] = {
            "pre_switch": [],
            "post_switch": [],
        }

        # Episode-level waypoint bonus tracking (prevents exploitation)
        # Cap ensures waypoints guide but don't dominate terminal rewards
        self.waypoint_bonus_this_episode = 0.0
        self.max_waypoint_bonus_per_episode = 20.0  # 40% of completion reward (50.0)

        # Sequential waypoint tracking for soft guidance (allows novel paths)
        # Tracks expected next waypoint for sequence multiplier calculation
        self._next_expected_waypoint_index: int = 0  # Next waypoint in optimal sequence
        self._waypoint_sequence: List[Any] = []  # Ordered waypoints by node_index
        self._last_collected_waypoint: Optional[Any] = None  # Last collected waypoint
        self._frames_since_last_collection: int = 0  # For exit direction bonus timing

        # Track diagnostic data for PBRS visualization (Phase 1 enhancement)
        self.velocity_history: List[Tuple[float, float]] = []
        self.alignment_history: List[float] = []
        self.path_gradient_history: List[Optional[Tuple[float, float]]] = []

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
        frames_executed: int = 1,
    ) -> float:
        """Calculate policy-invariant reward using true PBRS.

        Components:
        1. Terminal rewards (always active)
        2. Time penalty (curriculum-controlled via config, scaled by frames_executed)
        3. PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s) for dense path guidance

        PBRS ensures:
        - Dense reward signal at every step (not just improvements)
        - Automatic penalty for distance increases (backtracking)
        - Policy invariance (optimal policy unchanged)
        - Markov property (reward depends only on current state transition)

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
            obs: Current game state (must include _adjacency_graph, level_data)
            prev_obs: Previous game state
            action: Action taken (optional, unused)
            frames_executed: Number of frames executed in this action (for time penalty scaling)

        Returns:
            float: Total reward for the transition
        """
        self.steps_taken += frames_executed

        # OPTIMIZATION: Clear step-level cache at start of each step
        # This cache avoids duplicate distance calculations within the same step
        # (e.g., switch distance computed for PBRS is reused for diagnostics)
        if hasattr(self.pbrs_calculator, "path_calculator"):
            self.pbrs_calculator.path_calculator.clear_step_cache()

            # DIAGNOSTIC: Log cache statistics every 100 steps
            if self.steps_taken % 100 == 0 and self.steps_taken > 0:
                stats = self.pbrs_calculator.path_calculator.get_statistics()
                level_cache_stats = stats.get("level_cache", {})
                logger.info(
                    f"[CACHE_STATS @ step {self.steps_taken}] "
                    f"Step-level: {stats.get('step_cache_hits', 0)}/{stats.get('step_cache_hits', 0) + stats.get('step_cache_misses', 0)} "
                    f"({stats.get('step_cache_hit_rate', 0):.1%}) | "
                    f"Level: {level_cache_stats.get('geometric_hits', 0)}/{level_cache_stats.get('geometric_hits', 0) + level_cache_stats.get('geometric_misses', 0)} "
                    f"({level_cache_stats.get('geometric_hit_rate', 0):.1%})"
                )

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

        # === HIERARCHICAL EXPLORATION RESET ===
        # When switch activates, reset position tracker's phase-specific exploration tracking
        # This treats Phase 2 (→exit) as a fresh exploration problem
        if switch_just_activated:
            self.position_tracker.reset_for_goal_change()

        # === TERMINAL REWARDS (always active) ===
        # Get level_data for validation and pathfinding
        level_data = obs.get("level_data")

        # Death penalties - differentiated by cause for better learning signal
        if obs.get("player_dead", False):
            death_cause = obs.get("death_cause", None)

            # Determine base penalty based on death cause
            if death_cause == "impact":
                # High-velocity collision with ceiling/floor
                base_penalty = IMPACT_DEATH_PENALTY
            elif death_cause in ("mine", "drone", "thwump", "hazard"):
                # Contact with deadly entities (highly preventable)
                base_penalty = HAZARD_DEATH_PENALTY
            else:
                # Generic/unknown death cause
                base_penalty = DEATH_PENALTY

            # === PROGRESS-GATED DEATH PENALTY ===
            # Scale death penalty by progress made to allow safe exploration in early game
            # This prevents the death penalty from overwhelming PBRS gradients before
            # the agent has a chance to learn from forward progress.
            #
            # Justification:
            # - Early deaths (progress < 20%): 25% penalty = -10 instead of -40
            # - Mid deaths (20% < progress < 50%): 50% penalty = -20 instead of -40
            # - Late deaths (progress > 50%): Full penalty = -40
            #
            # This allows exploration without severe punishment while maintaining
            # full penalty when agent has demonstrated it can reach dangerous areas.
            combined_path_distance = obs.get("_pbrs_combined_path_distance", 1000.0)
            current_distance = self.closest_distance_this_episode

            # Calculate progress: 0.0 at spawn, 1.0 at goal
            if combined_path_distance > 0 and current_distance != float("inf"):
                # Progress = 1 - (closest_distance / combined_path)
                # Use closest distance this episode to reward any forward progress made
                progress = 1.0 - (current_distance / combined_path_distance)
                progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
            else:
                progress = 0.0  # No progress data available

            # Apply progress-based scaling
            if progress < 0.2:
                # Early exploration: 25% of base penalty
                penalty_scale = 0.25
            elif progress < 0.5:
                # Mid exploration: 50% of base penalty
                penalty_scale = 0.50
            else:
                # Late game: Full penalty
                penalty_scale = 1.0

            terminal_reward = base_penalty * penalty_scale

            # Add milestone reward to terminal reward (switch activated before death)
            total_terminal = terminal_reward + milestone_reward
            self.episode_terminal_reward = total_terminal

            # Apply global reward scaling for value function stability
            scaled_reward = total_terminal * GLOBAL_REWARD_SCALE

            # Populate PBRS components for TensorBoard logging (terminal state)
            # Log UNSCALED values for interpretability
            self.last_pbrs_components = {
                "terminal_reward": terminal_reward,
                "milestone_reward": milestone_reward,
                "pbrs_reward": 0.0,
                "time_penalty": 0.0,
                "revisit_penalty": 0.0,
                "total_reward": total_terminal,
                "scaled_reward": scaled_reward,
                "is_terminal": True,
                "terminal_type": "death",
                "death_cause": death_cause or "unknown",
                "switch_activated_on_death": switch_just_activated,
                # Progress-gated death penalty diagnostics
                "death_progress": progress,
                "death_penalty_scale": penalty_scale,
                "death_base_penalty": base_penalty,
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

        current_potential = self.pbrs_calculator.calculate_combined_potential(
            state=obs,
            adjacency=adjacency,
            level_data=level_data,
            graph_data=graph_data,
            objective_weight=objective_weight,
            scale_factor=self.config.pbrs_normalization_scale,
        )

        # Track actual path length for efficiency metrics
        current_player_pos = (obs["player_x"], obs["player_y"])
        movement_distance = 0.0
        ineffective_action_penalty = 0.0
        if self.prev_player_pos is not None:
            # Calculate Euclidean distance moved (approximation of actual path)
            dx = current_player_pos[0] - self.prev_player_pos[0]
            dy = current_player_pos[1] - self.prev_player_pos[1]
            movement_distance = (dx * dx + dy * dy) ** 0.5
            self.episode_path_length += movement_distance

            # === INEFFECTIVE ACTION PENALTY ===
            # Penalize actions that produce minimal displacement (targets stationary behavior)
            # Cannot be exploited: moving is always better than not moving
            if movement_distance < INEFFECTIVE_ACTION_THRESHOLD:
                ineffective_action_penalty = INEFFECTIVE_ACTION_PENALTY
                reward += ineffective_action_penalty
                logger.debug(
                    f"Ineffective action: displacement {movement_distance:.3f}px < "
                    f"{INEFFECTIVE_ACTION_THRESHOLD}px, penalty: {ineffective_action_penalty}"
                )

        # Track velocity for visualization (Phase 1)
        if self.prev_player_pos is not None and movement_distance > 0:
            dx = current_player_pos[0] - self.prev_player_pos[0]
            dy = current_player_pos[1] - self.prev_player_pos[1]
            self.velocity_history.append((dx, dy))
        else:
            self.velocity_history.append((0.0, 0.0))

        self.prev_player_pos = current_player_pos

        # === OSCILLATION DETECTION ===
        # Track positions over sliding window to detect oscillation (moving but no net progress)
        # This catches behavior that escapes ineffective action penalty:
        # e.g., LEFT 2px, RIGHT 2px, LEFT 2px... = movement but 0 net progress
        oscillation_penalty = 0.0
        self.position_history.append(current_player_pos)

        # Only check once we have enough history
        if len(self.position_history) >= OSCILLATION_WINDOW:
            # Calculate net displacement over window
            oldest_pos = self.position_history[-OSCILLATION_WINDOW]
            net_dx = current_player_pos[0] - oldest_pos[0]
            net_dy = current_player_pos[1] - oldest_pos[1]
            net_displacement = (net_dx * net_dx + net_dy * net_dy) ** 0.5

            # Apply penalty if net displacement is too low
            if net_displacement < NET_DISPLACEMENT_THRESHOLD:
                oscillation_penalty = OSCILLATION_PENALTY
                reward += oscillation_penalty
                self.oscillation_penalty_count += 1
                logger.debug(
                    f"Oscillation detected: net displacement {net_displacement:.2f}px < "
                    f"{NET_DISPLACEMENT_THRESHOLD}px over {OSCILLATION_WINDOW} actions. "
                    f"Penalty: {oscillation_penalty}"
                )

            # Keep window size bounded (remove oldest when over limit)
            if len(self.position_history) > OSCILLATION_WINDOW * 2:
                self.position_history = self.position_history[-OSCILLATION_WINDOW:]

        # Store optimal path length (combined path distance) on first step
        if self.optimal_path_length is None:
            self.optimal_path_length = obs.get("_pbrs_combined_path_distance", 0.0)

        # Calculate PBRS shaping reward: F(s,s') = γ * Φ(s') - Φ(s)
        pbrs_reward = 0.0

        # === HIERARCHICAL PBRS (Phase 3.1) ===
        # Continuous potential across switch activation - no reset needed
        # The calculate_combined_potential already handles both phases internally
        # by selecting appropriate goal (switch or exit) based on switch_activated
        #
        # OLD APPROACH (caused LSTM learning issues):
        #   if switch_just_activated: self.prev_potential = None
        #
        # NEW APPROACH: Never reset, continuous gradient field
        #   - Switch phase: potential based on switch distance
        #   - Exit phase: potential based on exit distance
        #   - Both are part of same hierarchical task, no discontinuity needed
        #   - LSTM learns temporal dependencies across full episode
        # Track potential change for diagnostics (must be computed before updating prev_potential)
        actual_potential_change = 0.0

        if self.prev_potential is not None:
            # Apply PBRS formula for policy-invariant shaping
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential

            # CRITICAL: Save potential change BEFORE updating prev_potential for correct logging
            actual_potential_change = current_potential - self.prev_potential

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
                        logger.info(
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

        # Track displacement from spawn for logging (gate removed - PBRS always active)
        # Previous displacement gate created "reward trap" near spawn:
        # - 10px threshold required 3-20 actions to escape (typical movement 0.5-3px)
        # - No PBRS in neutral zone made oscillation appear optimal
        # - PBRS is policy-invariant: backtracking naturally penalized via potential decrease
        displacement_from_spawn = 0.0
        if self.spawn_position is not None:
            dx = current_player_pos[0] - self.spawn_position[0]
            dy = current_player_pos[1] - self.spawn_position[1]
            displacement_from_spawn = (dx * dx + dy * dy) ** 0.5

        # === OSCILLATION AND STUCK DETECTION DIAGNOSTIC ===
        # Oscillation ratio: high ratio = lots of movement but no net progress
        # This is a key diagnostic for policy collapse where agent oscillates in place
        if displacement_from_spawn > 1.0:
            oscillation_ratio = self.episode_path_length / displacement_from_spawn
        else:
            oscillation_ratio = self.episode_path_length  # Avoid div by zero

        # Store for TensorBoard logging
        obs["_oscillation_ratio"] = oscillation_ratio
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

        # === VELOCITY ALIGNMENT REWARD (separate from PBRS potential) ===
        # CRITICAL FIX 2025-12-15: Moved from PBRS potential to separate reward component
        #
        # Problem: When velocity was part of PBRS potential, direction changes created
        # large rewards independent of progress:
        #   - Change from alignment=+1 to alignment=-1: potential swing of 2×weight (16.0!)
        #   - This incentivized oscillating directions near spawn
        #
        # Solution: Apply velocity alignment as INSTANTANEOUS reward (not differenced):
        #   - reward = velocity_weight × alignment (range: [-weight, +weight])
        #   - No differencing artifact from direction changes
        #   - Pure directional guidance signal
        #
        # This is semantically correct because:
        #   - PBRS potential Φ(s) = progress toward goal (position-based)
        #   - Velocity reward = instantaneous direction correctness
        #   - Both guide agent, but only PBRS uses differencing
        velocity_alignment_reward = 0.0
        velocity_alignment_value = obs.get("_pbrs_velocity_alignment", 0.0)
        velocity_weight = obs.get("_pbrs_velocity_weight", 0.0)

        if velocity_weight > 0 and abs(velocity_alignment_value) > 0.01:
            # Apply as instantaneous reward (not differenced like PBRS)
            velocity_alignment_reward = velocity_weight * velocity_alignment_value
            reward += velocity_alignment_reward

            logger.debug(
                f"Velocity alignment reward: alignment={velocity_alignment_value:.3f}, "
                f"weight={velocity_weight:.2f}, reward={velocity_alignment_reward:.4f}"
            )

        # === WAYPOINT BONUS (path-based, always enabled) ===
        # Reward reaching waypoints without penalizing novel paths
        waypoint_bonus = 0.0
        waypoint_metrics = {
            "sequence_multiplier": 0.0,
            "collection_type": "none",
            "skip_distance": 0,
        }

        if self.current_path_waypoints:
            # Use path-based waypoints (dense, deterministic from optimal paths)
            waypoint_bonus, waypoint_metrics = self._get_path_waypoint_bonus(
                current_pos=current_player_pos,
                previous_pos=self.prev_player_pos,
                switch_activated=obs.get("switch_activated", False),
            )

        # Apply episode-level cap to prevent waypoint bonus from dominating
        # Completion (50.0) >> Total waypoints (20.0 cap) >> Individual waypoint (~0.5)
        remaining_budget = (
            self.max_waypoint_bonus_per_episode - self.waypoint_bonus_this_episode
        )
        waypoint_bonus = min(waypoint_bonus, max(0.0, remaining_budget))
        self.waypoint_bonus_this_episode += waypoint_bonus

        reward += waypoint_bonus

        # === WAYPOINT APPROACH GRADIENT (continuous guidance) ===
        # Small continuous reward for moving toward next uncollected waypoint
        # Provides dense signal between waypoint collections
        waypoint_approach_gradient = 0.0
        if self.current_path_waypoints:
            current_phase = (
                "post_switch" if obs.get("switch_activated", False) else "pre_switch"
            )
            phase_waypoints = self.current_path_waypoints_by_phase.get(
                current_phase, []
            )

            waypoint_approach_gradient = self._get_waypoint_approach_gradient(
                current_pos=current_player_pos,
                previous_pos=self.prev_player_pos,
                phase_waypoints=phase_waypoints,
            )
            reward += waypoint_approach_gradient

        # === EXIT DIRECTION BONUS (post-collection continuation) ===
        # Small bonus for continuing in correct direction after collecting waypoint
        # Encourages agent to keep moving along path rather than stopping
        waypoint_exit_bonus = 0.0
        if self._last_collected_waypoint is not None:
            # Increment frames counter
            self._frames_since_last_collection += 1

            # Calculate exit direction bonus
            current_velocity = (obs["player_xspeed"], obs["player_yspeed"])
            waypoint_exit_bonus = self._get_exit_direction_bonus(current_velocity)
            reward += waypoint_exit_bonus

        # === WIRE WAYPOINTS TO PBRS FOR MULTI-STAGE ROUTING ===
        # Update PBRS calculator with waypoints for potential field routing
        # This enables potential field: current → waypoint → goal
        # Critical for inflection point navigation (e.g., go LEFT before RIGHT)

        if self.current_path_waypoints:
            # Get merged waypoints (path + demo) for current phase
            current_phase = (
                "post_switch" if obs.get("switch_activated", False) else "pre_switch"
            )
            path_wps = self.current_path_waypoints_by_phase.get(current_phase, [])
            demo_wps = self.demo_waypoints_by_phase.get(current_phase, [])

            # Merge waypoints (demo values enhance path waypoints)
            phase_waypoints = self._merge_waypoints(
                path_wps, demo_wps, cluster_radius=30.0
            )

            if phase_waypoints:
                # Determine current goal position
                if not obs.get("switch_activated", False):
                    waypoint_goal_pos = (obs["switch_x"], obs["switch_y"])
                else:
                    waypoint_goal_pos = (obs["exit_door_x"], obs["exit_door_y"])

                # Convert to momentum waypoint format
                momentum_waypoints = self._convert_path_waypoints_to_momentum(
                    phase_waypoints,
                    obs.get("switch_activated", False),
                    waypoint_goal_pos,
                )

                # Update PBRS calculator if waypoint count changed
                current_momentum_count = len(
                    self.pbrs_calculator.momentum_waypoints or []
                )
                if len(momentum_waypoints) != current_momentum_count:
                    self.pbrs_calculator.set_momentum_waypoints(
                        momentum_waypoints, waypoint_source="path"
                    )
                    logger.debug(
                        f"Updated PBRS with {len(momentum_waypoints)} path waypoints "
                        f"(phase: {current_phase})"
                    )

        # === POSITION TRACKING (revisit penalty + hierarchical exploration bonus) ===
        # Exploration bonus: PBRS-gated, hierarchical (resets on switch activation)
        # NOTE: Bonus disabled here (0.0) - using HierarchicalExplorationCallback instead
        # This avoids double-counting exploration bonuses at both reward and training levels
        (
            position_reward,
            exploration_reward,
            revisit_penalty,
            is_new_cell,
        ) = self.position_tracker.get_position_reward(
            position=(obs["player_x"], obs["player_y"]),
            revisit_penalty_weight=self.config.revisit_penalty_weight,
            exploration_bonus=0.0,  # DISABLED - using HierarchicalExplorationCallback
            pbrs_reward=pbrs_reward,  # Gate: only award if PBRS > 0
        )

        reward += position_reward

        # === VELOCITY-DIRECTION ALIGNMENT BONUS (Phase 2 Fix 4) ===
        # CRITICAL FIX 2025-12-09: Add reward when velocity aligns with optimal path direction
        # TensorBoard (npp-logs-129) showed agent moving RIGHT into mines instead of UP-LEFT
        # This bonus provides explicit reward for moving in the correct DIRECTION, not just
        # reducing distance. Essential for learning counter-intuitive navigation (opposite
        # to previous path hop).
        alignment_bonus = 0.0
        # Validate all required keys exist before computing alignment bonus
        if (
            "_next_hop_dir_x" in obs
            and "_next_hop_dir_y" in obs
            and "player_x" in obs
            and "player_y" in obs
            and "player_x" in prev_obs
            and "player_y" in prev_obs
        ):
            # Get next hop direction from graph (optimal path direction)
            next_hop_dir_x = obs["_next_hop_dir_x"]
            next_hop_dir_y = obs["_next_hop_dir_y"]

            # Calculate velocity from position change
            current_x = obs["player_x"]
            current_y = obs["player_y"]
            prev_x = prev_obs["player_x"]
            prev_y = prev_obs["player_y"]

            velocity_x = current_x - prev_x
            velocity_y = current_y - prev_y
            velocity_magnitude = (
                velocity_x * velocity_x + velocity_y * velocity_y
            ) ** 0.5

            # Only apply bonus when agent is moving (velocity > 1.0 pixels)
            if velocity_magnitude > 1.0:
                # Normalize velocity
                velocity_norm_x = velocity_x / velocity_magnitude
                velocity_norm_y = velocity_y / velocity_magnitude

                # Compute dot product: alignment in [-1, 1]
                # +1 = moving in optimal direction, -1 = moving opposite, 0 = perpendicular
                alignment = (
                    velocity_norm_x * next_hop_dir_x + velocity_norm_y * next_hop_dir_y
                )

                # Add bonus for positive alignment (moving toward goal)
                # Scale by 0.5 for moderate guidance, don't dominate PBRS
                if alignment > 0.0:
                    alignment_bonus = 0.5 * alignment * GLOBAL_REWARD_SCALE
                    reward += alignment_bonus

                    # Log when alignment is strong (helps debug navigation)
                    if alignment > 0.7 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"[ALIGNMENT_BONUS] Strong alignment: {alignment:.2f}, "
                            f"velocity=({velocity_x:.1f},{velocity_y:.1f}), "
                            f"next_hop=({next_hop_dir_x:.2f},{next_hop_dir_y:.2f}), "
                            f"bonus={alignment_bonus:.4f}"
                        )

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
        # Removed: progress_penalty was redundant with PBRS negative rewards
        progress_penalty = 0.0  # Kept for logging compatibility
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
        # This provides detailed breakdown of reward components for analysis
        # (milestone_reward was computed above, before terminal rewards check)

        # Extract diagnostic metrics from observation (set by objective_distance_potential)
        normalized_distance = obs.get("_pbrs_normalized_distance", 0.0)
        combined_path_distance = obs.get("_pbrs_combined_path_distance", 0.0)

        self.last_pbrs_components = {
            "pbrs_reward": pbrs_reward,
            "time_penalty": time_penalty,  # Use scaled penalty (per action, not per frame)
            "milestone_reward": milestone_reward,
            "revisit_penalty": revisit_penalty,
            "exploration_reward": exploration_reward,  # Hierarchical exploration bonus
            "position_reward": position_reward,  # Combined exploration + revisit
            "waypoint_bonus": waypoint_bonus,  # Phase 3: Adaptive waypoint bonus
            "alignment_bonus": alignment_bonus,  # Phase 2 Fix 4: Velocity-direction alignment
            "velocity_alignment_reward": velocity_alignment_reward,  # Phase 4 Fix: Separate from PBRS potential
            "exploration_cells_visited": len(self.position_tracker.visited_cells),
            "exploration_cells_current_phase": len(
                self.position_tracker.visited_cells_current_phase
            ),
            # Waypoint system metrics (soft sequential guidance)
            "waypoint_sequence_multiplier": waypoint_metrics.get(
                "sequence_multiplier", 0.0
            ),
            "waypoint_collection_type": waypoint_metrics.get("collection_type", "none"),
            "waypoint_skip_distance": waypoint_metrics.get("skip_distance", 0),
            "waypoint_approach_gradient": waypoint_approach_gradient,
            "waypoint_exit_bonus": waypoint_exit_bonus,
            "waypoints_collected_episode": len(
                getattr(self, "_collected_path_waypoints", set())
            ),
            "waypoints_total_available": len(self._waypoint_sequence),
            "next_expected_waypoint_index": self._next_expected_waypoint_index,
            "total_reward": reward,
            "is_terminal": False,
            # Include potential information for debugging
            "current_potential": current_potential
            if self.current_potential is not None
            else 0.0,
            "prev_potential": self.prev_potential
            if self.prev_potential is not None
            else 0.0,
            "potential_change": actual_potential_change,  # FIX: Use saved value from before prev_potential update
            # Enhanced diagnostic fields for PBRS verification
            "potential_gradient": abs(actual_potential_change),  # FIX: Use saved value
            "pbrs_gamma": self.pbrs_gamma,
            "theoretical_pbrs": actual_potential_change,  # FIX: For γ=1.0, this should equal pbrs_reward
            # New diagnostic fields for monitoring and analysis
            "distance_to_goal": distance_to_goal,
            "combined_path_distance": combined_path_distance,
            "normalized_distance": normalized_distance,
            # Frame skip information
            "frames_executed": frames_executed,
            # Position tracking
            "unique_positions_visited": len(self.position_tracker.position_counts),
            "total_position_visits": len(self.position_tracker.position_history),
            # Progress tracking
            "progress_penalty": progress_penalty,
            "closest_distance_episode": self.closest_distance_this_episode,
            # Displacement tracking (gate removed, kept for logging)
            "displacement_from_spawn": displacement_from_spawn,
            "displacement_threshold": PBRS_DISPLACEMENT_THRESHOLD,
            # Ineffective action penalty (catches stationary behavior)
            "ineffective_action_penalty": ineffective_action_penalty,
            "movement_distance": movement_distance,
            "ineffective_action_threshold": INEFFECTIVE_ACTION_THRESHOLD,
            # Oscillation penalty (catches moving but not progressing)
            "oscillation_penalty": oscillation_penalty,
            "oscillation_penalty_count": self.oscillation_penalty_count,
            # === PBRS PROFILING DATA ===
            # Timing data for performance analysis
            "_pbrs_cache_hit": obs.get("_pbrs_cache_hit", False),
            # === PBRS DIRECTION AND STUCK DIAGNOSTICS ===
            # For debugging policy collapse and wrong-direction PBRS
            "euclidean_alignment": obs.get("_pbrs_euclidean_alignment", 0.0),
            "path_euclidean_ratio": obs.get("_pbrs_path_euclidean_ratio", 1.0),
            "oscillation_ratio": obs.get("_oscillation_ratio", 0.0),
        }

        # === REWARD BOUNDS SAFEGUARD ===
        # Clamp reward to prevent extreme values that could cause gradient explosion
        # TensorBoard analysis showed policy_loss exploded to 25 million due to unstable rewards
        # from PBRS bugs (e.g., infinity path distances). Clamping provides defense-in-depth.
        # Bounds [-100, 100] are generous enough for any valid reward composition:
        #   Max positive: completion(50) + switch(5) + PBRS(15) + efficiency(25) = 95
        #   Max negative: death(-12) + time(-12) + revisit(-50) + oscillation(-20) = -94
        REWARD_CLAMP_MIN = -100.0
        REWARD_CLAMP_MAX = 100.0
        if reward < REWARD_CLAMP_MIN or reward > REWARD_CLAMP_MAX:
            logger.warning(
                f"Reward {reward:.2f} exceeds bounds, clamping to "
                f"[{REWARD_CLAMP_MIN}, {REWARD_CLAMP_MAX}]. "
                f"This may indicate a bug in reward calculation."
            )
            reward = max(REWARD_CLAMP_MIN, min(REWARD_CLAMP_MAX, reward))

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

            combined_dist = switch_dist + exit_dist
            logger.info(
                f"✓ Level validated as solvable:\n"
                f"   Spawn→Switch: {switch_dist:.0f}px\n"
                f"   Switch→Exit: {exit_dist:.0f}px\n"
                f"   Combined: {combined_dist:.0f}px\n"
                f"   Expected PBRS (weight={self.config.pbrs_objective_weight}): "
                f"{self.config.pbrs_objective_weight * (1.0 - 0.0):.1f}"
            )
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
        self._cached_velocity_alignment_weight = self.config.velocity_alignment_weight
        self._cached_mine_hazard_multiplier = self.config.mine_hazard_cost_multiplier
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

        # Reset unified position tracking (exploration + revisit)
        self.position_tracker.reset()

        # Reset oscillation detection
        self.position_history = []
        self.oscillation_penalty_count = 0

        # Reset PBRS calculator state
        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()

        # Reset progress tracking
        self.closest_distance_this_episode = float("inf")

        # Reset spawn position tracking for displacement gate
        self.spawn_position = None

        # Reset distance tracking for adaptive gating (Phase 2 enhancement)
        self._prev_distance_to_goal = None

        # Reset path waypoint collection tracking
        if hasattr(self, "_collected_path_waypoints"):
            self._collected_path_waypoints.clear()
        else:
            self._collected_path_waypoints = set()

        # Reset waypoint bonus budget for new episode
        self.waypoint_bonus_this_episode = 0.0

        # Reset sequential waypoint tracking for soft guidance
        self._next_expected_waypoint_index = 0
        self._last_collected_waypoint = None
        self._frames_since_last_collection = 0

        # Reset diagnostic tracking for visualization (Phase 1 enhancement)
        self.velocity_history = []
        self.alignment_history = []
        self.path_gradient_history = []
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
            logger.info(
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

    def get_waypoint_statistics(self) -> Dict[str, Any]:
        """Get waypoint system statistics for monitoring.

        Returns:
            Dictionary with waypoint statistics
        """
        stats = {}

        # Path waypoint statistics (always enabled in unified system)
        if self.path_waypoint_extractor:
            path_stats = self.path_waypoint_extractor.get_statistics()
            stats["path_waypoints"] = path_stats
            stats["path_waypoints_active"] = len(self.current_path_waypoints)
            stats["path_waypoints_collected"] = len(
                getattr(self, "_collected_path_waypoints", set())
            )

        return stats

    def extract_path_waypoints_for_level(
        self,
        level_data: Any,
        graph_data: Dict[str, Any],
        map_name: str,
    ) -> int:
        """Extract path-based waypoints from optimal A* paths for current level.

        Called during episode reset after graph is built. Extracts dense waypoints
        from spawn→switch and switch→exit paths for immediate guidance.

        Args:
            level_data: LevelData object with tiles and entities
            graph_data: Graph data dict with adjacency, physics_cache
            map_name: Level name for caching

        Returns:
            Number of waypoints extracted
        """
        if self.path_waypoint_extractor is None:
            return 0

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
                logger.warning("Cannot extract path waypoints: missing graph data")
                return 0

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
                return 0

            # Get switch position
            exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
            if not exit_switches:
                logger.warning("No exit switch found in level data")
                return 0
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
                return 0

            # Get exit door position
            exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
            if not exit_doors:
                logger.warning("No exit door found in level data")
                return 0
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
                return 0

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
                return 0

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
                return 0

            # Extract waypoints from both path segments
            waypoints = self.path_waypoint_extractor.extract_waypoints_from_paths(
                spawn_to_switch_path=spawn_to_switch_path,
                switch_to_exit_path=switch_to_exit_path,
                physics_cache=physics_cache,
                level_id=map_name,
                use_cache=True,
            )

            # Store waypoints
            self.current_path_waypoints = waypoints

            # Separate by phase for efficient filtering
            self.current_path_waypoints_by_phase = {
                "pre_switch": [wp for wp in waypoints if wp.phase == "pre_switch"],
                "post_switch": [wp for wp in waypoints if wp.phase == "post_switch"],
            }

            # Initialize waypoint sequence for soft sequential guidance
            # Sort all waypoints by node_index for sequential tracking
            self._waypoint_sequence = sorted(waypoints, key=lambda wp: wp.node_index)
            self._next_expected_waypoint_index = 0  # Start from first waypoint

            # Log spawn-proximity waypoint stats for diagnostics
            MIN_SPAWN_DISTANCE = 50.0
            spawn_pos = level_data.start_position

            near_spawn_count = 0
            for wp in waypoints:
                dx = wp.position[0] - spawn_pos[0]
                dy = wp.position[1] - spawn_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < MIN_SPAWN_DISTANCE:
                    near_spawn_count += 1

            if near_spawn_count > 0:
                logger.debug(
                    f"Extracted {len(waypoints)} waypoints, {near_spawn_count} within "
                    f"{MIN_SPAWN_DISTANCE}px of spawn (will not award bonus)"
                )

            logger.info(
                f"Extracted {len(waypoints)} path waypoints for level '{map_name}': "
                f"{len(self.current_path_waypoints_by_phase['pre_switch'])} pre-switch, "
                f"{len(self.current_path_waypoints_by_phase['post_switch'])} post-switch"
            )

            # CRITICAL: Wire waypoints to PBRS path calculator immediately after extraction
            # This ensures waypoints are cached BEFORE first calculate_reward() call
            # Without this, geometric distance calculations in PBRS potential have no waypoints cached
            if waypoints:
                # Use pre_switch waypoints for initial wiring (will update based on phase during calculate_reward)
                initial_waypoints = self.current_path_waypoints_by_phase.get(
                    "pre_switch", []
                )
                if initial_waypoints:
                    # Convert to momentum waypoint format (use placeholder goal position)
                    # This will be updated properly during calculate_reward() based on actual switch state
                    from ...constants.entity_types import EntityType

                    exit_switches = level_data.get_entities_by_type(
                        EntityType.EXIT_SWITCH
                    )
                    if exit_switches:
                        switch_pos = (
                            int(exit_switches[0].get("x", 0)),
                            int(exit_switches[0].get("y", 0)),
                        )
                        momentum_waypoints = self._convert_path_waypoints_to_momentum(
                            initial_waypoints,
                            switch_activated=False,
                            goal_pos=switch_pos,
                        )
                        self.pbrs_calculator.set_momentum_waypoints(
                            momentum_waypoints, waypoint_source="path"
                        )
                        logger.info(
                            f"Wired {len(momentum_waypoints)} pre-switch waypoints to PBRS calculator "
                            f"(will update based on phase during episode)"
                        )

            return len(waypoints)

        except Exception as e:
            logger.warning(f"Failed to extract path waypoints: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return 0

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

    def _find_nearest_uncollected_waypoint(
        self,
        phase_waypoints: List[Any],
        current_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[Any]:
        """Find nearest uncollected waypoint by spatial distance.

        BUG FIX: Previously sorted by node_index proximity (sequence position on path),
        which could return waypoints 500px+ away when there was a waypoint 14px away.
        This caused the approach gradient to always be 0 (outside 100px radius).

        Now sorts by spatial distance to agent position for continuous dense guidance.

        Args:
            phase_waypoints: List of waypoints for current phase
            current_pos: Current agent position for spatial sorting (required for approach gradient)

        Returns:
            Spatially nearest uncollected waypoint, or None if all collected
        """
        uncollected = [
            wp
            for wp in phase_waypoints
            if wp.position not in getattr(self, "_collected_path_waypoints", set())
        ]

        if not uncollected:
            return None

        # BUG FIX: Sort by SPATIAL distance, not node_index proximity
        # This ensures approach gradient targets actually nearby waypoints
        if current_pos is not None:
            uncollected.sort(
                key=lambda wp: (wp.position[0] - current_pos[0]) ** 2
                + (wp.position[1] - current_pos[1]) ** 2
            )
        else:
            # Fallback: sort by node_index if no position provided (backward compatibility)
            uncollected.sort(
                key=lambda wp: abs(wp.node_index - self._next_expected_waypoint_index)
            )
        return uncollected[0]

    def _get_waypoint_approach_gradient(
        self,
        current_pos: Tuple[float, float],
        previous_pos: Optional[Tuple[float, float]],
        phase_waypoints: List[Any],
    ) -> float:
        """Calculate continuous gradient toward nearest uncollected waypoint.

        Provides dense guidance signal between waypoint collections without
        blocking novel path discovery. Only applies within 100px of target waypoint.

        Args:
            current_pos: Current agent position
            previous_pos: Previous agent position
            phase_waypoints: List of waypoints for current phase

        Returns:
            Small gradient reward for approaching waypoint (typically 0.0-0.05)
        """
        if previous_pos is None or not phase_waypoints:
            return 0.0

        # Find nearest uncollected waypoint (by spatial distance)
        target_wp = self._find_nearest_uncollected_waypoint(
            phase_waypoints, current_pos=current_pos
        )
        if target_wp is None:
            return 0.0  # All waypoints collected

        # Calculate distance improvement
        prev_dist = math.sqrt(
            (previous_pos[0] - target_wp.position[0]) ** 2
            + (previous_pos[1] - target_wp.position[1]) ** 2
        )
        curr_dist = math.sqrt(
            (current_pos[0] - target_wp.position[0]) ** 2
            + (current_pos[1] - target_wp.position[1]) ** 2
        )

        distance_improvement = prev_dist - curr_dist

        # Only apply gradient within 200px and when moving closer (INCREASED from 100px)
        # Scale gradient by distance: stronger when closer for focused guidance
        if distance_improvement > 0 and curr_dist < 200.0:
            # Distance-scaled gradient: stronger when closer to waypoint
            # At 200px: 0.25x strength, at 100px: 0.5x, at 50px: 0.75x, at 0px: 1.0x
            distance_scale = 1.0 - (curr_dist / 200.0) * 0.75

            # Small gradient: 0.005 per pixel of progress, scaled by distance
            # This provides continuous guidance without dominating PBRS or terminal rewards
            gradient = 0.005 * distance_improvement * distance_scale
            return gradient

        return 0.0

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
        """
        metrics = {
            "sequence_multiplier": 0.0,
            "collection_type": "none",
            "skip_distance": 0,
        }

        if previous_pos is None or not self.current_path_waypoints:
            return 0.0, metrics

        # Define minimum spawn distance to prevent trivial early bonuses
        MIN_SPAWN_DISTANCE = 50.0

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
            return 0.0, metrics

        total_bonus = 0.0

        # Check each waypoint to see if we just crossed into its radius
        waypoint_radius = 18.0  # Match adaptive system

        for wp in phase_waypoints:
            wp_pos = wp.position

            # Check if already collected this episode
            if wp_pos in getattr(self, "_collected_path_waypoints", set()):
                continue

            # Calculate distances
            prev_dist = math.sqrt(
                (previous_pos[0] - wp_pos[0]) ** 2 + (previous_pos[1] - wp_pos[1]) ** 2
            )
            curr_dist = math.sqrt(
                (current_pos[0] - wp_pos[0]) ** 2 + (current_pos[1] - wp_pos[1]) ** 2
            )

            # Check if we just crossed into the waypoint radius
            if prev_dist > waypoint_radius and curr_dist <= waypoint_radius:
                # Calculate soft sequential multiplier
                sequence_multiplier = self._get_sequence_multiplier(wp.node_index)

                # Determine collection type for logging
                expected_index = self._next_expected_waypoint_index
                skip_distance = wp.node_index - expected_index

                if skip_distance == 0:
                    collection_type = "sequential"
                elif skip_distance == 1:
                    collection_type = "skip_1"
                elif skip_distance > 1:
                    collection_type = "skip_multiple"
                else:
                    collection_type = "backward"

                # Award bonus scaled by waypoint value and sequence multiplier
                base_bonus = wp.value * 0.3  # Scale to reasonable range (0.12-0.54)
                bonus = base_bonus * sequence_multiplier
                total_bonus += bonus

                # Mark as collected
                if not hasattr(self, "_collected_path_waypoints"):
                    self._collected_path_waypoints = set()
                self._collected_path_waypoints.add(wp_pos)

                # Update sequence tracking
                self._update_sequence_tracking(wp.node_index, wp)

                # Update metrics for TensorBoard logging
                metrics["sequence_multiplier"] = sequence_multiplier
                metrics["collection_type"] = collection_type
                metrics["skip_distance"] = skip_distance

                logger.debug(
                    f"Path waypoint reached: type={wp.waypoint_type}, "
                    f"value={wp.value:.2f}, base_bonus={base_bonus:.3f}, "
                    f"multiplier={sequence_multiplier:.2f}, final_bonus={bonus:.3f}, "
                    f"collection_type={collection_type}, skip_distance={skip_distance}, "
                    f"phase={wp.phase}, pos={wp_pos}"
                )

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
            logger.info("No demo waypoints to seed")
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

        logger.info(
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
            - last_next_hop_world: Last calculated next_hop position in world coords
            - last_next_hop_goal_id: Goal ID for last next_hop ("switch" or "exit")
        """
        return {
            "velocity_history": self.velocity_history,
            "alignment_history": self.alignment_history,
            "path_gradient_history": self.path_gradient_history,
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
        logger.info(
            f"[PBRS_EPISODE] Forward: {forward_progress_pct:.1f}% ({self.episode_forward_steps} steps), "
            f"Backtrack: {backtracking_pct:.1f}% ({self.episode_backtrack_steps} steps), "
            f"PBRS total: {pbrs_total:.2f}, Mean: {pbrs_mean:.4f}, "
            f"Path efficiency: {path_optimality:.3f} (optimal={self.optimal_path_length:.1f}, actual={self.episode_path_length:.1f})"
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
            # Exploration metrics (from unified position tracker)
            "exploration_cells_visited": len(self.position_tracker.visited_cells),
            "exploration_cells_current_phase": len(
                self.position_tracker.visited_cells_current_phase
            ),
            "exploration_hierarchical_resets": self.position_tracker.hierarchical_resets,
            "exploration_coverage": len(self.position_tracker.visited_cells)
            / max(1, (1056 // 12) * (600 // 12)),  # % of level explored (12px grid)
            # Waypoint metrics (path waypoints always enabled)
            "waypoints_reached_this_episode": len(
                getattr(self, "_collected_path_waypoints", set())
            ),
            "waypoints_active_current_level": len(self.current_path_waypoints),
            "waypoints_system": "path",
        }
