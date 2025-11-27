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
"""

import logging
from typing import Dict, Any, Optional
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
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


logger = logging.getLogger(__name__)


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
    ):
        """Initialize simplified reward calculator.

        Args:
            reward_config: RewardConfig instance managing curriculum-aware component lifecycle
            pbrs_gamma: Discount factor for PBRS (γ in F(s,s') = γ * Φ(s') - Φ(s))
        """
        # Single config object manages all curriculum logic
        self.config = reward_config or RewardConfig()

        # PBRS configuration (kept for backwards compatibility with path calculator)
        self.pbrs_gamma = pbrs_gamma

        # Create path calculator for PBRS
        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200, use_astar=True
        )
        self.pbrs_calculator = PBRSCalculator(path_calculator=path_calculator)

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

        # Step-level component tracking for TensorBoard callback
        # Always populated after calculate_reward() for info dict logging
        self.last_pbrs_components = {}

        # Position revisit tracking for oscillation penalty (Phase 4 improvement)
        self.position_visit_counts: Dict[tuple, int] = {}  # Grid cell visit counts
        self.position_history_window = 100  # Track last 100 positions
        self.position_history = deque(maxlen=100)  # Sliding window
        self.revisit_penalty_weight = 0.001  # Penalty per revisit

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

        # === TERMINAL REWARDS (always active) ===

        # Death penalties - differentiated by cause for better learning signal
        if obs.get("player_dead", False):
            death_cause = obs.get("death_cause", None)

            # Determine appropriate penalty based on death cause
            if death_cause == "impact":
                # High-velocity collision with ceiling/floor
                terminal_reward = IMPACT_DEATH_PENALTY
            elif death_cause in ("mine", "drone", "thwump", "hazard"):
                # Contact with deadly entities (highly preventable)
                terminal_reward = HAZARD_DEATH_PENALTY
            else:
                # Generic/unknown death cause
                terminal_reward = DEATH_PENALTY

            self.episode_terminal_reward = terminal_reward

            # Populate PBRS components for TensorBoard logging (terminal state)
            self.last_pbrs_components = {
                "terminal_reward": terminal_reward,
                "pbrs_reward": 0.0,
                "time_penalty": 0.0,
                "revisit_penalty": 0.0,
                "total_reward": terminal_reward,
                "is_terminal": True,
                "terminal_type": "death",
                "death_cause": death_cause or "unknown",
            }
            return terminal_reward

        # Completion reward
        if obs.get("player_won", False):
            self.episode_terminal_reward = LEVEL_COMPLETION_REWARD

            # Populate PBRS components for TensorBoard logging (terminal state)
            self.last_pbrs_components = {
                "terminal_reward": LEVEL_COMPLETION_REWARD,
                "pbrs_reward": 0.0,
                "time_penalty": 0.0,
                "total_reward": LEVEL_COMPLETION_REWARD,
                "is_terminal": True,
                "terminal_type": "win",
            }
            return LEVEL_COMPLETION_REWARD

        # === TIME PENALTY (curriculum-controlled, scaled by frame skip) ===
        # Config determines if active and magnitude based on training phase
        # Scale by frames_executed to account for frame skip (e.g., 4 frames per action)
        time_penalty = self.config.time_penalty_per_step * frames_executed
        reward = time_penalty

        # Track time penalty for logging (store scaled penalty per action)
        self.episode_time_penalties.append(time_penalty)

        # === MILESTONE REWARD ===
        switch_just_activated = obs.get("switch_activated", False) and not prev_obs.get(
            "switch_activated", False
        )
        if switch_just_activated:
            reward += SWITCH_ACTIVATION_REWARD

        # === PBRS OBJECTIVE POTENTIAL (curriculum-scaled, policy-invariant) ===
        # Calculate F(s,s') = γ * Φ(s') - Φ(s) for dense, policy-invariant path guidance
        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

        # Validate required data for PBRS calculation
        if not adjacency or not level_data:
            raise ValueError(
                "PBRS calculation requires adjacency graph and level_data in observation. "
                "Ensure graph building is enabled in environment config."
            )

        # Calculate current potential Φ(s')
        current_potential = self.pbrs_calculator.calculate_combined_potential(
            state=obs,
            adjacency=adjacency,
            level_data=level_data,
            graph_data=graph_data,
            objective_weight=self.config.pbrs_objective_weight,
            scale_factor=self.config.pbrs_normalization_scale,
        )

        # Track actual path length for efficiency metrics
        current_player_pos = (obs["player_x"], obs["player_y"])
        if self.prev_player_pos is not None:
            # Calculate Euclidean distance moved (approximation of actual path)
            dx = current_player_pos[0] - self.prev_player_pos[0]
            dy = current_player_pos[1] - self.prev_player_pos[1]
            movement_distance = (dx * dx + dy * dy) ** 0.5
            self.episode_path_length += movement_distance
        self.prev_player_pos = current_player_pos

        # Store optimal path length (combined path distance) on first step
        if self.optimal_path_length is None:
            self.optimal_path_length = obs.get("_pbrs_combined_path_distance", 0.0)

        # Calculate PBRS shaping reward: F(s,s') = γ * Φ(s') - Φ(s)
        pbrs_reward = 0.0

        # Handle goal transition (switch activation) for policy invariance
        if switch_just_activated:
            # When goal changes (switch → exit), reset potential tracking
            # This treats the transition like a hierarchical RL "option termination"
            # Prevents comparing incompatible potentials (switch-distance vs exit-distance)
            logger.debug(
                f"Switch activated: Resetting PBRS potential tracking for goal transition "
                f"(switch → exit). New potential Φ(s')={current_potential:.4f}"
            )
            self.prev_potential = None  # Reset for new goal
        elif self.prev_potential is not None:
            # Apply PBRS formula for policy-invariant shaping
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential

            # Log PBRS components for debugging and verification
            potential_change = current_potential - self.prev_potential
            logger.debug(
                f"PBRS: Φ(s)={self.prev_potential:.4f}, "
                f"Φ(s')={current_potential:.4f}, "
                f"ΔΦ={potential_change:.4f}, "
                f"F(s,s')={pbrs_reward:.4f} "
                f"(γ={self.pbrs_gamma})"
            )

            # Track forward/backtracking steps for logging
            if potential_change > 0.001:  # Forward progress
                self.episode_forward_steps += 1
            elif potential_change < -0.001:  # Backtracking
                self.episode_backtrack_steps += 1

            # Warn if potential decreased significantly (backtracking detected)
            if potential_change < -0.05:
                logger.debug(
                    f"Backtracking detected: potential decreased by {-potential_change:.4f} "
                    f"(PBRS penalty: {pbrs_reward:.4f})"
                )
            # Log significant progress
            elif potential_change > 0.05:
                logger.debug(
                    f"Progress detected: potential increased by {potential_change:.4f} "
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

        # Track PBRS reward for logging
        self.episode_pbrs_rewards.append(pbrs_reward)

        reward += pbrs_reward

        # === POSITION REVISIT PENALTY (oscillation deterrent) ===
        # Track position visits to discourage meandering and oscillation
        # Snap to 12px grid for coarse detection (sub-node size)
        grid_x = int(obs["player_x"] // 12) * 12
        grid_y = int(obs["player_y"] // 12) * 12
        grid_pos = (grid_x, grid_y)

        # Update position history (sliding window)
        self.position_history.append(grid_pos)
        visit_count = self.position_visit_counts.get(grid_pos, 0)
        self.position_visit_counts[grid_pos] = visit_count + 1

        # Apply revisit penalty (scaled by square root of frequency to avoid harsh penalties)
        revisit_penalty = 0.0
        if visit_count > 0:
            import math

            revisit_penalty = -self.revisit_penalty_weight * math.sqrt(visit_count)
            reward += revisit_penalty

        # Clear old visits when history full (maintain sliding window)
        if len(self.position_history) == self.position_history_window:
            oldest_pos = self.position_history[0]
            if oldest_pos in self.position_visit_counts:
                self.position_visit_counts[oldest_pos] -= 1
                if self.position_visit_counts[oldest_pos] == 0:
                    del self.position_visit_counts[oldest_pos]

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

        # Populate PBRS components for TensorBoard logging (non-terminal state)
        # This provides detailed breakdown of reward components for analysis
        milestone_reward = SWITCH_ACTIVATION_REWARD if switch_just_activated else 0.0

        # Extract diagnostic metrics from observation (set by objective_distance_potential)
        area_scale = obs.get("_pbrs_area_scale", 0.0)
        normalized_distance = obs.get("_pbrs_normalized_distance", 0.0)
        combined_path_distance = obs.get("_pbrs_combined_path_distance", 0.0)

        # Calculate current distance to goal (shortest path)
        if not obs.get("switch_activated", False):
            goal_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
            cache_key = "switch"
        else:
            goal_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))
            cache_key = "exit"

        player_pos = (int(obs["player_x"]), int(obs["player_y"]))

        # Get distance to goal (use cached value if available from PBRS calculation)
        try:
            from ...constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )

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
        except Exception:
            distance_to_goal = 0.0

        self.last_pbrs_components = {
            "pbrs_reward": pbrs_reward,
            "time_penalty": time_penalty,  # Use scaled penalty (per action, not per frame)
            "milestone_reward": milestone_reward,
            "revisit_penalty": revisit_penalty,  # New: position revisit penalty
            "total_reward": reward,
            "is_terminal": False,
            # Include potential information for debugging
            "current_potential": current_potential
            if self.current_potential is not None
            else 0.0,
            "prev_potential": self.prev_potential
            if self.prev_potential is not None
            else 0.0,
            "potential_change": (current_potential - self.prev_potential)
            if self.prev_potential is not None
            else 0.0,
            # Enhanced diagnostic fields for PBRS verification
            "potential_gradient": abs(current_potential - self.prev_potential)
            if self.prev_potential is not None
            else 0.0,
            "pbrs_gamma": self.pbrs_gamma,
            "theoretical_pbrs": (current_potential - self.prev_potential)
            if self.prev_potential is not None
            else 0.0,  # For γ=1.0, this should equal pbrs_reward
            # New diagnostic fields for monitoring and analysis
            "distance_to_goal": distance_to_goal,
            "area_scale": area_scale,
            "combined_path_distance": combined_path_distance,
            "normalized_distance": normalized_distance,
            # Frame skip information
            "frames_executed": frames_executed,
            # Position revisit tracking
            "unique_positions_visited": len(self.position_visit_counts),
            "total_position_visits": len(self.position_history),
        }

        return reward

    def reset(self):
        """Reset episode state for new episode."""
        self.steps_taken = 0

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

        # Reset position revisit tracking
        self.position_visit_counts.clear()
        self.position_history.clear()

        # Reset PBRS calculator state
        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()

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
