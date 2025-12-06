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
    """Position tracking for revisit penalties (oscillation deterrent).

    Tracks recent positions using sliding window to penalize oscillation/looping.
    Exploration is now handled by RND at the training level.
    """

    def __init__(self, grid_size: int = 24, window_size: int = 100):
        """Initialize position tracker.

        Args:
            grid_size: Cell size in pixels for position discretization (default: 24px tile size)
            window_size: Number of recent positions to track for revisit penalty
        """
        self.grid_size = grid_size
        self.visited_cells: set = set()  # Episode-wide tracking (for stats)
        self.position_history = deque(maxlen=window_size)  # Sliding window
        self.position_counts: Dict[tuple, int] = {}  # Visit counts in current window
        self.cells_visited_this_episode = 0

    def get_position_reward(
        self,
        position: tuple,
        revisit_penalty_weight: float = 0.001,
    ) -> tuple:
        """Calculate revisit penalty for current position.

        Exploration bonuses are disabled - RND handles exploration at training level.

        Uses LOGARITHMIC scaling with per-step soft cap to prevent accumulated
        revisit penalties from dominating terminal rewards:
        - Logarithmic: -weight × log(1 + visit_count) instead of linear
        - Soft cap: max penalty of -0.5 per step

        Args:
            position: (x, y) position in pixels
            revisit_penalty_weight: Penalty weight for revisits (default: 0.001)

        Returns:
            Tuple of (total_reward, exploration_reward=0, revisit_penalty, is_new_cell)
        """
        # Snap to grid
        grid_x = int(position[0] // self.grid_size) * self.grid_size
        grid_y = int(position[1] // self.grid_size) * self.grid_size
        cell = (grid_x, grid_y)

        total_reward = 0.0
        exploration_reward = 0.0  # Always 0 - RND handles exploration
        revisit_penalty = 0.0
        is_new = cell not in self.visited_cells

        # Track visited cells for stats (no reward)
        if is_new:
            self.visited_cells.add(cell)
            self.cells_visited_this_episode += 1

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

        return total_reward, exploration_reward, revisit_penalty, is_new

    def reset(self):
        """Reset tracking for new episode."""
        self.visited_cells.clear()
        self.position_history.clear()
        self.position_counts.clear()
        self.cells_visited_this_episode = 0


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

        # Position tracking for revisit penalty (oscillation deterrent)
        # Exploration is handled by RND at training level
        self.position_tracker = PositionTracker(grid_size=24, window_size=100)

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
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

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

            # === PBRS DIRECTION VALIDATION DIAGNOSTIC ===
            # Check if movement direction aligns with expected path direction
            # This catches the bug where PBRS rewards going TOWARD goal coordinates
            # when the actual path requires going AWAY from goal first
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
                    obs["_pbrs_potential_change"] = potential_change

                    # CRITICAL DIAGNOSTIC: Positive PBRS but moving away from goal (Euclidean)
                    # This is EXPECTED for levels where path goes away from goal first!
                    # But if this happens AND potential_change > 0, the path distance calc is correct
                    # If potential_change > 0 AND euclidean_alignment > 0.5,
                    # agent is moving toward goal - check if this is the correct path direction
                    if potential_change > 0.01 and euclidean_alignment > 0.7:
                        # Agent getting positive PBRS for moving toward goal (Euclidean)
                        # This might be correct OR might indicate path goes directly to goal
                        # Log for analysis
                        if self.steps_taken < 100:  # Only log early in episode
                            logger.warning(
                                f"[PBRS_DIRECTION] Early movement toward goal (Euclidean). "
                                f"step={self.steps_taken}, pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                f"move_dir=({move_dx:.2f},{move_dy:.2f}), "
                                f"goal_dir=({goal_dir_x:.2f},{goal_dir_y:.2f}), "
                                f"alignment={euclidean_alignment:.2f}, "
                                f"potential_change={potential_change:.4f}. "
                                f"If path should go AWAY from goal first, this is a bug!"
                            )
                    elif potential_change < -0.01 and euclidean_alignment < -0.5:
                        # Agent getting NEGATIVE PBRS for moving away from goal (Euclidean)
                        # This might be WRONG if the path requires going away from goal first!
                        if self.steps_taken < 100:  # Only log early in episode
                            logger.warning(
                                f"[PBRS_DIRECTION] Negative PBRS for moving away from goal (Euclidean). "
                                f"step={self.steps_taken}, pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                f"move_dir=({move_dx:.2f},{move_dy:.2f}), "
                                f"goal_dir=({goal_dir_x:.2f},{goal_dir_y:.2f}), "
                                f"alignment={euclidean_alignment:.2f}, "
                                f"potential_change={potential_change:.4f}. "
                                f"If path SHOULD go away from goal first, PBRS is penalizing correct behavior!"
                            )

            # === PATH DIRECTION VALIDATION (FIX FOR WRONG-DIRECTION PBRS) ===
            # Only give positive PBRS when moving toward the next hop on the optimal path
            # This prevents rewarding movement toward goal coordinates when the actual
            # path requires going away from the goal first (e.g., left when goal is right)
            if (
                pbrs_reward > 0.001  # Only gate positive rewards
                and self.prev_player_pos is not None
                and movement_distance > 0.5
            ):
                try:
                    # Get level cache from path calculator
                    path_calc = self.pbrs_calculator.path_calculator
                    level_cache = path_calc.level_cache if path_calc else None

                    if level_cache is not None:
                        # Determine current goal_id
                        direction_goal_id = (
                            "exit" if obs.get("switch_activated", False) else "switch"
                        )

                        # Find current node using the improved node selection
                        from nclone.graph.reachability.pathfinding_utils import (
                            find_ninja_node,
                            extract_spatial_lookups_from_graph_data,
                            NODE_WORLD_COORD_OFFSET,
                        )

                        spatial_hash, subcell_lookup = (
                            extract_spatial_lookups_from_graph_data(graph_data)
                        )

                        # Get current goal node
                        if direction_goal_id == "switch":
                            goal_node = level_cache._goal_id_to_goal_pos.get("switch")
                        else:
                            goal_node = level_cache._goal_id_to_goal_pos.get("exit")

                        if goal_node is not None:
                            # Find player's current node
                            player_node = find_ninja_node(
                                (
                                    int(current_player_pos[0]),
                                    int(current_player_pos[1]),
                                ),
                                adjacency,
                                spatial_hash=spatial_hash,
                                subcell_lookup=subcell_lookup,
                                ninja_radius=10.0,
                                goal_node=goal_node,
                                level_cache=level_cache,
                                goal_id=direction_goal_id,
                            )

                            if player_node is not None:
                                # Get next hop toward goal
                                next_hop = level_cache.get_next_hop(
                                    player_node, direction_goal_id
                                )

                                if next_hop is not None:
                                    # Calculate expected direction (to next hop in world coords)
                                    # Nodes are in tile data space, need to add offset
                                    next_hop_world_x = (
                                        next_hop[0] + NODE_WORLD_COORD_OFFSET
                                    )
                                    next_hop_world_y = (
                                        next_hop[1] + NODE_WORLD_COORD_OFFSET
                                    )

                                    expected_dx = (
                                        next_hop_world_x - current_player_pos[0]
                                    )
                                    expected_dy = (
                                        next_hop_world_y - current_player_pos[1]
                                    )
                                    expected_dist = (
                                        expected_dx**2 + expected_dy**2
                                    ) ** 0.5

                                    if expected_dist > 1.0:
                                        # Normalize expected direction
                                        expected_dir_x = expected_dx / expected_dist
                                        expected_dir_y = expected_dy / expected_dist

                                        # Movement direction
                                        move_dir_x = (
                                            current_player_pos[0]
                                            - self.prev_player_pos[0]
                                        ) / movement_distance
                                        move_dir_y = (
                                            current_player_pos[1]
                                            - self.prev_player_pos[1]
                                        ) / movement_distance

                                        # Dot product: +1 = aligned, -1 = opposite
                                        path_alignment = (
                                            move_dir_x * expected_dir_x
                                            + move_dir_y * expected_dir_y
                                        )

                                        # Store for diagnostics
                                        obs["_pbrs_path_alignment"] = path_alignment

                                        # If movement is significantly OPPOSITE to expected path direction,
                                        # zero out positive PBRS (conservative: only when clearly wrong)
                                        if path_alignment < -0.3:
                                            logger.warning(
                                                f"[PBRS_GATE] Zeroing positive PBRS due to wrong direction. "
                                                f"step={self.steps_taken}, "
                                                f"path_alignment={path_alignment:.2f}, "
                                                f"original_pbrs={pbrs_reward:.4f}, "
                                                f"move_dir=({move_dir_x:.2f},{move_dir_y:.2f}), "
                                                f"expected_dir=({expected_dir_x:.2f},{expected_dir_y:.2f}), "
                                                f"next_hop=({next_hop_world_x:.0f},{next_hop_world_y:.0f})"
                                            )
                                            pbrs_reward = 0.0  # Zero out wrong-direction positive PBRS

                                        # === ENHANCED DIAGNOSTICS FOR DIRECTION CHANGES ===
                                        # Store expected direction for tracking direction changes
                                        obs["_next_hop_dir_x"] = expected_dir_x
                                        obs["_next_hop_dir_y"] = expected_dir_y
                                        obs["_next_hop_pos"] = (
                                            next_hop_world_x,
                                            next_hop_world_y,
                                        )
                                        obs["_player_node"] = player_node

                                        # Detect if expected direction is primarily VERTICAL (up/down)
                                        # This helps identify when agent should be going UP instead of LEFT
                                        is_vertical_direction = abs(
                                            expected_dir_y
                                        ) > abs(expected_dir_x)
                                        is_upward = (
                                            expected_dir_y < -0.5
                                        )  # Y decreases going up

                                        # Log when agent is on LEFT side of level and should be going UP
                                        if (
                                            current_player_pos[0] < 150
                                        ):  # Left side of level
                                            if is_upward:
                                                # Expected direction is UP - agent should go up
                                                if (
                                                    self.steps_taken % 100 == 0
                                                ):  # Log every 100 steps
                                                    logger.warning(
                                                        f"[LEFT_WALL_DIAG] Agent on left side, should go UP. "
                                                        f"step={self.steps_taken}, "
                                                        f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                                        f"next_hop=({next_hop_world_x:.0f},{next_hop_world_y:.0f}), "
                                                        f"expected_dir=({expected_dir_x:.2f},{expected_dir_y:.2f}), "
                                                        f"path_alignment={path_alignment:.2f}, "
                                                        f"pbrs_reward={pbrs_reward:.4f}"
                                                    )
                                            elif (
                                                abs(expected_dir_x) > 0.7
                                                and expected_dir_x < 0
                                            ):
                                                # Still expecting LEFT movement at left wall = possible issue
                                                logger.warning(
                                                    f"[LEFT_WALL_BUG] Agent on left side but next_hop points LEFT! "
                                                    f"step={self.steps_taken}, "
                                                    f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                                    f"next_hop=({next_hop_world_x:.0f},{next_hop_world_y:.0f}), "
                                                    f"expected_dir=({expected_dir_x:.2f},{expected_dir_y:.2f}). "
                                                    f"Path may not connect to upper corridor!"
                                                )

                                        # Log significant direction changes (helps track path progress)
                                        prev_dir_x = getattr(
                                            self, "_prev_expected_dir_x", None
                                        )
                                        prev_dir_y = getattr(
                                            self, "_prev_expected_dir_y", None
                                        )
                                        if (
                                            prev_dir_x is not None
                                            and prev_dir_y is not None
                                        ):
                                            # Dot product of previous and current expected directions
                                            dir_change = (
                                                prev_dir_x * expected_dir_x
                                                + prev_dir_y * expected_dir_y
                                            )
                                            if (
                                                dir_change < 0.5
                                            ):  # Significant direction change (>60 degrees)
                                                logger.warning(
                                                    f"[PATH_TURN] Expected direction changed significantly. "
                                                    f"step={self.steps_taken}, "
                                                    f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                                    f"prev_dir=({prev_dir_x:.2f},{prev_dir_y:.2f}), "
                                                    f"new_dir=({expected_dir_x:.2f},{expected_dir_y:.2f}), "
                                                    f"dir_change_dot={dir_change:.2f}"
                                                )
                                        self._prev_expected_dir_x = expected_dir_x
                                        self._prev_expected_dir_y = expected_dir_y

                                else:
                                    # next_hop is None - agent might be at goal or path is broken
                                    if self.steps_taken % 100 == 0:
                                        logger.warning(
                                            f"[NO_NEXT_HOP] No next_hop found for player node. "
                                            f"step={self.steps_taken}, "
                                            f"pos=({current_player_pos[0]:.0f},{current_player_pos[1]:.0f}), "
                                            f"player_node={player_node}, "
                                            f"goal_id={direction_goal_id}"
                                        )
                except Exception as e:
                    # Don't fail reward calculation if path validation fails
                    logger.debug(f"Path direction validation failed: {e}")

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

        # === POSITION TRACKING (revisit penalty for oscillation deterrent) ===
        # Exploration is handled by RND at training level
        (
            position_reward,
            exploration_reward,
            revisit_penalty,
            is_new_cell,
        ) = self.position_tracker.get_position_reward(
            position=(obs["player_x"], obs["player_y"]),
            revisit_penalty_weight=self.config.revisit_penalty_weight,
        )
        reward += position_reward

        # === CALCULATE DISTANCE TO GOAL (needed for progress tracking) ===
        # Calculate current distance to goal (shortest path)
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
        #         )
        #     )
        # except Exception:
        #     geometric_path_distance = 0.0

        # FIX: Apply same radius adjustment to Euclidean for fair comparison
        # geometric_path_distance already has (ninja_radius + entity_radius) subtracted
        # so we must apply the same adjustment to Euclidean distance
        # combined_radius = NINJA_RADIUS + entity_radius
        # euclidean_adjusted = max(0.0, euclidean_to_current_goal_raw - combined_radius)

        # Only compute ratio for meaningful distances (> 30px to avoid grid-snapping artifacts)
        # Path distance is computed between 12px grid nodes, so small distances have high error
        # if euclidean_adjusted > 30.0 and geometric_path_distance > 0:
        #     divergence_ratio = geometric_path_distance / euclidean_adjusted

        #     # Store for TensorBoard logging
        #     obs["_pbrs_path_euclidean_ratio"] = divergence_ratio

        #     # Log anomalies: path should be longer than Euclidean for complex levels
        #     # Using 0.80 threshold (20% tolerance) because:
        #     # - Path is computed between 12px grid nodes, not exact positions
        #     # - Node snapping can cause up to ~17px discrepancy (diagonal of 12px cell)
        #     # - This is a diagnostic, not a critical error
        #     if divergence_ratio < 0.80:
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

        # # Populate PBRS components for TensorBoard logging (non-terminal state)
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
            "exploration_reward": exploration_reward,
            "position_reward": position_reward,  # Combined exploration + revisit
            "exploration_cells_visited": len(self.position_tracker.visited_cells),
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
            "_pbrs_time_ms": obs.get("_pbrs_time_ms", 0.0),
            "_pbrs_potential_calc_ms": obs.get("_pbrs_potential_calc_ms", 0.0),
            "_pbrs_node_find_ms": obs.get("_pbrs_node_find_ms", 0.0),
            "_pbrs_cache_update_ms": obs.get("_pbrs_cache_update_ms", 0.0),
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
            # Exploration metrics (from unified position tracker)
            "exploration_cells_visited": len(self.position_tracker.visited_cells),
            "exploration_coverage": len(self.position_tracker.visited_cells)
            / max(1, (1056 // 24) * (600 // 24)),  # % of level explored
        }
