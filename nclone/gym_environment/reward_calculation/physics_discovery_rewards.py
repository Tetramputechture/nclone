"""Efficiently computable physics-based rewards using existing state."""

import math
from collections import deque
from typing import Dict, Any, Optional


class PhysicsDiscoveryRewards:
    """Efficiently computable physics-based rewards using existing state.

    Provides alternative exploration signals to reduce over-reliance on PBRS by
    rewarding energy efficiency, physics state diversity, and special physics utilization.
    All calculations use existing observation data without expensive physics simulation.
    """

    def __init__(self):
        """Initialize physics discovery reward system."""
        # Baseline efficiency for comparison
        self.movement_efficiency_baseline = 0.5  # Distance per energy unit baseline

        # History tracking for diversity calculations
        self.energy_utilization_history = deque(maxlen=100)
        self.physics_state_history = deque(maxlen=50)

        # Special physics utilization tracking
        self.wall_jump_count = 0
        self.buffered_jump_count = 0
        self.energy_efficient_moves = 0

        # Diversity state tracking (simplified hash of physics state)
        self.visited_physics_states = set()

    def calculate_physics_rewards(
        self, obs: Dict[str, Any], prev_obs: Dict[str, Any], action: Optional[int]
    ) -> Dict[str, float]:
        """Calculate physics discovery rewards from existing observations.

        Args:
            obs: Current observation dict
            prev_obs: Previous observation dict
            action: Action taken (0-5: NOOP, LEFT, RIGHT, JUMP, JUMP+LEFT, JUMP+RIGHT)

        Returns:
            Dict of reward components: {'efficiency': float, 'diversity': float, 'utilization': float}
        """
        rewards = {}

        # 1. Energy efficiency bonus (using ninja's existing velocity/position)
        efficiency_reward = self._calculate_efficiency_bonus(obs, prev_obs)
        rewards["efficiency"] = efficiency_reward * 0.2

        # 2. Physics state diversity bonus (using game_state features)
        diversity_reward = self._calculate_diversity_bonus(obs["game_state"])
        rewards["diversity"] = diversity_reward * 0.1

        # 3. Special physics utilization (wall jumps, buffered jumps, etc.)
        utilization_reward = self._calculate_utilization_bonus(obs, prev_obs, action)
        rewards["utilization"] = utilization_reward * 0.3

        return rewards

    def _calculate_efficiency_bonus(
        self, obs: Dict[str, Any], prev_obs: Dict[str, Any]
    ) -> float:
        """Reward energy-efficient movement using existing position/velocity.

        Calculates movement efficiency as distance traveled per energy spent,
        comparing against a baseline to reward above-average efficiency.

        Args:
            obs: Current observation with player position and game_state
            prev_obs: Previous observation for comparison

        Returns:
            Efficiency bonus reward [0.0, 1.0]
        """
        # Distance traveled (already available in observations)
        distance = math.sqrt(
            (obs["player_x"] - prev_obs["player_x"]) ** 2
            + (obs["player_y"] - prev_obs["player_y"]) ** 2
        )

        # Energy change (from physics state - kinetic energy is feature 33)
        current_energy = obs["game_state"][32]  # kinetic_energy feature
        prev_energy = (
            prev_obs["game_state"][32] if len(prev_obs["game_state"]) > 32 else 0.0
        )
        energy_spent = (
            abs(current_energy - prev_energy) + 0.001
        )  # Avoid division by zero

        # Efficiency ratio
        efficiency = distance / energy_spent

        # Track efficiency history
        self.energy_utilization_history.append(efficiency)

        # Reward if above baseline (10% better than baseline)
        if efficiency > self.movement_efficiency_baseline * 1.1:
            bonus = min(
                (efficiency - self.movement_efficiency_baseline)
                / self.movement_efficiency_baseline,
                1.0,
            )
            self.energy_efficient_moves += 1
            return bonus

        return 0.0

    def _calculate_diversity_bonus(self, game_state) -> float:
        """Reward visiting novel physics states.

        Encourages exploration of diverse physics configurations by tracking
        visited physics states and rewarding novel combinations.

        Args:
            game_state: Current game state vector (40D enhanced physics state)

        Returns:
            Diversity bonus reward [0.0, 1.0]
        """
        # Create simplified physics state fingerprint (avoid floating point precision issues)
        # Focus on key physics indicators that define meaningful state differences

        if len(game_state) >= 40:
            # Use enhanced physics state features
            velocity_mag = int(game_state[0] * 10)  # Velocity magnitude (discretized)
            movement_state = int(
                game_state[3] + game_state[4] + game_state[5] + game_state[6]
            )  # Movement categories
            airborne = int(game_state[7] > 0)  # Airborne status
            contact_state = int(
                game_state[13] + game_state[14] + game_state[15]
            )  # Contact strengths
            energy_level = (
                int(game_state[32] * 10) if len(game_state) > 32 else 0
            )  # Kinetic energy

            # Create state fingerprint
            physics_fingerprint = (
                velocity_mag,
                movement_state,
                airborne,
                contact_state,
                energy_level,
            )
        else:
            # Fallback for basic state
            velocity_mag = int(game_state[0] * 10)
            movement_state = int(
                game_state[3] + game_state[4] + game_state[5] + game_state[6]
            )
            airborne = int(game_state[7] > 0)
            physics_fingerprint = (velocity_mag, movement_state, airborne, 0, 0)

        # Check if this is a novel physics state
        if physics_fingerprint not in self.visited_physics_states:
            self.visited_physics_states.add(physics_fingerprint)

            # Reward novelty, with diminishing returns as more states are discovered
            novelty_bonus = 1.0 / (1.0 + len(self.visited_physics_states) * 0.01)
            return min(novelty_bonus, 1.0)

        return 0.0

    def _calculate_utilization_bonus(
        self, obs: Dict[str, Any], prev_obs: Dict[str, Any], action: Optional[int]
    ) -> float:
        """Reward utilization of advanced physics mechanics.

        Provides bonuses for successfully using wall jumps, buffered inputs,
        and other advanced physics techniques that demonstrate mastery.

        Args:
            obs: Current observation
            prev_obs: Previous observation
            action: Action taken

        Returns:
            Utilization bonus reward [0.0, 1.0]
        """
        total_bonus = 0.0

        # Detect wall jump (airborne → touching wall + jump action → airborne with different velocity)
        was_airborne = prev_obs.get("ninja_airborne", False)
        is_airborne = obs.get("ninja_airborne", False)
        was_walled = prev_obs.get("ninja_walled", False)

        # Wall jump detection: was against wall, used jump action, now airborne with velocity change
        if (
            was_walled and action in [3, 4, 5] and is_airborne
        ):  # JUMP, JUMP+LEFT, JUMP+RIGHT
            velocity_change = abs(
                obs.get("player_xspeed", 0) - prev_obs.get("player_xspeed", 0)
            )
            if (
                velocity_change > 0.1
            ):  # Significant velocity change indicates successful wall jump
                total_bonus += 0.5
                self.wall_jump_count += 1

        # Detect buffered jump (using buffer system indicators from game state)
        if len(obs["game_state"]) >= 13:
            # Buffer features are at indices 10-12 (jump/floor/wall buffers)
            jump_buffer = obs["game_state"][10]
            floor_buffer = obs["game_state"][11]
            wall_buffer = obs["game_state"][12]

            # Reward active buffer usage (buffer values > -1 indicate active buffers)
            if (
                jump_buffer > -0.8 or floor_buffer > -0.8 or wall_buffer > -0.8
            ):  # Buffer is active
                if action in [3, 4, 5]:  # Jump action while buffer active
                    total_bonus += 0.3
                    self.buffered_jump_count += 1

        # Reward complex movement sequences (ground → air → wall → air combinations)
        if len(obs["game_state"]) >= 8:
            # Movement state categories from indices 3-7
            ground_movement = obs["game_state"][3] > 0
            air_movement = obs["game_state"][4] > 0
            wall_interaction = obs["game_state"][5] > 0

            prev_ground = (
                prev_obs["game_state"][3] > 0
                if len(prev_obs["game_state"]) >= 8
                else False
            )
            prev_air = (
                prev_obs["game_state"][4] > 0
                if len(prev_obs["game_state"]) >= 8
                else False
            )
            prev_wall = (
                prev_obs["game_state"][5] > 0
                if len(prev_obs["game_state"]) >= 8
                else False
            )

            # Reward transitions that show physics mastery
            if prev_ground and wall_interaction:  # Ground to wall
                total_bonus += 0.2
            elif prev_wall and air_movement:  # Wall to air
                total_bonus += 0.2
            elif (
                prev_air and ground_movement and action in [1, 2]
            ):  # Controlled landing
                total_bonus += 0.1

        return min(total_bonus, 1.0)

    def reset(self):
        """Reset episode-specific tracking for new episode."""
        # Keep long-term learning (visited_physics_states, efficiency baseline)
        # Reset episode-specific counters
        self.wall_jump_count = 0
        self.buffered_jump_count = 0
        self.energy_efficient_moves = 0

        # Clear recent history but keep some for cross-episode learning
        if len(self.energy_utilization_history) > 50:
            # Keep last 50 entries for baseline updating
            recent_history = list(self.energy_utilization_history)[-50:]
            self.energy_utilization_history.clear()
            self.energy_utilization_history.extend(recent_history)

    def get_physics_stats(self) -> Dict[str, Any]:
        """Get physics utilization statistics for monitoring.

        Returns:
            Dict with physics discovery statistics
        """
        return {
            "wall_jumps_used": self.wall_jump_count,
            "buffered_jumps_used": self.buffered_jump_count,
            "energy_efficient_moves": self.energy_efficient_moves,
            "unique_physics_states_visited": len(self.visited_physics_states),
            "average_efficiency": sum(self.energy_utilization_history)
            / max(len(self.energy_utilization_history), 1),
            "efficiency_baseline": self.movement_efficiency_baseline,
        }
