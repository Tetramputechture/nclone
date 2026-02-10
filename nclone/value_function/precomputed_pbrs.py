"""Runtime PBRS using precomputed optimal value functions.

This module provides PrecomputedValuePBRS, which loads precomputed
value functions V*(s) and provides O(1) potential lookups for PBRS.

The potential is derived from the optimal value function:
    Φ(s) = 1 - V*(s) / max_V

This provides:
- Exact optimal cost-to-go (no heuristic multipliers)
- Full state awareness (position, velocity, jump_frames)
- O(1) runtime queries via interpolation
- Guaranteed policy invariance for PBRS
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import os

from .state_discretization import StateDiscretizer, GridConfig
from .interpolation import ValueInterpolator

logger = logging.getLogger(__name__)


class PrecomputedValuePBRS:
    """PBRS potential function using precomputed optimal value function.

    This class loads a precomputed V*(s) grid and provides efficient
    potential lookups for PBRS. The potential Φ(s) = 1 - V*(s)/max_V
    is normalized to [0, 1] range for consistency with existing PBRS.

    Attributes:
        V: Value function array shape (nx, ny, nvx, nvy, njump)
        discretizer: State discretizer for grid operations
        interpolator: Value interpolator for continuous state queries
        goal_pos: Goal position (x, y) in pixels
        max_V: Maximum finite value in V (for normalization)
    """

    def __init__(self, value_function_path: str, interpolation_method: str = "linear"):
        """Load precomputed value function from disk.

        Args:
            value_function_path: Path to .npz file with value function
            interpolation_method: Interpolation method ('nearest', 'linear', 'hybrid')
        """
        if not os.path.exists(value_function_path):
            raise FileNotFoundError(f"Value function not found: {value_function_path}")

        # Load value function
        data = np.load(value_function_path, allow_pickle=True)
        self.V = data["V"]
        grid_cfg = data["grid_config"].item()
        self.goal_pos = tuple(data["goal_pos"])

        # Reconstruct grid config and discretizer
        config = GridConfig(**grid_cfg)
        self.discretizer = StateDiscretizer(config)

        # Create interpolator
        self.interpolator = ValueInterpolator(
            self.discretizer, method=interpolation_method
        )

        # Compute max value for normalization (exclude inf)
        finite_values = self.V[self.V < np.inf]
        if len(finite_values) > 0:
            self.max_V = np.max(finite_values)
            self.min_V = np.min(finite_values)
        else:
            logger.warning("No finite values in value function!")
            self.max_V = 1.0
            self.min_V = 0.0

        # Statistics
        self.total_states = np.prod(self.V.shape)
        self.reachable_states = np.sum(self.V < np.inf)
        self.goal_states = np.sum(self.V == 0)

        logger.info(f"Loaded precomputed value function from {value_function_path}")
        logger.info(f"  Grid shape: {self.V.shape}")
        logger.info(f"  Reachable: {self.reachable_states:,} / {self.total_states:,}")
        logger.info(f"  Goal states: {self.goal_states:,}")
        logger.info(f"  Value range: [{self.min_V:.1f}, {self.max_V:.1f}]")
        logger.info(f"  Goal position: {self.goal_pos}")
        logger.info(f"  Interpolation: {interpolation_method}")

    def get_potential(self, state: Dict[str, Any]) -> float:
        """Get PBRS potential for a game state.

        This method extracts position, velocity, and jump_duration from
        the game state dict and queries the value function.

        The potential is: Φ(s) = 1 - V*(s) / max_V
        - Φ = 1.0 at goal (V* = 0)
        - Φ = 0.0 far from goal (V* = max_V)
        - Φ approaches 0 for unreachable states (V* = inf)

        Args:
            state: Game state dictionary with keys:
                - player_x, player_y: Position in pixels
                - player_xspeed, player_yspeed: Velocity in pixels/frame
                - jump_duration_remaining: Normalized [0,1] or frames remaining

        Returns:
            Potential value in [0, 1]
        """
        # Extract state components
        x = state["player_x"]
        y = state["player_y"]
        vx = state.get("player_xspeed", 0.0)
        vy = state.get("player_yspeed", 0.0)

        # Extract jump duration
        # May be normalized [0,1] or raw frames - handle both
        jump_norm = state.get("jump_duration_remaining", 0.0)
        if 0.0 <= jump_norm <= 1.0:
            # Normalized - convert to frames
            from ..constants.physics_constants import MAX_JUMP_DURATION

            jump_frames = int(jump_norm * MAX_JUMP_DURATION)
        else:
            # Already in frames
            jump_frames = int(jump_norm)

        # Query value function
        continuous_state = (x, y, vx, vy, jump_frames)
        V_s = self.interpolator.interpolate(self.V, continuous_state)

        # Handle unreachable states
        if V_s == np.inf or V_s > self.max_V * 2:
            return 0.0  # No gradient for truly unreachable

        # Convert to potential: Φ(s) = 1 - V*(s) / max_V
        # Negate because lower V* = closer to goal = higher potential
        if self.max_V > 0:
            potential = 1.0 - (V_s / self.max_V)
        else:
            potential = 0.0

        # Clamp to [0, 1]
        return float(np.clip(potential, 0.0, 1.0))

    def get_potential_from_continuous_state(
        self, state: Tuple[float, float, float, float, int]
    ) -> float:
        """Get potential directly from continuous state tuple.

        Useful for testing and analysis without full game state dict.

        Args:
            state: (x, y, vx, vy, jump_duration) in continuous space

        Returns:
            Potential value in [0, 1]
        """
        V_s = self.interpolator.interpolate(self.V, state)

        if V_s == np.inf or V_s > self.max_V * 2:
            return 0.0

        if self.max_V > 0:
            potential = 1.0 - (V_s / self.max_V)
        else:
            potential = 0.0

        return float(np.clip(potential, 0.0, 1.0))

    def get_value_at_state(self, state: Dict[str, Any]) -> float:
        """Get raw value V*(s) at game state (before normalization).

        Useful for debugging and analysis.

        Args:
            state: Game state dictionary

        Returns:
            Raw value V*(s) in [0, inf]
        """
        x = state["player_x"]
        y = state["player_y"]
        vx = state.get("player_xspeed", 0.0)
        vy = state.get("player_yspeed", 0.0)

        jump_norm = state.get("jump_duration_remaining", 0.0)
        if 0.0 <= jump_norm <= 1.0:
            from ..constants.physics_constants import MAX_JUMP_DURATION

            jump_frames = int(jump_norm * MAX_JUMP_DURATION)
        else:
            jump_frames = int(jump_norm)

        continuous_state = (x, y, vx, vy, jump_frames)
        return float(self.interpolator.interpolate(self.V, continuous_state))

    def get_gradient(
        self, state: Dict[str, Any], epsilon: float = 1.0
    ) -> Tuple[float, float]:
        """Estimate potential gradient at state.

        Computes numerical gradient ∇Φ using finite differences.
        The gradient points in the direction of steepest potential increase
        (i.e., toward the goal in the value function sense).

        Args:
            state: Game state dictionary
            epsilon: Finite difference step size (pixels, default 1.0)

        Returns:
            (grad_x, grad_y): Gradient components (not normalized)
        """
        x = state["player_x"]
        y = state["player_y"]

        # Current potential
        phi = self.get_potential(state)

        # Perturb x
        state_plus_x = state.copy()
        state_plus_x["player_x"] = x + epsilon
        phi_plus_x = self.get_potential(state_plus_x)

        # Perturb y
        state_plus_y = state.copy()
        state_plus_y["player_y"] = y + epsilon
        phi_plus_y = self.get_potential(state_plus_y)

        # Finite difference
        grad_x = (phi_plus_x - phi) / epsilon
        grad_y = (phi_plus_y - phi) / epsilon

        return (grad_x, grad_y)

    def get_statistics(self) -> Dict[str, Any]:
        """Get value function statistics for logging.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_states": int(self.total_states),
            "reachable_states": int(self.reachable_states),
            "goal_states": int(self.goal_states),
            "min_value": float(self.min_V),
            "max_value": float(self.max_V),
            "goal_position": self.goal_pos,
            "grid_shape": self.V.shape,
            "memory_mb": self.V.nbytes / (1024 * 1024),
        }
