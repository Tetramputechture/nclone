"""Two-phase PBRS for levels with switch and exit goals.

This module extends the precomputed value function approach to handle
levels with two sequential objectives:
1. Reach switch to activate exit door
2. Reach exit door to complete level

Two value functions are maintained:
- V_switch*(s): Optimal cost-to-go to switch
- V_exit*(s): Optimal cost-to-go to exit

The potential blends both based on switch activation state, maintaining
continuity for temporal credit assignment in LSTM-based policies.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import os

from .precomputed_pbrs import PrecomputedValuePBRS
from .state_discretization import StateDiscretizer
from .interpolation import ValueInterpolator

logger = logging.getLogger(__name__)


class TwoPhaseValuePBRS:
    """Two-phase PBRS using separate value functions for switch and exit.

    This class manages two precomputed value functions:
    - V_switch: Cost-to-go to switch (phase 1)
    - V_exit: Cost-to-go to exit (phase 2)

    The combined potential maintains continuity across switch activation
    to preserve temporal credit assignment for LSTM policies.

    Potential composition:
        Φ_total(s) = Φ_switch(s) × (1 - flag) + Φ_exit(s) × flag

    where flag = switch_activated (0 or 1)

    This is still policy-invariant because:
    - Both Φ_switch and Φ_exit are individually policy-invariant
    - Linear combination preserves policy invariance
    - F(s,s') = γ × Φ_total(s') - Φ_total(s) remains valid PBRS

    Attributes:
        pbrs_switch: PrecomputedValuePBRS for switch goal
        pbrs_exit: PrecomputedValuePBRS for exit goal
        blend_mode: How to blend potentials ('continuous', 'discrete')
    """

    def __init__(
        self,
        switch_value_path: str,
        exit_value_path: str,
        interpolation_method: str = "linear",
        blend_mode: str = "continuous",
    ):
        """Load precomputed value functions for both phases.

        Args:
            switch_value_path: Path to switch value function .npz
            exit_value_path: Path to exit value function .npz
            interpolation_method: Interpolation method for both
            blend_mode: Blending strategy ('continuous' or 'discrete')
                - 'continuous': Smooth blend Φ_switch(1-flag) + Φ_exit(flag)
                - 'discrete': Hard switch between Φ_switch and Φ_exit
        """
        self.pbrs_switch = PrecomputedValuePBRS(switch_value_path, interpolation_method)
        self.pbrs_exit = PrecomputedValuePBRS(exit_value_path, interpolation_method)
        self.blend_mode = blend_mode

        if blend_mode not in ("continuous", "discrete"):
            raise ValueError(f"Unknown blend_mode: {blend_mode}")

        logger.info(f"Two-phase PBRS initialized (blend_mode={blend_mode})")
        logger.info(f"  Switch goal: {self.pbrs_switch.goal_pos}")
        logger.info(f"  Exit goal: {self.pbrs_exit.goal_pos}")

    def get_potential(self, state: Dict[str, Any]) -> float:
        """Get combined potential for two-phase navigation.

        Args:
            state: Game state dictionary with switch_activated flag

        Returns:
            Combined potential in [0, 1]
        """
        switch_activated = state.get("switch_activated", False)

        if self.blend_mode == "discrete":
            # Hard switch between phases
            if not switch_activated:
                # Phase 1: Navigate to switch
                potential_switch = self.pbrs_switch.get_potential(state)
                # Scale to [0, 0.5] for phase 1
                return 0.5 * potential_switch
            else:
                # Phase 2: Navigate to exit
                potential_exit = self.pbrs_exit.get_potential(state)
                # Scale to [0.5, 1.0] for phase 2
                return 0.5 + 0.5 * potential_exit

        else:  # continuous blend
            # Compute both potentials
            potential_switch = self.pbrs_switch.get_potential(state)
            potential_exit = self.pbrs_exit.get_potential(state)

            # Blend: weight by switch activation state
            # Before switch: mostly Φ_switch, slightly Φ_exit (anticipatory)
            # After switch: mostly Φ_exit, slightly Φ_switch (memory)
            flag = 1.0 if switch_activated else 0.0

            # Smooth blend with scaling to maintain [0, 1] range
            # Phase 1 (flag=0): 0.5 * Φ_switch + 0.0 * Φ_exit ∈ [0, 0.5]
            # Phase 2 (flag=1): 0.0 * Φ_switch + 0.5 * Φ_exit + 0.5 ∈ [0.5, 1.0]
            potential = (
                0.5 * potential_switch * (1 - flag)
                + (0.5 + 0.5 * potential_exit) * flag
            )

            return float(np.clip(potential, 0.0, 1.0))

    def get_gradient(
        self, state: Dict[str, Any], epsilon: float = 1.0
    ) -> Tuple[float, float]:
        """Estimate potential gradient (direction toward goal).

        Args:
            state: Game state dictionary
            epsilon: Finite difference step size

        Returns:
            (grad_x, grad_y): Gradient of combined potential
        """
        # Use the active PBRS (switch or exit)
        switch_activated = state.get("switch_activated", False)

        if not switch_activated:
            return self.pbrs_switch.get_gradient(state, epsilon)
        else:
            return self.pbrs_exit.get_gradient(state, epsilon)

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics for both value functions.

        Returns:
            Dictionary with statistics for both phases
        """
        return {
            "switch_phase": self.pbrs_switch.get_statistics(),
            "exit_phase": self.pbrs_exit.get_statistics(),
            "blend_mode": self.blend_mode,
        }


def create_two_phase_pbrs(
    level_id: str,
    value_function_dir: str = "value_functions",
    interpolation_method: str = "linear",
    blend_mode: str = "discrete",
) -> TwoPhaseValuePBRS:
    """Factory function to create two-phase PBRS from level ID.

    Args:
        level_id: Level identifier (e.g., "006", "train_simple_100011")
        value_function_dir: Directory containing value function files
        interpolation_method: Interpolation method
        blend_mode: Blending strategy

    Returns:
        TwoPhaseValuePBRS instance

    Raises:
        FileNotFoundError: If value function files don't exist
    """
    switch_path = os.path.join(value_function_dir, f"{level_id}_switch.npz")
    exit_path = os.path.join(value_function_dir, f"{level_id}_exit.npz")

    if not os.path.exists(switch_path):
        raise FileNotFoundError(f"Switch value function not found: {switch_path}")
    if not os.path.exists(exit_path):
        raise FileNotFoundError(f"Exit value function not found: {exit_path}")

    return TwoPhaseValuePBRS(
        switch_path,
        exit_path,
        interpolation_method=interpolation_method,
        blend_mode=blend_mode,
    )
