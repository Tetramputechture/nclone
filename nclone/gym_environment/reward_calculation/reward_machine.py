"""Reward Machine for N++ task progression tracking.

Implements a 3-state finite automaton that tracks task phase:
  u0 (start) -> u1 (switch collected) -> u2 (exit reached, terminal)

Transitions:
  u0 ->[agent touches switch] u1    (sparse reward: R_SWITCH)
  u1 ->[agent touches exit]   u2    (sparse reward: R_EXIT)
  Death or timeout resets to u0.

The reward machine provides:
1. Sparse rewards on state transitions (switch collection, exit reached)
2. Phase-dependent potential bonus Phi_phase for PBRS continuity
3. Terminal state enforcement: Phi = 0 at all terminal states

Reference: Icarte et al. (2018) "Using Reward Machines for High-Level Task
Specification and Decomposition in Reinforcement Learning"
"""

import logging
from typing import Set, Tuple

logger = logging.getLogger(__name__)

# Reward machine states
RM_STATE_START = 0          # Seeking switch
RM_STATE_SWITCH_COLLECTED = 1  # Switch collected, seeking exit
RM_STATE_EXIT_REACHED = 2     # Terminal success

# Default sparse rewards for transitions
DEFAULT_R_SWITCH = 100.0   # Reward for collecting switch
DEFAULT_R_EXIT = 200.0     # Reward for reaching exit

# Phase bonus for PBRS: Phi_phase(u=1) = C_PHASE
# This bonus rewards having collected the switch, providing
# a smooth potential increase at the transition point.
DEFAULT_C_PHASE = 0.5


class RewardMachine:
    """3-state reward machine tracking switch -> exit task progression.

    States:
        0 (RM_STATE_START): Agent seeking switch
        1 (RM_STATE_SWITCH_COLLECTED): Switch collected, seeking exit
        2 (RM_STATE_EXIT_REACHED): Exit reached (terminal)

    The reward machine computes:
    - Sparse rewards on transitions (configurable)
    - Phase potential bonus Phi_phase(u) for PBRS composition
    - Terminal Phi = 0 enforcement

    Total PBRS potential: Phi(s, u) = Phi_state(s, u) + Phi_phase(u)
    where Phi_state comes from precomputed grid and Phi_phase from this class.
    """

    def __init__(
        self,
        r_switch: float = DEFAULT_R_SWITCH,
        r_exit: float = DEFAULT_R_EXIT,
        c_phase: float = DEFAULT_C_PHASE,
    ):
        """Initialize reward machine.

        Args:
            r_switch: Sparse reward for collecting switch (u0 -> u1)
            r_exit: Sparse reward for reaching exit (u1 -> u2)
            c_phase: Phase potential bonus for switch-collected state (u1)
        """
        self.r_switch = r_switch
        self.r_exit = r_exit
        self.c_phase = c_phase
        self.state: int = RM_STATE_START

    def step(self, events: Set[str]) -> float:
        """Process events and return sparse reward for any transition.

        Args:
            events: Set of event strings that occurred this frame.
                    Recognized events: 'switch_collected', 'exit_reached'

        Returns:
            Sparse reward for this transition (0.0 if no transition)
        """
        if self.state == RM_STATE_START and 'switch_collected' in events:
            self.state = RM_STATE_SWITCH_COLLECTED
            logger.debug("[RM] Transition u0 -> u1 (switch collected)")
            return self.r_switch

        if self.state == RM_STATE_SWITCH_COLLECTED and 'exit_reached' in events:
            self.state = RM_STATE_EXIT_REACHED
            logger.debug("[RM] Transition u1 -> u2 (exit reached)")
            return self.r_exit

        return 0.0

    def get_phi_phase(self) -> float:
        """Get the phase-dependent potential bonus.

        Returns:
            Phi_phase(u):
                u=0: 0.0 (no bonus before switch)
                u=1: c_phase (bonus for having collected switch)
                u=2: 0.0 (terminal state, Phi MUST be 0)
        """
        if self.state == RM_STATE_SWITCH_COLLECTED:
            return self.c_phase
        return 0.0

    def get_total_potential(
        self,
        phi_state: float,
        is_terminal: bool = False,
    ) -> float:
        """Compute total PBRS potential: Phi(s, u) = Phi_state + Phi_phase.

        CRITICAL: Returns 0.0 for all terminal states (death, timeout, success).
        This is required for PBRS policy invariance (Ng et al. 1999).

        Args:
            phi_state: Position-based potential from Phi grid (in [-1, 0])
            is_terminal: Whether the current state is terminal (death/timeout/success)

        Returns:
            Total potential. 0.0 for terminal states.
        """
        if is_terminal:
            return 0.0

        return phi_state + self.get_phi_phase()

    @property
    def is_terminal(self) -> bool:
        """Whether the reward machine is in a terminal state."""
        return self.state == RM_STATE_EXIT_REACHED

    @property
    def switch_collected(self) -> bool:
        """Whether the switch has been collected."""
        return self.state >= RM_STATE_SWITCH_COLLECTED

    def reset(self) -> None:
        """Reset reward machine to initial state (u0).

        Called on episode start and on death/timeout.
        """
        self.state = RM_STATE_START
