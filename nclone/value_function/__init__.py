"""Value function precomputation for optimal PBRS.

This module implements precomputed optimal value functions V*(s) for
Potential-Based Reward Shaping (PBRS). Instead of using heuristic
edge costs on a position-only graph, we compute the true optimal
value function over the full state space (position, velocity, jump_frames)
via backward dynamic programming.

Key components:
- PhysicsStepWrapper: Stateless interface to ninja physics
- StateDiscretization: Conversion between continuous and discrete states
- ValueIteration: Bellman updates for computing V*
- PrecomputedPBRS: Runtime lookup of precomputed potentials
- TwoPhaseValuePBRS: Two-goal support (switch + exit)
"""

from .physics_step_wrapper import PhysicsStepWrapper
from .state_discretization import StateDiscretizer, GridConfig
from .value_iteration import ValueIterator, save_value_function, load_value_function
from .interpolation import ValueInterpolator
from .precomputed_pbrs import PrecomputedValuePBRS
from .two_phase_pbrs import TwoPhaseValuePBRS, create_two_phase_pbrs
from .transition_table import TransitionTable, value_iteration_with_table
from .continuous_transition_table import (
    ContinuousTransitionTable,
    sl_value_iteration_with_table,
)
from .position_validity import compute_position_validity_mask
from .gpu_value_iteration import (
    is_gpu_available,
    get_gpu_info,
    sl_value_iteration_gpu,
    benchmark_gpu_vs_cpu,
)

__all__ = [
    'PhysicsStepWrapper',
    'StateDiscretizer',
    'GridConfig',
    'ValueIterator',
    'ValueInterpolator',
    'PrecomputedValuePBRS',
    'TwoPhaseValuePBRS',
    'TransitionTable',
    'ContinuousTransitionTable',
    'save_value_function',
    'load_value_function',
    'create_two_phase_pbrs',
    'value_iteration_with_table',
    'sl_value_iteration_with_table',
    'compute_position_validity_mask',
    # GPU acceleration
    'is_gpu_available',
    'get_gpu_info',
    'sl_value_iteration_gpu',
    'benchmark_gpu_vs_cpu',
]
