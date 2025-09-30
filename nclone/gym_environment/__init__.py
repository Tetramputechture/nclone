# Environment utilities

# Main environment class with integrated functionality
from .npp_environment import NppEnvironment

# Factory functions for easy environment creation
from .environment_factory import (
    create_training_env,
    create_evaluation_env,
    create_research_env,
    create_minimal_env,
    make_vectorizable_env,
    create_vectorized_training_envs,
    benchmark_environment_performance,
    validate_environment,
    # Backward compatibility aliases
    create_dynamic_graph_env,
    create_reachability_aware_env,
)

# Note: Legacy wrapper classes have been removed.
# All functionality is now integrated directly into NppEnvironment.
# Use the factory functions above for easy environment creation.

__all__ = [
    # Main environment
    "NppEnvironment",
    # Factory functions (recommended)
    "create_training_env",
    "create_evaluation_env",
    "create_research_env",
    "create_minimal_env",
    "make_vectorizable_env",
    "create_vectorized_training_envs",
    "benchmark_environment_performance",
    "validate_environment",
    # Backward compatibility
    "create_dynamic_graph_env",
    "create_reachability_aware_env",
    # Legacy wrappers have been removed - use factory functions instead
]

# Legacy wrapper functionality has been fully integrated into NppEnvironment.
# The wrapper classes have been removed to simplify the codebase.
#
# Migration guide:
# - DynamicGraphWrapper → create_training_env(enable_graph_updates=True)
# - ReachabilityWrapper → create_training_env(enable_reachability=True)
# - VectorizationWrapper → create_vectorized_training_envs()
# - All wrapper combinations → Use appropriate factory function
#
# For custom configurations, use NppEnvironment directly with the desired parameters.
