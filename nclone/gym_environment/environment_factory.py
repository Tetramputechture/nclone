"""
Environment factory functions for creating NPP-RL environments with different configurations.

This module provides convenient factory functions to create NppEnvironment instances
with common configurations for training, evaluation, and research.
"""

from typing import Optional, Callable, Dict, Any
from .npp_environment import NppEnvironment
from .config import EnvironmentConfig
from .frame_stack_wrapper import FrameStackWrapper


def create_training_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create an environment optimized for training."""
    if config is None:
        config = EnvironmentConfig.for_training()
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def create_evaluation_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create an environment optimized for evaluation."""
    if config is None:
        config = EnvironmentConfig.for_evaluation()
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def create_research_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create an environment optimized for research and debugging."""
    if config is None:
        config = EnvironmentConfig.for_research()
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def create_visual_testing_env(
    config: Optional[EnvironmentConfig] = None,
) -> NppEnvironment:
    """Create an environment optimized for testing rendering and visualization."""
    if config is None:
        config = EnvironmentConfig.for_visual_testing()
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def create_minimal_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create a minimal environment with all advanced features disabled."""
    if config is None:
        config = EnvironmentConfig.minimal()
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def make_vectorizable_env(
    env_factory: Callable[[], NppEnvironment],
) -> Callable[[], NppEnvironment]:
    """
    Factory function to create a vectorizable environment.

    This wraps an environment factory to ensure proper initialization
    for SubprocVecEnv compatibility.

    Args:
        env_factory: Function that returns an NppEnvironment instance

    Returns:
        Callable that returns NppEnvironment instance
    """

    def _make_env():
        return env_factory()

    return _make_env


def create_vectorized_training_envs(
    num_envs: int = 4,
    env_kwargs: Optional[Dict[str, Any]] = None,
    enable_graph_updates: bool = True,
    enable_reachability: bool = True,
    debug: bool = False,
) -> list:
    """
    Create multiple training environments for vectorization.

    Args:
        num_envs: Number of environments to create
        env_kwargs: Environment configuration parameters
        enable_graph_updates: Whether to enable dynamic graph updates
        enable_reachability: Whether to enable reachability analysis
        debug: Enable debug logging for graph operations

    Returns:
        List of environment factory functions
    """
    env_factories = []

    for i in range(num_envs):
        # Create a copy of env_kwargs for each environment
        local_env_kwargs = env_kwargs.copy() if env_kwargs else None

        # Add a unique seed for each environment
        if local_env_kwargs is None:
            local_env_kwargs = {}
        local_env_kwargs["seed"] = i

        # Create factory function
        def make_env(kwargs=local_env_kwargs):
            return create_training_env(
                env_kwargs=kwargs,
                enable_graph_updates=enable_graph_updates,
                enable_reachability=enable_reachability,
                debug=debug,
            )

        env_factories.append(make_vectorizable_env(make_env))

    return env_factories


def create_hierarchical_env(
    config: Optional[EnvironmentConfig] = None,
    completion_planner: Optional[Any] = None,
) -> NppEnvironment:
    """Create an environment optimized for hierarchical RL training."""
    if config is None:
        config = EnvironmentConfig.for_hierarchical_training(
            completion_planner=completion_planner
        )
    elif completion_planner is not None:
        config.hierarchical.completion_planner = completion_planner
    env = NppEnvironment(config)
    
    # Apply frame stacking wrapper if enabled
    if config.frame_stack.enable_visual_frame_stacking or config.frame_stack.enable_state_stacking:
        env = FrameStackWrapper(
            env,
            visual_stack_size=config.frame_stack.visual_stack_size,
            state_stack_size=config.frame_stack.state_stack_size,
            enable_visual_stacking=config.frame_stack.enable_visual_frame_stacking,
            enable_state_stacking=config.frame_stack.enable_state_stacking,
            padding_type=config.frame_stack.padding_type,
        )
    
    return env


def benchmark_environment_performance(
    env: NppEnvironment, num_steps: int = 1000, target_fps: float = 60.0
) -> Dict[str, Any]:
    """
    Benchmark environment performance.

    Args:
        env: Environment to benchmark
        num_steps: Number of steps to run
        target_fps: Target FPS for performance evaluation

    Returns:
        Performance benchmark results
    """
    import time

    # Reset environment
    env.reset()

    # Warm up
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)

    # Benchmark
    start_time = time.time()
    update_times = []

    for step in range(num_steps):
        step_start = time.time()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_time = time.time() - step_start
        update_times.append(step_time * 1000)  # Convert to ms

        if terminated or truncated:
            env.reset()

    total_time = time.time() - start_time

    # Calculate statistics
    avg_step_time_ms = sum(update_times) / len(update_times)
    max_step_time_ms = max(update_times)
    min_step_time_ms = min(update_times)

    target_step_time_ms = 1000.0 / target_fps
    performance_ratio = target_step_time_ms / avg_step_time_ms

    # Get performance stats from environment
    graph_stats = {}
    reachability_stats = {}

    if hasattr(env, "get_graph_performance_stats"):
        graph_stats = env.get_graph_performance_stats()

    if hasattr(env, "get_reachability_performance_stats"):
        reachability_stats = env.get_reachability_performance_stats()

    results = {
        "total_steps": num_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": avg_step_time_ms,
        "max_step_time_ms": max_step_time_ms,
        "min_step_time_ms": min_step_time_ms,
        "target_step_time_ms": target_step_time_ms,
        "performance_ratio": performance_ratio,
        "meets_target_fps": performance_ratio >= 1.0,
        "graph_stats": graph_stats,
        "reachability_stats": reachability_stats,
    }

    print(
        f"Environment Benchmark Results:\n"
        f"  Average step time: {avg_step_time_ms:.2f}ms\n"
        f"  Target step time: {target_step_time_ms:.2f}ms\n"
        f"  Performance ratio: {performance_ratio:.2f}\n"
        f"  Meets target FPS: {results['meets_target_fps']}\n"
    )

    if graph_stats:
        print(f"  Graph update time: {graph_stats.get('avg_update_time_ms', 0):.2f}ms")

    if reachability_stats:
        print(f"  Reachability time: {reachability_stats.get('avg_time_ms', 0):.2f}ms")

    return results


def validate_environment(env: NppEnvironment) -> bool:
    """
    Validate that an environment is properly configured.

    Args:
        env: Environment to validate

    Returns:
        True if environment is valid, False otherwise
    """
    import logging

    # Check if environment has proper observation space
    if not hasattr(env, "observation_space"):
        logging.error("Environment missing observation_space")
        return False

    # Test environment reset and step
    try:
        obs, info = env.reset()

        # Check for required observation components
        required_keys = ["player_frame", "global_view", "game_state"]
        if isinstance(obs, dict):
            for key in required_keys:
                if key not in obs:
                    logging.error(f"Missing required observation key: {key}")
                    return False

        # Test a single step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logging.info("Environment validation passed")
        return True

    except Exception as e:
        logging.error(f"Environment validation failed: {e}")
        return False
