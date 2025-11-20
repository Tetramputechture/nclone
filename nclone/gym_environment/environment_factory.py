"""
Environment factory functions for creating NPP-RL environments with different configurations.

This module provides convenient factory functions to create NppEnvironment instances
with common configurations for training, evaluation, and research.
"""

from typing import Optional, Callable, Dict, Any
from .npp_environment import NppEnvironment
from .config import EnvironmentConfig


def create_training_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create an environment optimized for training."""
    if config is None:
        config = EnvironmentConfig.for_training()
    env = NppEnvironment(config)

    # Apply frame stacking wrapper if enabled
    if (
        config.enable_visual_observations
        and config.frame_stack.enable_visual_frame_stacking
        or config.frame_stack.enable_state_stacking
    ):
        from .frame_stack_wrapper import FrameStackWrapper

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
    if (
        config.frame_stack.enable_visual_frame_stacking
        or config.frame_stack.enable_state_stacking
    ):
        from .frame_stack_wrapper import FrameStackWrapper

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
    if (
        config.frame_stack.enable_visual_frame_stacking
        or config.frame_stack.enable_state_stacking
    ):
        from .frame_stack_wrapper import FrameStackWrapper

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
    debug: bool = False,
) -> list:
    """
    Create multiple training environments for vectorization.

    Args:
        num_envs: Number of environments to create
        env_kwargs: Environment configuration parameters
        debug: Enable debug logging for graph operations

    Returns:
        List of environment factory functions
    """
    env_factories = []

    for i in range(num_envs):
        # Create config with specified graph settings
        config = EnvironmentConfig.for_training()
        config.graph.debug = debug

        # Add a unique seed for each environment
        if env_kwargs and "seed" in env_kwargs:
            config.seed = env_kwargs["seed"]
        else:
            config.seed = i

        # Create factory function with proper closure
        def make_env_factory(seed_value):
            def make_env():
                cfg = EnvironmentConfig.for_training()
                cfg.graph.debug = debug
                cfg.seed = seed_value
                return create_training_env(config=cfg)

            return make_env

        env_factories.append(make_vectorizable_env(make_env_factory(i)))

    return env_factories
