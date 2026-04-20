"""Environment factory functions for creating NPP-RL environments.

Post strict-Layer-1 cleanup: no visual observations, no frame stacking, no
graph / reachability / level-context plumbing. All helpers return a plain
:class:`NppEnvironment` driven by an :class:`EnvironmentConfig` that emits the
4-key Layer 1 policy observation dict.
"""

from typing import Any, Callable, Dict, Optional
from .npp_environment import NppEnvironment
from .config import EnvironmentConfig


def create_training_env(config: Optional[EnvironmentConfig] = None) -> NppEnvironment:
    """Create an environment optimized for training."""
    if config is None:
        config = EnvironmentConfig.for_training()
    return NppEnvironment(config)


def create_evaluation_env(
    config: Optional[EnvironmentConfig] = None,
) -> NppEnvironment:
    """Create an environment optimized for evaluation."""
    if config is None:
        config = EnvironmentConfig.for_evaluation()
    return NppEnvironment(config)


def create_visual_testing_env(
    config: Optional[EnvironmentConfig] = None,
) -> NppEnvironment:
    """Create an environment configured for human playback / video capture.

    The policy observation is identical; this only enables the pygame window
    via `RenderConfig(render_mode='human')` for interactive debugging.
    """
    if config is None:
        config = EnvironmentConfig.for_visual_testing()
    return NppEnvironment(config)


def make_vectorizable_env(
    env_factory: Callable[[], NppEnvironment],
) -> Callable[[], NppEnvironment]:
    """Wrap an environment factory for SubprocVecEnv compatibility."""

    def _make_env():
        return env_factory()

    return _make_env


def create_vectorized_training_envs(
    num_envs: int = 4,
    env_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> list:
    """Create multiple training environment factories for vectorization."""
    env_factories = []

    for i in range(num_envs):
        seed_value = (
            env_kwargs["seed"]
            if env_kwargs is not None and "seed" in env_kwargs
            else i
        )

        def make_env_factory(seed: int):
            def make_env():
                cfg = EnvironmentConfig.for_training()
                cfg.graph.debug = debug
                cfg.seed = seed
                return create_training_env(config=cfg)

            return make_env

        env_factories.append(make_vectorizable_env(make_env_factory(seed_value)))

    return env_factories
