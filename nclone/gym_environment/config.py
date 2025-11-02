"""
Configuration classes for NPP-RL environment.

This module provides structured configuration management for the NppEnvironment
and its various components, replacing the previous parameter explosion pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging


@dataclass
class FrameStackConfig:
    """Configuration for frame stacking to capture temporal information.

    Frame stacking is a technique commonly used in Deep RL (especially DQN and its variants)
    to provide temporal information to the policy network. By stacking consecutive frames,
    the agent can infer velocity, acceleration, and motion patterns.

    References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.
      Original DQN paper using 4-frame stacking for Atari games.
    - Machado et al. (2018). "Revisiting the Arcade Learning Environment." IJCAI.
      Analysis of frame stacking and preprocessing techniques.

    Visual frames (player_frame, global_view) and game state can be stacked independently
    to capture different temporal scales. For example:
    - Visual frames: capture recent motion (2-4 frames typical)
    - Game state: capture physics trends (2-12 frames possible)
    """

    enable_visual_frame_stacking: bool = False
    visual_stack_size: int = 4  # Number of frames to stack (2-12)

    enable_state_stacking: bool = False
    state_stack_size: int = 4  # Number of game states to stack (2-12)

    padding_type: str = "zero"  # "zero", "repeat" - padding for initial frames

    def __post_init__(self):
        """Validate frame stacking configuration."""
        if self.visual_stack_size < 1 or self.visual_stack_size > 12:
            raise ValueError("visual_stack_size must be between 1 and 12")

        if self.state_stack_size < 1 or self.state_stack_size > 12:
            raise ValueError("state_stack_size must be between 1 and 12")

        valid_padding = ["zero", "repeat"]
        if self.padding_type not in valid_padding:
            raise ValueError(f"padding_type must be one of {valid_padding}")


@dataclass
class AugmentationConfig:
    """Configuration for frame augmentation performance.

    Augmentation represents ~45% of runtime in profiling analysis.
    Disabling validation can save ~12% of total execution time.

    Note: When frame stacking is enabled with augmentation, augmentation is applied
    consistently across the entire stack to maintain temporal coherence.
    """

    enable_augmentation: bool = True
    disable_validation: bool = True  # Disable validation in training for performance
    intensity: str = "medium"
    p: float = 0.5

    def __post_init__(self):
        """Validate augmentation configuration."""
        valid_intensities = ["light", "medium", "strong"]
        if self.intensity not in valid_intensities:
            raise ValueError(f"intensity must be one of {valid_intensities}")

        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")


@dataclass
class RenderConfig:
    """Configuration for rendering and visualization."""

    render_mode: str = "grayscale_array"
    enable_animation: bool = False
    enable_debug_overlay: bool = False

    def __post_init__(self):
        """Validate render configuration."""
        valid_modes = ["human", "grayscale_array"]
        if self.render_mode not in valid_modes:
            raise ValueError(f"render_mode must be one of {valid_modes}")


@dataclass
class PBRSConfig:
    """Configuration for Potential-Based Reward Shaping."""

    pbrs_gamma: float = 0.99

    def __post_init__(self):
        """Validate PBRS configuration."""
        if self.pbrs_gamma < 0 or self.pbrs_gamma > 1:
            raise ValueError("pbrs_gamma must be between 0 and 1")


@dataclass
class GraphConfig:
    """Configuration for graph-based features.

    - enable_graph_for_observations: Add graph observations to observation space
    """

    enable_graph_for_observations: bool = True
    debug: bool = False

    def __post_init__(self):
        """Validate graph configuration."""
        if self.debug:
            logging.info("Graph debug mode enabled")


@dataclass
class ReachabilityConfig:
    """Configuration for reachability analysis."""

    enable_reachability: bool = True
    debug: bool = False

    def __post_init__(self):
        """Validate reachability configuration."""
        if self.debug:
            logging.info("Reachability debug mode enabled")


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical RL functionality."""

    enable_hierarchical: bool = False
    completion_planner: Optional[Any] = None
    enable_subtask_rewards: bool = True
    subtask_reward_scale: float = 0.1
    max_subtask_steps: int = 1000
    debug: bool = False

    # Hierarchical constants
    SUBTASK_COMPLETION_BONUS: float = 0.5
    SUBTASK_TIMEOUT_PENALTY: float = 0.1
    SUBTASK_TIMEOUT_THRESHOLD: int = 500
    SIGNIFICANT_MOVEMENT_THRESHOLD: float = 5.0
    DISTANCE_REWARD_SCALE: float = 0.01
    MINE_AVOIDANCE_REWARD_SCALE: float = 0.005

    def __post_init__(self):
        """Validate hierarchical configuration."""
        if self.subtask_reward_scale < 0:
            raise ValueError("subtask_reward_scale must be non-negative")

        if self.max_subtask_steps <= 0:
            raise ValueError("max_subtask_steps must be positive")

        if self.enable_hierarchical and self.debug:
            logging.info("Hierarchical RL debug mode enabled")


@dataclass
class EnvironmentConfig:
    """Main configuration class for NppEnvironment."""

    # Basic environment settings
    seed: Optional[int] = None
    eval_mode: bool = False
    custom_map_path: Optional[str] = None
    test_dataset_path: Optional[str] = None  # Path to test dataset for evaluation
    enable_logging: bool = False
    enable_short_episode_truncation: bool = False

    # Component configurations
    frame_stack: FrameStackConfig = field(default_factory=FrameStackConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    pbrs: PBRSConfig = field(default_factory=PBRSConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    def __post_init__(self):
        """Validate environment configuration."""
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO)
            logging.info("Environment logging enabled")

    @classmethod
    def for_training(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for training.

        Performance optimizations enabled:
        - Debug overlay disabled (saves ~14% runtime)
        - Augmentation validation disabled (saves ~12% runtime)
        - RGB array rendering (no display overhead)
        - Frame stabilization optimized (67% fewer calls)

        Memory optimizations enabled:
        - Animation disabled (saves ~500 KB per environment)
        - Optimized for parallel training with 50+ environments

        Expected performance: ~50-60% faster than unoptimized configuration.
        Expected memory savings: ~500 KB per environment instance.
        """
        config = cls(
            augmentation=AugmentationConfig(
                enable_augmentation=True,
                disable_validation=True,
                intensity="medium",
                p=0.5,
            ),
            render=RenderConfig(
                render_mode="grayscale_array",
                enable_animation=False,
            ),
            graph=GraphConfig(
                enable_graph_for_pbrs=True, enable_graph_for_observations=False
            ),
            reachability=ReachabilityConfig(enable_reachability=True),
            enable_short_episode_truncation=False,
            **kwargs,
        )
        return config

    @classmethod
    def for_evaluation(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for evaluation.

        Performance optimizations enabled:
        - Debug overlay disabled (saves ~14% runtime)
        - Augmentation validation disabled (saves ~12% runtime)
        - Clean evaluation without PBRS
        """
        config = cls(
            augmentation=AugmentationConfig(
                enable_augmentation=False,
                disable_validation=True,
                intensity="medium",
                p=0.5,
            ),
            render=RenderConfig(render_mode="grayscale_array"),
            pbrs=PBRSConfig(enable_pbrs=True),
            graph=GraphConfig(
                enable_graph_for_pbrs=True, enable_graph_for_observations=False
            ),
            reachability=ReachabilityConfig(enable_reachability=True),
            eval_mode=True,
            enable_short_episode_truncation=False,  # Let episodes run to completion
            **kwargs,
        )
        return config

    @classmethod
    def for_research(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for research and debugging.

        Validation ENABLED for safety during development.
        Debug overlay and logging enabled for detailed inspection.
        """
        config = cls(
            augmentation=AugmentationConfig(
                enable_augmentation=True,
                disable_validation=False,  # Keep validation for debugging
                intensity="medium",
                p=0.5,
            ),
            render=RenderConfig(
                render_mode="human", enable_animation=True, enable_debug_overlay=True
            ),
            graph=GraphConfig(
                enable_graph_for_observations=True,
                debug=True,
            ),
            reachability=ReachabilityConfig(enable_reachability=True, debug=True),
            enable_logging=True,
            **kwargs,
        )
        return config

    @classmethod
    def for_visual_testing(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for testing rendering and visualization."""
        config = cls(
            enable_logging=False,
            render=RenderConfig(
                render_mode="human", enable_animation=True, enable_debug_overlay=False
            ),
            graph=GraphConfig(
                enable_graph_for_observations=False,
                debug=False,
            ),
            reachability=ReachabilityConfig(enable_reachability=False, debug=False),
            **kwargs,
        )
        return config

    @classmethod
    def for_hierarchical_training(
        cls, completion_planner=None, **kwargs
    ) -> "EnvironmentConfig":
        """Create configuration optimized for hierarchical RL training.

        Performance optimizations enabled for fast training.
        """
        config = cls(
            augmentation=AugmentationConfig(
                enable_augmentation=True,
                disable_validation=True,  # Performance optimization
                intensity="medium",
                p=0.5,
            ),
            render=RenderConfig(render_mode="grayscale_array"),
            graph=GraphConfig(enable_graph_for_observations=True),
            reachability=ReachabilityConfig(enable_reachability=True),
            hierarchical=HierarchicalConfig(
                enable_hierarchical=True,
                completion_planner=completion_planner,
                enable_subtask_rewards=True,
            ),
            enable_short_episode_truncation=True,
            **kwargs,
        )
        return config

    @classmethod
    def minimal(cls, **kwargs) -> "EnvironmentConfig":
        """Create minimal configuration with all advanced features disabled.

        Maximum performance - validation disabled, no augmentation.
        """
        config = cls(
            augmentation=AugmentationConfig(
                enable_augmentation=False,  # No augmentation for minimal config
                disable_validation=True,
                intensity="light",
                p=0.0,
            ),
            render=RenderConfig(render_mode="grayscale_array"),
            graph=GraphConfig(enable_graph_for_observations=False),
            reachability=ReachabilityConfig(enable_reachability=False),
            hierarchical=HierarchicalConfig(enable_hierarchical=False),
            **kwargs,
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for backward compatibility."""
        return {
            # Basic settings
            "seed": self.seed,
            "eval_mode": self.eval_mode,
            "custom_map_path": self.custom_map_path,
            "enable_logging": self.enable_logging,
            "enable_short_episode_truncation": self.enable_short_episode_truncation,
            # Frame stacking settings
            "enable_visual_frame_stacking": self.frame_stack.enable_visual_frame_stacking,
            "visual_stack_size": self.frame_stack.visual_stack_size,
            "enable_state_stacking": self.frame_stack.enable_state_stacking,
            "state_stack_size": self.frame_stack.state_stack_size,
            "frame_stack_padding_type": self.frame_stack.padding_type,
            # Augmentation settings
            "enable_augmentation": self.augmentation.enable_augmentation,
            "augmentation_disable_validation": self.augmentation.disable_validation,
            "augmentation_intensity": self.augmentation.intensity,
            "augmentation_p": self.augmentation.p,
            # Render settings
            "render_mode": self.render.render_mode,
            "enable_animation": self.render.enable_animation,
            "enable_debug_overlay": self.render.enable_debug_overlay,
            # PBRS settings
            "pbrs_gamma": self.pbrs.pbrs_gamma,
            # Graph settings
            "enable_graph_for_observations": self.graph.enable_graph_for_observations,
            # Reachability settings
            "enable_reachability": self.reachability.enable_reachability,
            # Hierarchical settings
            "enable_hierarchical": self.hierarchical.enable_hierarchical,
            "completion_planner": self.hierarchical.completion_planner,
            "enable_subtask_rewards": self.hierarchical.enable_subtask_rewards,
            "subtask_reward_scale": self.hierarchical.subtask_reward_scale,
            "max_subtask_steps": self.hierarchical.max_subtask_steps,
            # Debug settings
            "debug": self.graph.debug
            or self.reachability.debug
            or self.hierarchical.debug,
        }


def validate_config(config: EnvironmentConfig) -> bool:
    """
    Validate environment configuration for common issues.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check for conflicting settings
        if (
            config.hierarchical.enable_hierarchical
            and not config.reachability.enable_reachability
        ):
            raise ValueError(
                "Hierarchical RL requires reachability analysis to be enabled. "
                "Set reachability.enable_reachability=True in your configuration."
            )

        if config.render.render_mode == "human" and not config.render.enable_animation:
            logging.info(
                "Human render mode without animation - consider enabling animation for better visualization"
            )

        # Check for performance implications
        if config.render.enable_debug_overlay and not config.enable_logging:
            logging.warning(
                "Debug overlay enabled without logging - consider enabling logging for debug info"
            )

        return True

    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
