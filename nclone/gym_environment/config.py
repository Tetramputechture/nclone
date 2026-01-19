"""
Configuration classes for NPP-RL environment.

This module provides structured configuration management for the NppEnvironment
and its various components, replacing the previous parameter explosion pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import logging

# Import RewardConfig for reward system integration
from .reward_calculation.reward_config import RewardConfig


class ObservationMode(Enum):
    """Observation space mode selection."""
    FULL = "full"           # Current: 41 game_state + 38 reach + 96 spatial + 3 sdf
    MINIMAL = "minimal"     # New: 40 dims total (physics + path + mines + buffers)


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
        valid_modes = ["human", "grayscale_array", "rgb_array"]
        if self.render_mode not in valid_modes:
            raise ValueError(f"render_mode must be one of {valid_modes}")


@dataclass
class PBRSConfig:
    """Configuration for Potential-Based Reward Shaping.

    For heuristic potential functions (path distance), γ=1.0 eliminates negative bias.
    Policy invariance holds for ANY γ, but γ=1.0 ensures clean telescoping in episodic tasks.
    """

    pbrs_gamma: float = 1.0  # Changed from 0.995 to eliminate systematic negative bias

    def __post_init__(self):
        """Validate PBRS configuration."""
        if self.pbrs_gamma < 0 or self.pbrs_gamma > 1:
            raise ValueError("pbrs_gamma must be between 0 and 1")


@dataclass
class GraphConfig:
    """Configuration for graph-based features."""

    debug: bool = False

    def __post_init__(self):
        """Validate graph configuration."""
        if self.debug:
            logging.info("Graph debug mode enabled")


@dataclass
class ReachabilityConfig:
    """Configuration for reachability analysis."""

    debug: bool = False

    def __post_init__(self):
        """Validate reachability configuration."""
        if self.debug:
            logging.info("Reachability debug mode enabled")


@dataclass
class EnvironmentConfig:
    """Main configuration class for NppEnvironment."""

    # Basic environment settings
    seed: Optional[int] = None
    eval_mode: bool = False
    custom_map_path: Optional[str] = None
    test_dataset_path: Optional[str] = None  # Path to test dataset for evaluation
    enable_logging: bool = False
    enable_profiling: bool = (
        False  # Enable detailed performance profiling (~5% overhead)
    )
    enable_visual_observations: bool = (
        False  # If False, skip rendering entirely (graph+state+reachability sufficient)
    )
    enable_graph_observations: bool = True  # If False, skip graph arrays in observation space (spatial_context sufficient)  # Memory optimization: saves ~21KB per observation when using graph_free architecture
    observation_mode: ObservationMode = ObservationMode.FULL  # Observation space mode

    # Component configurations
    frame_stack: FrameStackConfig = field(default_factory=FrameStackConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    pbrs: PBRSConfig = field(default_factory=PBRSConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)

    # Reward system configuration (curriculum-aware)
    reward_config: Optional[RewardConfig] = None

    # Goal curriculum configuration (for intermediate goal curriculum learning)
    goal_curriculum_config: Optional[Any] = None

    # Shared level cache (for zero-copy multi-worker training on same level)
    shared_level_cache: Optional[Any] = None
    # Multi-stage shared caches (for goal curriculum with shared memory)
    shared_level_caches_by_stage: Optional[Dict[int, Any]] = None
    # Curriculum map cache (for pre-modified map_data per curriculum stage)
    curriculum_map_cache: Optional[Any] = None

    # Frame skip
    frame_skip: int = 4

    def __post_init__(self):
        """Validate environment configuration."""
        if self.enable_logging:
            logging.basicConfig(level=logging.DEBUG)
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
            eval_mode=True,
            **kwargs,
        )
        return config

    def for_visual_testing(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for visual testing."""
        config = cls(
            **kwargs,
            render=RenderConfig(render_mode="human"),
            enable_visual_observations=True,
            frame_skip=1,
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
            # Reward system settings
            "reward_config_enabled": self.reward_config is not None,
            "reward_phase": self.reward_config.training_phase
            if self.reward_config
            else "N/A",
            "pbrs_objective_weight": self.reward_config.pbrs_objective_weight
            if self.reward_config
            else "N/A",
            "time_penalty_per_step": self.reward_config.time_penalty_per_step
            if self.reward_config
            else "N/A",
            # Debug settings
            "debug": self.graph.debug or self.reachability.debug,
            # Frame skip settings
            "frame_skip": self.frame_skip,
            # Graph observation settings
            "enable_graph_observations": self.enable_graph_observations,
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
