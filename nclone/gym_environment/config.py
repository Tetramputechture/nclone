"""
Configuration classes for NPP-RL environment.

This module provides structured configuration management for the NppEnvironment
and its various components, replacing the previous parameter explosion pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

# Import RewardConfig for reward system integration
from .reward_calculation.reward_config import RewardConfig


@dataclass
class RenderConfig:
    """Configuration for rendering and visualization.

    Rendering is only used for human playback, video capture, and debug overlays.
    It is decoupled from the policy observation path (which is structured tensors
    only — see Layer 1 observation contract).
    """

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
    use_precomputed_pbrs: bool = False  # Use precomputed surface-graph PBRS (.npz) for per-frame rewards
    value_function_dir: Optional[str] = None  # Directory containing {level_id}_surface_pbrs.npz files

    def __post_init__(self):
        """Validate PBRS configuration."""
        if self.pbrs_gamma < 0 or self.pbrs_gamma > 1:
            raise ValueError("pbrs_gamma must be between 0 and 1")


@dataclass
class GraphConfig:
    """Configuration for internal graph infrastructure used by reward shaping.

    The policy observation does NOT include graph tensors. Graph state is only
    retained as reward-side infrastructure (A* cache, path distance calculator,
    PBRS). See Linear I.2 (Retain Surface Graph as Offline-Only Curriculum Tool).
    """

    debug: bool = False

    def __post_init__(self):
        """Validate graph configuration."""
        if self.debug:
            logging.info("Graph debug mode enabled")


@dataclass
class ReachabilityConfig:
    """Configuration for reachability analysis (reward-side only)."""

    debug: bool = False

    def __post_init__(self):
        """Validate reachability configuration."""
        if self.debug:
            logging.info("Reachability debug mode enabled")


@dataclass
class EnvironmentConfig:
    """Main configuration class for NppEnvironment.

    The policy observation dict is fixed to the Layer 1 contract
    (`tile_grid`, `entity_overlays`, `hazard_map`, `game_state`) plus
    `action_mask` / `duration_mask`. There are no visual, graph, reachability,
    or spatial_context observations exposed to the policy.
    """

    # Basic environment settings
    seed: Optional[int] = None
    eval_mode: bool = False
    custom_map_path: Optional[str] = None
    test_dataset_path: Optional[str] = None  # Path to test dataset for evaluation
    enable_logging: bool = False
    enable_profiling: bool = (
        False  # Enable detailed performance profiling (~5% overhead)
    )

    # Component configurations
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

    # Layer 3 FiGAR macro-action safety masking
    enable_layer3_macro_action_mask: bool = False
    layer3_duration_bins: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        16,
        20,
        24,
        32,
        48,
        64,
    )
    layer3_all_masked_fallback: str = "noop"

    def __post_init__(self):
        """Validate environment configuration."""
        if self.enable_logging:
            logging.basicConfig(level=logging.DEBUG)
            logging.info("Environment logging enabled")

        if self.layer3_all_masked_fallback not in {"noop"}:
            raise ValueError(
                "layer3_all_masked_fallback must currently be 'noop'"
            )

    @classmethod
    def for_training(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for training."""
        return cls(**kwargs)

    @classmethod
    def for_evaluation(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for evaluation."""
        return cls(eval_mode=True, **kwargs)

    @classmethod
    def for_visual_testing(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration for human playback / video capture.

        The policy observation is identical; this only enables the pygame window
        for interactive debugging.
        """
        return cls(
            render=RenderConfig(render_mode="human"),
            frame_skip=1,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for backward compatibility."""
        return {
            "seed": self.seed,
            "eval_mode": self.eval_mode,
            "custom_map_path": self.custom_map_path,
            "enable_logging": self.enable_logging,
            "render_mode": self.render.render_mode,
            "enable_animation": self.render.enable_animation,
            "enable_debug_overlay": self.render.enable_debug_overlay,
            "pbrs_gamma": self.pbrs.pbrs_gamma,
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
            "debug": self.graph.debug or self.reachability.debug,
            "frame_skip": self.frame_skip,
            "enable_layer3_macro_action_mask": self.enable_layer3_macro_action_mask,
            "layer3_duration_bins": self.layer3_duration_bins,
            "layer3_all_masked_fallback": self.layer3_all_masked_fallback,
        }


def validate_config(config: EnvironmentConfig) -> bool:
    """Validate environment configuration for common issues."""
    try:
        if config.render.render_mode == "human" and not config.render.enable_animation:
            logging.info(
                "Human render mode without animation - consider enabling animation for better visualization"
            )

        if config.render.enable_debug_overlay and not config.enable_logging:
            logging.warning(
                "Debug overlay enabled without logging - consider enabling logging for debug info"
            )

        return True

    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
