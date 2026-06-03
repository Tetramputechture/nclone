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

    pbrs_gamma: float = 1.0  # Retained for BSRS γ matching on the trainer side

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
    # When True, compute full next-hop + directional-platform features (3–6) in
    # ``compute_reachability_features_from_graph``. Layer 1 policy does not use
    # them; default False saves several ms per step on reachability cache miss.
    compute_navigation_features: bool = False

    # --- Lattice d* reward potential (substitute + geometric fallback) ------
    # When True, the HWM distance signal (_cached_switch/exit_distance) is taken
    # from the physics-aware lattice d* cost-to-go where the spawn has a finite
    # plan, falling back to the geometric A* otherwise (robust to no-route
    # levels).  Default False: opt-in per rollout so existing training dynamics
    # are unchanged until validated.  See value_function/lattice_distance.py.
    use_lattice_potential: bool = False
    # When True (and use_lattice_potential is on), the substituted lattice
    # distance includes an orienteering gold tail-cost (NN tour over only the
    # WORTHWHILE remaining gold, i.e. gold whose detour costs < lattice_gold_value
    # frames) so the PBRS potential guides toward gold worth collecting.  The tail
    # depends on the remaining-gold SET, so it's a valid (policy-invariant)
    # potential — see LatticeDistanceSource.gold_cost_to_go.  Default off (parity).
    lattice_gold_potential: bool = False
    lattice_gold_weight: float = 1.0
    # Per-gold time-saving exchange rate (frames) used by the orienteering filter;
    # mirror of reward_constants.GOLD_FRAME_VALUE. A gold enters the potential only
    # if its detour costs fewer than this many frames.
    lattice_gold_value: float = 120.0
    # Planner objective knobs threaded to the per-level precompute
    # (completion>loops>speed).  None => edge_risk module defaults.
    lattice_w_risk: Optional[float] = None
    lattice_w_loop: Optional[float] = None
    lattice_risk_cap: Optional[float] = None
    lattice_use_native: bool = True
    # When True, expose a small physics-aware navigation hint to the POLICY obs
    # under the ``lattice_nav`` key ([dx_hop, dy_hop, cost_to_go_norm, phase]).
    # Default False keeps the strict Layer-1 contract; enabling it requires the
    # feature extractor to accept the extra key.  Builds the lattice source even
    # if use_lattice_potential is False.
    expose_lattice_features: bool = False

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
    # When True, skip BaseNppEnvironment's initial ``load_map()``; the first level is
    # supplied via ``reset(options={"map_data": ...})`` (benchmark / payload-driven).
    defer_initial_map_load: bool = False
    enable_logging: bool = False
    enable_profiling: bool = (
        False  # Enable detailed performance profiling (~5% overhead)
    )
    # Seconds between "ENVIRONMENT PROFILING REPORT" blocks on stdout; 0 disables
    # printing (timings still recorded for get_profiling_summary). Use 0 for
    # multi-worker runs to avoid interleaved garbage on the console.
    profiling_console_interval: float = 10.0

    # Component configurations
    render: RenderConfig = field(default_factory=RenderConfig)
    pbrs: PBRSConfig = field(default_factory=PBRSConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)

    # Reward system configuration
    reward_config: Optional[RewardConfig] = None

    # Shared level cache (for zero-copy multi-worker training on same level)
    shared_level_cache: Optional[Any] = None

    # Physics backend: "python" (NPlayHeadless) or "native" (Rust NativePlayHeadless).
    # Native is a drop-in for the training hot path (much faster, no rendering); use
    # "python" for human render / video capture. See base_environment for the gaps.
    sim_backend: str = "python"

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
