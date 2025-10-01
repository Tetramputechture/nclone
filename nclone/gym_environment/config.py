"""
Configuration classes for NPP-RL environment.

This module provides structured configuration management for the NppEnvironment
and its various components, replacing the previous parameter explosion pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging


@dataclass
class RenderConfig:
    """Configuration for rendering and visualization."""
    render_mode: str = "rgb_array"
    enable_animation: bool = False
    enable_debug_overlay: bool = False
    
    def __post_init__(self):
        """Validate render configuration."""
        valid_modes = ["human", "rgb_array"]
        if self.render_mode not in valid_modes:
            raise ValueError(f"render_mode must be one of {valid_modes}")


@dataclass
class PBRSConfig:
    """Configuration for Potential-Based Reward Shaping."""
    enable_pbrs: bool = True
    pbrs_weights: Dict[str, float] = field(default_factory=lambda: {
        "objective_weight": 1.0,
        "hazard_weight": 0.5,
        "impact_weight": 0.3,
        "exploration_weight": 0.2,
    })
    pbrs_gamma: float = 0.99
    
    def __post_init__(self):
        """Validate PBRS configuration."""
        if self.pbrs_gamma < 0 or self.pbrs_gamma > 1:
            raise ValueError("pbrs_gamma must be between 0 and 1")
        
        required_weights = ["objective_weight", "hazard_weight", "impact_weight", "exploration_weight"]
        for weight in required_weights:
            if weight not in self.pbrs_weights:
                logging.warning(f"Missing PBRS weight: {weight}, using default 0.0")
                self.pbrs_weights[weight] = 0.0


@dataclass
class GraphConfig:
    """Configuration for graph-based features."""
    enable_graph_updates: bool = True
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
    enable_logging: bool = False
    enable_short_episode_truncation: bool = False
    
    # Component configurations
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
    def for_training(cls, **kwargs) -> 'EnvironmentConfig':
        """Create configuration optimized for training."""
        config = cls(
            render=RenderConfig(render_mode="rgb_array"),
            pbrs=PBRSConfig(enable_pbrs=True),
            graph=GraphConfig(enable_graph_updates=True),
            reachability=ReachabilityConfig(enable_reachability=True),
            enable_short_episode_truncation=True,
            **kwargs
        )
        return config
    
    @classmethod
    def for_evaluation(cls, **kwargs) -> 'EnvironmentConfig':
        """Create configuration optimized for evaluation."""
        config = cls(
            render=RenderConfig(render_mode="rgb_array"),
            pbrs=PBRSConfig(enable_pbrs=False),  # Clean evaluation
            graph=GraphConfig(enable_graph_updates=True),
            reachability=ReachabilityConfig(enable_reachability=True),
            eval_mode=True,
            enable_short_episode_truncation=False,  # Let episodes run to completion
            **kwargs
        )
        return config
    
    @classmethod
    def for_research(cls, **kwargs) -> 'EnvironmentConfig':
        """Create configuration optimized for research and debugging."""
        config = cls(
            render=RenderConfig(
                render_mode="human",
                enable_animation=True,
                enable_debug_overlay=True
            ),
            pbrs=PBRSConfig(enable_pbrs=True),
            graph=GraphConfig(enable_graph_updates=True, debug=True),
            reachability=ReachabilityConfig(enable_reachability=True, debug=True),
            enable_logging=True,
            **kwargs
        )
        return config
    
    @classmethod
    def for_hierarchical_training(cls, completion_planner=None, **kwargs) -> 'EnvironmentConfig':
        """Create configuration optimized for hierarchical RL training."""
        config = cls(
            render=RenderConfig(render_mode="rgb_array"),
            pbrs=PBRSConfig(enable_pbrs=True),
            graph=GraphConfig(enable_graph_updates=True),
            reachability=ReachabilityConfig(enable_reachability=True),
            hierarchical=HierarchicalConfig(
                enable_hierarchical=True,
                completion_planner=completion_planner,
                enable_subtask_rewards=True
            ),
            enable_short_episode_truncation=True,
            **kwargs
        )
        return config
    
    @classmethod
    def minimal(cls, **kwargs) -> 'EnvironmentConfig':
        """Create minimal configuration with all advanced features disabled."""
        config = cls(
            render=RenderConfig(render_mode="rgb_array"),
            pbrs=PBRSConfig(enable_pbrs=False),
            graph=GraphConfig(enable_graph_updates=False),
            reachability=ReachabilityConfig(enable_reachability=False),
            hierarchical=HierarchicalConfig(enable_hierarchical=False),
            **kwargs
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
            
            # Render settings
            "render_mode": self.render.render_mode,
            "enable_animation": self.render.enable_animation,
            "enable_debug_overlay": self.render.enable_debug_overlay,
            
            # PBRS settings
            "enable_pbrs": self.pbrs.enable_pbrs,
            "pbrs_weights": self.pbrs.pbrs_weights,
            "pbrs_gamma": self.pbrs.pbrs_gamma,
            
            # Graph settings
            "enable_graph_updates": self.graph.enable_graph_updates,
            
            # Reachability settings
            "enable_reachability": self.reachability.enable_reachability,
            
            # Hierarchical settings
            "enable_hierarchical": self.hierarchical.enable_hierarchical,
            "completion_planner": self.hierarchical.completion_planner,
            "enable_subtask_rewards": self.hierarchical.enable_subtask_rewards,
            "subtask_reward_scale": self.hierarchical.subtask_reward_scale,
            "max_subtask_steps": self.hierarchical.max_subtask_steps,
            
            # Debug settings
            "debug": self.graph.debug or self.reachability.debug or self.hierarchical.debug,
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
        if config.hierarchical.enable_hierarchical and not config.reachability.enable_reachability:
            raise ValueError(
                "Hierarchical RL requires reachability analysis to be enabled. "
                "Set reachability.enable_reachability=True in your configuration."
            )
        
        if config.render.render_mode == "human" and not config.render.enable_animation:
            logging.info("Human render mode without animation - consider enabling animation for better visualization")
        
        # Check for performance implications
        if config.render.enable_debug_overlay and not config.enable_logging:
            logging.warning("Debug overlay enabled without logging - consider enabling logging for debug info")
        
        return True
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False