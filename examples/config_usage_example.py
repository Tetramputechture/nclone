#!/usr/bin/env python3
"""
Example demonstrating the new configuration system for NPP-RL environments.

This example shows how to use the new EnvironmentConfig classes instead of
the old parameter explosion pattern.
"""

from nclone.gym_environment.config import EnvironmentConfig, HierarchicalConfig
from nclone.gym_environment.environment_factory import create_hierarchical_env, create_training_env


def main():
    print("=== NPP-RL Environment Configuration Examples ===\n")
    
    # Example 1: Using predefined configurations
    print("1. Using predefined configurations:")
    
    # Training configuration
    training_config = EnvironmentConfig.for_training()
    print(f"   Training config - PBRS enabled: {training_config.pbrs.enable_pbrs}")
    print(f"   Training config - Render mode: {training_config.render.render_mode}")
    
    # Evaluation configuration
    eval_config = EnvironmentConfig.for_evaluation()
    print(f"   Evaluation config - PBRS enabled: {eval_config.pbrs.enable_pbrs}")
    print(f"   Evaluation config - Eval mode: {eval_config.eval_mode}")
    
    # Research configuration
    research_config = EnvironmentConfig.for_research()
    print(f"   Research config - Render mode: {research_config.render.render_mode}")
    print(f"   Research config - Debug overlay: {research_config.render.enable_debug_overlay}")
    
    # Hierarchical configuration
    hierarchical_config = EnvironmentConfig.for_hierarchical_training()
    print(f"   Hierarchical config - Hierarchical enabled: {hierarchical_config.hierarchical.enable_hierarchical}")
    print(f"   Hierarchical config - Subtask rewards: {hierarchical_config.hierarchical.enable_subtask_rewards}")
    
    print()
    
    # Example 2: Custom configuration
    print("2. Creating custom configurations:")
    
    custom_config = EnvironmentConfig(
        seed=42,
        eval_mode=False,
        enable_logging=True
    )
    
    # Customize hierarchical settings
    custom_config.hierarchical.enable_hierarchical = True
    custom_config.hierarchical.subtask_reward_scale = 0.2
    custom_config.hierarchical.max_subtask_steps = 500
    
    # Customize render settings
    custom_config.render.render_mode = "human"
    custom_config.render.enable_animation = True
    
    print(f"   Custom config - Seed: {custom_config.seed}")
    print(f"   Custom config - Hierarchical scale: {custom_config.hierarchical.subtask_reward_scale}")
    print(f"   Custom config - Max subtask steps: {custom_config.hierarchical.max_subtask_steps}")
    
    print()
    
    # Example 3: Using configurations with factory functions
    print("3. Using configurations with factory functions:")
    
    # New way (preferred)
    print("   Creating environment with config object...")
    env1 = create_hierarchical_env(config=hierarchical_config)
    print(f"   Environment created successfully")
    
    # Backward compatibility (deprecated but still works)
    print("   Creating environment with individual parameters (deprecated)...")
    env2 = create_hierarchical_env(
        enable_subtask_rewards=True,
        subtask_reward_scale=0.15,
        debug=True
    )
    print(f"   Environment created successfully")
    
    print()
    
    # Example 4: Configuration validation
    print("4. Configuration validation:")
    
    try:
        # This should work fine
        valid_config = EnvironmentConfig.for_training()
        print("   Valid configuration created successfully")
        
        # This should raise a validation error
        invalid_config = EnvironmentConfig()
        invalid_config.hierarchical.subtask_reward_scale = -1.0  # Invalid negative value
        invalid_config.hierarchical.__post_init__()  # Trigger validation
        
    except ValueError as e:
        print(f"   Validation caught invalid configuration: {e}")
    
    print()
    
    # Example 5: Converting config to dict for backward compatibility
    print("5. Converting config to dictionary:")
    
    config_dict = training_config.to_dict()
    print(f"   Config as dict has {len(config_dict)} keys")
    print(f"   Sample keys: {list(config_dict.keys())[:5]}...")
    
    print("\n=== Configuration Examples Complete ===")


if __name__ == "__main__":
    main()