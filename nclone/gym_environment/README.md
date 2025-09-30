# NPP-RL Environment Integration

This document describes the integrated NPP-RL environment system that combines all wrapper functionality directly into the base `NppEnvironment` class.

## Overview

The NPP-RL environment system has been redesigned to integrate all advanced functionality directly into the base environment class, eliminating the need for multiple wrapper layers. This provides:

- **Simplified API**: Single environment class with all features
- **Better Performance**: No wrapper overhead
- **Easier Configuration**: Factory functions for common use cases
- **Vectorization Support**: Built-in support for parallel training
- **Backward Compatibility**: Legacy wrapper interfaces still work (with deprecation warnings)

## Key Features

### 1. Dynamic Graph Updates
- Real-time graph construction using nclone's hierarchical graph builder
- Switch state tracking for dynamic connectivity updates
- HGT-compatible graph observations
- Sub-millisecond performance for real-time RL training

### 2. Reachability Analysis
- Simplified 8-dimensional reachability features
- Flood-fill based connectivity analysis
- Performance-optimized with caching
- Strategic information for path planning

### 3. Vectorization Support
- Built-in pickle/unpickle support for SubprocVecEnv
- Proper initialization and cleanup
- Factory functions for creating multiple environments

## Quick Start

### Basic Usage

```python
from nclone.gym_environment import create_training_env

# Create environment with all features enabled
env = create_training_env(
    enable_graph_updates=True,
    enable_reachability=True,
    debug=False
)

# Use like any Gym environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Factory Functions

```python
from nclone.gym_environment import (
    create_training_env,      # For training with PBRS and all features
    create_evaluation_env,    # For evaluation without PBRS
    create_research_env,      # For research with debug features
    create_minimal_env,       # Minimal configuration for baselines
)

# Training environment
train_env = create_training_env()

# Evaluation environment (no PBRS, full episodes)
eval_env = create_evaluation_env()

# Research environment (visual rendering, debug overlay)
research_env = create_research_env(enable_debug_overlay=True)

# Minimal environment (no advanced features)
baseline_env = create_minimal_env()
```

### Vectorized Training

```python
from nclone.gym_environment import create_vectorized_training_envs
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create multiple environment factories
env_factories = create_vectorized_training_envs(
    num_envs=8,
    enable_graph_updates=True,
    enable_reachability=True
)

# Create vectorized environment
vec_env = SubprocVecEnv(env_factories)

# Use with any RL library
obs = vec_env.reset()
actions = [vec_env.action_space.sample() for _ in range(8)]
obs, rewards, dones, infos = vec_env.step(actions)
```

## Observation Space

The integrated environment provides a rich observation space:

### Core Observations
- `player_frame`: Player-centered view (84x84x12)
- `global_view`: Full level view (176x100x1)  
- `game_state`: Enhanced ninja and entity states (3891-dim)

### Graph Observations (if enabled)
- `graph_node_feats`: Node features [N_MAX_NODES, 3]
- `graph_edge_index`: Edge connectivity [2, E_MAX_EDGES]
- `graph_edge_feats`: Edge features [E_MAX_EDGES, 1]
- `graph_node_mask`: Valid node indicators [N_MAX_NODES]
- `graph_edge_mask`: Valid edge indicators [E_MAX_EDGES]
- `graph_node_types`: Node type IDs [N_MAX_NODES]
- `graph_edge_types`: Edge type IDs [E_MAX_EDGES]

### Reachability Observations (if enabled)
- `reachability_features`: 8-dimensional reachability analysis
  1. Reachable area ratio (0-1)
  2. Distance to nearest switch (normalized)
  3. Distance to exit (normalized)
  4. Reachable switches count (normalized)
  5. Reachable hazards count (normalized)
  6. Connectivity score (0-1)
  7. Exit reachable flag (0-1)
  8. Switch-to-exit path exists (0-1)

## Configuration Options

### Environment Parameters

```python
env = NppEnvironment(
    # Core settings
    render_mode="rgb_array",           # "rgb_array" or "human"
    enable_animation=False,            # Enable visual animations
    enable_logging=False,              # Enable debug logging
    enable_debug_overlay=False,        # Enable visual debug overlay
    seed=None,                         # Random seed
    
    # Episode settings
    eval_mode=False,                   # Use evaluation maps
    enable_short_episode_truncation=False,  # Truncate on lack of progress
    custom_map_path=None,              # Path to custom map
    
    # Reward shaping
    enable_pbrs=True,                  # Enable potential-based reward shaping
    pbrs_weights={                     # PBRS component weights
        "objective_weight": 1.0,
        "hazard_weight": 0.5,
        "impact_weight": 0.3,
        "exploration_weight": 0.2,
    },
    pbrs_gamma=0.99,                   # PBRS discount factor
    
    # Integrated features
    enable_graph_updates=True,         # Enable dynamic graph updates
    enable_reachability=True,          # Enable reachability analysis
    debug=False,                       # Enable debug logging for graph ops
)
```

## Performance

The integrated environment is designed for high performance:

- **Graph Updates**: Sub-millisecond performance for real-time training
- **Reachability Analysis**: <1ms computation with caching
- **Vectorization**: Efficient parallel processing support
- **Memory Usage**: Optimized for batch processing

### Performance Monitoring

```python
# Get performance statistics
graph_stats = env.get_graph_performance_stats()
reachability_stats = env.get_reachability_performance_stats()

# Benchmark environment
from nclone.gym_environment import benchmark_environment_performance
results = benchmark_environment_performance(env, num_steps=1000, target_fps=60.0)
```

## Migration from Wrappers

If you were using the old wrapper system, migration is straightforward:

### Old Code (No longer available)
```python
# This approach is no longer supported - wrapper classes have been removed
from nclone.gym_environment import (
    NppEnvironment,
    DynamicGraphWrapper,      # REMOVED
    ReachabilityWrapper,      # REMOVED  
    VectorizationWrapper      # REMOVED
)

base_env = NppEnvironment()
graph_env = DynamicGraphWrapper(base_env)  # No longer works
reach_env = ReachabilityWrapper(graph_env)  # No longer works
final_env = VectorizationWrapper(reach_env)  # No longer works
```

### New Code (Recommended)
```python
from nclone.gym_environment import create_training_env

# All functionality integrated
env = create_training_env(
    enable_graph_updates=True,
    enable_reachability=True
)
```

## Migration from Legacy Wrappers

Legacy wrapper classes have been **removed** and are no longer available. The functionality has been fully integrated into `NppEnvironment`. Here's the migration guide:

**Removed Classes:**
- `DynamicGraphWrapper` → Use `create_training_env(enable_graph_updates=True)`
- `ReachabilityWrapper` → Use `create_training_env(enable_reachability=True)`
- `VectorizationWrapper` → Use `create_vectorized_training_envs()`

**Legacy Factory Functions (still available):**
- `create_dynamic_graph_env` → Use `create_training_env()` (backward compatible)
- `create_reachability_aware_env` → Use `create_training_env()` (backward compatible)

**Benefits of the integrated approach:**
- Simplified codebase with no wrapper overhead
- Better performance through direct integration
- Easier configuration and debugging
- All features available in a single environment class

## Advanced Usage

### Custom Environment Configuration

```python
from nclone.gym_environment import NppEnvironment

# Create custom environment
env = NppEnvironment(
    render_mode="human",
    enable_graph_updates=True,
    enable_reachability=True,
    enable_debug_overlay=True,
    debug=True,
    pbrs_weights={
        "objective_weight": 2.0,  # Custom weights
        "hazard_weight": 0.1,
        "impact_weight": 0.5,
        "exploration_weight": 0.8,
    }
)
```

### Accessing Advanced Features

```python
# Get hierarchical graph data
hierarchical_graph = env.get_hierarchical_graph_data()
if hierarchical_graph:
    fine_graph = hierarchical_graph.fine_graph
    medium_graph = hierarchical_graph.medium_graph
    coarse_graph = hierarchical_graph.coarse_graph

# Force graph update
env.force_graph_update()

# Get current graph
current_graph = env.get_current_graph()
```

### Debug Visualization

```python
# Enable various debug overlays
env.set_graph_debug_enabled(True)
env.set_exploration_debug_enabled(True)
env.set_reachability_debug_enabled(True)

# Set reachability data for visualization
env.set_reachability_data(reachability_state, subgoals, frontiers)
```

## Testing

To test the integrated environment:

```python
from nclone.gym_environment import validate_environment, create_training_env

env = create_training_env()
is_valid = validate_environment(env)
print(f"Environment valid: {is_valid}")
```

## Performance Tips

1. **Disable unused features**: Set `enable_graph_updates=False` or `enable_reachability=False` if not needed
2. **Use minimal config for baselines**: Use `create_minimal_env()` for maximum performance
3. **Enable caching**: Reachability analysis uses caching automatically
4. **Monitor performance**: Use performance stats to identify bottlenecks
5. **Vectorize training**: Use `create_vectorized_training_envs()` for parallel training

## Troubleshooting

### Common Issues

1. **Missing observations**: Ensure features are enabled in environment configuration
2. **Performance issues**: Check if debug mode is enabled in production
3. **Vectorization problems**: Use factory functions for proper initialization
4. **Graph update failures**: Check debug logs for detailed error information

### Debug Mode

Enable debug mode for detailed logging:

```python
env = create_training_env(debug=True)
```

This will provide detailed information about graph updates, reachability computation, and performance metrics.
