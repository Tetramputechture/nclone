# nclone: N++ Simulator for Deep RL

N++ game simulator with Gym-compatible interface for Deep Reinforcement Learning research.

## Overview

`nclone` is a high-performance Python-based N++ simulator designed specifically for DRL training. It provides:

- **Accurate physics simulation** matching N++ gameplay mechanics
- **Gym-compatible environment** for standard RL frameworks
- **Headless rendering** for fast training (1000+ FPS)
- **Multi-modal observations** (visual frames, physics state, graph representations)

## Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/Tetramputechture/nclone.git
cd nclone
pip install -e .

# Or install directly
pip install git+https://github.com/Tetramputechture/nclone.git
```

**Dependencies:**
- Python 3.8+
- pygame>=2.5.0
- numpy>=1.24.0
- opencv-python>=4.8.0
- pycairo>=1.25.0
- gymnasium>=0.29.0

## Quick Start

### Interactive Play

Test the environment with keyboard controls:

```bash
# Run test environment
python -m nclone.test_environment

# Controls:
#   Arrow keys / WASD: Move
#   Space / Up: Jump
#   R: Reset level
```

### RL Training Integration

```python
from nclone import NppEnvironment

# Create environment
env = NppEnvironment(
    render_mode="grayscale_array",  # Headless for training
    dataset_dir="datasets/train",
    config=EnvironmentConfig(
        graph=GraphConfig()  # Graph for PBRS and observations
    ),
    curriculum_level=0  # Difficulty level
)

# Standard Gym interface
obs, info = env.reset()
done = False

while not done:
    action = policy(obs)  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Environment Interface

### Action Space

Discrete(6) actions:
- 0: No-op
- 1: Left
- 2: Right
- 3: Jump
- 4: Left + Jump
- 5: Right + Jump

### Observation Space

Multi-modal observations for CNN, MLP, and GNN architectures.

```python
{
    # Visual (CNNs)
    'player_frame': Box(0, 255, (84, 84, 1), uint8),  # or (stack_size, 84, 84, 1) if stacked
    'global_view': Box(0, 255, (176, 100, 1), uint8),  # or (stack_size, 176, 100, 1) if stacked
    
    # State vectors (MLPs)
    'game_state': Box(-1, 1, (26,), float32),  # or (stack_size, 26) if stacked
    'reachability_features': Box(0, 1, (8,), float32),
    'entity_positions': Box(0, 1, (6,), float32),
    'switch_states': Box(0, 1, (25,), float32),
    
    # Graph (GNNs, optional - memory-optimized)
    'graph_node_feats': Box(-inf, inf, (max_nodes, 4), float32),  # Memory-optimized: 4 dims
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), uint16),
    ...
}
```

**Validate observations:**
```bash
python tools/validate_observations.py --episodes 3
```

**Full documentation:** [OBSERVATION_SPACE_README.md](OBSERVATION_SPACE_README.md)

### Frame Stacking

Frame stacking provides temporal information to the policy by stacking consecutive observations. This technique, popularized by DQN (Mnih et al., 2015), allows the agent to infer velocity, acceleration, and motion dynamics.

**Configuration:**

```python
from nclone.gym_environment import create_training_env, EnvironmentConfig, FrameStackConfig

# Enable frame stacking
config = EnvironmentConfig.for_training()
config.frame_stack = FrameStackConfig(
    enable_visual_frame_stacking=True,
    visual_stack_size=4,  # Stack 4 visual frames
    enable_state_stacking=True,
    state_stack_size=4,  # Stack 4 game states
    padding_type="zero"  # or "repeat"
)

env = create_training_env(config)
```

**Key Features:**
- **Independent stacking**: Visual frames and game states can be stacked independently with different stack sizes (2-12 frames)
- **Consistent augmentation**: When frame augmentation is enabled, the same transform is applied across all frames in the stack to maintain temporal coherence
- **Configurable padding**: Choose between zero padding or repeating the initial frame

**References:**
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature* 518, 529-533. https://doi.org/10.1038/nature14236
- Machado et al. (2018). "Revisiting the Arcade Learning Environment." *IJCAI* 61, 523-562.

### Reward Structure

nclone implements a multi-component reward system designed for effective RL training, following best practices from reward shaping research.

**Quick Reference:**

| Component | Value | Description |
|-----------|-------|-------------|
| **Level Completion** | +1.0 | Primary success signal |
| **Death** | -0.5 | Moderate penalty for failure |
| **Switch Activation** | +0.1 | Intermediate milestone |
| **Time Penalty** | -0.01/step | Encourages efficiency |
| **Navigation** | ~0.0001 | Distance-based shaping |
| **Exploration** | 0.001-0.004 | Multi-scale spatial coverage |

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,  # Fast completion (default)
    get_safe_navigation_config,      # Safety-constrained navigation
    get_exploration_focused_config,  # Maximum exploration
    get_minimal_shaping_config,      # Sparse rewards only
)

### Curriculum Levels

Progressive difficulty stages:

| Level | Description | Entity Count | Complexity |
|-------|-------------|--------------|------------|
| 0 | Simple navigation | Exit only | Low |
| 1 | Switch activation | Exit + Switch | Low-Med |
| 2 | Basic hazards | + 2-3 Mines | Medium |
| 3 | Complex hazards | + 5-7 Mines | Med-High |
| 4 | Advanced puzzles | + Locked doors | High |

## Performance

### Benchmarks

| Mode | FPS | Use Case |
|------|-----|----------|
| Headless (grayscale_array) | 1000-2000 | RL training |
| Rendered (human) | 60 | Interactive play/debug |
| Multi-process (64 envs) | 30K+ steps/sec | Distributed training |

**Reachability Analysis:**
- Computation time: <1ms per step
- Uses OpenCV flood fill for efficiency
- Cached and updated incrementally

### Memory Usage

- Single environment: ~200MB RAM
- 64 parallel environments: ~6GB RAM
- Graph observations add ~10% overhead

## Datasets

### Level Format

Levels stored as JSON with Metanet format compatibility:

```json
{
    "id": "level_001",
    "tiles": [...],
    "objects": [
        {"type": "exit", "position": [x, y]},
        {"type": "switch", "position": [x, y]},
        {"type": "mine", "position": [x, y], "active": true}
    ],
    "spawn_point": [x, y]
}
```

### Creating Custom Datasets

```python
from nclone.level_loader import LevelLoader

# Load custom levels
loader = LevelLoader(dataset_dir="my_levels/")
levels = loader.load_all_levels()

# Use in environment
env = NPPEnvironment(dataset_dir="my_levels/")
```

## Advanced Features

### Graph Observations

Enable structural level understanding:

```python
env = NPPEnvironment(
    config=EnvironmentConfig(
        graph=GraphConfig()
    ),
    graph_config={
        'node_features': 56,    # Comprehensive node features (reduced from 61)
        'edge_features': 6,     # Comprehensive edge features
        'max_nodes': 100,       # Maximum nodes in graph
        'max_edges': 400        # Maximum edges in graph
    }
)

# Note: Feature dimensions come from nclone.graph.common:
# - NODE_FEATURE_DIM = 50 (3 spatial + 7 type + 5 entity + 34 tile + 1 reachability)
# - EDGE_FEATURE_DIM = 6 (edge type, connectivity)
```

Graph structure:
- Nodes represent level geometry and entities
- Edges represent spatial relationships
- Automatically computed and cached

## Development

### Running Tests

```bash
# Run test suite
pytest tests/

# Run specific test
pytest tests/test_npp_environment.py -xvs

# With coverage
pytest --cov=nclone --cov-report=html
```

### Code Quality

```bash
# Install dev tools
make dev-setup

# Lint code
make lint

# Auto-fix issues
make fix

# Remove unused imports
make imports
```

### Profiling

```bash
# Profile environment performance
python -m cProfile -o profile.out -m nclone.test_environment
python -m pstats profile.out

# Log frame times
python -m nclone.test_environment --log-frametimes
```

## Integration with npp-rl

This simulator is designed to work with the `npp-rl` training framework:

```bash
# Install both repositories
git clone https://github.com/Tetramputechture/nclone.git
git clone https://github.com/Tetramputechture/npp-rl.git

cd nclone && pip install -e . && cd ..
cd npp-rl && pip install -r requirements.txt && cd ..

# Train agent
cd npp-rl
python scripts/train_and_compare.py \
    --architectures full_hgt \
    --train-dataset ../nclone/datasets/train \
    --total-timesteps 20000000
```

See `npp-rl/README.md` for full training instructions.
