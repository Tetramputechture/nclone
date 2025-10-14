# nclone: N++ Simulator for Deep RL

N++ game simulator with Gym-compatible interface for Deep Reinforcement Learning research.

## Overview

`nclone` is a high-performance Python-based N++ simulator designed specifically for DRL training. It provides:

- **Accurate physics simulation** matching N++ gameplay mechanics
- **Gym-compatible environment** for standard RL frameworks
- **Headless rendering** for fast training (1000+ FPS)
- **Multi-modal observations** (visual frames, physics state, graph representations)
- **Fast reachability analysis** (<1ms using OpenCV flood fill)

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
    render_mode="rgb_array",  # Headless for training
    dataset_dir="datasets/train",
    enable_graph_updates=True,  # Graph observations
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

**Comprehensive multi-modal observations for CNN, MLP, and GNN architectures.**

See **[OBSERVATION_SPACE_README.md](OBSERVATION_SPACE_README.md)** for complete documentation.

Multi-modal Dict observation:

```python
{
    # Visual Modalities (CNNs)
    'player_frame': Box(0, 255, (84, 84, 12), uint8),  # Local 12-frame temporal
    'global_view': Box(0, 255, (176, 100, 1), uint8),  # Global strategic view
    
    # State Modalities (MLPs)
    'game_state': Box(-1, 1, (26+N,), float32),        # Physics + entities
    'reachability_features': Box(0, 1, (8,), float32), # Path planning
    'entity_positions': Box(0, 1, (6,), float32),      # Key positions
    
    # Graph Modality (GNNs)
    'graph_node_feats': Box(-inf, inf, (N_MAX_NODES, F_node), float32),
    'graph_edge_index': Box(0, N_MAX_NODES-1, (2, E_MAX_EDGES), int32),
    'graph_edge_feats': Box(0, 1, (E_MAX_EDGES, F_edge), float32),
    'graph_node_mask': Box(0, 1, (N_MAX_NODES,), int32),
    'graph_edge_mask': Box(0, 1, (E_MAX_EDGES,), int32),
}
```

**game_state vector (39 dims):**
- Ninja state (12): position, velocity, contact, jump state, etc.
- Exit state (2): normalized position
- Switch state (3): position + activation status
- Mine states (20): positions + activation for 10 mines
- Time remaining (1)
- Vector to switch (2)
- Vector to exit (2)

**reachability_features (8 dims):**
- Area ratio (explored vs total)
- Distance to switch (normalized)
- Distance to exit (normalized)
- Reachable switches count
- Reachable hazards count
- Connectivity score
- Exit reachable (boolean)
- Path exists to exit (boolean)

**entity_positions (6 dims):**
- `[0:2]`: Ninja (x, y)
- `[2:4]`: Switch (x, y)
- `[4:6]`: Exit (x, y)

### Reward Structure

nclone implements a sophisticated multi-component reward system designed for effective RL training, following best practices from reward shaping research.

**Quick Reference:**

| Component | Value | Description |
|-----------|-------|-------------|
| **Level Completion** | +1.0 | Primary success signal |
| **Death** | -0.5 | Moderate penalty for failure |
| **Switch Activation** | +0.1 | Intermediate milestone |
| **Time Penalty** | -0.01/step | Encourages efficiency |
| **Navigation** | ~0.0001 | Distance-based shaping |
| **Exploration** | 0.001-0.004 | Multi-scale spatial coverage |

**Key Features:**
- ✅ **No Magic Numbers**: All constants documented in `reward_constants.py`
- ✅ **PBRS Theory**: Policy-invariant reward shaping (Ng et al. 1999)
- ✅ **Multi-Scale**: Terminal, milestone, and dense shaping signals
- ✅ **Production Ready**: Validation, presets, comprehensive testing

**Configuration Presets:**

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,  # Fast completion (default)
    get_safe_navigation_config,      # Safety-constrained navigation
    get_exploration_focused_config,  # Maximum exploration
    get_minimal_shaping_config,      # Sparse rewards only
)

# Use preset configuration
config = get_completion_focused_config()
```

**Detailed Documentation:** See [docs/REWARD_SYSTEM.md](docs/REWARD_SYSTEM.md) for comprehensive reward system documentation including:
- Theoretical foundation (PBRS, ICM, count-based exploration)
- All reward constants with rationale
- Configuration presets and examples
- Best practices and troubleshooting
- Research references

## Configuration

### Environment Creation Options

```python
env = NPPEnvironment(
    render_mode="rgb_array",        # "human" or "rgb_array"
    dataset_dir="datasets/train",   # Level dataset directory
    enable_graph_updates=True,      # Enable graph observations
    curriculum_level=0,             # Curriculum stage (0-4)
    enable_mines=True,              # Include mine entities
    max_episode_steps=20000,        # Timeout (frames at 60 FPS)
    frame_skip=1,                   # Action repeat
    temporal_frames=12,             # Temporal stack size
)
```

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
| Headless (rgb_array) | 1000-2000 | RL training |
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
    enable_graph_updates=True,
    graph_config={
        'node_features': 8,     # Simplified node features
        'edge_features': 4,     # Simplified edge features
        'max_nodes': 100,       # Maximum nodes in graph
        'max_edges': 400        # Maximum edges in graph
    }
)
```

Graph structure:
- Nodes represent level geometry and entities
- Edges represent spatial relationships
- Automatically computed and cached

### Mine State Tracking

Detailed mine interaction tracking:

```python
from nclone.gym_environment.mine_state_processor import MineStateProcessor

processor = MineStateProcessor()
obs = env.reset()

# Access mine states
mine_states = processor.extract_mine_states(obs)
# Returns: [(x, y, active, toggled), ...] for each mine
```

### Frame Augmentation

Consistent data augmentation for training:

```python
from nclone.gym_environment.frame_augmentation import (
    apply_consistent_augmentation,
    get_recommended_config
)

config = get_recommended_config()
augmented_frame = apply_consistent_augmentation(frame, config, seed=42)
```

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
