# nclone Repository Guide

## Repository Purpose
nclone is a Pygame-based simulation of the game N++ designed specifically for Deep Reinforcement Learning (DRL) research. It features a hierarchical graph system for AI pathfinding, custom reward systems for DRL agents, and supports both interactive and headless modes for training and experimentation.

## Key Features
- **N++ Game Simulation**: Replicates core N++ gameplay mechanics with physics, entities, and level progression
- **Deep RL Environment**: Gym-compatible environments with custom reward systems for DRL agent training
- **Hierarchical Graph System**: Multi-resolution graph processing (6px, 24px, 96px) for AI pathfinding and navigation
- **Headless Mode**: Fast training without graphics for DRL experimentation
- **Graph Visualization**: Debug overlay system for analyzing pathfinding graphs and AI behavior

## Setup Instructions

### Prerequisites
- Python 3.8+ (tested with 3.8-3.11)
- Virtual environment recommended

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package (editable for development)
pip install -e .

# Verify installation
python -m nclone.test_environment --help
```

### Development Setup
```bash
# Install development tools
make dev-setup

# Run linting
make lint

# Auto-fix code issues
make fix
```

## Repository Structure

### Core Package (`nclone/`)
- **Simulation Core**:
  - `nsim.py`: Physics and game logic engine
  - `ninja.py`: Player character state machine and controls
  - `physics.py`: Physics constants and helpers
  - `entities.py`: Game entity definitions and interactions

- **Graph System** (Primary Architecture):
  - `graph/hierarchical_builder.py`: Multi-resolution graph builder
  - `graph/edge_building.py`: Edge creation with functional/walkable/jump edges
  - `graph/pathfinding.py`: A* pathfinding with graph navigation
  - `graph/precise_collision.py`: Collision detection with ninja radius awareness
  - `graph/visualization.py`: Graph debug overlay rendering

- **Environments** (Gym-compatible):
  - `nclone_environments/basic_level_no_gold/`: Main RL environment
  - `test_environment.py`: Interactive testing environment

- **Rendering**:
  - `nsim_renderer.py`: Main game renderer
  - `debug_overlay_renderer.py`: Graph visualization overlay
  - `tile_renderer.py`, `entity_renderer.py`: Component renderers

### Testing & Debugging
- **`tests/`**: Comprehensive test suite with 23+ test files
  - `test_graph_fixes_unit_tests.py`: **Main test suite** for graph system
  - `run_tests.py`: Test runner script
- **`debug/`**: 45+ debugging and analysis scripts
  - `final_validation.py`: Comprehensive system validation
  - Various analysis scripts for graph connectivity, pathfinding, etc.

### Documentation (`docs/`)
- `sim_mechanics_doc.md`: Detailed simulation mechanics
- `GRAPH_VISUALIZATION_GUIDE.md`: Graph system documentation
- `pathfinding_strategy.md`: Pathfinding design notes
- `FILE_INDEX.md`: Module descriptions

## Development Guidelines

### Code Quality
- Use `make lint` before committing
- Run `make fix` to auto-fix issues including unused imports
- Follow Python 3.8+ compatibility
- Maintain test coverage for new features

### Testing
```bash
# Run main test suite
python tests/test_graph_fixes_unit_tests.py

# Run all tests
python tests/run_tests.py

# Run specific test categories
python -m pytest tests/ -v
```

### Graph System Development
- The graph system is the core architecture for AI pathfinding
- Always test graph changes with `debug/final_validation.py`
- Use graph visualization overlay to debug pathfinding issues
- Maintain compatibility with multi-resolution levels (6px, 24px, 96px)

### Common Development Tasks

#### Running the Simulation
```bash
# Interactive mode with keyboard controls
python -m nclone.test_environment

# With frametime logging
python -m nclone.test_environment --log-frametimes
```

#### Headless Mode for Training
```bash
# Single headless simulation
python -m nclone.nplay_headless

# Multiple parallel simulations
python -m nclone.run_multiple_headless --num-simulations 4 --num-steps 50000
```

#### Graph System Debugging
```bash
# Comprehensive validation
python debug/final_validation.py

# Specific graph analysis
python debug/analyze_graph_fragmentation.py
python debug/debug_pathfinding.py
```

## Important Technical Details

### Graph System Architecture
- **Multi-resolution**: 6px (fine), 24px (medium), 96px (coarse) grid levels
- **Edge Types**: WALK (green), JUMP (orange), FALL (blue), FUNCTIONAL (yellow)
- **Collision Detection**: 10-pixel ninja radius awareness for traversability
- **Pathfinding**: A* algorithm with graph-based navigation

### Environment Integration
- Gym-compatible environments in `nclone_environments/`
- Reward systems designed for DRL training
- Observation processing for RL agents
- Action space: movement and jumping controls

### Performance Considerations
- Headless mode for fast training (no rendering overhead)
- Multi-process support for parallel simulations
- Efficient graph caching and performance optimizations
- Sub-grid resolution for accurate spatial representation

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Ensure `pip install -e .` was run in project root
2. **Graph Visualization Issues**: Use debug scripts in `debug/` directory
3. **Pathfinding Problems**: Run `debug/final_validation.py` for comprehensive analysis
4. **Performance Issues**: Use headless mode and check for efficient graph usage

### Dependencies
- **Core**: numpy, pygame, gymnasium
- **Development**: pytest, ruff (linting), black (formatting)
- **Optional**: pytest-cov (coverage), sphinx (docs)

## Git Workflow
- Work on feature branches, not main
- Use descriptive commit messages
- Run tests before pushing
- The repository uses standard GitHub workflow (no CI/CD currently configured)

## Related Repositories
- **npp-rl**: Sibling repository containing the DRL training stack and PPO agents
- This repository provides the environment; npp-rl provides the training infrastructure