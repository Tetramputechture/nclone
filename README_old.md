# nclone

`nclone` is a Pygame-based simulation of the game N++. This repository is tailored for Deep Reinforcement Learning (DRL) research. It features a custom reward system designed for DRL agents and supports headless mode for faster training and experimentation.

## Features

*   **N++ Simulation:** Replicates core gameplay mechanics of N++.
*   **Pygame-based:** Built using the Pygame library for rendering and interaction.
*   **Deep RL Focus:** Includes a reward system to guide DRL agent learning, and serves as the environment for the RL agent developed in the `npp-rl` subdirectory.
*   **Headless Mode:** Allows the simulation to run without a graphical interface, significantly speeding up DRL training processes.
*   **Customizable Environments:** The environment (`gym_environment/npp_environment.py`) can be configured for different experimental setups.
*   **Simplified Reachability System:** Ultra-fast OpenCV flood fill reachability analysis (<1ms) with 8-dimensional strategic features optimized for RL training.
*   **Hierarchical Graph Processing:** Multi-resolution graph system (6px, 24px, 96px) for efficient pathfinding and AI navigation.

## Deep Reinforcement Learning Agent

The DRL training stack lives in a separate sibling repository `npp-rl` (not vendored inside this repo). Refer to that repository for PPO training scripts, policies, and usage instructions. This repository exposes Gym-compatible environments that `npp-rl` consumes.

## Installation

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    The project uses `setuptools` for packaging. You can install it directly using pip:
    ```bash
    pip install .
    ```
    For development, you might prefer an editable install:
    ```bash
    pip install -e .
    ```
    This will install all dependencies listed in `pyproject.toml`, including Pygame, NumPy, PyCairo, and Stable Baselines3 (required for the RL agent in the `npp-rl` subdirectory).

3.  **Verify the installation:**
    After installation, you can verify that the package is correctly installed and the test environment can be found by running:
    ```bash
    python -m nclone.test_environment --help
    ```
    This command should print the help message for `test_environment.py`. If you see a `ModuleNotFoundError`, please refer to the Troubleshooting section below.

## Development: Linting and code cleanup

Use the Makefile targets to lint the codebase recursively and remove unused imports:

```bash
make dev-setup    # installs/updates Ruff in your active environment
make lint         # lint without modifying files
make fix          # auto-fix issues (includes removing unused imports)
make imports      # remove only unused imports
```

Ensure you are working inside an activated virtual environment.

## Running the Simulation (Base Environment)

After installing the package as described above, you can run the base simulation (without the RL agent directly controlling it).
To test the environment and see the simulation in action, you can run the `test_environment.py` script:

```bash
python -m nclone.test_environment
```

This script initializes the `NppEnvironment` environment in human-render mode, allowing you to control the ninja using keyboard inputs:
*   **Left/Right Arrow Keys (or A/D):** Move left/right.
*   **Space/Up Arrow Key:** Jump.
*   **R Key:** Reset the environment.

You can also run with frametime logging:
```bash
python -m nclone.test_environment --log-frametimes
```

To train or run the RL agent, please refer to the instructions in `npp-rl/README.md`.

## Headless Mode (Base Environment)

The environment can be initialized in `rgb_array` mode for headless operation, which is crucial for DRL training. This is configured in the environment's constructor. See `gym_environment/npp_environment.py` for an example of how the `render_mode` is set.

## Running Multiple Headless Simulations (Base Environment)

To leverage multi-core processors for large-scale experiments or data collection (e.g., for DRL), you can run multiple headless simulations concurrently using the `run_multiple_headless.py` script.

```bash
python -m nclone.run_multiple_headless --num-simulations 4 --num-steps 50000
```

This command will launch 4 independent headless simulations, each running for 50,000 steps. You can adjust these parameters as needed:

*   `--num-simulations`: Specifies the number of concurrent simulation instances.
*   `--num-steps`: Specifies the number of simulation steps each instance will run.

Each simulation runs in its own process, allowing for parallel execution.

## Gym Environment Observation Space

The `NppEnvironment` provides a multi-modal observation space for RL agents:

- **player_frame**: (12, 84, 84) - Temporal stack of 12 grayscale frames centered on player
- **global_view**: (176, 100) - Downsampled grayscale view of entire level
- **game_state**: (30+,) - Physics state vector including velocities, positions, forces
- **reachability_features**: (8,) - Strategic features from ultra-fast flood fill analysis
- **entity_positions**: (6,) - Normalized [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y] for hierarchical planning

All observations are normalized and ready for neural network processing. See `gym_environment/observation_processor.py` for implementation details.

## Simplified Reachability System

The reachability system has been simplified to reduce overengineering while maintaining optimal performance for RL training.

### Key Components

#### ReachabilitySystem (`nclone/graph/reachability/reachability_system.py`)
- Ultra-fast OpenCV flood fill implementation (<1ms performance)
- Tile-aware and door/switch state-aware analysis
- Clean interface for RL integration

#### Simplified Feature Extraction (`nclone/graph/reachability/compact_features.py`)
- **8-dimensional strategic features** (reduced from 64)
- Focus on connectivity and strategic information
- Lets RL system learn movement patterns through experience

#### Streamlined Edge Building (`nclone/graph/edge_building.py`)
- Basic connectivity using WALK and JUMP edge types
- Direct use of flood fill results
- Simplified from complex physics calculations

### 8-Dimensional Feature Set

1. **Reachable Area Ratio** - Proportion of level currently accessible
2. **Objective Distance** - Normalized distance to current objective  
3. **Switch Accessibility** - Fraction of important switches reachable
4. **Exit Accessibility** - Whether exit is currently reachable
5. **Hazard Proximity** - Distance to nearest reachable hazard
6. **Connectivity Score** - Overall connectivity measure
7. **Analysis Confidence** - Confidence in reachability analysis
8. **Computation Time** - Performance metric (normalized)

### Performance Characteristics

- **Reachability Analysis**: <1ms (OpenCV flood fill)
- **Feature Extraction**: <5ms (8D features)
- **Total Pipeline**: <10ms (meets RL training requirements)
- **Memory Usage**: 32 bytes per feature vector (vs 256 bytes previously)

### Usage

```python
from nclone.graph.reachability.feature_extractor import ReachabilityFeatureExtractor

# Initialize simplified extractor
extractor = ReachabilityFeatureExtractor(debug=True)

# Extract 8-dimensional strategic features
features = extractor.extract_features(
    ninja_position=(120, 120),
    level_data=level_data,
    entities=entities,
    switch_states=switch_states
)

print(f"Features shape: {features.shape}")  # (8,)
print(f"Feature names: {extractor.get_feature_names()}")
```


## Troubleshooting

### `ModuleNotFoundError: No module named 'nclone'` or `No module named 'nclone.maps'`

If you encounter these errors when trying to run the simulation (e.g., `python -m nclone.test_environment`):

1.  **Ensure you are in the correct directory:** Your terminal should be in the root of the `nclone` project directory (where `pyproject.toml` is located) when you run the installation command.
2.  **Ensure your virtual environment is activated:** If you created one, make sure it's active.
3.  **Perform a clean reinstallation:** Sometimes, previous build artifacts or installations can cause issues. Try the following:
    *   Deactivate and remove your virtual environment (if applicable): `deactivate` (if active), then `rm -rf venv`.
    *   Clean build artifacts: Remove `build/`, `dist/`, and `nclone.egg-info/` directories from the project root if they exist.
    *   Uninstall any existing nclone package: `pip uninstall nclone` (you might need to run this multiple times if it reports 'not installed' but issues persist).
    *   Re-create the virtual environment (see step 2 in Installation).
    *   Re-install the package (see step 3 in Installation), preferably with `pip install -e .`.
4.  **Check Python version:** Ensure you are using Python 3.9 or newer as specified in `pyproject.toml`.

If problems persist, please open an issue in the repository.

## Project Structure (Key Files & Directories)

Top-level:

- `pyproject.toml`: Package metadata and dependencies.
- `README.md`: This overview.
- `docs/`: Additional documentation.
  - `sim_mechanics_doc.md`: Detailed simulation mechanics.
  - `FILE_INDEX.md`: One-line descriptions of key modules/files.

Package `nclone/`:

- Core simulation and I/O
  - `nsim.py`: Physics and game logic core.
  - `physics.py`: Physics helpers/constants.
  - `ninja.py`: Player character state machine and control logic.
  - `entities.py`: Entity definitions and interactions.
  - `maps/`: Map data.
  - `map_loader.py`: Map loading utilities.
  - `tile_definitions.py`: Tile collision geometry definitions.
  - `render_utils.py`, `tile_renderer.py`, `entity_renderer.py`, `nsim_renderer.py`: Rendering.
  - `nplay.py`, `nplay_headless.py`: Interactive and headless runners.
  - `run_multiple_headless.py`: Multi-process headless runner.

- Environments (Gym-compatible)
  - `gym_environment/base_environment.py`
  - `gym_environment/`
    - `npp_environment.py`, `observation_processor.py`, `constants.py`, `reward_calculation/`

- Content generation
  - `map_generation/`: Procedural map generators and constants.
  - `map_augmentation/`: Map transforms (e.g., mirroring).

- **Reachability System** (Primary Architecture)
  - `graph/reachability/reachability_system.py`: Multi-tier reachability coordinator
  - `graph/reachability/opencv_flood_fill.py`: Fast OpenCV-based analysis
  - `graph/subgoal_planner.py`: Hierarchical subgoal planning for level completion
  - `graph/common.py`: Shared graph components, data structures, and constants

- Utilities
  - `constants.py`, `sim_config.py`, `debug_overlay_renderer.py`, `ntrace.py`, `test_environment.py`.

## Documentation

- **Simulation mechanics**: `docs/sim_mechanics_doc.md` - Core N++ gameplay mechanics and physics
- **File index**: `docs/FILE_INDEX.md` - Navigation guide for key modules
- **Task definitions**: `docs/tasks/` - Implementation task specifications