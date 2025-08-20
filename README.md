# nclone

`nclone` is a Pygame-based simulation of the game N++. This repository is tailored for Deep Reinforcement Learning (DRL) research. It features a custom reward system designed for DRL agents and supports headless mode for faster training and experimentation.

## Features

*   **N++ Simulation:** Replicates core gameplay mechanics of N++.
*   **Pygame-based:** Built using the Pygame library for rendering and interaction.
*   **Deep RL Focus:** Includes a reward system to guide DRL agent learning, and serves as the environment for the RL agent developed in the `npp-rl` subdirectory.
*   **Headless Mode:** Allows the simulation to run without a graphical interface, significantly speeding up DRL training processes.
*   **Customizable Environments:** The environment (`nclone_environments/basic_level_no_gold/basic_level_no_gold.py`) can be configured for different experimental setups.

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

This script initializes the `BasicLevelNoGold` environment in human-render mode, allowing you to control the ninja using keyboard inputs:
*   **Left/Right Arrow Keys (or A/D):** Move left/right.
*   **Space/Up Arrow Key:** Jump.
*   **R Key:** Reset the environment.

You can also run with frametime logging:
```bash
python -m nclone.test_environment --log-frametimes
```

To train or run the RL agent, please refer to the instructions in `npp-rl/README.md`.

## Headless Mode (Base Environment)

The environment can be initialized in `rgb_array` mode for headless operation, which is crucial for DRL training. This is configured in the environment's constructor. See `nclone_environments/basic_level_no_gold/basic_level_no_gold.py` for an example of how the `render_mode` is set.

## Running Multiple Headless Simulations (Base Environment)

To leverage multi-core processors for large-scale experiments or data collection (e.g., for DRL), you can run multiple headless simulations concurrently using the `run_multiple_headless.py` script.

```bash
python -m nclone.run_multiple_headless --num-simulations 4 --num-steps 50000
```

This command will launch 4 independent headless simulations, each running for 50,000 steps. You can adjust these parameters as needed:

*   `--num-simulations`: Specifies the number of concurrent simulation instances.
*   `--num-steps`: Specifies the number of simulation steps each instance will run.

Each simulation runs in its own process, allowing for parallel execution.

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
  - `pathfinding_strategy.md`: Design notes for the pathfinding system.
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
  - `nclone_environments/base_environment.py`
  - `nclone_environments/basic_level_no_gold/`
    - `basic_level_no_gold.py`, `observation_processor.py`, `constants.py`, `reward_calculation/`

- Content generation
  - `map_generation/`: Procedural map generators and constants.
  - `map_augmentation/`: Map transforms (e.g., mirroring).

- Pathfinding (experimental tools/designs)
  - `pathfinding/`: Surface parsing, navigation graph, A* search, and visualization helpers.

- Utilities
  - `constants.py`, `sim_config.py`, `debug_overlay_renderer.py`, `ntrace.py`, `test_environment.py`.

## Documentation

- Simulation mechanics: `docs/sim_mechanics_doc.md`
- Pathfinding design: `docs/pathfinding_strategy.md`
- File index: `docs/FILE_INDEX.md`