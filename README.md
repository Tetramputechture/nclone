# nclone

`nclone` is a Pygame-based simulation of the game N++. This repository is a fork specifically tailored for Deep Reinforcement Learning (DRL) research. It features a custom reward system designed for DRL agents and supports headless mode for faster training and experimentation.

## Features

*   **N++ Simulation:** Replicates core gameplay mechanics of N++.
*   **Pygame-based:** Built using the Pygame library for rendering and interaction.
*   **Deep RL Focus:** Includes a reward system to guide DRL agent learning, and serves as the environment for the RL agent developed in the `npp-rl` subdirectory.
*   **Headless Mode:** Allows the simulation to run without a graphical interface, significantly speeding up DRL training processes.
*   **Customizable Environments:** The environment (`nclone_environments/basic_level_no_gold/basic_level_no_gold.py`) can be configured for different experimental setups.

## Deep Reinforcement Learning Agent

This repository includes a sophisticated Deep Reinforcement Learning agent designed to play N++. The agent, based on Proximal Policy Optimization (PPO), is located in the `npp-rl` subdirectory.

For detailed information about the RL agent's architecture, features (including 3D convolutions for temporal modeling, adaptive exploration strategies, scaled network architectures), training procedures, and usage instructions, please refer to the `README.md` file within the `npp-rl` directory:

[**Navigate to npp-rl/README.md**](./npp-rl/README.md)

Key aspects of the RL agent include:
*   Multi-modal input processing (visual frames and game state vectors).
*   Advanced feature extraction using 3D or enhanced 2D CNNs.
*   Research-backed hyperparameter configurations.
*   Optional adaptive exploration mechanisms (e.g., Intrinsic Curiosity Module).
*   Comprehensive training scripts and monitoring tools (Tensorboard integration).

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

*   `nclone/`: Main game simulation package directory.
    *   `nplay_headless.py`: Core headless simulation runner.
    *   `nsim.py`: The underlying N++ physics and game logic simulator.
    *   `nsim_renderer.py`: Handles rendering of the simulation state.
    *   `run_multiple_headless.py`: Script to run multiple headless simulations concurrently.
    *   `nclone_environments/`: Contains Gym-compatible environments.
        *   `basic_level_no_gold/`: A specific environment configuration.
            *   `basic_level_no_gold.py`: The main environment class.
            *   `reward_calculation/`: Logic for calculating rewards.
            *   `constants.py`: Environment-specific constants (e.g., `TEMPORAL_FRAMES`).
    *   `maps/`: Contains map files.
    *   `map_generation/`: Scripts for procedural map generation.
*   `npp-rl/`: Directory for the Reinforcement Learning agent and training. **See `npp-rl/README.md` for details on the RL agent.**
    *   `agents/`:
        *   `enhanced_training.py`: Main script for training with current features.
        *   `npp_agent_ppo.py`: Original PPO training script (updated to use current features).
        *   `enhanced_feature_extractor.py`: Contains feature extractor classes.
        *   `adaptive_exploration.py`: Implements exploration strategies.
        *   `hyperparameters/ppo_hyperparameters.py`: Stores PPO hyperparameters.
    *   *(Other potential subdirectories for RL components)*
*   `test_environment.py`: Example script to run and test the base environment.
*   `pyproject.toml`: Project metadata and dependencies.
*   `README.md`: This file (overview of the `nclone` simulator).

## Sim Mechanics Doc

[Navigate to sim_mechanics_doc.md](./sim_mechanics_doc.md) to read about detailed mechanics of the simulation.