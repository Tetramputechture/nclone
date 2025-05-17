# nclone

`nclone` is a Pygame-based simulation of the game N++. This repository is a fork specifically tailored for Deep Reinforcement Learning (DRL) research. It features a custom reward system designed for DRL agents and supports headless mode for faster training and experimentation.

## Features

*   **N++ Simulation:** Replicates core gameplay mechanics of N++.
*   **Pygame-based:** Built using the Pygame library for rendering and interaction.
*   **Deep RL Focus:** Includes a reward system to guide DRL agent learning.
*   **Headless Mode:** Allows the simulation to run without a graphical interface, significantly speeding up training processes.
*   **Customizable Environments:** The environment (`nclone_environments/basic_level_no_gold/basic_level_no_gold.py`) can be configured for different experimental setups.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SimonV42/nclone.git
    cd nclone
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project uses `setuptools` for packaging. You can install it directly using pip:
    ```bash
    pip install .
    ```
    For development, you might prefer an editable install:
    ```bash
    pip install -e .
    ```
    This will install all dependencies listed in `pyproject.toml`, including Pygame, NumPy, and PyCairo.

## Running the Simulation

To test the environment and see the simulation in action, you can run the `test_environment.py` script:

```bash
PYTHONPATH=.. python -m nclone.test_environment
```

This script initializes the `BasicLevelNoGold` environment in human-render mode, allowing you to control the ninja using keyboard inputs:
*   **Left/Right Arrow Keys (or A/D):** Move left/right.
*   **Space/Up Arrow Key:** Jump.
*   **R Key:** Reset the environment.

You can also run with frametime logging:
```bash
python nclone/test_environment.py --log-frametimes
```

## Headless Mode

The environment can be initialized in `rgb_array` mode for headless operation, which is crucial for DRL training. This is configured in the environment's constructor. See `nclone_environments/basic_level_no_gold/basic_level_no_gold.py` for an example of how the `render_mode` is set.

## Project Structure (Key Files & Directories)

*   `nclone/`: Main package directory.
    *   `nplay_headless.py`: Core headless simulation runner.
    *   `nsim.py`: The underlying N++ physics and game logic simulator.
    *   `nsim_renderer.py`: Handles rendering of the simulation state.
    *   `nclone_environments/`: Contains Gym-compatible environments.
        *   `basic_level_no_gold/`: A specific environment configuration.
            *   `basic_level_no_gold.py`: The main environment class.
            *   `reward_calculation/`: Logic for calculating rewards.
    *   `maps/`: Contains map files.
    *   `map_generation/`: Scripts for procedural map generation.
*   `test_environment.py`: Example script to run and test the environment.
*   `pyproject.toml`: Project metadata and dependencies.
*   `README.md`: This file.

This provides a foundation for training reinforcement learning agents to play N++.