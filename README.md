# nclone (Pyglet Refactor)

`nclone` is a **Pyglet-based** simulation of the game N++. This repository is a fork specifically tailored for Deep Reinforcement Learning (DRL) research. It features a custom reward system designed for DRL agents and supports headless mode for faster training and experimentation. This version has been refactored from Pygame to Pyglet to leverage OpenGL for improved rendering performance, crucial for high-throughput DRL applications.

## Features

*   **N++ Simulation:** Replicates core gameplay mechanics of N++.
*   **Pyglet-based:** Built using the Pyglet library, leveraging OpenGL for high-performance rendering.
*   **Deep RL Focus:** Includes a reward system to guide DRL agent learning.
*   **Headless Mode (`rgb_array`):** Allows the simulation to run without a graphical interface, significantly speeding up training processes by rendering directly to NumPy arrays.
*   **Customizable Environments:** The environment (`src/nclone/nclone_environments/basic_level_no_gold/basic_level_no_gold.py`) can be configured for experimental setups.

## Installation

This project uses [Poetry](https.python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tetramputechture/nclone.git # Or your fork's URL
    cd nclone
    ```

2.  **Install Poetry (if you haven't already):**
    Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies using Poetry:**
    This command will create a virtual environment (if one isn't active) and install all dependencies listed in `pyproject.toml`.
    ```bash
    poetry install
    ```

## Running the Simulation

All commands should be run from the project root directory using `poetry run`.

### Interactive Mode (Human Rendering)

To run the simulation with visual output and keyboard control:
```bash
poetry run xvfb-run -a python -m nclone.nplay 
```
*   `xvfb-run -a` is used to provide a virtual display, necessary for Pyglet in headless server environments. If running on a desktop with a display server, you might not need `xvfb-run`.
*   **Controls (default in `nplay.py`):**
    *   **Left/Right Arrow Keys:** Move
    *   **Up Arrow Key:** Jump
    *   **R Key:** Reset simulation
    *   **Escape Key:** Quit

### Test Environment Script

The `test_environment.py` script allows testing the `BasicLevelNoGold` environment:
```bash
# Human mode
poetry run xvfb-run -a python test_environment.py

# Headless mode (runs for a set number of frames, e.g., for profiling)
poetry run python test_environment.py --headless --profile-frames 1000
```
*   **Controls (in human mode for `test_environment.py`):**
    *   **A/D or Left/Right Arrow Keys:** Move
    *   **Space/Up Arrow Key:** Jump
    *   **R Key:** Reset environment
    *   **Escape Key:** Quit

### Headless Mode for DRL

The `NPlayHeadless` class (`src/nclone/nplay_headless.py`) is designed for DRL. It's used by environments like `BasicLevelNoGold` when `render_mode='rgb_array'.

### Running Multiple Headless Simulations

For large-scale experiments, use `run_multiple_headless.py`:
```bash
poetry run python run_multiple_headless.py --num-simulations 4 --num-steps 50000 --record-dir ./videos
```
*   `--num-simulations`: Number of concurrent simulation instances.
*   `--num-steps`: Simulation steps per instance.
*   `--record-dir`: (Optional) Directory to save MP4 recordings of simulations.

## Project Structure (Key Files & Directories)

*   `src/nclone/`: Main Python package directory.
    *   `nplay.py`: Main interactive (human-render) game/simulation runner.
    *   `nplay_headless.py`: Core class for headless simulation control, providing `rgb_array` rendering.
    *   `nsim.py`: The underlying N++ physics and game logic simulator.
    *   `nsim_renderer.py`: Handles rendering of the simulation state using Pyglet.
        *   `tile_renderer.py`, `entity_renderer.py`, `debug_overlay_renderer.py`: Component renderers.
    *   `nclone_environments/`: Contains Gym-compatible environments.
        *   `basic_level_no_gold/`: A specific environment configuration.
    *   `maps/`: (Located at project root) Contains map files.
    *   `map_generation/`: (Sub-package in `src/nclone/`) Scripts for procedural map generation.
*   `test_environment.py`: Script to test the `BasicLevelNoGold` environment.
*   `run_multiple_headless.py`: Script to run multiple headless simulations concurrently.
*   `pyproject.toml`: Project metadata, dependencies, and build configuration (Poetry).
*   `README.md`: This file.

## Refactoring Notes (Pygame to Pyglet)

*   **Rendering:** Pygame's `Surface.blit` and `pygame.draw` were replaced with Pyglet Sprites, Batches, and direct OpenGL drawing (via Pyglet primitives or cairo for complex shapes subsequently converted to textures).
*   **Windowing & Events:** `pygame.display` and `pygame.event` were replaced by `pyglet.window.Window` and Pyglet's event loop (`pyglet.app.run()`, event handlers like `on_draw`, `on_key_press`).
*   **Image Loading:** `pygame.image.load` replaced by `pyglet.image.load`.
*   **Performance:** The primary motivation was to improve rendering speed for DRL by leveraging Pyglet's OpenGL backend. Offscreen rendering to NumPy arrays (`rgb_array` mode) is implemented using Pyglet Framebuffer Objects (FBOs).
*   **Headless Operation:** For `rgb_array` mode, Pyglet requires an active GL context. This is managed by creating an invisible window or using a provided window. For server environments, `xvfb-run` is typically needed.
*   **Packaging:** Switched from `setuptools` (implied by `pip install .`) to `Poetry` for more robust dependency management and packaging. Project structure was changed to `src/` layout.
