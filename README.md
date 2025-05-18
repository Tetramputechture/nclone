# NClone CPP

A C++ port of the NClone simulation with Python bindings.
Made so that we can run our simulation at extreme volumes and speeds to enable
training for an ML agent.

## Prerequisites

- CMake (>= 3.10)
- SFML 3 development libraries
- Python 3.7+ with development headers
- Cython
- numpy

### Installing Prerequisites

On Ubuntu/Debian:
```bash
sudo apt-get install cmake libsfml-dev python3-dev
pip install cython numpy
```

On macOS:
```bash
brew install cmake sfml
pip install cython numpy
```

## Installation

There are two ways to use this project:

### 1. As a standalone C++ application

```bash
cd nclone-sim-sfml
mkdir build
cd build
cmake ..
make
```

The executable will be available at `build/nclone-cpp`

### 2. As a Python module

#### Development Installation (Editable Mode)
If you're developing or modifying the module:

```bash
# Clone the repository
git clone https://github.com/yourusername/nclone-sim-sfml.git
cd nclone-sim-sfml

# Install in editable mode
cd python-bindings
pip install -e .
```

#### Regular Installation
If you just want to use the module:

```bash
cd nclone-sim-sfml/python-bindings
pip install .
```

## Using the Python Module

Here's a complete example of how to use the module in your Python script:

```python
from nplay_headless_cpp import NPlayHeadlessCpp
import numpy as np

def run_simulation():
    # Create simulation with options
    sim = NPlayHeadlessCpp(
        enable_debug_overlay=False,  # Enable/disable debug visualization
        basic_sim=False,            # Use simplified simulation
        full_export=False,          # Export full state information
        tolerance=1.0,              # Physics tolerance
        enable_anim=True,           # Enable animation
        log_data=False             # Enable data logging
    )

    # Load a map
    with open("path/to/map.dat", "rb") as f:
        sim.load_map(f.read())

    # Game loop
    while True:
        # Get current state
        ninja_pos = sim.get_ninja_position()  # Returns (x, y) tuple
        ninja_vel = sim.get_ninja_velocity()  # Returns (vx, vy) tuple
        in_air = sim.is_ninja_in_air()
        walled = sim.is_ninja_walled()
        
        # Get additional state information
        gold_collected = sim.get_gold_collected()
        doors_opened = sim.get_doors_opened()
        total_gold = sim.get_total_gold_available()
        frame_number = sim.get_sim_frame()
        
        # Get normalized state vectors (useful for ML)
        ninja_state = sim.get_ninja_state()  # 10-element normalized state vector
        entity_states = sim.get_entity_states()  # All entity states
        full_state = sim.get_state_vector()  # Complete game state
        
        # Get exit-related information
        switch_activated = sim.exit_switch_activated()
        switch_pos = sim.exit_switch_position()
        door_pos = sim.exit_door_position()

        # Simulate one step (example: move right and jump)
        # hor_input: float between -1.0 and 1.0 (-1=left, 1=right)
        # jump_input: int (0=no jump, 1=jump)
        sim.tick(1.0, 1)

        # Get rendered frame (if needed)
        frame = sim.render()  # Returns (600, 1056, 3) float32 array
        
        # Check win/death conditions
        if sim.has_won():
            print("Level complete!")
            break
        elif sim.has_died():
            print("Ninja died!")
            sim.reset()  # Reset the simulation

# Run the simulation
if __name__ == "__main__":
    run_simulation()
```

### State Vector Information

The module provides several ways to get the game state:

1. `get_ninja_state()`: Returns a 10-element normalized vector containing:
   - Position (x, y)
   - Velocity (vx, vy)
   - In air status
   - Walled status
   - Additional ninja properties

2. `get_entity_states(only_exit_and_switch=False)`: Returns a vector containing all entity states
   - If only_exit_and_switch=True, returns only exit door and switch states
   - Each entity's state includes position, type, and other properties

3. `get_state_vector(only_exit_and_switch=False)`: Returns a complete state representation
   - Combines ninja state and entity states
   - Useful for machine learning applications

## Project Structure

- `src/` - C++ source files
  - `simulation.cpp/hpp` - Core simulation logic
  - `renderer.cpp/hpp` - SFML-based rendering
  - `sim_wrapper.cpp/hpp` - C++ wrapper for Python bindings
  - `entities/` - Game entity implementations
  - `physics/` - Physics engine components
- `python-bindings/` - Python package
  - `src/nplay_headless_cpp/` - Python module source
    - `nplay_headless_cpp.pyx` - Cython interface
    - `__init__.py` - Package initialization
  - `setup.py` - Python build configuration
- `CMakeLists.txt` - CMake build configuration
- `Makefile` - Common build commands

## Common Issues

1. **SFML not found**: 
   - Ensure SFML development libraries are installed
   - Check paths in setup.py match your system
   - Try: `sudo apt-get install libsfml-dev`

2. **Build errors**: 
   - Make sure all prerequisites are installed
   - Check CMake version: `cmake --version`
   - Try cleaning build: `rm -rf build/ && mkdir build`

3. **Import errors**: 
   - Ensure the module is in your Python path
   - Try: `pip install -e .` in python-bindings directory
   - Check: `python -c "import nplay_headless_cpp"`

4. **Runtime errors**:
   - Check SFML library paths in setup.py
   - Ensure map file exists and is readable
   - Verify Python version compatibility

5. **Display issues**:
    - Run `export LIBGL_ALWAYS_INDIRECT=0` before running the simulation