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

## Building

There are two ways to build this project:

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

```bash
# Build the Python extension
make python-build

# Or manually:
python setup.py build_ext --inplace
```

## Using from Python

After building the Python extension, you can use it like this:

```python
from nclone.nplay_headless_cpp import NPlayHeadlessCpp

# Create simulation
sim = NPlayHeadlessCpp(enable_debug_overlay=False)

# Load a map
with open("path/to/map.dat", "rb") as f:
    sim.load_map(f.read())

# Run simulation
sim.tick(1.0, 1)  # Move right and jump

# Get state
pos = sim.get_ninja_position()  # Returns (x, y) tuple
vel = sim.get_ninja_velocity()  # Returns (vx, vy) tuple
in_air = sim.is_ninja_in_air()
walled = sim.is_ninja_walled()

# Get rendered frame as numpy array
frame = sim.render()  # Returns (600, 1056, 3) float32 array with values in [0,1]

# Check win/death conditions
if sim.has_won():
    print("Level complete!")
elif sim.has_died():
    print("Ninja died!")
```

## Project Structure

- `src/` - C++ source files
  - `simulation.cpp/hpp` - Core simulation logic
  - `renderer.cpp/hpp` - SFML-based rendering
  - `sim_wrapper.cpp/hpp` - C++ wrapper for Python bindings
  - `entities/` - Game entity implementations
  - `physics/` - Physics engine components
- `nclone/` - Python package
  - `nplay_headless_cpp.pyx` - Cython interface
- `setup.py` - Python build configuration
- `CMakeLists.txt` - CMake build configuration
- `Makefile` - Common build commands

## Common Issues

1. SFML not found: Make sure SFML development libraries are installed and the paths in setup.py are correct
2. Build errors: Ensure you have all prerequisites installed
3. Import errors: Make sure the Python module is built and in your Python path
