from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

# Run CMake build first
subprocess.check_call(['make', 'cpp'], cwd='..')

# Get paths to SFML built by CMake
SFML_INCLUDE = [
    "../src",  # Main source directory
    "../src/entities",  # Add entities directory for entity headers
    "../src/physics",   # Add physics directory for physics headers
    "../build/_deps/sfml-src/include",  # SFML headers
    np.get_include()
]

SFML_LIB = ["../build/_deps/sfml-build/lib"]  # SFML libraries built by CMake

# Get all cpp source files
def get_cpp_sources():
    sources = [
        "src/nplay_headless_cpp/nplay_headless_cpp.pyx",  # Cython source
        "../src/sim_wrapper.cpp",
        "../src/renderer.cpp",
        "../src/simulation.cpp",
        "../src/ninja.cpp",
        "../src/sim_config.cpp",
        "../src/physics/physics.cpp",
        "../src/physics/segment.cpp",
        "../src/physics/grid_segment_linear.cpp",
        "../src/physics/grid_segment_circular.cpp",
        "../src/entities/boost_pad.cpp",
        "../src/entities/bounce_block.cpp",
        "../src/entities/death_ball.cpp",
        "../src/entities/door_base.cpp",
        "../src/entities/door_regular.cpp",
        "../src/entities/door_locked.cpp",
        "../src/entities/door_trap.cpp",
        "../src/entities/drone_base.cpp",
        "../src/entities/drone_chaser.cpp",
        "../src/entities/drone_zap.cpp",
        "../src/entities/entity.cpp",
        "../src/entities/exit_door.cpp",
        "../src/entities/exit_switch.cpp",
        "../src/entities/gold.cpp",
        "../src/entities/laser.cpp",
        "../src/entities/launch_pad.cpp",
        "../src/entities/mini_drone.cpp",
        "../src/entities/one_way_platform.cpp",
        "../src/entities/shove_thwump.cpp",
        "../src/entities/thwump.cpp",
        "../src/entities/toggle_mine.cpp"
    ]
    return sources

# Define the extension
extension = Extension(
    "nplay_headless_cpp.nplay_headless_cpp",
    sources=get_cpp_sources(),
    include_dirs=SFML_INCLUDE,
    library_dirs=SFML_LIB,
    libraries=[
        'sfml-graphics-d',
        'sfml-window-d', 
        'sfml-system-d'
    ],
    language="c++",
    extra_compile_args=["-std=c++17"],
    runtime_library_dirs=[os.path.abspath("../build/_deps/sfml-build/lib")]  # Help find SFML libs at runtime
)

setup(
    name="nplay_headless_cpp",
    version="0.0.1",
    description="Wrapper for NClone-CPP",
    ext_modules=cythonize(
        [extension],
        language_level=3,
        compiler_directives={"linetrace": True}
    ),
    zip_safe=False,
    packages=["nplay_headless_cpp"],
    package_dir={"": "src"},
    setup_requires=[
        "cython>=0.29.0",
        "numpy"
    ],
    install_requires=[
        "numpy"
    ]
) 