from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get SFML include and lib paths - you may need to modify these for your system
SFML_INCLUDE = "/usr/include"  # Default Linux path
SFML_LIB = "/usr/lib"         # Default Linux path

# For Windows, uncomment and modify these:
# SFML_INCLUDE = "C:/SFML/include"
# SFML_LIB = "C:/SFML/lib"

# Define the extension
extensions = [
    Extension(
        "nclone.nplay_headless_cpp",
        ["nclone/nplay_headless_cpp.pyx"],
        include_dirs=[
            "nclone-sim-sfml/src",
            SFML_INCLUDE,
            np.get_include()
        ],
        library_dirs=[SFML_LIB],
        libraries=[
            'sfml-graphics',
            'sfml-window',
            'sfml-system'
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="nclone",
    ext_modules=cythonize(extensions),
    zip_safe=False,
    packages=["nclone"],
) 