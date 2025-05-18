from setuptools import setup, find_packages

setup(
    name='nclone_environments',
    version='0.1.0',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        'pygame', # From the traceback
        # Add any other direct dependencies of your nclone_environments package here
    ],
    description='N-Clone Game Environments',
    long_description='Python environments for the N-Clone game, designed for testing and AI agent interaction.',
    author='Your Name / Tetramputechture',
    author_email='your.email@example.com',
    url='https://github.com/Tetramputechture/nclone-cpp', # Or the specific URL for this Python part if different
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Assuming MIT from LICENSE.md, please verify
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
