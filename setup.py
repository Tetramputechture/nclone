from setuptools import setup, find_packages
import os

setup(
    name="nclone",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pygame",
        "pycairo",
        "numpy",
    ],
    author="SimonV42",
    description="An N++ simulator with headless mode support",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists(
        "README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/SimonV42/nclone",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
