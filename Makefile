.PHONY: all clean cpp python-build python-install test

# Default target
all: cpp python-build

# Build C++ project
cpp:
	mkdir -p build
	cd build && cmake .. && make

# Build Python extension
python-build:
	python setup.py build_ext --inplace

# Install Python package
python-install:
	pip install -e .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf __pycache__/
	rm -f nclone/*.so
	rm -f nclone/*.cpp
	find . -name "*.pyc" -delete

# Run tests (placeholder)
test: python-build
	python -m pytest tests/

# Development setup
dev-setup:
	pip install -r requirements-dev.txt

# Create requirements file
requirements:
	pip freeze > requirements.txt

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Build both C++ and Python components"
	@echo "  cpp            - Build C++ standalone application"
	@echo "  python-build   - Build Python extension"
	@echo "  python-install - Install Python package"
	@echo "  clean          - Clean build artifacts"
	@echo "  test           - Run tests"
	@echo "  dev-setup     - Install development dependencies"
	@echo "  requirements  - Update requirements.txt" 