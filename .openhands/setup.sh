#!/bin/bash

# OpenHands setup script for nclone repository
# This script runs every time OpenHands begins working with the repository
# It installs dependencies and sets up the environment for development

set -e  # Exit on any error

echo "🚀 Setting up nclone development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install the nclone package in editable mode with all dependencies
echo "📦 Installing nclone package with dependencies..."
pip install -e . --quiet

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install -e ".[dev]" --quiet

# Install test dependencies
echo "🧪 Installing test dependencies..."
pip install -e ".[test]" --quiet

# Install linting tools (Ruff)
echo "🔍 Installing linting tools..."
pip install ruff --quiet

# Set up environment variables for headless operation
echo "🖥️  Setting up environment variables..."
export SDL_VIDEODRIVER=dummy
export XDG_RUNTIME_DIR=/tmp/runtime-$(whoami)
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Verify installation by running a quick test
echo "✅ Verifying installation..."
if python -c "import nclone; print('nclone package imported successfully')" 2>/dev/null; then
    echo "✅ nclone package installation verified"
else
    echo "❌ Error: nclone package installation failed"
    exit 1
fi

# Test pygame import (critical for the simulation)
if python -c "import pygame; print('pygame imported successfully')" 2>/dev/null; then
    echo "✅ pygame installation verified"
else
    echo "❌ Error: pygame installation failed"
    exit 1
fi

# Test numpy import (required for graph operations)
if python -c "import numpy; print('numpy imported successfully')" 2>/dev/null; then
    echo "✅ numpy installation verified"
else
    echo "❌ Error: numpy installation failed"
    exit 1
fi

# Test gymnasium import (required for RL environments)
if python -c "import gymnasium; print('gymnasium imported successfully')" 2>/dev/null; then
    echo "✅ gymnasium installation verified"
else
    echo "❌ Error: gymnasium installation failed"
    exit 1
fi

# Display helpful information
echo ""
echo "🎉 nclone development environment setup complete!"
echo ""
echo "📋 Quick reference:"
echo "   • Run tests: python tests/test_graph_fixes_unit_tests.py"
echo "   • Run all tests: python tests/run_tests.py"
echo "   • Test environment: python -m nclone.test_environment"
echo "   • Lint code: make lint"
echo "   • Fix code: make fix"
echo "   • Debug graph: python debug/final_validation.py"
echo ""
echo "📁 Key directories:"
echo "   • nclone/graph/: Core graph system implementation"
echo "   • tests/: Comprehensive test suite"
echo "   • debug/: Debugging and analysis scripts"
echo "   • docs/: Documentation and guides"
echo ""
echo "🔧 Development workflow:"
echo "   1. Make changes to nclone/graph/ files"
echo "   2. Run tests: python tests/test_graph_fixes_unit_tests.py"
echo "   3. Debug issues: python debug/final_validation.py"
echo "   4. Lint code: make lint"
echo "   5. Commit changes with descriptive messages"
echo ""