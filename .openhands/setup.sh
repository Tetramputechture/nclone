#!/bin/bash

# OpenHands setup script for nclone repository
# This script runs every time OpenHands begins working with the repository
# It installs dependencies and sets up the environment for development

set -e  # Exit on any error

echo "🚀 Setting up nclone development environment..."
echo "📁 Project root: $(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
echo "📝 Log file: $(dirname "${BASH_SOURCE[0]}")/.setup.log"
echo "⏰ Started at: $(date)"
echo ""

# Change to project root
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" || {
    echo -e "\033[0;31m❌ Failed to change to project root directory\033[0m"
    exit 1
}

# Check Python version
echo -e "\033[0;34mℹ️  Checking Python version...\033[0m"
if ! python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2); then
    echo -e "\033[0;31m❌ Failed to get Python version\033[0m"
    exit 1
fi

if [[ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]]; then
    echo -e "\033[0;31m❌ Python 3.8 or higher is required. Found: $python_version\033[0m"
    exit 1
fi

echo -e "\033[0;32m✅ Python version check passed: $python_version\033[0m"

# Create necessary directories
echo -e "\033[0;34mℹ️  Creating necessary directories...\033[0m"
mkdir -p nclone/maps/official nclone/maps/eval
echo -e "\033[0;32m✅ Necessary directories created\033[0m"

# Set up environment variables for headless operation
echo -e "\033[0;34mℹ️  Setting up environment variables...\033[0m"
export SDL_VIDEODRIVER=dummy
export XDG_RUNTIME_DIR=/tmp/runtime-$(whoami)
mkdir -p "/tmp/runtime-$(whoami)"
chmod 700 "/tmp/runtime-$(whoami)"
echo -e "\033[0;32m✅ Environment variables configured\033[0m"

# Install build dependencies first
echo -e "\033[0;34mℹ️  Installing build dependencies...\033[0m"

if pip install --timeout=60 --retries=2 --disable-pip-version-check --no-warn-script-location --progress-bar=off \
   "setuptools>=45" "wheel" "setuptools_scm[toml]>=6.2"; then
    echo -e "\033[0;32m✅ Build dependencies installed successfully\033[0m"
else
    echo -e "\033[0;31m❌ Failed to install build dependencies\033[0m"
    exit 1
fi

# Install nclone package in editable mode
echo -e "\033[0;34mℹ️  Installing nclone package in editable mode...\033[0m"
if pip install -e .; then
    echo -e "\033[0;32m✅ nclone installed successfully\033[0m"
else
    echo -e "\033[0;31m❌ Failed to install nclone package\033[0m"
    exit 1
fi

# Install additional development tools
echo -e "\033[0;34mℹ️  Installing additional development tools...\033[0m"
if pip install --timeout=60 --retries=2 --disable-pip-version-check --no-warn-script-location --progress-bar=off ruff; then
    echo -e "\033[0;32m✅ Additional tools installed successfully\033[0m"
else
    echo -e "\033[1;33m⚠️  Some additional tools failed to install, but continuing...\033[0m"
fi

# Verify installations
echo -e "\033[0;34mℹ️  Verifying package installations...\033[0m"

verification_failed=false

# Test critical imports
for module in "nclone:nclone package" "pygame:pygame" "numpy:numpy" "gymnasium:gymnasium"; do
    module_name="${module%%:*}"
    description="${module##*:}"
    
    echo -e "\033[0;34mℹ️  Testing $description import...\033[0m"
    
    if timeout 30 python -c "import $module_name; print('$description imported successfully')" >/dev/null 2>&1; then
        echo -e "\033[0;32m✅ $description installation verified\033[0m"
    else
        echo -e "\033[0;31m❌ $description installation failed\033[0m"
        verification_failed=true
    fi
done

if [[ "$verification_failed" == "true" ]]; then
    echo -e "\033[0;31m❌ Package verification failed\033[0m"
    exit 1
fi

# Clone the npp-rl repository in the parent directory
echo -e "\033[0;34mℹ️  Cloning npp-rl repository...\033[0m"
if git clone https://github.com/tetramputechture/npp-rl.git ../npp-rl; then
    echo -e "\033[0;32m✅ npp-rl repository cloned successfully\033[0m"
else
    echo -e "\033[0;31m❌ Failed to clone npp-rl repository\033[0m"
    exit 1
fi

# Success message
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