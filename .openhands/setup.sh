#!/bin/bash

# OpenHands setup script for nclone repository
# This script runs every time OpenHands begins working with the repository
# It installs dependencies and sets up the environment for development

# Check if we're being sourced in a shell with conflicting readonly variables
# If so, re-execute in a clean subshell
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # We're being sourced
    conflicting_vars=()
    for var in SCRIPT_DIR PROJECT_ROOT LOG_FILE PID_FILE LOCK_FILE PIP_TIMEOUT IMPORT_TIMEOUT CLEANUP_TIMEOUT MIN_DISK_SPACE MIN_MEMORY RED GREEN YELLOW BLUE NC; do
        if [[ -n "${!var:-}" ]] && [[ "$(declare -p "$var" 2>/dev/null)" =~ declare\ -r ]]; then
            conflicting_vars+=("$var")
        fi
    done
    
    if [[ ${#conflicting_vars[@]} -gt 0 ]]; then
        echo "🔄 Detected readonly variables: ${conflicting_vars[*]}" >&2
        echo "🚀 Re-executing in clean subshell to avoid conflicts..." >&2
        exec bash "${BASH_SOURCE[0]}" "$@"
    fi
    
    # Clean up the temporary array
    unset conflicting_vars var
fi

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# Configuration - now safe to set readonly variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly LOG_FILE="${SCRIPT_DIR}/setup.log"
readonly PID_FILE="${SCRIPT_DIR}/setup.pid"
readonly LOCK_FILE="${SCRIPT_DIR}/setup.lock"

# Timeout settings (in seconds)
readonly PIP_TIMEOUT=300  # 5 minutes
readonly IMPORT_TIMEOUT=30  # 30 seconds
readonly CLEANUP_TIMEOUT=60  # 1 minute

# Resource requirements (in MB)
readonly MIN_DISK_SPACE=1000  # 1GB
readonly MIN_MEMORY=512       # 512MB

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color


# Initialize logging
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Store PID for cleanup
echo $$ > "$PID_FILE"

# Cleanup function
cleanup() {
    local exit_code=$?
    echo -e "${YELLOW}🧹 Performing cleanup...${NC}"
    
    # Remove lock file
    [[ -f "$LOCK_FILE" ]] && rm -f "$LOCK_FILE"
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    
    # Kill any hanging pip processes
    pkill -f "pip install" 2>/dev/null || true
    
    if [[ $exit_code -ne 0 ]]; then
        echo -e "${RED}❌ Setup failed with exit code $exit_code${NC}"
        echo -e "${YELLOW}📋 Check the log file for details: $LOG_FILE${NC}"
    fi
    
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT
trap 'echo "Interrupted by user"; exit 130' INT TERM

# Utility functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Progress indicator for long operations
show_progress() {
    local pid=$1
    local message=$2
    local chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local delay=0.1
    
    while kill -0 $pid 2>/dev/null; do
        for (( i=0; i<${#chars}; i++ )); do
            printf "\r${BLUE}${chars:$i:1} $message${NC}"
            sleep $delay
            if ! kill -0 $pid 2>/dev/null; then
                break 2
            fi
        done
    done
    printf "\r"
}

# Check if another instance is running
check_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid=$(cat "$LOCK_FILE" 2>/dev/null)
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another instance of setup is already running (PID: $lock_pid)"
            exit 1
        else
            log_warning "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

# Check system resources
check_resources() {
    log_info "Checking system resources..."
    
    # Check available disk space
    local disk_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print int($4/1024)}')
    if [[ $disk_space -lt $MIN_DISK_SPACE ]]; then
        log_error "Insufficient disk space. Required: ${MIN_DISK_SPACE}MB, Available: ${disk_space}MB"
        exit 1
    fi
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        local available_memory=$(free -m | awk 'NR==2{print $7}')
        if [[ -n "$available_memory" ]] && [[ $available_memory -lt $MIN_MEMORY ]]; then
            log_warning "Low available memory. Available: ${available_memory}MB, Recommended: ${MIN_MEMORY}MB"
        fi
    fi
    
    log_success "Resource check passed"
}

# Run command with timeout
run_with_timeout() {
    local timeout_duration=$1
    local error_message=$2
    shift 2
    
    if timeout "$timeout_duration" "$@"; then
        return 0
    else
        local exit_code=$?
        log_error "$error_message"
        if [[ $exit_code -eq 124 ]]; then
            log_error "Command timed out after ${timeout_duration} seconds"
        fi
        return $exit_code
    fi
}

# Test Python import with timeout
test_import_with_timeout() {
    local module=$1
    local description=$2
    
    log_info "Testing $description import..."
    
    if run_with_timeout $IMPORT_TIMEOUT "Failed to import $module" \
       python -c "import $module; print('$description imported successfully')" >/dev/null 2>&1; then
        log_success "$description installation verified"
        return 0
    else
        log_error "$description installation failed"
        return 1
    fi
}

# Enhanced pip install with progress and timeout
pip_install_with_progress() {
    local packages=("$@")
    local pip_args=(
        --timeout=60
        --retries=2
        --disable-pip-version-check
        --no-warn-script-location
        --progress-bar=off
    )
    
    log_info "Installing packages: ${packages[*]}"
    
    # Run pip install directly with timeout (no background process)
    if timeout $PIP_TIMEOUT pip install "${pip_args[@]}" "${packages[@]}"; then
        log_success "Package installation completed"
        return 0
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "Package installation timed out after ${PIP_TIMEOUT} seconds"
        else
            log_error "Package installation failed with exit code: $exit_code"
        fi
        return 1
    fi
}

# Main setup function
main() {
    local start_time=$(date +%s)
    
    echo "🚀 Setting up nclone development environment..."
    echo "📁 Project root: $PROJECT_ROOT"
    echo "📝 Log file: $LOG_FILE"
    echo "⏰ Started at: $(date)"
    echo ""
    
    # Preliminary checks
    check_lock
    check_resources
    
    # Check Python version
    log_info "Checking Python version..."
    if ! python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2); then
        log_error "Failed to get Python version"
        exit 1
    fi
    
    required_version="3.8"
    if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
        log_error "Python $required_version or higher is required. Found: $python_version"
        exit 1
    fi
    
    log_success "Python version check passed: $python_version"
    
    # Create necessary directories
    log_info "Creating necessary directories..."
    cd "$PROJECT_ROOT" || {
        log_error "Failed to change to project root directory: $PROJECT_ROOT"
        exit 1
    }
    mkdir -p nclone/maps/official nclone/maps/eval
    log_success "Necessary directories created"
    
    # Set up environment variables for headless operation
    log_info "Setting up environment variables..."
    export SDL_VIDEODRIVER=dummy
    export XDG_RUNTIME_DIR=/tmp/runtime-$(whoami)
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    log_success "Environment variables configured"
    
    # Install build dependencies first
    log_info "Installing build dependencies..."
    
    local build_packages=(
        "setuptools>=45"
        "wheel"
        "setuptools_scm[toml]>=6.2"
    )
    
    if pip_install_with_progress "${build_packages[@]}"; then
        log_success "Build dependencies installed successfully"
    else
        log_error "Failed to install build dependencies"
        exit 1
    fi
    
    # Install nclone with all dependencies
    log_info "Installing nclone with all dependencies..."
    
    # Change to project root directory for editable install
    cd "$PROJECT_ROOT" || {
        log_error "Failed to change to project root directory: $PROJECT_ROOT"
        exit 1
    }
    
    # Install nclone package in editable mode
    log_info "Installing nclone package in editable mode..."
    if pip install -e "$PROJECT_ROOT"; then
        log_success "nclone installed successfully"
    else
        log_error "Failed to install nclone package"
        exit 1
    fi
    
    # Install additional development tools
    log_info "Installing additional development tools..."
    if pip_install_with_progress "ruff"; then
        log_success "Additional tools installed successfully"
    else
        log_warning "Some additional tools failed to install, but continuing..."
    fi
    
    # Verify installations
    log_info "Verifying package installations..."
    
    local verification_failed=false
    
    # Test critical imports
    test_import_with_timeout "nclone" "nclone package" || verification_failed=true
    test_import_with_timeout "pygame" "pygame" || verification_failed=true
    test_import_with_timeout "numpy" "numpy" || verification_failed=true
    test_import_with_timeout "gymnasium" "gymnasium" || verification_failed=true
    
    if [[ "$verification_failed" == "true" ]]; then
        log_error "Package verification failed"
        exit 1
    fi
    
    # Calculate setup time
    local end_time=$(date +%s)
    local setup_duration=$((end_time - start_time))
    
    # Success message
    echo ""
    echo "🎉 nclone development environment setup complete!"
    echo "⏱️  Setup completed in ${setup_duration} seconds"
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
    echo "📝 Setup log saved to: $LOG_FILE"
    echo ""
}

# Run main function
main "$@"