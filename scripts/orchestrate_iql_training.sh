#!/bin/bash
###############################################################################
# IQL Offline RL Remote Training Orchestration Script
#
# Automates end-to-end IQL training on a remote GPU instance:
# - Filters attracts locally (reducing 250k files to qualifying subset)
# - Transfers only filtered attracts + .replay files (avoids full attracts transfer)
# - Copies nclone code to remote
# - Builds HDF5 dataset on remote from demos
# - Trains IQL on remote GPU
# - Downloads checkpoints and logs
#
# Prerequisites:
#   - Run this script from the nclone directory
#   - Remote instance must have Python 3.11+, pip, and SSH access
#   - Remote instance should have GPU with CUDA drivers
#   - Remote instance should have build tools: build-essential, python3-dev,
#     libcairo2-dev, pkg-config
#
# Usage:
#   cd /path/to/nclone
#   ./scripts/orchestrate_iql_training.sh \
#     --instance-ip 54.123.45.67 \
#     --experiment-name iql_run_2025 \
#     --attracts-dir ../attracts \
#     --replay-dir ./my_replays
#
# With pre-filtered data:
#   ./scripts/orchestrate_iql_training.sh \
#     --instance-ip 54.123.45.67 \
#     --experiment-name iql_run_2025 \
#     --skip-filter \
#     --filtered-dir /path/to/attracts_filtered \
#     --replay-dir ./my_replays
#
###############################################################################

set -e
set -o pipefail

# ============================================================================
# Color codes
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Defaults and globals
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCLONE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_LOG_DIR="${NCLONE_DIR}/logs/iql_training_${TIMESTAMP}"
SSH_OPTS="-o ServerAliveInterval=60 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes"

INSTANCE_IP=""
EXPERIMENT_NAME=""
ATTRACTS_DIR=""
REPLAY_DIR=""
FILTERED_DIR=""
SSH_KEY="${HOME}/.ssh/id_rsa"
SSH_USER="ubuntu"
SKIP_FILTER=false
RESUME=false
FILTER_WORKERS=8
BUILD_WORKERS=8

REMOTE_BASE_DIR="~/nclone-iql-training"
REMOTE_NCLONE_DIR="${REMOTE_BASE_DIR}/nclone"
REMOTE_DEMOS_DIR="${REMOTE_BASE_DIR}/demos"
REMOTE_DATASET="${REMOTE_BASE_DIR}/iql_dataset.h5"
REMOTE_CHECKPOINTS="${REMOTE_BASE_DIR}/checkpoints"
REMOTE_VENV="${REMOTE_BASE_DIR}/venv"

# ============================================================================
# Logging
# ============================================================================
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        INFO)
            echo -e "${BLUE}[${timestamp}]${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] ✓${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        WARNING)
            echo -e "${YELLOW}[${timestamp}] ⚠${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] ✗${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        HEADER)
            echo -e "\n${CYAN}========================================${NC}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            echo -e "${CYAN}${message}${NC}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            echo -e "${CYAN}========================================${NC}\n" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        *)
            echo -e "${timestamp} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
    esac
}

# ============================================================================
# SSH helpers
# ============================================================================
ssh_cmd() {
    ssh -i "$SSH_KEY" ${SSH_OPTS} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR "${SSH_USER}@${INSTANCE_IP}" "export PATH=\$HOME/.local/bin:\$PATH; $*"
}

scp_to_remote() {
    local local_path=$1
    local remote_path=$2
    scp -i "$SSH_KEY" ${SSH_OPTS} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -r "$local_path" "${SSH_USER}@${INSTANCE_IP}:${remote_path}"
}

scp_from_remote() {
    local remote_path=$1
    local local_path=$2
    scp -i "$SSH_KEY" ${SSH_OPTS} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -r "${SSH_USER}@${INSTANCE_IP}:${remote_path}" "$local_path"
}

rsync_to_remote() {
    local local_path=$1
    local remote_path=$2
    # -L: copy symlink referent (dereference) so remote gets actual files
    rsync -avzL --progress \
        -e "ssh -i $SSH_KEY ${SSH_OPTS} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        "$local_path" "${SSH_USER}@${INSTANCE_IP}:${remote_path}"
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
  --instance-ip IP          Remote GPU instance IP address
  --experiment-name NAME     Unique experiment identifier
  --attracts-dir PATH       Path to full attracts folder (~250k files)
                            (ignored if --skip-filter)

Optional:
  --filtered-dir PATH       When --skip-filter: use this as demo source
  --replay-dir PATH         Path to folder containing .replay files to include
  --skip-filter             Use pre-filtered dir via --filtered-dir
  --ssh-key PATH            Path to SSH private key (default: ~/.ssh/id_rsa)
  --ssh-user USER           SSH username (default: ubuntu)
  --resume                  Skip dataset build if HDF5 exists, skip training if checkpoints exist
  --filter-workers N        Filter workers (default: 8)
  --build-workers N         Dataset build workers on remote (default: 8)
  --help                    Show this help message

Example:
  cd /path/to/nclone
  $0 \\
    --instance-ip 54.123.45.67 \\
    --experiment-name iql_run_2025 \\
    --attracts-dir ../attracts \\
    --replay-dir ./my_replays

EOF
    exit 1
}

# ============================================================================
# Argument parsing
# ============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --instance-ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            --experiment-name)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --attracts-dir)
                ATTRACTS_DIR="$2"
                shift 2
                ;;
            --filtered-dir)
                FILTERED_DIR="$2"
                shift 2
                ;;
            --replay-dir)
                REPLAY_DIR="$2"
                shift 2
                ;;
            --skip-filter)
                SKIP_FILTER=true
                shift
                ;;
            --ssh-key)
                SSH_KEY="$2"
                shift 2
                ;;
            --ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --filter-workers)
                FILTER_WORKERS="$2"
                shift 2
                ;;
            --build-workers)
                BUILD_WORKERS="$2"
                shift 2
                ;;
            --help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    if [[ -z "$INSTANCE_IP" ]] || [[ -z "$EXPERIMENT_NAME" ]]; then
        echo "Error: Missing required arguments (--instance-ip, --experiment-name)"
        usage
    fi

    if [[ "$SKIP_FILTER" == true ]]; then
        if [[ -z "$FILTERED_DIR" ]]; then
            echo "Error: --filtered-dir required when --skip-filter"
            usage
        fi
        if [[ ! -d "$FILTERED_DIR" ]]; then
            echo "Error: Filtered dir not found: $FILTERED_DIR"
            exit 1
        fi
    else
        if [[ -z "$ATTRACTS_DIR" ]]; then
            echo "Error: --attracts-dir required (or use --skip-filter --filtered-dir)"
            usage
        fi
        if [[ ! -d "$ATTRACTS_DIR" ]]; then
            echo "Error: Attracts dir not found: $ATTRACTS_DIR"
            exit 1
        fi
    fi

    if [[ -n "$REPLAY_DIR" ]] && [[ ! -d "$REPLAY_DIR" ]]; then
        echo "Error: Replay dir not found: $REPLAY_DIR"
        exit 1
    fi

    if [[ ! -f "$SSH_KEY" ]]; then
        echo "Error: SSH key not found at $SSH_KEY"
        exit 1
    fi
}

# ============================================================================
# Validate we're in nclone
# ============================================================================
validate_local() {
    if [[ ! -f "${NCLONE_DIR}/pyproject.toml" ]] || [[ ! -d "${NCLONE_DIR}/nclone" ]]; then
        log ERROR "Must run from nclone directory (or have scripts/ as subdir)"
        log ERROR "NCLONE_DIR: ${NCLONE_DIR}"
        exit 1
    fi
    log SUCCESS "nclone directory: ${NCLONE_DIR}"
}

# ============================================================================
# Filter attracts locally
# ============================================================================
run_filter() {
    local staging="$1"
    local filtered_out="${staging}/filtered"
    mkdir -p "$filtered_out"

    log HEADER "Filtering attracts locally"
    log INFO "Source: ${ATTRACTS_DIR}"
    log INFO "Output: ${filtered_out}"
    log INFO "Workers: ${FILTER_WORKERS}"

    (cd "${NCLONE_DIR}" && python -m nclone.offline_rl.dataset.filter_attracts \
        --source "${ATTRACTS_DIR}" \
        --output "${filtered_out}" \
        --workers "${FILTER_WORKERS}") || {
        log ERROR "Filter failed"
        exit 1
    }
    log SUCCESS "Filter complete"
    echo "${filtered_out}"
}

# ============================================================================
# Merge .replay files into demo dir
# ============================================================================
merge_replays() {
    local demo_dir=$1
    if [[ -z "$REPLAY_DIR" ]]; then
        return
    fi
    log INFO "Copying .replay files from ${REPLAY_DIR} into ${demo_dir}"
    local count
    count=$(find "$REPLAY_DIR" -maxdepth 1 -name "*.replay" -type f 2>/dev/null | wc -l)
    if [[ "$count" -eq 0 ]]; then
        log WARNING "No .replay files found in ${REPLAY_DIR}"
        return
    fi
    for f in "${REPLAY_DIR}"/*.replay; do
        [[ -f "$f" ]] && cp "$f" "$demo_dir"
    done
    log SUCCESS "Merged $count .replay files"
}

# ============================================================================
# Copy nclone to remote
# ============================================================================
copy_nclone_to_remote() {
    log HEADER "Copying nclone to remote"

    ssh_cmd "mkdir -p ${REMOTE_BASE_DIR}"

    log INFO "Creating tarball (excluding large/cache dirs)..."
    tar -czf /tmp/nclone-iql-${TIMESTAMP}.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.so' \
        --exclude='*.egg-info' \
        --exclude='build' \
        --exclude='dist' \
        --exclude='attract_*' \
        --exclude='attracts' \
        --exclude='*_videos' \
        --exclude='value_functions' \
        --exclude='simulation_phi_cache' \
        --exclude='path_analysis_export_*.png' \
        --exclude='iql_*.h5' \
        --exclude='iql_checkpoints' \
        --exclude='nclone-native/target' \
        -C "$(dirname "$NCLONE_DIR")" \
        "$(basename "$NCLONE_DIR")" 2>/dev/null || true

    log INFO "Uploading tarball..."
    scp_to_remote "/tmp/nclone-iql-${TIMESTAMP}.tar.gz" "${REMOTE_BASE_DIR}/"
    ssh_cmd "cd ${REMOTE_BASE_DIR} && tar -xzf nclone-iql-${TIMESTAMP}.tar.gz && rm nclone-iql-${TIMESTAMP}.tar.gz"
    rm -f /tmp/nclone-iql-${TIMESTAMP}.tar.gz
    log SUCCESS "nclone copied to ${REMOTE_NCLONE_DIR}"
}

# ============================================================================
# Sync demos to remote
# ============================================================================
sync_demos_to_remote() {
    local local_demo_dir=$1
    log HEADER "Syncing demos to remote (rsync -L to dereference symlinks)"

    local file_count
    file_count=$(find "$local_demo_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    log INFO "Transferring ${file_count} demo files to ${REMOTE_DEMOS_DIR}"

    ssh_cmd "mkdir -p ${REMOTE_DEMOS_DIR}"
    rsync_to_remote "${local_demo_dir}/" "${REMOTE_DEMOS_DIR}"
    log SUCCESS "Demos synced"
}

# ============================================================================
# Install system dependencies (cairo, build tools for pycairo)
# ============================================================================
install_system_dependencies() {
    log INFO "Verifying system dependencies on remote..."

    local missing_deps=()
    if ! ssh_cmd "pkg-config --exists cairo" > /dev/null 2>&1; then
        missing_deps+=("libcairo2-dev")
    fi
    if ! ssh_cmd "which pkg-config" > /dev/null 2>&1; then
        missing_deps+=("pkg-config")
    fi
    if ! ssh_cmd "which gcc" > /dev/null 2>&1; then
        missing_deps+=("build-essential")
    fi
    if ! ssh_cmd "ls /usr/include/python3.*/Python.h 2>/dev/null | grep -q ." 2>/dev/null; then
        missing_deps+=("python3-dev")
    fi

    if [[ ${#missing_deps[@]} -eq 0 ]]; then
        log SUCCESS "All system dependencies already installed"
        return 0
    fi

    log WARNING "Missing system dependencies: ${missing_deps[*]}"
    if ssh_cmd "sudo -n true" > /dev/null 2>&1; then
        log INFO "Installing: ${missing_deps[*]}"
        ssh_cmd "sudo apt-get update -qq && sudo apt-get install -y -qq ${missing_deps[*]}" || {
            log ERROR "Failed to install system dependencies"
            log ERROR "Please manually run: sudo apt-get install libcairo2-dev pkg-config build-essential python3-dev"
            exit 1
        }
        log SUCCESS "System dependencies installed"
    else
        log ERROR "Missing dependencies and no sudo access"
        log ERROR "Please manually run: sudo apt-get install libcairo2-dev pkg-config build-essential python3-dev"
        exit 1
    fi
}

# ============================================================================
# Setup remote environment
# Uses --system-site-packages venv to inherit the instance's CUDA PyTorch
# (same approach as npp-rl orchestration for Lambda Stack / GPU instances)
# ============================================================================
setup_remote() {
    log HEADER "Setting up remote environment"

    install_system_dependencies

    # Verify system PyTorch has CUDA before proceeding
    log INFO "Checking system PyTorch CUDA support..."
    local sys_cuda
    sys_cuda=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null" || echo "MISSING")
    if [[ "$sys_cuda" == "True" ]]; then
        local sys_torch_ver
        sys_torch_ver=$(ssh_cmd "python3 -c 'import torch; print(torch.__version__)'" 2>/dev/null || echo "unknown")
        local gpu_info
        gpu_info=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.get_device_name(0))'" 2>/dev/null || echo "unknown")
        log SUCCESS "System PyTorch ${sys_torch_ver} has CUDA (${gpu_info})"
    elif [[ "$sys_cuda" == "False" ]]; then
        log ERROR "System PyTorch is installed but has NO CUDA support"
        log ERROR "The remote instance needs a CUDA-enabled PyTorch (Lambda Stack, NGC container, etc.)"
        exit 1
    else
        log ERROR "System PyTorch not found on remote instance"
        log ERROR "The remote instance must have PyTorch with CUDA pre-installed"
        exit 1
    fi

    # Ensure venv was created with --system-site-packages (delete stale isolated venvs)
    if ssh_cmd "test -f ${REMOTE_VENV}/bin/activate" 2>/dev/null; then
        local has_system
        has_system=$(ssh_cmd "grep -c 'include-system-site-packages = true' ${REMOTE_VENV}/pyvenv.cfg 2>/dev/null" || echo "0")
        if [[ "$has_system" == "0" ]]; then
            log WARNING "Existing venv is isolated (no --system-site-packages). Recreating..."
            ssh_cmd "rm -rf ${REMOTE_VENV}"
        else
            log INFO "Virtual environment already exists at ${REMOTE_VENV}"
        fi
    fi

    if ! ssh_cmd "test -f ${REMOTE_VENV}/bin/activate" 2>/dev/null; then
        log INFO "Creating virtual environment with --system-site-packages..."
        ssh_cmd "python3 -m venv --system-site-packages ${REMOTE_VENV}" || {
            log ERROR "Failed to create venv"
            exit 1
        }
    fi

    local VENV_RUN="source ${REMOTE_VENV}/bin/activate &&"

    log INFO "Upgrading pip..."
    ssh_cmd "${VENV_RUN} pip install --upgrade pip setuptools wheel" > /dev/null 2>&1

    # Write a constraints file to keep numpy<2 across ALL pip installs.
    # System PyTorch (Lambda Stack) is compiled against numpy 1.x; numpy 2.x
    # causes "numpy.dtype size changed" crashes and breaks torch.cuda.
    log INFO "Pinning numpy<2 (system PyTorch requires numpy 1.x)..."
    ssh_cmd "echo 'numpy>=1.24,<2' > ${REMOTE_BASE_DIR}/constraints.txt"

    log INFO "Installing NumPy and h5py..."
    ssh_cmd "${VENV_RUN} pip install -c ${REMOTE_BASE_DIR}/constraints.txt 'numpy>=1.24,<2' 'h5py>=3.7'" || {
        log ERROR "Failed to install numpy/h5py"
        exit 1
    }

    # Install nclone (--no-deps to avoid pip pulling a CPU-only torch)
    log INFO "Installing nclone (--no-deps to preserve system PyTorch)..."
    ssh_cmd "${VENV_RUN} cd ${REMOTE_NCLONE_DIR} && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NCLONE=0.1.0 pip install --no-deps -e ." || {
        log ERROR "nclone install failed"
        exit 1
    }

    # Install remaining nclone deps (excluding torch and opencv which come from system).
    # Use --constraint to prevent any transitive dep from upgrading numpy to 2.x.
    log INFO "Installing nclone dependencies (excluding torch, using system opencv)..."
    ssh_cmd "${VENV_RUN} pip install -c ${REMOTE_BASE_DIR}/constraints.txt \
        'pygame>=2.1.0' 'gymnasium>=0.26.0' 'pycairo>=1.28.0' \
        'albumentations>=2.0.8' 'pillow>=10.0.0' 'tensorboard>=2.11.0'" || {
        log WARNING "Some optional deps failed, continuing..."
    }

    # Safety net: force numpy back to <2 in case a dep ignored the constraint
    ssh_cmd "${VENV_RUN} pip install 'numpy>=1.24,<2'" > /dev/null 2>&1

    # Verify CUDA is still working inside the venv
    log INFO "Verifying PyTorch CUDA inside venv..."
    local venv_cuda
    venv_cuda=$(ssh_cmd "${VENV_RUN} python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'" 2>/dev/null || echo "False 0")
    local cuda_avail
    cuda_avail=$(echo "$venv_cuda" | awk '{print $1}')
    local gpu_count
    gpu_count=$(echo "$venv_cuda" | awk '{print $2}')

    if [[ "$cuda_avail" == "True" ]]; then
        log SUCCESS "PyTorch CUDA available in venv (${gpu_count} GPU(s))"
    else
        log ERROR "PyTorch CUDA not available inside venv!"
        log ERROR "Check if a pip package shadowed system torch or upgraded numpy to 2.x"
        log ERROR "Debug: source ${REMOTE_VENV}/bin/activate && python -c 'import numpy; print(numpy.__version__); import torch; print(torch.__file__, torch.cuda.is_available())'"
        exit 1
    fi

    log SUCCESS "Remote setup complete"
}

# ============================================================================
# Build nclone-native Rust simulator (required for transition extraction)
# ============================================================================
build_nclone_native() {
    log HEADER "Building nclone-native Rust simulator"

    if ssh_cmd "source ${REMOTE_VENV}/bin/activate && python -c 'import nclone_native' 2>/dev/null"; then
        log INFO "nclone_native already installed, skipping build"
        return 0
    fi

    log INFO "Installing Rust toolchain (if needed)..."
    if ! ssh_cmd "test -f ~/.cargo/bin/rustc 2>/dev/null || command -v rustc >/dev/null 2>&1"; then
        ssh_cmd "curl -sSf https://sh.rustup.rs | sh -s -- -y -q" || {
            log ERROR "Failed to install Rust"
            exit 1
        }
    fi

    log INFO "Installing maturin..."
    ssh_cmd "source ${REMOTE_VENV}/bin/activate && pip install maturin" || {
        log ERROR "Failed to install maturin"
        exit 1
    }

    log INFO "Building nclone-native (this may take a few minutes)..."
    ssh_cmd "export PATH=\"\$HOME/.cargo/bin:\$PATH\"; source ${REMOTE_VENV}/bin/activate && cd ${REMOTE_NCLONE_DIR}/nclone-native && maturin develop --release" || {
        log ERROR "nclone-native build failed"
        log ERROR "Transition extraction requires nclone_native. Build manually: cd nclone-native && maturin develop --release"
        exit 1
    }

    log SUCCESS "nclone-native built"
}

# ============================================================================
# Build dataset on remote
# ============================================================================
build_dataset_remote() {
    log HEADER "Building HDF5 dataset on remote"

    if [[ "$RESUME" == true ]]; then
        if ssh_cmd "test -f ${REMOTE_DATASET}" 2>/dev/null; then
            log INFO "Resume: dataset exists, skipping build"
            return 0
        fi
    fi

    ssh_cmd "source ${REMOTE_VENV}/bin/activate && cd ${REMOTE_NCLONE_DIR} && python -m nclone.offline_rl.dataset.build_dataset_cli \
        --demo-dir ${REMOTE_DEMOS_DIR} \
        --output ${REMOTE_DATASET} \
        --workers ${BUILD_WORKERS}" || {
        log ERROR "Dataset build failed"
        exit 1
    }
    log SUCCESS "Dataset built: ${REMOTE_DATASET}"
}

# ============================================================================
# Validate dataset has required keys (tile_grids, etc.)
# ============================================================================
validate_dataset_remote() {
    if ! ssh_cmd "test -f ${REMOTE_DATASET}" 2>/dev/null; then
        return 1
    fi
    local check
    check=$(ssh_cmd "source ${REMOTE_VENV}/bin/activate && python -c \"
import h5py
import os
path = os.path.expanduser('${REMOTE_DATASET}')
try:
    with h5py.File(path, 'r') as f:
        ok = 'tile_grids' in f and 'switch_overlays' in f and 'exit_overlays' in f
        print('OK' if ok else 'INVALID')
except Exception as e:
    print('ERROR:', e)
\" 2>/dev/null" 2>/dev/null) || true
    if [[ "$check" == "OK" ]]; then
        return 0
    fi
    return 1
}

# ============================================================================
# Train IQL on remote
# ============================================================================
train_iql_remote() {
    log HEADER "Training IQL on remote"

    if [[ "$RESUME" == true ]]; then
        if ssh_cmd "test -d ${REMOTE_CHECKPOINTS} && test -n \"\$(ls -A ${REMOTE_CHECKPOINTS} 2>/dev/null)\"" 2>/dev/null; then
            log INFO "Resume: checkpoints exist, skipping training"
            return 0
        fi
    fi

    log INFO "Validating dataset before training..."
    if ! validate_dataset_remote; then
        log WARNING "Dataset missing tile_grids (incomplete/corrupt build). Rebuilding..."
        ssh_cmd "rm -f ${REMOTE_DATASET}"
        build_dataset_remote
    fi

    ssh_cmd "mkdir -p ${REMOTE_CHECKPOINTS}"
    echo "source ${REMOTE_VENV}/bin/activate && cd ${REMOTE_NCLONE_DIR} && python -m nclone.offline_rl.train_cli \
        --dataset ${REMOTE_DATASET} \
        --output-dir ${REMOTE_CHECKPOINTS} \
        --device cuda \
        --amp \
        --prefetch"
    log SUCCESS "IQL training complete"
}

# ============================================================================
# Download results
# ============================================================================
download_results() {
    log HEADER "Downloading results"

    local dest="${LOCAL_LOG_DIR}/checkpoints_${EXPERIMENT_NAME}_${TIMESTAMP}"
    mkdir -p "$dest"

    log INFO "Downloading checkpoints to ${dest}"
    scp_from_remote "${REMOTE_CHECKPOINTS}/" "$dest" 2>/dev/null || log WARNING "Could not download checkpoints"
    scp_from_remote "${REMOTE_BASE_DIR}/*.h5" "$dest" 2>/dev/null || true
    log SUCCESS "Results saved to ${dest}"
}

# ============================================================================
# Main
# ============================================================================
main() {
    mkdir -p "$LOCAL_LOG_DIR"

    log HEADER "IQL Offline RL Remote Training Orchestration"
    parse_args "$@"

    log INFO "Configuration:"
    log INFO "  Instance: ${INSTANCE_IP}"
    log INFO "  Experiment: ${EXPERIMENT_NAME}"
    log INFO "  SSH Key: ${SSH_KEY}"
    log INFO "  SSH User: ${SSH_USER}"
    log INFO "  Skip filter: ${SKIP_FILTER}"
    log INFO "  Resume: ${RESUME}"
    echo ""

    validate_local

    # Staging dir for filtered demos (always use staging so we never modify user dirs)
    STAGING_DIR=$(mktemp -d)
    trap "rm -rf ${STAGING_DIR}" EXIT
    DEMO_DIR="${STAGING_DIR}/filtered"
    mkdir -p "$DEMO_DIR"

    if [[ "$SKIP_FILTER" == true ]]; then
        log INFO "Copying pre-filtered dir: ${FILTERED_DIR} -> ${DEMO_DIR}"
        cp -a "${FILTERED_DIR}/." "$DEMO_DIR/"
    else
        run_filter "$STAGING_DIR" > /dev/null
        # run_filter creates STAGING_DIR/filtered
    fi

    merge_replays "$DEMO_DIR"

    log INFO "Validating SSH connection..."
    if ! ssh_cmd "echo 'SSH OK'" > /dev/null 2>&1; then
        log ERROR "SSH connection failed"
        exit 1
    fi
    log SUCCESS "SSH connection OK"

    copy_nclone_to_remote
    sync_demos_to_remote "$DEMO_DIR"
    setup_remote
    build_nclone_native
    build_dataset_remote
    train_iql_remote
    download_results

    log SUCCESS "Orchestration complete!"
}

# ============================================================================
# Entry point
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
