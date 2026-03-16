#!/bin/bash
###############################################################################
# build_augmented_dataset.sh
#
# End-to-end pipeline: filtered attract corpus → augmented HDF5 datasets.
#
# Steps:
#   1. split   -- split ~18k filtered attracts into train/val by level ID
#   2. augment -- run all 3 augmentations on each split independently
#   3. merge   -- combine originals + augmented into combined_train/val dirs
#   4. build   -- build iql_train_dataset.h5 and iql_val_dataset.h5
#   5. validate-- check reward structure and structural integrity
#
# Prerequisites:
#   - Run from the nclone/ project root
#   - Python env with nclone and nclone_native installed
#   - FILTERED_DIR must point to the output of filter_attracts.py (T1.1)
#
# Usage:
#   ./scripts/build_augmented_dataset.sh \
#       --filtered-dir /path/to/filtered_attracts \
#       --output-dir   /path/to/dataset_workspace \
#       --workers      8
#
# Optional flags:
#   --skip-split       Skip step 1 (splits already exist in --output-dir)
#   --skip-augment     Skip step 2 (augmented dirs already exist)
#   --skip-merge       Skip step 3 (combined dirs already exist)
#   --skip-build       Skip step 4 (HDF5 files already exist)
#   --skip-validate    Skip step 5
#   --val-fraction F   Fraction for val split (default: 0.10)
#   --seed N           RNG seed for reproducible split (default: 42)
#   --verify-sample N  Number of augmented files to verify via Rust sim (default: 200)
###############################################################################

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
FILTERED_DIR=""
OUTPUT_DIR=""
WORKERS=8
VAL_FRACTION="0.10"
SEED=42
VERIFY_SAMPLE=200
SKIP_SPLIT=false
SKIP_AUGMENT=false
SKIP_MERGE=false
SKIP_BUILD=false
SKIP_VALIDATE=false

# ============================================================================
# Argument parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --filtered-dir)  FILTERED_DIR="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --workers)       WORKERS="$2";       shift 2 ;;
        --val-fraction)  VAL_FRACTION="$2";  shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --verify-sample) VERIFY_SAMPLE="$2"; shift 2 ;;
        --skip-split)    SKIP_SPLIT=true;    shift ;;
        --skip-augment)  SKIP_AUGMENT=true;  shift ;;
        --skip-merge)    SKIP_MERGE=true;    shift ;;
        --skip-build)    SKIP_BUILD=true;    shift ;;
        --skip-validate) SKIP_VALIDATE=true; shift ;;
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$FILTERED_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --filtered-dir and --output-dir are required." >&2
    exit 1
fi

FILTERED_DIR="$(realpath "$FILTERED_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

TRAIN_DIR="${OUTPUT_DIR}/train"
VAL_DIR="${OUTPUT_DIR}/val"
AUG_TRAIN_DIR="${OUTPUT_DIR}/augmented_train"
AUG_VAL_DIR="${OUTPUT_DIR}/augmented_val"
COMBINED_TRAIN_DIR="${OUTPUT_DIR}/combined_train"
COMBINED_VAL_DIR="${OUTPUT_DIR}/combined_val"
TRAIN_H5="${OUTPUT_DIR}/iql_train_dataset.h5"
VAL_H5="${OUTPUT_DIR}/iql_val_dataset.h5"

# ============================================================================
# Helpers
# ============================================================================
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Step 1 — Split
# ============================================================================
if [[ "$SKIP_SPLIT" == "false" ]]; then
    log "=== Step 1: Splitting filtered corpus by level ID ==="
    python -m nclone.offline_rl.dataset.prepare_splits split \
        --source   "$FILTERED_DIR" \
        --output   "$OUTPUT_DIR" \
        --val-fraction "$VAL_FRACTION" \
        --seed     "$SEED"
    log "Split complete."
else
    log "=== Step 1: Skipping split (--skip-split) ==="
    [[ -d "$TRAIN_DIR" ]] || die "Expected train dir not found: $TRAIN_DIR"
    [[ -d "$VAL_DIR"   ]] || die "Expected val dir not found:   $VAL_DIR"
fi

# ============================================================================
# Step 2 — Augment
# ============================================================================
if [[ "$SKIP_AUGMENT" == "false" ]]; then
    log "=== Step 2a: Augmenting train split ==="
    python -m nclone.offline_rl.dataset.augmentation.augment_cli \
        --source       "$TRAIN_DIR" \
        --output       "$AUG_TRAIN_DIR" \
        --augmentations horizontal_mirror tile_rand translate \
        --workers      "$WORKERS" \
        --verify-sample "$VERIFY_SAMPLE" \
        --verify-remove-failures \
        --seed         "$SEED"

    log "=== Step 2b: Augmenting val split ==="
    VAL_VERIFY=$(( VERIFY_SAMPLE / 4 ))
    python -m nclone.offline_rl.dataset.augmentation.augment_cli \
        --source       "$VAL_DIR" \
        --output       "$AUG_VAL_DIR" \
        --augmentations horizontal_mirror tile_rand translate \
        --workers      "$WORKERS" \
        --verify-sample "$VAL_VERIFY" \
        --verify-remove-failures \
        --seed         "$SEED"
    log "Augmentation complete."
else
    log "=== Step 2: Skipping augmentation (--skip-augment) ==="
    [[ -d "$AUG_TRAIN_DIR" ]] || die "Expected augmented_train not found: $AUG_TRAIN_DIR"
    [[ -d "$AUG_VAL_DIR"   ]] || die "Expected augmented_val not found:   $AUG_VAL_DIR"
fi

# ============================================================================
# Step 3 — Merge originals + augmented
# ============================================================================
if [[ "$SKIP_MERGE" == "false" ]]; then
    log "=== Step 3: Merging originals and augmented files ==="
    python -m nclone.offline_rl.dataset.prepare_splits merge \
        --splits-dir "$OUTPUT_DIR"
    log "Merge complete."
else
    log "=== Step 3: Skipping merge (--skip-merge) ==="
    [[ -d "$COMBINED_TRAIN_DIR" ]] || die "Expected combined_train not found: $COMBINED_TRAIN_DIR"
    [[ -d "$COMBINED_VAL_DIR"   ]] || die "Expected combined_val not found:   $COMBINED_VAL_DIR"
fi

# ============================================================================
# Step 4 — Build HDF5 datasets
# ============================================================================
if [[ "$SKIP_BUILD" == "false" ]]; then
    log "=== Step 4a: Building training HDF5 ==="
    python -m nclone.offline_rl.dataset.build_dataset_cli \
        --demo-dir "$COMBINED_TRAIN_DIR" \
        --output   "$TRAIN_H5" \
        --workers  "$WORKERS" \
        --compression gzip

    log "=== Step 4b: Building validation HDF5 ==="
    HALF_WORKERS=$(( WORKERS / 2 ))
    HALF_WORKERS=$(( HALF_WORKERS < 1 ? 1 : HALF_WORKERS ))
    python -m nclone.offline_rl.dataset.build_dataset_cli \
        --demo-dir "$COMBINED_VAL_DIR" \
        --output   "$VAL_H5" \
        --workers  "$HALF_WORKERS" \
        --compression gzip
    log "HDF5 build complete."
else
    log "=== Step 4: Skipping HDF5 build (--skip-build) ==="
    [[ -f "$TRAIN_H5" ]] || die "Expected train HDF5 not found: $TRAIN_H5"
    [[ -f "$VAL_H5"   ]] || die "Expected val HDF5 not found:   $VAL_H5"
fi

# ============================================================================
# Step 5 — Validate datasets
# ============================================================================
if [[ "$SKIP_VALIDATE" == "false" ]]; then
    log "=== Step 5: Validating datasets ==="
    python -m nclone.offline_rl.dataset.validate_dataset \
        --train "$TRAIN_H5" \
        --val   "$VAL_H5" \
        --sample 100 \
        --seed "$SEED" \
        --check-overlap
    log "Validation complete."
else
    log "=== Step 5: Skipping validation (--skip-validate) ==="
fi

# ============================================================================
# Summary
# ============================================================================
log "==================================================================="
log "Pipeline complete!"
log "  Train HDF5: $TRAIN_H5"
log "  Val HDF5:   $VAL_H5"
log "==================================================================="
