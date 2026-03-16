#!/usr/bin/env bash
# Post-training automation: export V-network and generate value landscapes.
# Run this script after IQL training completes (train_cli.py exits cleanly).
#
# Usage (from nclone project root):
#   bash scripts/post_training.sh
#
# Requires:
#   iql_checkpoints/iql_best_val.pt  (created by train_cli.py T2.3)
#   datasets/iql_pipeline_v2/iql_val_dataset.h5

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="iql_checkpoints/iql_best_val.pt"
VAL_DATASET="datasets/iql_pipeline_v2/iql_val_dataset.h5"
V_NETWORK_OUT="v_network.pt"
LANDSCAPE_DIR="docs/generalizable_agent/02-offline-pretraining/value-landscapes"

echo "=== T2.3: Exporting V-network ==="
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: $CHECKPOINT not found. Did training complete?" >&2
    exit 1
fi

python -m nclone.offline_rl.export \
    --checkpoint "$CHECKPOINT" \
    --output "$V_NETWORK_OUT"

echo ""
echo "=== T2.5: Generating value landscape heatmaps ==="
python scripts/generate_value_landscapes.py \
    --checkpoint "$V_NETWORK_OUT" \
    --val-dataset "$VAL_DATASET" \
    --output-dir "$LANDSCAPE_DIR" \
    --device cuda \
    --step-px 12

echo ""
echo "=== T2.4: Running NeuralValuePBRS integration tests ==="
python -m pytest nclone/value_function/tests/test_neural_pbrs_integration.py -v \
    --tb=short

echo ""
echo "=== Post-training steps complete ==="
echo "  v_network.pt        → $V_NETWORK_OUT"
echo "  Value landscapes    → $LANDSCAPE_DIR/"
echo "  Update training-run-notes.md with final results from TensorBoard."
