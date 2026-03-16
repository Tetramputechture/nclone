#!/usr/bin/env python3
"""Generate V(s) value landscape heatmaps for 5 representative held-out levels.

Produces dual-phase (phase 0 = before switch, phase 1 = after switch) heatmaps
for 5 levels from the validation HDF5, including at least one mine-heavy level.

Requires:
  - Trained and exported v_network.pt (from T2.3)
  - datasets/iql_pipeline_v2/iql_val_dataset.h5

Output saved to:
  docs/generalizable_agent/02-offline-pretraining/value-landscapes/

Usage (from the nclone project root)::

    python scripts/generate_value_landscapes.py \\
        --checkpoint v_network.pt \\
        --val-dataset datasets/iql_pipeline_v2/iql_val_dataset.h5 \\
        --output-dir docs/generalizable_agent/02-offline-pretraining/value-landscapes \\
        --device cuda
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nclone.value_function.value_landscape import NeuralValueLandscape

_TILE_SIZE_PX = 24
_GRID_W = 42
_GRID_H = 23
_LEVEL_W_PX = _GRID_W * _TILE_SIZE_PX
_LEVEL_H_PX = _GRID_H * _TILE_SIZE_PX


def _count_mines_in_level(mine_feature_block: np.ndarray) -> int:
    """Estimate unique mine count from mine features stored in the HDF5.

    Mine features per transition are 8 nearest mines × 3 features. We infer
    level mine count from the number of distinct (dx, dy) pairs observed.
    This is a rough proxy — use the actual level entity data when available.
    """
    # Not directly available; return 0 (mines are looked up per-level separately)
    return 0


def _pick_level_indices(f: h5py.File, n: int = 5) -> list:
    """Pick n level indices from the val HDF5.

    Strategy:
      - At least 1 mine-heavy level: levels where mine_features has non-zero
        entries for many transitions (proxy for >50 mines).
      - The remaining levels are spread across the level-ID range.
    """
    n_levels = f["tile_grids"].shape[0]
    level_ids = f["level_ids"][:]

    # Count transitions per level
    transitions_per_level = np.bincount(level_ids.astype(np.int64), minlength=n_levels)

    # Find mine-heavy candidate: levels with non-zero mine features
    mine_feats = f["mine_features"][:]  # (N, 24)
    mine_nonzero = (mine_feats != 0).any(axis=1)  # (N,) bool
    level_has_mines = np.zeros(n_levels, dtype=bool)
    for lid in range(n_levels):
        mask = level_ids == lid
        if mask.any() and mine_nonzero[mask].mean() > 0.5:
            level_has_mines[lid] = True

    mine_heavy_candidates = np.where(level_has_mines)[0]
    chosen = []

    if len(mine_heavy_candidates) > 0:
        chosen.append(int(mine_heavy_candidates[0]))

    # Fill the rest by spreading evenly over the level count
    step = max(1, n_levels // (n - len(chosen) + 1))
    for i in range(0, n_levels, step):
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)

    return chosen[:n]


def _level_index_to_tile_data(f: h5py.File, level_idx: int):
    """Return (tile_grid, switch_pos, exit_pos) for a level by its level index."""
    tile_grid = f["tile_grids"][level_idx].astype(np.int64)
    switch_overlay = f["switch_overlays"][level_idx].astype(np.float32)
    exit_overlay = f["exit_overlays"][level_idx].astype(np.float32)

    sw_idx = np.unravel_index(np.argmax(switch_overlay), switch_overlay.shape)
    ex_idx = np.unravel_index(np.argmax(exit_overlay), exit_overlay.shape)

    switch_pos = (
        sw_idx[0] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
        sw_idx[1] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
    )
    exit_pos = (
        ex_idx[0] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
        ex_idx[1] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
    )
    return tile_grid, switch_pos, exit_pos


def run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1
    if not os.path.exists(args.val_dataset):
        print(f"ERROR: val dataset not found: {args.val_dataset}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading validation dataset: {args.val_dataset}")
    with h5py.File(args.val_dataset, "r") as f:
        n_levels = f["tile_grids"].shape[0]
        print(f"  Val levels: {n_levels}")
        chosen = _pick_level_indices(f, n=5)
        print(f"  Selected level indices: {chosen}")
        level_data = [_level_index_to_tile_data(f, idx) for idx in chosen]

    landscape = NeuralValueLandscape(
        args.checkpoint,
        device=args.device,
        step_px=args.step_px,
    )

    for i, (level_idx, (tile_grid, switch_pos, exit_pos)) in enumerate(
        zip(chosen, level_data)
    ):
        print(f"\n[{i+1}/5] Level index {level_idx}  "
              f"switch={switch_pos}  exit={exit_pos}")

        landscape.set_level(
            tile_grid=tile_grid,
            switch_pos=switch_pos,
            exit_pos=exit_pos,
            mine_positions=[],
        )

        base = os.path.join(args.output_dir, f"level_{level_idx:04d}")
        landscape.render_dual(
            output_path=f"{base}_dual.png",
            level_label=f"val level {level_idx}",
        )
        landscape.render_heatmap(phase=0, output_path=f"{base}_phase0.png")
        landscape.render_heatmap(phase=1, output_path=f"{base}_phase1.png")

    print(f"\nAll landscapes saved to {args.output_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V(s) value landscape heatmaps for 5 val levels"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to exported v_network.pt",
    )
    parser.add_argument(
        "--val-dataset", required=True,
        help="Path to validation HDF5 dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/generalizable_agent/02-offline-pretraining/value-landscapes",
        help="Directory to save PNG heatmaps",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--step-px", type=int, default=12,
        help="Pixel stride for sweep grid (default: 12)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
