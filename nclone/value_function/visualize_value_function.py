#!/usr/bin/env python3
"""CLI entry point for value-function visualization.

Dispatches to ``SurfaceGraphVisualizer`` (surface-graph PBRS) or
``NeuralValueLandscape`` (IQL V-network heatmap) depending on the
``--mode`` argument.

Surface-graph modes (require *_surface_pbrs.npz)::

    python -m nclone.value_function.visualize_value_function \\
        --data value_functions/rampjump_surface_pbrs.npz --mode heatmap

    python -m nclone.value_function.visualize_value_function \\
        --data value_functions/rampjump_surface_pbrs.npz --mode dashboard

Neural V-network modes (require exported v_network.pt and HDF5 dataset)::

    python -m nclone.value_function.visualize_value_function \\
        --mode neural \\
        --checkpoint v_network.pt \\
        --dataset datasets/iql_pipeline_v2/iql_train_dataset.h5 \\
        --level-idx 0 \\
        --output docs/generalizable_agent/02-offline-pretraining/value-landscapes/level0.png
"""

import argparse
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PBRS / IQL value functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "edges", "dashboard", "neural",
                 "arc_heatmap", "combined", "mine_masking"],
        default="heatmap",
        help=(
            "Visualization mode (default: heatmap). "
            "arc_heatmap: two-panel arc cache min-cost + coverage density. "
            "combined: single panel showing both surface d* and arc cache potential "
            "on the same normalised scale. "
            "mine_masking: overlay mine kill-radius circles and highlight "
            "blocked nodes/edges so you can verify that mine-proximity masking "
            "correctly removes dangerous routes."
        ),
    )
    # Surface-graph options
    parser.add_argument("--data", default=None, help="Path to *_surface_pbrs.npz")
    parser.add_argument("--map-path", default=None, help="Optional .map file for tile overlay")
    parser.add_argument(
        "--phase",
        choices=["switch", "exit"],
        default="switch",
        help="d_star phase for heatmap mode (default: switch)",
    )
    parser.add_argument(
        "--max-edges", type=int, default=5000,
        help="Max edges to draw in edge mode (default: 5000)",
    )
    # Neural landscape options
    parser.add_argument("--checkpoint", default=None, help="Path to exported v_network.pt")
    parser.add_argument(
        "--dataset", default=None,
        help="Path to HDF5 dataset (for tile grid / level data)",
    )
    parser.add_argument(
        "--level-idx", type=int, default=0,
        help="Level index within the HDF5 dataset (default: 0)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for neural mode (default: cpu)",
    )
    parser.add_argument(
        "--step-px", type=int, default=12,
        help="Pixel stride for neural sweep grid (default: 12)",
    )
    # Shared
    parser.add_argument("--output", default=None, help="Save figure to file")
    return parser.parse_args()


def _run_surface(args: argparse.Namespace) -> int:
    from .value_landscape import SurfaceGraphVisualizer
    if not args.data or not os.path.exists(args.data):
        print(f"ERROR: --data file not found: {args.data}")
        return 1
    viz = SurfaceGraphVisualizer(args.data, map_path=args.map_path)
    if args.mode == "heatmap":
        viz.visualize_heatmap(phase=args.phase, output_path=args.output)
    elif args.mode == "edges":
        viz.visualize_edges(max_edges=args.max_edges, output_path=args.output)
    elif args.mode == "dashboard":
        viz.visualize_dashboard(output_path=args.output)
    elif args.mode == "arc_heatmap":
        viz.visualize_arc_heatmap(phase=args.phase, output_path=args.output)
    elif args.mode == "combined":
        viz.visualize_combined_potential(phase=args.phase, output_path=args.output)
    elif args.mode == "mine_masking":
        viz.visualize_mine_masking(phase=args.phase, output_path=args.output)
    return 0


def _run_neural(args: argparse.Namespace) -> int:
    import h5py
    import numpy as np
    from .value_landscape import NeuralValueLandscape

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"ERROR: --checkpoint not found: {args.checkpoint}")
        return 1
    if not args.dataset or not os.path.exists(args.dataset):
        print(f"ERROR: --dataset not found: {args.dataset}")
        return 1

    _TILE_SIZE_PX = 24

    with h5py.File(args.dataset, "r") as f:
        idx = args.level_idx
        tile_grid = f["tile_grids"][idx].astype(np.int64)
        switch_overlay = f["switch_overlays"][idx].astype(np.float32)
        exit_overlay = f["exit_overlays"][idx].astype(np.float32)

    sw_idx = np.unravel_index(np.argmax(switch_overlay), switch_overlay.shape)
    ex_idx = np.unravel_index(np.argmax(exit_overlay), exit_overlay.shape)
    switch_pos = (sw_idx[0] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
                  sw_idx[1] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2)
    exit_pos = (ex_idx[0] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2,
                ex_idx[1] * _TILE_SIZE_PX + _TILE_SIZE_PX / 2)

    landscape = NeuralValueLandscape(
        args.checkpoint, device=args.device, step_px=args.step_px,
    )
    landscape.set_level(
        tile_grid=tile_grid,
        switch_pos=switch_pos,
        exit_pos=exit_pos,
        mine_positions=[],
    )

    output = args.output
    if output:
        base, ext = os.path.splitext(output)
        if not ext:
            ext = ".png"
        landscape.render_dual(
            output_path=f"{base}_dual{ext}",
            level_label=f"level {idx}",
        )
        landscape.render_heatmap(phase=0, output_path=f"{base}_phase0{ext}")
        landscape.render_heatmap(phase=1, output_path=f"{base}_phase1{ext}")
    else:
        landscape.render_dual(level_label=f"level {idx}")

    return 0


def main() -> int:
    args = _parse_args()
    if args.mode == "neural":
        return _run_neural(args)
    return _run_surface(args)


if __name__ == "__main__":
    sys.exit(main())
