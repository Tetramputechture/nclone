#!/usr/bin/env python3
"""Visualization tool for simulation-based phi grids.

Generates a 6-panel dashboard to validate phi grid correctness:
1. Grid classification (valid/invalid cells)
2. Reachable cells from spawn
3. Phi switch heatmap
4. Phi exit heatmap
5. Raw distance (d_switch) heatmap
6. Statistics panel

Usage:
    # Build and visualize inline (exposes intermediate diagnostics)
    python -m nclone.simulation_pbrs.visualize_phi_grid \\
        --map-path test-single-level/rampjump \\
        --save phi_grid_rampjump.png

    # Visualize from cached .npz
    python -m nclone.simulation_pbrs.visualize_phi_grid \\
        --map-path test-single-level/rampjump \\
        --cache-dir simulation_phi_cache/ \\
        --save phi_grid_rampjump.png
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for WSL/CI
import matplotlib.pyplot as plt

from .grid_classifier import GridClassifier, GRID_WIDTH, GRID_HEIGHT
from .transition_enumerator import TransitionEnumerator
from .backward_dijkstra import backward_dijkstra
from .phi_grid_builder import (
    SimulationPhiGridBuilder,
    _compute_cache_key,
    _flood_fill_from_spawn,
    _normalize_distance_to_phi,
)

logger = logging.getLogger(__name__)

# Extent for imshow: pixel coordinates [x_min, x_max, y_max, y_min] (y inverted)
EXTENT = [0, 1056, 600, 0]


def _create_sim_and_goals(map_path: str, use_native: bool = True):
    """Create simulator and extract switch/exit/spawn positions."""
    if use_native:
        try:
            from .native_sim_adapter import create_native_sim_from_map_path
            sim, switch_pos, exit_pos = create_native_sim_from_map_path(map_path)
            spawn_pos = (sim.ninja.xpos, sim.ninja.ypos)
            return sim, switch_pos, exit_pos, spawn_pos
        except ImportError:
            pass

    from ..nplay_headless import NPlayHeadless
    nplay = NPlayHeadless(enable_rendering=False)
    nplay.load_map(map_path)
    nplay.reset()
    switch_pos = nplay.exit_switch_position()
    exit_pos = nplay.exit_door_position()
    spawn_pos = (nplay.sim.ninja.xpos, nplay.sim.ninja.ypos)
    return nplay.sim, switch_pos, exit_pos, spawn_pos


def _build_and_collect_diagnostics(
    map_path: str,
    use_native: bool = True,
) -> Dict:
    """Run the full phi grid pipeline and collect intermediate results."""
    sim, switch_pos, exit_pos, spawn_pos = _create_sim_and_goals(
        map_path, use_native=use_native
    )

    valid_mask = GridClassifier.classify(sim.tile_dic)
    enumerator = TransitionEnumerator(max_frames=50)
    adjacency = enumerator.enumerate(sim, valid_mask)

    spawn_cell = GridClassifier.pixel_to_cell(spawn_pos[0], spawn_pos[1])
    reachable_cells = _flood_fill_from_spawn(spawn_cell, adjacency)

    reachable_mask = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)
    for ix, iy in reachable_cells:
        if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
            reachable_mask[ix, iy] = True

    filtered_adjacency = {}
    for src, neighbors in adjacency.items():
        if src not in reachable_cells:
            continue
        filtered_neighbors = [
            (dest, cost) for dest, cost in neighbors if dest in reachable_cells
        ]
        if filtered_neighbors:
            filtered_adjacency[src] = filtered_neighbors

    d_switch = backward_dijkstra(filtered_adjacency, switch_pos, reachable_mask)
    d_exit = backward_dijkstra(filtered_adjacency, exit_pos, reachable_mask)
    phi_switch = _normalize_distance_to_phi(d_switch)
    phi_exit = _normalize_distance_to_phi(d_exit)

    return {
        "tile_dic": sim.tile_dic,
        "valid_mask": valid_mask,
        "reachable_mask": reachable_mask,
        "reachable_cells": reachable_cells,
        "adjacency": filtered_adjacency,
        "d_switch": d_switch,
        "d_exit": d_exit,
        "phi_switch": phi_switch,
        "phi_exit": phi_exit,
        "switch_pos": switch_pos,
        "exit_pos": exit_pos,
        "spawn_pos": spawn_pos,
    }


def _load_from_cache(
    map_path: str,
    cache_dir: str,
) -> Optional[Dict]:
    """Load phi grids from cache and derive diagnostics from map."""
    sim, switch_pos, exit_pos, spawn_pos = _create_sim_and_goals(map_path)
    cache_key = _compute_cache_key(
        sim.tile_dic, switch_pos, exit_pos, spawn_pos
    )
    cache_path = os.path.join(cache_dir, f"sim_phi_{cache_key}.npz")

    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path)
    phi_switch = data["phi_switch"]
    phi_exit = data["phi_exit"]
    d_switch = data["d_switch"]
    d_exit = data["d_exit"]

    valid_mask = GridClassifier.classify(sim.tile_dic)
    if "reachable_mask" in data:
        reachable_mask = data["reachable_mask"]
    else:
        # Fallback for old caches: use valid_mask (all valid cells)
        reachable_mask = valid_mask.copy()

    return {
        "tile_dic": sim.tile_dic,
        "valid_mask": valid_mask,
        "reachable_mask": reachable_mask,
        "reachable_cells": None,
        "adjacency": None,
        "d_switch": d_switch,
        "d_exit": d_exit,
        "phi_switch": phi_switch,
        "phi_exit": phi_exit,
        "switch_pos": switch_pos,
        "exit_pos": exit_pos,
        "spawn_pos": spawn_pos,
    }


def _render_tiles_to_axis(ax, tile_dic: Dict[Tuple[int, int], int]) -> None:
    """Render level tiles to matplotlib axis."""
    from ..value_function.visualize_value_function import render_tiles_to_axis as _render
    _render(ax, tile_dic, tile_size=24.0, tile_color="#404040", alpha=0.8)


class PhiGridVisualizer:
    """Generates a 6-panel dashboard for phi grid validation."""

    def __init__(self, diagnostics: Dict):
        self.diag = diagnostics

    def _render_classification_panel(self, ax) -> None:
        """Panel 1: Valid (open) vs invalid (solid) cells."""
        _render_tiles_to_axis(ax, self.diag["tile_dic"])
        mask = self.diag["valid_mask"]
        display = np.where(mask, 1.0, 0.0)
        display = np.ma.masked_where(~mask, display)
        ax.imshow(
            display.T,
            extent=EXTENT,
            cmap="Greens",
            vmin=0,
            vmax=1,
            alpha=0.5,
            interpolation="nearest",
            aspect="equal",
            origin="upper",
        )
        ax.set_title("Grid Classification (valid=green)")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)

    def _render_reachable_panel(self, ax) -> None:
        """Panel 2: Cells reachable from spawn."""
        _render_tiles_to_axis(ax, self.diag["tile_dic"])
        mask = self.diag["reachable_mask"]
        display = np.where(mask, 1.0, 0.0)
        display = np.ma.masked_where(~mask, display)
        ax.imshow(
            display.T,
            extent=EXTENT,
            cmap="Blues",
            vmin=0,
            vmax=1,
            alpha=0.5,
            interpolation="nearest",
            aspect="equal",
            origin="upper",
        )
        sx, sy = self.diag["spawn_pos"]
        ax.plot(sx, sy, "y*", markersize=15, label="Spawn")
        ax.set_title("Reachable from Spawn")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.legend(loc="upper right")

    def _render_phi_panel(
        self,
        ax,
        phi: np.ndarray,
        goal_pos: Tuple[float, float],
        title: str,
        goal_label: str,
    ) -> None:
        """Render a phi heatmap with tile overlay and goal marker."""
        _render_tiles_to_axis(ax, self.diag["tile_dic"])
        masked = np.ma.masked_where(
            ~self.diag["reachable_mask"],
            phi,
        )
        im = ax.imshow(
            masked.T,
            extent=EXTENT,
            cmap="viridis",
            vmin=-1,
            vmax=0,
            alpha=0.7,
            interpolation="nearest",
            aspect="equal",
            origin="upper",
        )
        ax.plot(
            goal_pos[0],
            goal_pos[1],
            "r*",
            markersize=15,
            label=goal_label,
        )
        ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label="phi [-1, 0]")

    def _render_distance_panel(self, ax) -> None:
        """Panel 5: Raw frame-count distance to switch."""
        _render_tiles_to_axis(ax, self.diag["tile_dic"])
        d = self.diag["d_switch"]
        masked = np.ma.masked_where(
            ~self.diag["reachable_mask"],
            d,
        )
        im = ax.imshow(
            masked.T,
            extent=EXTENT,
            cmap="coolwarm",
            vmin=0,
            vmax=float(np.percentile(d[self.diag["reachable_mask"]], 95)),
            alpha=0.7,
            interpolation="nearest",
            aspect="equal",
            origin="upper",
        )
        sx, sy = self.diag["switch_pos"]
        ax.plot(sx, sy, "r*", markersize=15, label="Switch")
        ax.set_title("Raw Distance to Switch (frames)")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label="frames")

    def _render_stats_panel(self, ax) -> None:
        """Panel 6: Key metrics."""
        ax.axis("off")
        valid_count = int(self.diag["valid_mask"].sum())
        reachable_count = int(self.diag["reachable_mask"].sum())
        total = GRID_WIDTH * GRID_HEIGHT

        phi_s = self.diag["phi_switch"]
        phi_e = self.diag["phi_exit"]
        d_s = self.diag["d_switch"]
        d_e = self.diag["d_exit"]
        reachable = self.diag["reachable_mask"]

        phi_s_reach = phi_s[reachable]
        phi_e_reach = phi_e[reachable]
        d_s_reach = d_s[reachable]
        d_e_reach = d_e[reachable]

        edge_count = 0
        if self.diag.get("adjacency"):
            edge_count = sum(
                len(neighbors) for neighbors in self.diag["adjacency"].values()
            )

        stats = f"""
PHI GRID STATISTICS
{"=" * 30}

Grid: {GRID_WIDTH} x {GRID_HEIGHT} ({total} cells)
Valid cells: {valid_count} ({100*valid_count/total:.1f}%)
Reachable from spawn: {reachable_count} ({100*reachable_count/total:.1f}%)
Edges: {edge_count if edge_count else "N/A (from cache)"}

Phi Switch:
  Range: [{phi_s_reach.min():.3f}, {phi_s_reach.max():.3f}]
  Mean: {phi_s_reach.mean():.3f}

Phi Exit:
  Range: [{phi_e_reach.min():.3f}, {phi_e_reach.max():.3f}]
  Mean: {phi_e_reach.mean():.3f}

Distance to Switch:
  d_max: {d_s_reach.max():.0f} frames
  Mean: {d_s_reach.mean():.1f} frames

Distance to Exit:
  d_max: {d_e_reach.max():.0f} frames
  Mean: {d_e_reach.mean():.1f} frames

Positions (px):
  Spawn: ({self.diag["spawn_pos"][0]:.0f}, {self.diag["spawn_pos"][1]:.0f})
  Switch: ({self.diag["switch_pos"][0]:.0f}, {self.diag["switch_pos"][1]:.0f})
  Exit: ({self.diag["exit_pos"][0]:.0f}, {self.diag["exit_pos"][1]:.0f})
"""
        ax.text(
            0.05,
            0.95,
            stats,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
        )

    def generate_dashboard(self) -> plt.Figure:
        """Assemble the 6-panel figure."""
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        self._render_classification_panel(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._render_reachable_panel(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._render_phi_panel(
            ax3,
            self.diag["phi_switch"],
            self.diag["switch_pos"],
            "Phi Switch",
            "Switch",
        )

        ax4 = fig.add_subplot(gs[1, 0])
        self._render_phi_panel(
            ax4,
            self.diag["phi_exit"],
            self.diag["exit_pos"],
            "Phi Exit",
            "Exit",
        )

        ax5 = fig.add_subplot(gs[1, 1])
        self._render_distance_panel(ax5)

        ax6 = fig.add_subplot(gs[1, 2])
        self._render_stats_panel(ax6)

        fig.suptitle("Phi Grid Validation Dashboard", fontsize=14, fontweight="bold")
        return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize simulation-based phi grids for validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else None,
    )
    parser.add_argument(
        "--map-path",
        required=True,
        help="Path to .map file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Load from cached .npz (faster, no intermediate diagnostics)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to PNG file",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="Use Python simulator instead of native",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.exists(args.map_path):
        logger.error("Map file not found: %s", args.map_path)
        return 1

    use_native = not args.no_native

    if args.cache_dir:
        logger.info("Loading from cache: %s", args.cache_dir)
        diag = _load_from_cache(args.map_path, args.cache_dir)
        if diag is None:
            logger.error(
                "Cache miss. Run precompute_cli first or omit --cache-dir to build inline."
            )
            return 1
    else:
        logger.info("Building phi grid inline (full pipeline)...")
        diag = _build_and_collect_diagnostics(args.map_path, use_native=use_native)

    viz = PhiGridVisualizer(diag)
    fig = viz.generate_dashboard()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", args.save)
    else:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
