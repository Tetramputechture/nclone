"""Value-landscape heatmap generators for surface-graph and neural PBRS.

``SurfaceGraphVisualizer``: surface-graph PBRS visualizer (legacy approach).
``NeuralValueLandscape``: IQL V-network spatial heatmap over the level grid.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")  # non-interactive backend for headless/server
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from .level_renderer import render_tile_grid_to_axis, mark_entities

# ---------------------------------------------------------------------------
# NeuralValueLandscape
# ---------------------------------------------------------------------------

# Grid sample resolution in game-units (pixels). 12px ≈ half a tile.
_SWEEP_STEP_PX = 12

# Tile geometry from game constants (physics world = 44×25 tiles including 1-tile border;
# tile_grid stores only the inner 42×23 tiles: tile_grid[i,j] = game tile (i+1, j+1)).
from nclone.constants import (
    LEVEL_WIDTH_PX as _LEVEL_W_PX, LEVEL_HEIGHT_PX as _LEVEL_H_PX,
    TILE_PIXEL_SIZE as _TILE_SIZE_PX, MAP_TILE_WIDTH as _GRID_W, MAP_TILE_HEIGHT as _GRID_H,
)


class NeuralValueLandscape:
    """Renders spatial V(s) heatmaps from a trained IQL V-network.

    Sweeps (x, y) positions at ``step_px`` resolution across the level, calls
    ``NeuralValuePBRS.get_potential()`` at each point, and produces a heatmap.
    Solid tiles (type 1) are masked out so the overlay does not cover geometry.

    Usage::

        landscape = NeuralValueLandscape("v_network.pt", device="cpu")
        landscape.set_level(tile_grid, switch_pos, exit_pos, mine_positions)
        landscape.render_heatmap(phase=0, output_path="phase0.png")
        landscape.render_heatmap(phase=1, output_path="phase1.png")
        landscape.render_dual(output_path="dual.png")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        step_px: int = _SWEEP_STEP_PX,
    ):
        """Load frozen IQL V-network for landscape generation.

        Args:
            checkpoint_path: Path to exported v_network.pt.
            device: Torch device for inference.
            step_px: Pixel stride for the (x, y) sweep grid.
        """
        from .two_phase_pbrs import create_neural_pbrs
        self._pbrs = create_neural_pbrs(checkpoint_path, device=device)
        self._step = step_px
        self._tile_grid: Optional[np.ndarray] = None
        self._switch_pos: Optional[Tuple[float, float]] = None
        self._exit_pos: Optional[Tuple[float, float]] = None
        self._mine_positions: List[Tuple[float, float]] = []
        self._solid_mask: Optional[np.ndarray] = None

    def set_level(
        self,
        tile_grid: np.ndarray,
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float],
        mine_positions: Optional[List[Tuple[float, float, int]]] = None,
    ) -> None:
        """Cache level data and precompute the CNN tile encoding.

        Args:
            tile_grid: (42, 23) integer array of tile type IDs.
            switch_pos: (x, y) pixel position of exit switch.
            exit_pos: (x, y) pixel position of exit door.
            mine_positions: List of (x, y, entity_type) for toggle mines.
        """
        self._tile_grid = tile_grid
        self._switch_pos = switch_pos
        self._exit_pos = exit_pos
        self._mine_positions = [(mx, my) for mx, my, _ in (mine_positions or [])]

        # Precompute CNN encoding (cached; only MLP runs per sweep point)
        self._pbrs.set_level(
            tile_grid=tile_grid,
            switch_pos=switch_pos,
            exit_pos=exit_pos,
            mine_positions=mine_positions or [],
        )

        # Build solid mask in pixel space for overlay masking
        self._solid_mask = (tile_grid == 1)  # (42, 23) bool

    def _compute_grid(self, switch_activated: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sweep the level and return xs, ys, values arrays.

        Args:
            switch_activated: Phase flag (False = phase 0, True = phase 1).

        Returns:
            xs, ys, values as 1D arrays over all valid (non-solid) sample points.
        """
        xs_list, ys_list, vals_list = [], [], []

        # Sweep the inner playable area (skip 1-tile solid border on each edge).
        # tile_grid[i,j] = game tile (i+1, j+1); physics col = x//tile_size.
        # Correct tile_grid index: col_idx = physics_col - 1.
        x_samples = np.arange(_TILE_SIZE_PX + self._step, _LEVEL_W_PX - _TILE_SIZE_PX, self._step)
        y_samples = np.arange(_TILE_SIZE_PX + self._step, _LEVEL_H_PX - _TILE_SIZE_PX, self._step)

        for px in x_samples:
            col_idx = int(px) // _TILE_SIZE_PX - 1  # tile_grid col index (no border offset)
            for py in y_samples:
                row_idx = int(py) // _TILE_SIZE_PX - 1  # tile_grid row index
                if (0 <= col_idx < _GRID_W and 0 <= row_idx < _GRID_H
                        and self._solid_mask is not None
                        and self._solid_mask[col_idx, row_idx]):
                    continue

                state = {
                    "player_x": float(px),
                    "player_y": float(py),
                    "player_xspeed": 0.0,
                    "player_yspeed": 0.0,
                    "on_ground": True,
                    "wall_sliding": False,
                    "jump_duration_remaining": 0.0,
                    "switch_activated": switch_activated,
                    "floor_normal_x": 0.0,
                    "floor_normal_y": -1.0,
                    "wall_normal": 0.0,
                }
                val = self._pbrs.get_potential(state)
                xs_list.append(px)
                ys_list.append(py)
                vals_list.append(val)

        return np.array(xs_list), np.array(ys_list), np.array(vals_list)

    def render_heatmap(
        self,
        phase: int = 0,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (14, 8),
    ) -> None:
        """Render a single-phase V(s) heatmap.

        Args:
            phase: 0 = before switch collected, 1 = after switch collected.
            output_path: Save path (.png/.pdf). If None, calls plt.show().
            title: Figure title. Defaults to "V(s) Phase {phase}".
            figsize: Matplotlib figure size.
        """
        assert self._tile_grid is not None, "call set_level() first"
        switch_activated = bool(phase)
        xs, ys, vals = self._compute_grid(switch_activated)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, _LEVEL_W_PX)
        ax.set_ylim(_LEVEL_H_PX, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_title(title or f"V(s) Phase {phase} ({'switch collected' if phase else 'no switch'})")

        # Tile geometry background
        render_tile_grid_to_axis(ax, self._tile_grid, tile_color="#333333", alpha=0.5)

        # Value scatter
        if len(vals) > 0:
            finite_vals = vals[np.isfinite(vals)]
            vmin = float(np.percentile(finite_vals, 2)) if len(finite_vals) > 0 else -1.0
            vmax = float(np.percentile(finite_vals, 98)) if len(finite_vals) > 0 else 0.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(xs, ys, c=vals, cmap="RdYlGn", norm=norm,
                            s=18, marker="s", alpha=0.75, zorder=4)
            plt.colorbar(sc, ax=ax, label="V(s)")

        # Entity markers
        mark_entities(
            ax,
            switch_pos=self._switch_pos if phase == 0 else None,
            exit_pos=self._exit_pos if phase == 1 else None,
            mine_positions=self._mine_positions if self._mine_positions else None,
        )
        if self._mine_positions or self._switch_pos or self._exit_pos:
            ax.legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    def render_dual(
        self,
        output_path: Optional[str] = None,
        level_label: str = "",
        figsize: Tuple[float, float] = (20, 8),
    ) -> None:
        """Render side-by-side phase 0 and phase 1 heatmaps.

        Args:
            output_path: Save path. If None, calls plt.show().
            level_label: Extra label appended to the suptitle.
            figsize: Matplotlib figure size.
        """
        assert self._tile_grid is not None, "call set_level() first"

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, switch_activated, phase_label in zip(
            axes,
            [False, True],
            ["Phase 0: switch not collected", "Phase 1: switch collected"],
        ):
            xs, ys, vals = self._compute_grid(switch_activated)

            ax.set_xlim(0, _LEVEL_W_PX)
            ax.set_ylim(_LEVEL_H_PX, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_title(phase_label)

            render_tile_grid_to_axis(ax, self._tile_grid, tile_color="#333333", alpha=0.5)

            if len(vals) > 0:
                finite_vals = vals[np.isfinite(vals)]
                vmin = float(np.percentile(finite_vals, 2)) if len(finite_vals) > 0 else -1.0
                vmax = float(np.percentile(finite_vals, 98)) if len(finite_vals) > 0 else 0.0
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                sc = ax.scatter(xs, ys, c=vals, cmap="RdYlGn", norm=norm,
                                s=18, marker="s", alpha=0.75, zorder=4)
                plt.colorbar(sc, ax=ax, label="V(s)")

            mark_entities(
                ax,
                switch_pos=self._switch_pos if not switch_activated else None,
                exit_pos=self._exit_pos if switch_activated else None,
                mine_positions=self._mine_positions if self._mine_positions else None,
            )
            if any([self._mine_positions, self._switch_pos, self._exit_pos]):
                ax.legend(loc="upper right", fontsize=7)

        suptitle = f"V(s) landscape — {level_label}" if level_label else "V(s) landscape"
        plt.suptitle(suptitle, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# SurfaceGraphVisualizer (surface-graph PBRS, legacy approach)
# ---------------------------------------------------------------------------

class SurfaceGraphVisualizer:
    """Visualizer for surface-graph PBRS data (.npz files)."""

    def __init__(self, data_path: str, map_path: Optional[str] = None):
        print(f"Loading surface graph from: {data_path}")
        self.graph, self.d_switch, self.d_exit, self.meta = (
            _load_surface_pbrs(data_path)
        )

        sp = self.graph.surface_points
        self.xs = np.array([p.x for p in sp])
        self.ys = np.array([p.y for p in sp])
        from .surface_extraction import SURFACE_FLOOR
        self.stypes = np.array([p.surface_type for p in sp])
        self.is_edge = np.array([p.is_edge for p in sp])
        self._SURFACE_FLOOR = SURFACE_FLOOR

        self.tile_dic = None
        if map_path and os.path.exists(map_path):
            self.tile_dic = _load_tile_dict(map_path)

        n = len(sp)
        n_edges = len(self.graph.edges_src)
        print(f"  Surface points: {n:,}  |  Graph nodes: {self.graph.n_nodes:,}  |  Edges: {n_edges:,}")

    def _setup_axis(self, ax, title: str = "") -> None:
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        if title:
            ax.set_title(title)
        if self.tile_dic is not None:
            from .level_renderer import render_tiles_to_axis
            render_tiles_to_axis(ax, self.tile_dic, alpha=0.4)

    def visualize_heatmap(
        self, phase: str = "switch", output_path: Optional[str] = None,
    ) -> None:
        """Show surface points coloured by d_star."""
        d_star = self.d_switch if phase == "switch" else self.d_exit
        max_bins = self.graph.max_vel_bins
        costs = np.array([
            float(d_star[i * max_bins]) if i * max_bins < len(d_star) else np.inf
            for i in range(len(self.graph.surface_points))
        ])
        finite = costs[np.isfinite(costs)]
        if len(finite) == 0:
            print("No finite d* values found.")
            return

        fig, ax = plt.subplots(figsize=(14, 9))
        self._setup_axis(ax, f"Surface d* ({phase})")
        norm = mcolors.Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
        c = np.where(np.isfinite(costs), costs, np.max(finite))
        sc = ax.scatter(self.xs, self.ys, c=c, cmap="viridis_r", norm=norm, s=8, zorder=3)
        plt.colorbar(sc, ax=ax, label="d* (frames)")
        mask_e = self.is_edge
        ax.scatter(self.xs[mask_e], self.ys[mask_e],
                   facecolors="none", edgecolors="red",
                   s=20, linewidths=0.5, zorder=4, label="edge points")
        ax.legend(loc="upper right")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

    def visualize_edges(
        self, max_edges: int = 5000, output_path: Optional[str] = None,
    ) -> None:
        """Overlay graph edges on surface points."""
        from nclone.value_function.surface_graph import node_id_to_parts
        fig, ax = plt.subplots(figsize=(14, 9))
        self._setup_axis(ax, "Surface Graph Edges")
        sp = self.graph.surface_points
        max_bins = self.graph.max_vel_bins
        n_show = min(max_edges, len(self.graph.edges_src))
        indices = (
            np.random.choice(len(self.graph.edges_src), n_show, replace=False)
            if n_show < len(self.graph.edges_src)
            else np.arange(n_show)
        )
        segments, weights = [], []
        for idx in indices:
            si = int(self.graph.edges_src[idx])
            di = int(self.graph.edges_dst[idx])
            s_pt, _ = node_id_to_parts(si, max_bins)
            d_pt, _ = node_id_to_parts(di, max_bins)
            if s_pt >= len(sp) or d_pt >= len(sp):
                continue
            segments.append([(sp[s_pt].x, sp[s_pt].y), (sp[d_pt].x, sp[d_pt].y)])
            weights.append(float(self.graph.edges_weight[idx]))
        if segments:
            w_arr = np.array(weights)
            norm = mcolors.Normalize(vmin=w_arr.min(), vmax=w_arr.max())
            lc = LineCollection(segments, cmap="plasma", norm=norm,
                                linewidths=0.5, alpha=0.4, zorder=2)
            lc.set_array(w_arr)
            ax.add_collection(lc)
            plt.colorbar(lc, ax=ax, label="Edge weight (frames)")
        ax.scatter(self.xs, self.ys, c="cyan", s=3, zorder=3, alpha=0.6)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

    def visualize_dashboard(self, output_path: Optional[str] = None) -> None:
        """Multi-panel dashboard with statistics."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_dstar_scatter(ax1, self.d_switch, "d* switch")
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_dstar_scatter(ax2, self.d_exit, "d* exit")
        ax3 = fig.add_subplot(gs[0, 2])
        self._setup_axis(ax3, "Surface Types")
        colors = np.where(self.stypes == self._SURFACE_FLOOR, "green", "blue")
        ax3.scatter(self.xs, self.ys, c=colors, s=5, alpha=0.6, zorder=3)
        ax3.scatter([], [], c="green", label="Floor")
        ax3.scatter([], [], c="blue", label="Wall")
        ax3.legend()
        ax4 = fig.add_subplot(gs[1, 0])
        max_bins = self.graph.max_vel_bins
        d_sw = np.array([
            float(self.d_switch[i * max_bins])
            if i * max_bins < len(self.d_switch) else np.inf
            for i in range(len(self.graph.surface_points))
        ])
        finite_sw = d_sw[np.isfinite(d_sw)]
        if len(finite_sw) > 0:
            ax4.hist(finite_sw, bins=50, color="skyblue", edgecolor="black")
        ax4.set_xlabel("d* switch (frames)")
        ax4.set_ylabel("Count")
        ax4.set_title("Switch Cost Distribution")
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self.graph.edges_weight) > 0:
            ax5.hist(self.graph.edges_weight, bins=50, color="salmon", edgecolor="black")
        ax5.set_xlabel("Edge weight (frames)")
        ax5.set_ylabel("Count")
        ax5.set_title("Edge Weight Distribution")
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        d_max_sw = self.meta.get("d_max_switch", "?")
        d_max_ex = self.meta.get("d_max_exit", "?")
        txt = (
            f"STATISTICS\n{'=' * 30}\n\n"
            f"Surface points: {len(self.graph.surface_points):,}\n"
            f"  Floor: {int(np.sum(self.stypes == 0)):,}\n"
            f"  Wall:  {int(np.sum(self.stypes == 1)):,}\n"
            f"  Edge:  {int(np.sum(self.is_edge)):,}\n\n"
            f"Graph nodes: {self.graph.n_nodes:,}\n"
            f"Edges:       {len(self.graph.edges_src):,}\n\n"
            f"d_max switch: {d_max_sw}\n"
            f"d_max exit:   {d_max_ex}\n"
        )
        ax6.text(0.05, 0.95, txt, transform=ax6.transAxes,
                 fontsize=10, va="top", family="monospace")
        plt.suptitle("Surface-Graph PBRS Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

    def _draw_dstar_scatter(self, ax, d_star, title: str) -> None:
        self._setup_axis(ax, title)
        max_bins = self.graph.max_vel_bins
        costs = np.array([
            float(d_star[i * max_bins]) if i * max_bins < len(d_star) else np.inf
            for i in range(len(self.graph.surface_points))
        ])
        finite = costs[np.isfinite(costs)]
        if len(finite) == 0:
            return
        c = np.where(np.isfinite(costs), costs, np.max(finite))
        norm = mcolors.Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
        sc = ax.scatter(self.xs, self.ys, c=c, cmap="viridis_r", norm=norm, s=6, zorder=3)
        plt.colorbar(sc, ax=ax, label="d* (frames)")


# ---------------------------------------------------------------------------
def _load_surface_pbrs(path: str):
    """Load surface-graph PBRS data from .npz."""
    from .surface_extraction import SurfacePoint
    from .surface_graph import build_graph
    data = np.load(path, allow_pickle=True)
    n = len(data["surface_points_x"])
    surface_points = [
        SurfacePoint(
            x=float(data["surface_points_x"][i]),
            y=float(data["surface_points_y"][i]),
            normal_x=float(data["surface_normals_x"][i]),
            normal_y=float(data["surface_normals_y"][i]),
            surface_type=int(data["surface_types"][i]),
            segment_id=int(data["surface_segment_ids"][i]),
            is_edge=bool(data["surface_is_edge"][i]),
        )
        for i in range(n)
    ]
    graph = build_graph(surface_points)
    if "edges_src" in data:
        graph.edges_src = data["edges_src"]
        graph.edges_dst = data["edges_dst"]
        graph.edges_weight = data["edges_weight"]
    d_star_switch = data["d_star_switch"].astype(np.float32)
    d_star_exit = data["d_star_exit"].astype(np.float32)
    meta = data["metadata"].item() if "metadata" in data else {}
    return graph, d_star_switch, d_star_exit, meta


def _load_tile_dict(map_path: str) -> Optional[dict]:
    """Try to load tile dictionary from a .map file."""
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=map_path)
        return pw.tile_dic
    except Exception as e:
        print(f"Warning: could not load tiles from {map_path}: {e}")
        return None
