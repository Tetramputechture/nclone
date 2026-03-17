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
        (self.graph, self.d_switch, self.d_exit,
         self.arc_keys_switch, self.arc_vals_switch,
         self.arc_keys_exit, self.arc_vals_exit,
         self.meta) = _load_surface_pbrs(data_path)

        sp = self.graph.surface_points
        self.xs = np.array([p.x for p in sp])
        self.ys = np.array([p.y for p in sp])
        from .surface_extraction import SURFACE_FLOOR
        self.stypes = np.array([p.surface_type for p in sp])
        self.is_edge = np.array([p.is_edge for p in sp])
        self._SURFACE_FLOOR = SURFACE_FLOOR

        self.tile_dic = None
        self.switch_pos: Optional[Tuple[float, float]] = None
        self.exit_pos: Optional[Tuple[float, float]] = None
        self.mine_positions: List[Tuple[float, float]] = []

        if map_path and os.path.exists(map_path):
            self.tile_dic = _load_tile_dict(map_path)
            sw, ex, mines = _load_entity_positions(map_path)
            self.switch_pos = sw
            self.exit_pos = ex
            self.mine_positions = mines

        # Also load mine positions from the .npz when available (stored since
        # the runtime-masking refactor), so they show up even without --map-path.
        if not self.mine_positions and "mine_positions_x" in data and len(data["mine_positions_x"]) > 0:
            mine_xs = data["mine_positions_x"]
            mine_ys = data["mine_positions_y"]
            self.mine_positions = [
                (float(mine_xs[i]), float(mine_ys[i]))
                for i in range(len(mine_xs))
            ]

        n = len(sp)
        n_edges = len(self.graph.edges_src)
        n_arc_sw = len(self.arc_vals_switch)
        n_arc_ex = len(self.arc_vals_exit)
        print(f"  Surface points: {n:,}  |  Graph nodes: {self.graph.n_nodes:,}  |  Edges: {n_edges:,}")
        print(f"  Arc cache switch: {n_arc_sw:,} cells  |  Arc cache exit: {n_arc_ex:,} cells")

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
        # Always overlay entity positions when available
        from .level_renderer import mark_entities
        mark_entities(
            ax,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions if self.mine_positions else None,
        )

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
        ax.legend(loc="upper right", fontsize=8)
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
        ax.scatter(self.xs, self.ys, c="cyan", s=3, zorder=3, alpha=0.6, label="surface pts")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

    def visualize_arc_heatmap(
        self,
        phase: str = "switch",
        output_path: Optional[str] = None,
    ) -> None:
        """Two-panel arc cache visualization.

        Left panel: arc cache cells coloured by minimum cost-to-go across all
        velocity bins at that (x, y) position.  Shows how well the arc cache
        covers airborne space and what costs it assigns.

        Right panel: number of (vx, vy) entries per spatial cell (coverage
        density).  Sparse cells are more likely to fall back to the
        pessimistic heuristic.
        """
        arc_keys = self.arc_keys_switch if phase == "switch" else self.arc_keys_exit
        arc_vals = self.arc_vals_switch if phase == "switch" else self.arc_vals_exit
        d_max = float(self.meta.get(
            "d_max_switch" if phase == "switch" else "d_max_exit", 1.0))

        if len(arc_vals) == 0:
            print(f"No arc cache data for phase '{phase}'.")
            return

        POSITION_BIN_SIZE = 12.0

        # Aggregate per (x_bin, y_bin): minimum cost and entry count
        spatial: dict = {}
        for i in range(len(arc_vals)):
            xb, yb = int(arc_keys[i, 0]), int(arc_keys[i, 1])
            val = float(arc_vals[i])
            if (xb, yb) not in spatial:
                spatial[(xb, yb)] = [val, 1]
            else:
                if val < spatial[(xb, yb)][0]:
                    spatial[(xb, yb)][0] = val
                spatial[(xb, yb)][1] += 1

        coords = np.array(list(spatial.keys()), dtype=np.float32)
        min_costs = np.array([v[0] for v in spatial.values()], dtype=np.float32)
        counts = np.array([v[1] for v in spatial.values()], dtype=np.float32)

        # Convert bin coordinates to pixel centres
        px_x = (coords[:, 0] + 0.5) * POSITION_BIN_SIZE
        px_y = (coords[:, 1] + 0.5) * POSITION_BIN_SIZE

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
        phase_label = "switch" if phase == "switch" else "exit"

        # --- Left: minimum cost heatmap ---
        self._setup_axis(ax1, f"Arc Cache — Min Cost to {phase_label.title()} (phase: {phase})")
        finite_mask = np.isfinite(min_costs)
        if finite_mask.any():
            finite_costs = min_costs[finite_mask]
            norm = mcolors.Normalize(
                vmin=float(np.percentile(finite_costs, 2)),
                vmax=float(np.percentile(finite_costs, 98)),
            )
            sc1 = ax1.scatter(
                px_x[finite_mask], px_y[finite_mask],
                c=min_costs[finite_mask],
                cmap="viridis_r", norm=norm, s=10, marker="s", alpha=0.7, zorder=3,
            )
            plt.colorbar(sc1, ax=ax1, label="Min cost-to-go (frames)")

        ax1.set_title(
            f"Arc Cache Min Cost — {phase_label} phase\n"
            f"({len(spatial):,} unique spatial cells, {len(arc_vals):,} total entries)",
            fontsize=10,
        )
        ax1.legend(loc="upper right", fontsize=7)

        # --- Right: coverage density ---
        self._setup_axis(ax2, f"Arc Cache — Coverage Density (phase: {phase})")
        norm2 = mcolors.LogNorm(vmin=max(1, float(counts.min())),
                                vmax=float(counts.max()))
        sc2 = ax2.scatter(
            px_x, px_y, c=counts,
            cmap="plasma", norm=norm2, s=10, marker="s", alpha=0.7, zorder=3,
        )
        plt.colorbar(sc2, ax=ax2, label="Velocity entries per spatial cell")
        ax2.set_title(
            f"Arc Cache Coverage Density — {phase_label} phase\n"
            f"(higher = more velocity contexts covered)", fontsize=10,
        )
        ax2.legend(loc="upper right", fontsize=7)

        plt.suptitle(
            f"Arc Cache Airborne Potential  —  {phase_label} phase  "
            f"|  d_max = {d_max:.0f} frames",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        _save_or_show(fig, output_path)

    def visualize_combined_potential(
        self,
        phase: str = "switch",
        output_path: Optional[str] = None,
    ) -> None:
        """Single-panel combined PBRS potential plot.

        Shows what the agent actually experiences:
        - Surface nodes as small circles coloured by normalised d* (-d*/d_max)
        - Arc cache spatial cells as squares coloured by normalised arc cost
          (-min_cost/d_max)
        Both use the same colormap and scale so potentials are directly
        comparable across contact regimes.
        """
        arc_keys = self.arc_keys_switch if phase == "switch" else self.arc_keys_exit
        arc_vals = self.arc_vals_switch if phase == "switch" else self.arc_vals_exit
        d_star = self.d_switch if phase == "switch" else self.d_exit
        d_max = float(self.meta.get(
            "d_max_switch" if phase == "switch" else "d_max_exit", 1.0))
        if d_max <= 0:
            d_max = 1.0

        POSITION_BIN_SIZE = 12.0
        max_bins = self.graph.max_vel_bins

        # Surface potential: use velocity bin 0 (zero velocity) per surface point
        surf_costs = np.array([
            float(d_star[i * max_bins]) if i * max_bins < len(d_star) else np.nan
            for i in range(len(self.graph.surface_points))
        ])
        finite_surf = np.isfinite(surf_costs)
        surf_phi = np.where(finite_surf, -surf_costs / d_max, np.nan)
        surf_phi = np.clip(surf_phi, -1.0, 1.0)

        # Arc cache potential: project to 2D min per spatial cell
        arc_phi_map: dict = {}
        for i in range(len(arc_vals)):
            xb, yb = int(arc_keys[i, 0]), int(arc_keys[i, 1])
            val = float(arc_vals[i])
            if (xb, yb) not in arc_phi_map or val < arc_phi_map[(xb, yb)]:
                arc_phi_map[(xb, yb)] = val

        arc_coords = np.array(list(arc_phi_map.keys()), dtype=np.float32) if arc_phi_map else np.empty((0, 2))
        arc_costs_arr = np.array(list(arc_phi_map.values()), dtype=np.float32) if arc_phi_map else np.empty(0)
        arc_px_x = (arc_coords[:, 0] + 0.5) * POSITION_BIN_SIZE if len(arc_coords) else np.empty(0)
        arc_px_y = (arc_coords[:, 1] + 0.5) * POSITION_BIN_SIZE if len(arc_coords) else np.empty(0)
        arc_phi = np.clip(-arc_costs_arr / d_max, -1.0, 1.0) if len(arc_costs_arr) else np.empty(0)

        # Unified colormap range
        all_phi = np.concatenate([surf_phi[finite_surf], arc_phi[np.isfinite(arc_phi)]])
        if len(all_phi) == 0:
            print("No finite potential values found.")
            return
        vmin = float(np.percentile(all_phi, 2))
        vmax = float(np.percentile(all_phi, 98))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(16, 9))
        self._setup_axis(ax, "")

        # Arc cache cells (rendered first, underneath surface points)
        if len(arc_phi) > 0:
            sc_arc = ax.scatter(
                arc_px_x, arc_px_y, c=arc_phi,
                cmap="RdYlGn", norm=norm, s=16, marker="s", alpha=0.55, zorder=3,
                label=f"Airborne (arc cache, {len(arc_phi):,} cells)",
            )

        # Surface points on top
        if finite_surf.any():
            sc_surf = ax.scatter(
                self.xs[finite_surf], self.ys[finite_surf],
                c=surf_phi[finite_surf],
                cmap="RdYlGn", norm=norm, s=18, marker="o", alpha=0.85, zorder=4,
                label=f"Surface (d*, {int(finite_surf.sum()):,} pts)",
            )
            plt.colorbar(sc_surf, ax=ax, label="Normalised potential Φ(s)")

        ax.legend(loc="upper right", fontsize=8)

        phase_label = "switch" if phase == "switch" else "exit"
        ax.set_title(
            f"Combined PBRS Potential — {phase_label} phase  "
            f"(squares = airborne arc cache,  circles = surface d*)\n"
            f"d_max = {d_max:.0f} frames",
            fontsize=11,
        )
        plt.tight_layout()
        _save_or_show(fig, output_path)

    def visualize_mine_masking(
        self,
        phase: str = "switch",
        output_path: Optional[str] = None,
    ) -> None:
        """Visualize mine-proximity masking on the surface graph.

        Shows:
        - Surface points coloured by d* (same as heatmap)
        - Red X markers for surface nodes that fall inside any mine's kill
          radius (NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED = 14 px)
        - Translucent red circles at each mine position with the kill radius
        - Graph edges in two layers: unmasked (blue/plasma by weight) and
          masked — either source or destination is blocked (red dashed)
        - A statistics text box showing blocked node / edge counts

        Args:
            phase: ``"switch"`` or ``"exit"`` — which d* array to use for
                colouring unblocked surface points.
            output_path: Save path.  If None, calls ``plt.show()``.
        """
        from nclone.value_function.surface_graph import (
            find_mine_blocked_nodes,
            node_id_to_parts,
        )
        from nclone.constants.physics_constants import (
            NINJA_RADIUS,
            TOGGLE_MINE_RADIUS_TOGGLED,
        )

        d_star = self.d_switch if phase == "switch" else self.d_exit
        d_max = float(self.meta.get(
            "d_max_switch" if phase == "switch" else "d_max_exit", 1.0))
        if d_max <= 0:
            d_max = 1.0

        kill_radius = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px
        mine_pos_list = [(mx, my) for mx, my in self.mine_positions]
        blocked = find_mine_blocked_nodes(
            self.graph, mine_pos_list, kill_radius=kill_radius)

        sp = self.graph.surface_points
        max_bins = self.graph.max_vel_bins

        # Per-point: is ANY velocity bin for this surface point blocked?
        n_pts = len(sp)
        pt_blocked = np.zeros(n_pts, dtype=bool)
        for nid in blocked:
            pt_idx, _ = node_id_to_parts(nid, max_bins)
            if pt_idx < n_pts:
                pt_blocked[pt_idx] = True

        costs = np.array([
            float(d_star[i * max_bins]) if i * max_bins < len(d_star) else np.inf
            for i in range(n_pts)
        ])

        fig, ax = plt.subplots(figsize=(16, 9))
        phase_label = "switch" if phase == "switch" else "exit"
        self._setup_axis(ax, "")

        # Unblocked surface points coloured by d*
        unblocked_mask = ~pt_blocked
        if unblocked_mask.any():
            finite_u = costs[unblocked_mask & np.isfinite(costs)]
            norm = mcolors.Normalize(
                vmin=float(np.min(finite_u)) if len(finite_u) else 0,
                vmax=float(np.max(finite_u)) if len(finite_u) else 1,
            )
            sc = ax.scatter(
                self.xs[unblocked_mask], self.ys[unblocked_mask],
                c=costs[unblocked_mask],
                cmap="viridis_r", norm=norm,
                s=8, zorder=3, alpha=0.8,
                label=f"Surface pts ({int(unblocked_mask.sum()):,})",
            )
            plt.colorbar(sc, ax=ax, label="d* (frames)")

        # Blocked surface points — red X
        if pt_blocked.any():
            ax.scatter(
                self.xs[pt_blocked], self.ys[pt_blocked],
                c="red", marker="x", s=30, linewidths=0.8,
                zorder=5, label=f"Blocked pts ({int(pt_blocked.sum()):,})",
            )

        # Mine kill-radius circles
        for mx, my in mine_pos_list:
            circle = plt.Circle(
                (mx, my), kill_radius,
                color="red", fill=True, alpha=0.15, zorder=4,
            )
            ax.add_patch(circle)
            circle_border = plt.Circle(
                (mx, my), kill_radius,
                color="red", fill=False, linewidth=1.0, alpha=0.7, zorder=4,
            )
            ax.add_patch(circle_border)
            ax.plot(mx, my, "r+", markersize=6, zorder=5)

        # Edges: split into unmasked (blue) and masked (red dashed)
        edges_src = self.graph.edges_src
        edges_dst = self.graph.edges_dst
        edges_weight = self.graph.edges_weight
        n_edges = len(edges_src)
        MAX_DISPLAY = 8000
        if n_edges > MAX_DISPLAY:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_edges, MAX_DISPLAY, replace=False)
        else:
            indices = np.arange(n_edges)

        segs_ok, segs_blocked = [], []
        w_ok, w_blocked = [], []
        for idx in indices:
            si = int(edges_src[idx])
            di = int(edges_dst[idx])
            s_pt, _ = node_id_to_parts(si, max_bins)
            d_pt, _ = node_id_to_parts(di, max_bins)
            if s_pt >= n_pts or d_pt >= n_pts:
                continue
            seg = [(sp[s_pt].x, sp[s_pt].y), (sp[d_pt].x, sp[d_pt].y)]
            w = float(edges_weight[idx])
            if si in blocked or di in blocked:
                segs_blocked.append(seg)
                w_blocked.append(w)
            else:
                segs_ok.append(seg)
                w_ok.append(w)

        if segs_ok:
            w_arr = np.array(w_ok)
            lc_ok = LineCollection(
                segs_ok, cmap="plasma",
                norm=mcolors.Normalize(vmin=w_arr.min(), vmax=w_arr.max()),
                linewidths=0.4, alpha=0.3, zorder=2,
                label=f"Unblocked edges ({len(segs_ok):,})",
            )
            lc_ok.set_array(w_arr)
            ax.add_collection(lc_ok)

        if segs_blocked:
            lc_bl = LineCollection(
                segs_blocked, colors="red",
                linewidths=0.6, linestyles="dashed", alpha=0.5, zorder=3,
                label=f"Blocked edges ({len(segs_blocked):,})",
            )
            ax.add_collection(lc_bl)

        n_masked_nodes = int(pt_blocked.sum())
        n_total_nodes = len(blocked)
        n_masked_edges_total = sum(
            1 for i in range(n_edges)
            if int(edges_src[i]) in blocked or int(edges_dst[i]) in blocked
        )
        ax.set_title(
            f"Mine Masking — {phase_label} phase\n"
            f"Blocked nodes: {n_total_nodes:,} ({n_masked_nodes:,} pts)  |  "
            f"Masked edges: {n_masked_edges_total:,}/{n_edges:,}  |  "
            f"Mines: {len(mine_pos_list)}  |  Kill radius: {kill_radius:.1f} px",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        _save_or_show(fig, output_path)

    def visualize_dashboard(self, output_path: Optional[str] = None) -> None:
        """Multi-panel dashboard with statistics and arc cache coverage."""
        POSITION_BIN_SIZE = 12.0

        # Precompute arc cache spatial coverage for switch phase
        arc_coverage_xy: dict = {}
        for i in range(len(self.arc_vals_switch)):
            xb, yb = int(self.arc_keys_switch[i, 0]), int(self.arc_keys_switch[i, 1])
            key = (xb, yb)
            arc_coverage_xy[key] = arc_coverage_xy.get(key, 0) + 1

        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)

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
        ax3.legend(loc="upper right", fontsize=7)

        # Arc cache spatial coverage density (new panel)
        ax4 = fig.add_subplot(gs[0, 3])
        self._setup_axis(ax4, "Arc Cache Coverage (switch phase)")
        if arc_coverage_xy:
            coords = np.array(list(arc_coverage_xy.keys()), dtype=np.float32)
            counts = np.array(list(arc_coverage_xy.values()), dtype=np.float32)
            arc_px_x = (coords[:, 0] + 0.5) * POSITION_BIN_SIZE
            arc_px_y = (coords[:, 1] + 0.5) * POSITION_BIN_SIZE
            norm4 = mcolors.LogNorm(vmin=max(1.0, float(counts.min())),
                                    vmax=float(counts.max()))
            sc4 = ax4.scatter(arc_px_x, arc_px_y, c=counts, cmap="hot_r",
                              norm=norm4, s=8, marker="s", alpha=0.75, zorder=3)
            plt.colorbar(sc4, ax=ax4, label="# velocity entries")
        ax4.set_title(
            f"Arc Cache Coverage\n"
            f"({len(arc_coverage_xy):,} spatial cells, "
            f"{len(self.arc_vals_switch):,} total entries)",
            fontsize=9,
        )
        ax4.legend(loc="upper right", fontsize=7)

        ax5 = fig.add_subplot(gs[1, 0])
        max_bins = self.graph.max_vel_bins
        d_sw = np.array([
            float(self.d_switch[i * max_bins])
            if i * max_bins < len(self.d_switch) else np.inf
            for i in range(len(self.graph.surface_points))
        ])
        finite_sw = d_sw[np.isfinite(d_sw)]
        if len(finite_sw) > 0:
            ax5.hist(finite_sw, bins=50, color="skyblue", edgecolor="black")
        ax5.set_xlabel("d* switch (frames)")
        ax5.set_ylabel("Count")
        ax5.set_title("Switch Cost Distribution")

        ax6 = fig.add_subplot(gs[1, 1])
        if len(self.graph.edges_weight) > 0:
            ax6.hist(self.graph.edges_weight, bins=50, color="salmon", edgecolor="black")
        ax6.set_xlabel("Edge weight (frames)")
        ax6.set_ylabel("Count")
        ax6.set_title("Edge Weight Distribution")

        # Arc cache min-cost distribution
        ax7 = fig.add_subplot(gs[1, 2])
        if len(self.arc_vals_switch) > 0:
            finite_arc = self.arc_vals_switch[np.isfinite(self.arc_vals_switch)]
            if len(finite_arc) > 0:
                ax7.hist(finite_arc, bins=50, color="mediumpurple", edgecolor="black")
        ax7.set_xlabel("Arc cache cost (frames)")
        ax7.set_ylabel("Count")
        ax7.set_title("Arc Cache Cost Distribution (switch)")

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis("off")
        d_max_sw = self.meta.get("d_max_switch", "?")
        d_max_ex = self.meta.get("d_max_exit", "?")
        sw_str = (f"({self.switch_pos[0]:.0f}, {self.switch_pos[1]:.0f})"
                  if self.switch_pos else "unknown")
        ex_str = (f"({self.exit_pos[0]:.0f}, {self.exit_pos[1]:.0f})"
                  if self.exit_pos else "unknown")
        mine_str = (", ".join(f"({m[0]:.0f},{m[1]:.0f})" for m in self.mine_positions)
                    or "none")
        txt = (
            f"STATISTICS\n{'=' * 30}\n\n"
            f"Switch:  {sw_str}\n"
            f"Exit:    {ex_str}\n"
            f"Mines:   {mine_str}\n\n"
            f"Surface points: {len(self.graph.surface_points):,}\n"
            f"  Floor: {int(np.sum(self.stypes == 0)):,}\n"
            f"  Wall:  {int(np.sum(self.stypes == 1)):,}\n"
            f"  Edge:  {int(np.sum(self.is_edge)):,}\n\n"
            f"Graph nodes: {self.graph.n_nodes:,}\n"
            f"Edges:       {len(self.graph.edges_src):,}\n\n"
            f"d_max switch: {d_max_sw}\n"
            f"d_max exit:   {d_max_ex}\n\n"
            f"Arc cache (switch):\n"
            f"  Spatial cells: {len(arc_coverage_xy):,}\n"
            f"  Total entries: {len(self.arc_vals_switch):,}\n"
            f"Arc cache (exit):\n"
            f"  Total entries: {len(self.arc_vals_exit):,}\n"
        )
        ax8.text(0.05, 0.95, txt, transform=ax8.transAxes,
                 fontsize=10, va="top", family="monospace")

        plt.suptitle("Surface-Graph PBRS Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save_or_show(fig, output_path)

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
    """Load surface-graph PBRS data from .npz.

    Returns:
        (graph, d_star_switch, d_star_exit, arc_keys_switch, arc_vals_switch,
         arc_keys_exit, arc_vals_exit, meta)
    """
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

    arc_keys_switch = (data["arc_cache_keys_switch"]
                       if "arc_cache_keys_switch" in data else np.empty((0, 4), dtype=np.int32))
    arc_vals_switch = (data["arc_cache_values_switch"]
                       if "arc_cache_values_switch" in data else np.empty(0, dtype=np.float32))
    arc_keys_exit = (data["arc_cache_keys_exit"]
                     if "arc_cache_keys_exit" in data else np.empty((0, 4), dtype=np.int32))
    arc_vals_exit = (data["arc_cache_values_exit"]
                     if "arc_cache_values_exit" in data else np.empty(0, dtype=np.float32))

    return (graph, d_star_switch, d_star_exit,
            arc_keys_switch, arc_vals_switch,
            arc_keys_exit, arc_vals_exit,
            meta)


def _load_tile_dict(map_path: str) -> Optional[dict]:
    """Try to load tile dictionary from a .map file."""
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=map_path)
        return pw.tile_dic
    except Exception as e:
        print(f"Warning: could not load tiles from {map_path}: {e}")
        return None


def _load_entity_positions(
    map_path: str,
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    List[Tuple[float, float]],
]:
    """Load switch, exit, and mine positions from a map file.

    Returns:
        (switch_pos, exit_pos, mine_positions) — positions as (x, y) pixel
        tuples.  Any that cannot be found are returned as None / empty list.
    """
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=map_path)
        switch_pos = pw.get_goal_position("switch")
        exit_pos = pw.get_goal_position("exit")
        mine_positions: List[Tuple[float, float]] = []
        for e in pw.entity_dic.get(21, []):
            mine_positions.append((e.xpos, e.ypos))
        for e in pw.entity_dic.get(1, []):
            mine_positions.append((e.xpos, e.ypos))
        return switch_pos, exit_pos, mine_positions
    except Exception as e:
        print(f"Warning: could not load entity positions from {map_path}: {e}")
        return None, None, []



def _save_or_show(fig, output_path: Optional[str]) -> None:
    """Save figure to file or display it."""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
