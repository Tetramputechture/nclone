#!/usr/bin/env python3
"""Visualization tool for precomputed surface-graph PBRS data.

Provides multiple views to inspect and validate the precomputed surface
graph: surface points coloured by d*, edges, arc trajectories, and
potential gradient fields overlaid on level geometry.

Usage:
    # Surface point heatmap with d_star colouring
    python -m nclone.value_function.visualize_value_function \\
        --data value_functions/rampjump_surface_pbrs.npz --mode heatmap

    # Edges / arc overlay
    python -m nclone.value_function.visualize_value_function \\
        --data value_functions/rampjump_surface_pbrs.npz --mode edges

    # Dashboard with statistics
    python -m nclone.value_function.visualize_value_function \\
        --data value_functions/rampjump_surface_pbrs.npz --mode dashboard
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.value_function.surface_extraction import (
    SurfacePoint, SURFACE_FLOOR,
)
from nclone.value_function.surface_graph import (
    build_graph, node_id_to_parts,
)

# ---------------------------------------------------------------------------
# Tile rendering (matches npp_rl.rendering.matplotlib_tile_renderer)
# ---------------------------------------------------------------------------

def _group_tiles_by_type(
    tile_dic: Dict[Tuple[int, int], int],
) -> Dict[int, list]:
    """Group tiles by type for efficient batch rendering."""
    tile_groups: Dict[int, list] = {}
    for coords, tile_val in tile_dic.items():
        if tile_val != 0 and tile_val <= 33:
            if tile_val not in tile_groups:
                tile_groups[tile_val] = []
            tile_groups[tile_val].append(coords)
    return tile_groups


def _draw_complex_tile(
    ax,
    tile_type: int,
    x: int,
    y: int,
    tile_size: float,
    tile_color: str,
    alpha: float,
) -> None:
    """Draw a complex tile shape (slopes, curves, etc.)."""
    base_x = x * tile_size
    base_y = y * tile_size

    if tile_type < 10:
        # Triangular tiles (types 6-9) - 45 degree slopes
        dx1 = 0
        dy1 = tile_size if tile_type == 8 else 0
        dx2 = 0 if tile_type == 9 else tile_size
        dy2 = tile_size if tile_type == 9 else 0
        dx3 = 0 if tile_type == 6 else tile_size
        dy3 = tile_size

        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)

    elif tile_type < 14:
        # Quarter circle tiles (types 10-13) - convex corners
        dx = tile_size if (tile_type == 11 or tile_type == 12) else 0
        dy = tile_size if (tile_type == 12 or tile_type == 13) else 0

        theta1 = 90 * (tile_type - 10)
        theta2 = 90 * (tile_type - 9)

        wedge = mpatches.Wedge(
            (base_x + dx, base_y + dy),
            tile_size,
            theta1,
            theta2,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(wedge)
        wedge.set_zorder(0)

    elif tile_type < 18:
        # Inverted quarter circle tiles (types 14-17) - concave corners
        dx1 = tile_size if (tile_type == 15 or tile_type == 16) else 0
        dy1 = tile_size if (tile_type == 16 or tile_type == 17) else 0
        dx2 = tile_size if (tile_type == 14 or tile_type == 17) else 0
        dy2 = tile_size if (tile_type == 14 or tile_type == 15) else 0

        theta1 = 180 + 90 * (tile_type - 10)
        theta2 = 180 + 90 * (tile_type - 9)

        start_x = base_x + dx1
        start_y = base_y + dy1
        center_x = base_x + dx2
        center_y = base_y + dy2

        angles = np.linspace(np.radians(theta1), np.radians(theta2), 20)
        arc_x = center_x + tile_size * np.cos(angles)
        arc_y = center_y + tile_size * np.sin(angles)

        vertices = [(start_x, start_y)]
        for ax_pt, ay_pt in zip(arc_x, arc_y):
            vertices.append((ax_pt, ay_pt))

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)

    elif tile_type < 22:
        # Sloped triangular tiles (types 18-21) - mild slopes
        dx1 = 0
        dy1 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx2 = tile_size
        dy2 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx3 = tile_size if (tile_type == 19 or tile_type == 20) else 0
        dy3 = tile_size / 2

        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)

    elif tile_type < 26:
        # Quadrilateral tiles (types 22-25) - raised mild slopes
        dx1 = 0
        dy1 = tile_size / 2 if (tile_type == 23 or tile_type == 24) else 0
        dx2 = 0 if tile_type == 23 else tile_size
        dy2 = tile_size / 2 if tile_type == 25 else 0
        dx3 = tile_size
        dy3 = (tile_size / 2 if tile_type == 22 else 0) if tile_type < 24 else tile_size
        dx4 = tile_size if tile_type == 23 else 0
        dy4 = tile_size

        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
            (base_x + dx4, base_y + dy4),
        ]

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)

    elif tile_type < 30:
        # Triangular tiles with midpoint (types 26-29) - steep slopes
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 28 or tile_type == 29) else 0
        dx2 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy2 = 0
        dx3 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy3 = tile_size

        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)

    elif tile_type <= 33:
        # Complex quadrilateral tiles (types 30-33) - raised steep slopes
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 30 or tile_type == 31) else 0
        dx2 = tile_size if (tile_type == 31 or tile_type == 33) else 0
        dy2 = tile_size
        dx3 = tile_size if (tile_type == 31 or tile_type == 32) else 0
        dy3 = tile_size if (tile_type == 32 or tile_type == 33) else 0
        dx4 = tile_size if (tile_type == 30 or tile_type == 32) else 0
        dy4 = 0

        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
            (base_x + dx4, base_y + dy4),
        ]

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(0)


def render_tiles_to_axis(
    ax,
    tile_dic: Dict[Tuple[int, int], int],
    tile_size: float = 24.0,
    tile_color: str = "#404040",
    alpha: float = 0.8,
) -> None:
    """Render level tiles to a matplotlib axis.

    Uses the same vertex geometry as npp_rl.rendering.matplotlib_tile_renderer
    for correct slope and curve orientation.
    """
    tile_groups = _group_tiles_by_type(tile_dic)

    for tile_type, coords_list in tile_groups.items():
        if tile_type == 1:
            for x, y in coords_list:
                rect = mpatches.Rectangle(
                    (x * tile_size, y * tile_size),
                    tile_size,
                    tile_size,
                    facecolor=tile_color,
                    edgecolor=None,
                    linewidth=0,
                    alpha=alpha,
                )
                ax.add_patch(rect)
                rect.set_zorder(0)
        elif tile_type < 6:
            for x, y in coords_list:
                dx = tile_size / 2 if tile_type == 3 else 0
                dy = tile_size / 2 if tile_type == 4 else 0
                w = tile_size if tile_type % 2 == 0 else tile_size / 2
                h = tile_size / 2 if tile_type % 2 == 0 else tile_size

                rect = mpatches.Rectangle(
                    (x * tile_size + dx, y * tile_size + dy),
                    w,
                    h,
                    facecolor=tile_color,
                    edgecolor=None,
                    linewidth=0,
                    alpha=alpha,
                )
                ax.add_patch(rect)
                rect.set_zorder(0)
        else:
            for x, y in coords_list:
                _draw_complex_tile(ax, tile_type, x, y, tile_size, tile_color, alpha)


# ---------------------------------------------------------------------------
# Surface graph loading
# ---------------------------------------------------------------------------

def _load_surface_pbrs(path: str):
    """Load surface-graph PBRS data from .npz."""
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


# ---------------------------------------------------------------------------
# Visualization modes
# ---------------------------------------------------------------------------

class SurfaceGraphVisualizer:
    """Visualizer for surface-graph PBRS data."""

    def __init__(self, data_path: str, map_path: Optional[str] = None):
        print(f"Loading surface graph from: {data_path}")
        self.graph, self.d_switch, self.d_exit, self.meta = (
            _load_surface_pbrs(data_path))

        sp = self.graph.surface_points
        self.xs = np.array([p.x for p in sp])
        self.ys = np.array([p.y for p in sp])
        self.stypes = np.array([p.surface_type for p in sp])
        self.is_edge = np.array([p.is_edge for p in sp])

        self.tile_dic = None
        if map_path and os.path.exists(map_path):
            self.tile_dic = _load_tile_dict(map_path)

        n = len(sp)
        n_edges = len(self.graph.edges_src)
        print(f"  Surface points: {n:,}")
        print(f"  Floor: {int(np.sum(self.stypes == 0)):,}  "
              f"Wall: {int(np.sum(self.stypes == 1)):,}  "
              f"Edge: {int(np.sum(self.is_edge)):,}")
        print(f"  Graph nodes: {self.graph.n_nodes:,}")
        print(f"  Edges: {n_edges:,}")

    def _setup_axis(self, ax, title: str = ""):
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        if title:
            ax.set_title(title)
        if self.tile_dic is not None:
            render_tiles_to_axis(ax, self.tile_dic, alpha=0.4)

    # ---- heatmap ---------------------------------------------------------
    def visualize_heatmap(
        self, phase: str = "switch", output_path: Optional[str] = None,
    ) -> None:
        """Show surface points coloured by d_star."""
        d_star = self.d_switch if phase == "switch" else self.d_exit
        # Per-point d*: take the zero-velocity-bin value
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

        norm = mcolors.Normalize(vmin=float(np.min(finite)),
                                  vmax=float(np.max(finite)))
        c = np.where(np.isfinite(costs), costs, np.max(finite))
        sc = ax.scatter(self.xs, self.ys, c=c, cmap="viridis_r",
                        norm=norm, s=8, zorder=3)
        plt.colorbar(sc, ax=ax, label="d* (frames)")

        # Highlight edge points
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

    # ---- edges -----------------------------------------------------------
    def visualize_edges(
        self, max_edges: int = 5000, output_path: Optional[str] = None,
    ) -> None:
        """Overlay graph edges on surface points."""
        fig, ax = plt.subplots(figsize=(14, 9))
        self._setup_axis(ax, "Surface Graph Edges")

        sp = self.graph.surface_points
        max_bins = self.graph.max_vel_bins

        n_show = min(max_edges, len(self.graph.edges_src))
        indices = np.random.choice(
            len(self.graph.edges_src), n_show, replace=False) if n_show < len(
                self.graph.edges_src) else np.arange(n_show)

        segments = []
        weights = []
        for idx in indices:
            si = int(self.graph.edges_src[idx])
            di = int(self.graph.edges_dst[idx])
            s_pt, _ = node_id_to_parts(si, max_bins)
            d_pt, _ = node_id_to_parts(di, max_bins)
            if s_pt >= len(sp) or d_pt >= len(sp):
                continue
            segments.append([
                (sp[s_pt].x, sp[s_pt].y),
                (sp[d_pt].x, sp[d_pt].y),
            ])
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

    # ---- dashboard -------------------------------------------------------
    def visualize_dashboard(self, output_path: Optional[str] = None) -> None:
        """Multi-panel dashboard with statistics."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. Switch d* heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_dstar_scatter(ax1, self.d_switch, "d* switch")

        # 2. Exit d* heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_dstar_scatter(ax2, self.d_exit, "d* exit")

        # 3. Surface types
        ax3 = fig.add_subplot(gs[0, 2])
        self._setup_axis(ax3, "Surface Types")
        colors = np.where(self.stypes == SURFACE_FLOOR, "green", "blue")
        ax3.scatter(self.xs, self.ys, c=colors, s=5, alpha=0.6, zorder=3)
        ax3.scatter([], [], c="green", label="Floor")
        ax3.scatter([], [], c="blue", label="Wall")
        ax3.legend()

        # 4. d* distribution
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

        # 5. Edge weight distribution
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self.graph.edges_weight) > 0:
            ax5.hist(self.graph.edges_weight, bins=50,
                     color="salmon", edgecolor="black")
        ax5.set_xlabel("Edge weight (frames)")
        ax5.set_ylabel("Count")
        ax5.set_title("Edge Weight Distribution")

        # 6. Stats text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        d_max_sw = self.meta.get("d_max_switch", "?")
        d_max_ex = self.meta.get("d_max_exit", "?")
        txt = (
            f"STATISTICS\n"
            f"{'=' * 30}\n\n"
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

        plt.suptitle("Surface-Graph PBRS Dashboard", fontsize=14,
                      fontweight="bold")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

    # ---- internal --------------------------------------------------------
    def _draw_dstar_scatter(self, ax, d_star, title):
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
        norm = mcolors.Normalize(vmin=float(np.min(finite)),
                                  vmax=float(np.max(finite)))
        sc = ax.scatter(self.xs, self.ys, c=c, cmap="viridis_r",
                        norm=norm, s=6, zorder=3)
        plt.colorbar(sc, ax=ax, label="d* (frames)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize surface-graph PBRS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to *_surface_pbrs.npz file",
    )
    parser.add_argument(
        "--map-path", default=None,
        help="Optional .map file for tile overlay",
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "edges", "dashboard"],
        default="heatmap",
        help="Visualization mode (default: heatmap)",
    )
    parser.add_argument(
        "--phase",
        choices=["switch", "exit"],
        default="switch",
        help="Which d_star phase to show (default: switch)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save figure to file instead of showing",
    )
    parser.add_argument(
        "--max-edges", type=int, default=5000,
        help="Max edges to draw in edge mode (default: 5000)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        return 1

    viz = SurfaceGraphVisualizer(args.data, map_path=args.map_path)

    if args.mode == "heatmap":
        viz.visualize_heatmap(phase=args.phase, output_path=args.output)
    elif args.mode == "edges":
        viz.visualize_edges(max_edges=args.max_edges, output_path=args.output)
    elif args.mode == "dashboard":
        viz.visualize_dashboard(output_path=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
