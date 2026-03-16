"""Tile and entity rendering helpers for value-function visualization.

Provides ``render_tiles_to_axis`` and the supporting geometry helpers used by
``SurfaceGraphVisualizer`` and ``NeuralValueLandscape``.  Tile vertex geometry
matches ``npp_rl.rendering.matplotlib_tile_renderer``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tile geometry helpers
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
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)

    elif tile_type < 14:
        dx = tile_size if (tile_type == 11 or tile_type == 12) else 0
        dy = tile_size if (tile_type == 12 or tile_type == 13) else 0
        theta1 = 90 * (tile_type - 10)
        theta2 = 90 * (tile_type - 9)
        wedge = mpatches.Wedge(
            (base_x + dx, base_y + dy), tile_size, theta1, theta2,
            facecolor=tile_color, edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(wedge)
        wedge.set_zorder(0)

    elif tile_type < 18:
        dx1 = tile_size if (tile_type == 15 or tile_type == 16) else 0
        dy1 = tile_size if (tile_type == 16 or tile_type == 17) else 0
        dx2 = tile_size if (tile_type == 14 or tile_type == 17) else 0
        dy2 = tile_size if (tile_type == 14 or tile_type == 15) else 0
        theta1 = 180 + 90 * (tile_type - 10)
        theta2 = 180 + 90 * (tile_type - 9)
        center_x = base_x + dx2
        center_y = base_y + dy2
        angles = np.linspace(np.radians(theta1), np.radians(theta2), 20)
        arc_x = center_x + tile_size * np.cos(angles)
        arc_y = center_y + tile_size * np.sin(angles)
        vertices = [(base_x + dx1, base_y + dy1)] + list(zip(arc_x, arc_y))
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)

    elif tile_type < 22:
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
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)

    elif tile_type < 26:
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
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)

    elif tile_type < 30:
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
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)

    elif tile_type <= 33:
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
        poly = mpatches.Polygon(
            vertices, closed=True, facecolor=tile_color,
            edgecolor=None, linewidth=0, alpha=alpha,
        )
        ax.add_patch(poly)
        poly.set_zorder(0)


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

    Args:
        ax: Matplotlib axis to draw on.
        tile_dic: Mapping of (tile_col, tile_row) -> tile_type (0–37).
        tile_size: Pixel size of one tile.
        tile_color: Fill colour for solid tiles.
        alpha: Tile fill opacity.
    """
    tile_groups = _group_tiles_by_type(tile_dic)

    for tile_type, coords_list in tile_groups.items():
        if tile_type == 1:
            for x, y in coords_list:
                rect = mpatches.Rectangle(
                    (x * tile_size, y * tile_size), tile_size, tile_size,
                    facecolor=tile_color, edgecolor=None, linewidth=0, alpha=alpha,
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
                    (x * tile_size + dx, y * tile_size + dy), w, h,
                    facecolor=tile_color, edgecolor=None, linewidth=0, alpha=alpha,
                )
                ax.add_patch(rect)
                rect.set_zorder(0)
        else:
            for x, y in coords_list:
                _draw_complex_tile(ax, tile_type, x, y, tile_size, tile_color, alpha)


def render_tile_grid_to_axis(
    ax,
    tile_grid: np.ndarray,
    tile_size: float = 24.0,
    tile_color: str = "#404040",
    alpha: float = 0.6,
) -> None:
    """Render a (42, 23) tile grid array directly to an axis.

    Converts the (W, H) array format used by the IQL dataset into the
    ``{(col, row): tile_type}`` dict format expected by
    ``render_tiles_to_axis``, with a +1 offset so tiles are drawn at
    their correct physics-world pixel positions.

    The tile_grid stores the 42×23 *inner* tiles only (excluding the
    1-tile solid border).  ``tile_grid[i, j]`` corresponds to the game
    tile at column ``i+1``, row ``j+1`` in physics coordinates (where
    game col 0 and col 43 are the solid left/right borders).  Using
    ``(col+1, row+1)`` as the dict key causes ``render_tiles_to_axis``
    to draw each tile at ``((i+1)*tile_size, (j+1)*tile_size)``, which
    matches the tile's physics pixel range.

    Args:
        ax: Matplotlib axis.
        tile_grid: Integer array of shape (42, 23) with tile type IDs.
        tile_size: Pixel size per tile.
        tile_color: Solid tile fill colour.
        alpha: Tile opacity.
    """
    tile_dic: Dict[Tuple[int, int], int] = {}
    w, h = tile_grid.shape
    for col in range(w):
        for row in range(h):
            tv = int(tile_grid[col, row])
            if tv != 0:
                # Shift by +1 so the tile is drawn at its physics pixel position:
                # tile_grid[col, row] = game tile (col+1, row+1) which occupies
                # physics pixels [(col+1)*tile_size, (col+2)*tile_size).
                tile_dic[(col + 1, row + 1)] = tv
    render_tiles_to_axis(ax, tile_dic, tile_size=tile_size,
                         tile_color=tile_color, alpha=alpha)


def mark_entities(
    ax,
    switch_pos: Optional[Tuple[float, float]] = None,
    exit_pos: Optional[Tuple[float, float]] = None,
    mine_positions: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """Overlay switch, exit door, and mine markers on an axis.

    Args:
        ax: Matplotlib axis.
        switch_pos: (x, y) pixel coords of the exit switch.
        exit_pos: (x, y) pixel coords of the exit door.
        mine_positions: List of (x, y) pixel coords for mines.
    """
    if mine_positions:
        xs = [m[0] for m in mine_positions]
        ys = [m[1] for m in mine_positions]
        ax.scatter(xs, ys, marker="x", color="red", s=20, linewidths=0.8,
                   zorder=6, label=f"Mines ({len(mine_positions)})")

    if switch_pos is not None:
        ax.scatter(*switch_pos, marker="*", color="yellow", s=150,
                   edgecolors="black", linewidths=0.5, zorder=7, label="Switch")

    if exit_pos is not None:
        ax.scatter(*exit_pos, marker="D", color="lime", s=80,
                   edgecolors="black", linewidths=0.5, zorder=7, label="Exit")
