"""
Core tile rendering utilities for consistent tile rendering across the application.

This module provides the fundamental tile rendering functions that can be used by both
the main game renderer and the visualization/debug systems to ensure consistency.
All tile rendering logic is centralized here to eliminate duplication.
"""

import pygame
import cairo
import math
from . import render_utils


def group_tiles_by_type(tile_dic):
    """Group tiles by type for efficient batch rendering."""
    tile_groups = {}
    for coords, tile_val in tile_dic.items():
        if tile_val != 0:
            if tile_val not in tile_groups:
                tile_groups[tile_val] = []
            tile_groups[tile_val].append(coords)
    return tile_groups


def render_tile_group_to_cairo(ctx, tile_type, coords_list, tile_size):
    """Render a group of tiles of the same type to a Cairo context."""
    if tile_type == 1 or tile_type > 33:
        # Simple rectangular tiles - batch render
        for x, y in coords_list:
            ctx.rectangle(x * tile_size, y * tile_size, tile_size, tile_size)
        ctx.fill()
    elif tile_type < 6:
        # Half tiles (types 2-5) - batch render
        dx = tile_size / 2 if tile_type == 3 else 0
        dy = tile_size / 2 if tile_type == 4 else 0
        w = tile_size if tile_type % 2 == 0 else tile_size / 2
        h = tile_size / 2 if tile_type % 2 == 0 else tile_size
        for x, y in coords_list:
            ctx.rectangle(x * tile_size + dx, y * tile_size + dy, w, h)
        ctx.fill()
    else:
        # Complex tiles (slopes, curves, etc.) - render individually
        for x, y in coords_list:
            draw_complex_tile_to_cairo(ctx, tile_type, x, y, tile_size)


def draw_complex_tile_to_cairo(ctx, tile_type, x, y, tile_size):
    """Draw a complex tile shape to a Cairo context."""
    if tile_type < 10:
        # Triangular tiles (types 6-9)
        dx1 = 0
        dy1 = tile_size if tile_type == 8 else 0
        dx2 = 0 if tile_type == 9 else tile_size
        dy2 = tile_size if tile_type == 9 else 0
        dx3 = 0 if tile_type == 6 else tile_size
        dy3 = tile_size
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.line_to(x * tile_size + dx2, y * tile_size + dy2)
        ctx.line_to(x * tile_size + dx3, y * tile_size + dy3)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 14:
        # Quarter circle tiles (types 10-13)
        dx = tile_size if (tile_type == 11 or tile_type == 12) else 0
        dy = tile_size if (tile_type == 12 or tile_type == 13) else 0
        a1 = (math.pi / 2) * (tile_type - 10)
        a2 = (math.pi / 2) * (tile_type - 9)
        ctx.move_to(x * tile_size + dx, y * tile_size + dy)
        ctx.arc(x * tile_size + dx, y * tile_size + dy, tile_size, a1, a2)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 18:
        # Inverted quarter circle tiles (types 14-17)
        dx1 = tile_size if (tile_type == 15 or tile_type == 16) else 0
        dy1 = tile_size if (tile_type == 16 or tile_type == 17) else 0
        dx2 = tile_size if (tile_type == 14 or tile_type == 17) else 0
        dy2 = tile_size if (tile_type == 14 or tile_type == 15) else 0
        a1 = math.pi + (math.pi / 2) * (tile_type - 10)
        a2 = math.pi + (math.pi / 2) * (tile_type - 9)
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.arc(x * tile_size + dx2, y * tile_size + dy2, tile_size, a1, a2)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 22:
        # Sloped triangular tiles (types 18-21)
        dx1 = 0
        dy1 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx2 = tile_size
        dy2 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx3 = tile_size if (tile_type == 19 or tile_type == 20) else 0
        dy3 = tile_size / 2
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.line_to(x * tile_size + dx2, y * tile_size + dy2)
        ctx.line_to(x * tile_size + dx3, y * tile_size + dy3)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 26:
        # Quadrilateral tiles (types 22-25)
        dx1 = 0
        dy1 = tile_size / 2 if (tile_type == 23 or tile_type == 24) else 0
        dx2 = 0 if tile_type == 23 else tile_size
        dy2 = tile_size / 2 if tile_type == 25 else 0
        dx3 = tile_size
        dy3 = (tile_size / 2 if tile_type == 22 else 0) if tile_type < 24 else tile_size
        dx4 = tile_size if tile_type == 23 else 0
        dy4 = tile_size
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.line_to(x * tile_size + dx2, y * tile_size + dy2)
        ctx.line_to(x * tile_size + dx3, y * tile_size + dy3)
        ctx.line_to(x * tile_size + dx4, y * tile_size + dy4)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 30:
        # Triangular tiles with midpoint (types 26-29)
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 28 or tile_type == 29) else 0
        dx2 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy2 = 0
        dx3 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy3 = tile_size
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.line_to(x * tile_size + dx2, y * tile_size + dy2)
        ctx.line_to(x * tile_size + dx3, y * tile_size + dy3)
        ctx.close_path()
        ctx.fill()
    elif tile_type < 34:
        # Complex quadrilateral tiles (types 30-33)
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 30 or tile_type == 31) else 0
        dx2 = tile_size if (tile_type == 31 or tile_type == 33) else 0
        dy2 = tile_size
        dx3 = tile_size if (tile_type == 31 or tile_type == 32) else 0
        dy3 = tile_size if (tile_type == 32 or tile_type == 33) else 0
        dx4 = tile_size if (tile_type == 30 or tile_type == 32) else 0
        dy4 = 0
        ctx.move_to(x * tile_size + dx1, y * tile_size + dy1)
        ctx.line_to(x * tile_size + dx2, y * tile_size + dy2)
        ctx.line_to(x * tile_size + dx3, y * tile_size + dy3)
        ctx.line_to(x * tile_size + dx4, y * tile_size + dy4)
        ctx.close_path()
        ctx.fill()


def render_tiles_to_cairo_surface(
    tile_dic, surface_width, surface_height, tile_size, tile_color_rgb
):
    """
    Render tiles to a Cairo surface using the core tile rendering logic.

    Args:
        tile_dic: Dictionary mapping (x, y) coordinates to tile type values
        surface_width: Width of the target surface
        surface_height: Height of the target surface
        tile_size: Size of each tile in pixels
        tile_color_rgb: RGB color tuple for tiles

    Returns:
        cairo.ImageSurface with rendered tiles
    """
    # Create Cairo surface for precise rendering
    cairo_surface = cairo.ImageSurface(
        cairo.Format.ARGB32, surface_width, surface_height
    )
    ctx = cairo.Context(cairo_surface)

    # Clear surface
    ctx.set_operator(cairo.Operator.CLEAR)
    ctx.paint()
    ctx.set_operator(cairo.Operator.SOURCE)

    # Set tile color
    ctx.set_source_rgb(*tile_color_rgb)

    # Group tiles by type for efficient rendering
    tile_groups = group_tiles_by_type(tile_dic)

    # Render each tile type group
    for tile_type, coords_list in tile_groups.items():
        render_tile_group_to_cairo(ctx, tile_type, coords_list, tile_size)

    return cairo_surface


class SharedTileRenderer:
    """Convenience wrapper for tile rendering functionality."""

    @staticmethod
    def render_tiles_to_pygame_surface(
        tile_dic, surface_width, surface_height, tile_size=24, tile_color_rgb=None
    ):
        """
        Render tiles to a pygame surface using the core tile rendering logic.

        Args:
            tile_dic: Dictionary mapping (x, y) coordinates to tile type values
            surface_width: Width of the target surface
            surface_height: Height of the target surface
            tile_size: Size of each tile in pixels (default 24)
            tile_color_rgb: RGB color tuple for tiles (default uses TILECOLOR_RGB)

        Returns:
            pygame.Surface with rendered tiles
        """
        if tile_color_rgb is None:
            tile_color_rgb = render_utils.TILECOLOR_RGB

        # Use the core rendering function
        cairo_surface = render_tiles_to_cairo_surface(
            tile_dic, surface_width, surface_height, tile_size, tile_color_rgb
        )

        # Convert Cairo surface to pygame surface
        buffer = cairo_surface.get_data()
        pygame_surface = pygame.image.frombuffer(
            buffer, (cairo_surface.get_width(), cairo_surface.get_height()), "BGRA"
        )

        return pygame_surface
