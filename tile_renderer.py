import cairo
import pygame
import math
from . import render_utils

class TileRenderer:
    def __init__(self, sim, screen, adjust):
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.render_surface = None
        self.render_context = None

    def draw_tiles(self, init: bool, tile_color: str = render_utils.TILECOLOR) -> pygame.Surface:
        """Draws all tiles onto a surface.

        Args:
            init: Boolean indicating if this is the first draw call.
            tile_color: The color to use for the tiles.

        Returns:
            A pygame.Surface with the rendered tiles.
        """
        if init or self.render_surface is None:
            self.render_surface = cairo.ImageSurface(
                cairo.Format.RGB24, *self.screen.get_size())
            self.render_context = cairo.Context(self.render_surface)

        tilesize = 24 * self.adjust

        # Clear the entire surface first
        self.render_context.set_operator(cairo.Operator.CLEAR)
        self.render_context.paint()  # Use paint() instead of rectangle + fill for full clear
        self.render_context.set_operator(cairo.Operator.SOURCE)

        # Use pre-calculated RGB values
        if tile_color == render_utils.TILECOLOR:
            self.render_context.set_source_rgb(*render_utils.TILECOLOR_RGB)
        else:
            self.render_context.set_source_rgb(*render_utils.hex2float(tile_color))

        # Group tiles by type for batch rendering
        tile_groups = {}
        for coords, tile in self.sim.tile_dic.items():
            if tile != 0 and tile not in tile_groups:  # Skip empty tiles
                tile_groups[tile] = []
            if tile != 0:  # Skip empty tiles
                tile_groups[tile].append(coords)

        # Batch render similar tiles
        for tile_type, coords_list in tile_groups.items():
            if tile_type == 1 or tile_type > 33:  # Full tiles
                for x, y in coords_list:
                    self.render_context.rectangle(x * tilesize, y * tilesize,
                                                  tilesize, tilesize)
                self.render_context.fill()
            elif tile_type < 6:  # Half tiles
                dx = tilesize / 2 if tile_type == 3 else 0
                dy = tilesize / 2 if tile_type == 4 else 0
                w = tilesize if tile_type % 2 == 0 else tilesize / 2
                h = tilesize / 2 if tile_type % 2 == 0 else tilesize
                for x, y in coords_list:
                    self.render_context.rectangle(x * tilesize + dx,
                                                  y * tilesize + dy, w, h)
                    self.render_context.fill()
            else:
                # Complex shapes
                for x, y in coords_list:
                    self._draw_complex_tile(tile_type, x, y, tilesize)

        buffer = self.render_surface.get_data()
        return pygame.image.frombuffer(buffer, self.screen.get_size(), "BGRA")

    def _draw_complex_tile(self, tile_type, x, y, tilesize):
        """Draw a complex tile shape."""
        if tile_type < 10:
            dx1 = 0
            dy1 = tilesize if tile_type == 8 else 0
            dx2 = 0 if tile_type == 9 else tilesize
            dy2 = tilesize if tile_type == 9 else 0
            dx3 = 0 if tile_type == 6 else tilesize
            dy3 = tilesize
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 14:
            dx = tilesize if (tile_type == 11 or tile_type == 12) else 0
            dy = tilesize if (tile_type == 12 or tile_type == 13) else 0
            a1 = (math.pi / 2) * (tile_type - 10)
            a2 = (math.pi / 2) * (tile_type - 9)
            self.render_context.move_to(x * tilesize + dx, y * tilesize + dy)
            self.render_context.arc(
                x * tilesize + dx, y * tilesize + dy, tilesize, a1, a2)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 18:
            dx1 = tilesize if (tile_type == 15 or tile_type == 16) else 0
            dy1 = tilesize if (tile_type == 16 or tile_type == 17) else 0
            dx2 = tilesize if (tile_type == 14 or tile_type == 17) else 0
            dy2 = tilesize if (tile_type == 14 or tile_type == 15) else 0
            a1 = math.pi + (math.pi / 2) * (tile_type - 10)
            a2 = math.pi + (math.pi / 2) * (tile_type - 9)
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.arc(
                x * tilesize + dx2, y * tilesize + dy2, tilesize, a1, a2)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 22:
            dx1 = 0
            dy1 = tilesize if (tile_type == 20 or tile_type == 21) else 0
            dx2 = tilesize
            dy2 = tilesize if (tile_type == 20 or tile_type == 21) else 0
            dx3 = tilesize if (tile_type == 19 or tile_type == 20) else 0
            dy3 = tilesize / 2
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 26:
            dx1 = 0
            dy1 = tilesize / 2 if (tile_type == 23 or tile_type == 24) else 0
            dx2 = 0 if tile_type == 23 else tilesize
            dy2 = tilesize / 2 if tile_type == 25 else 0
            dx3 = tilesize
            dy3 = (tilesize / 2 if tile_type ==
                   22 else 0) if tile_type < 24 else tilesize
            dx4 = tilesize if tile_type == 23 else 0
            dy4 = tilesize
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.line_to(x * tilesize + dx4, y * tilesize + dy4)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 30:
            dx1 = tilesize / 2
            dy1 = tilesize if (tile_type == 28 or tile_type == 29) else 0
            dx2 = tilesize if (tile_type == 27 or tile_type == 28) else 0
            dy2 = 0
            dx3 = tilesize if (tile_type == 27 or tile_type == 28) else 0
            dy3 = tilesize
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 34:
            dx1 = tilesize / 2
            dy1 = tilesize if (tile_type == 30 or tile_type == 31) else 0
            dx2 = tilesize if (tile_type == 31 or tile_type == 33) else 0
            dy2 = tilesize
            dx3 = tilesize if (tile_type == 31 or tile_type == 32) else 0
            dy3 = tilesize if (tile_type == 32 or tile_type == 33) else 0
            dx4 = tilesize if (tile_type == 30 or tile_type == 32) else 0
            dy4 = 0
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.line_to(x * tilesize + dx4, y * tilesize + dy4)
            self.render_context.close_path()
            self.render_context.fill() 