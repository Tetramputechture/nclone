import cairo
import pygame
import math
from . import render_utils

class TileRenderer:
    def __init__(self, sim, screen, adjust):
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.render_surface = None # Cairo surface for drawing
        self.render_context = None
        self.cached_tile_pygame_surface = None # Pygame surface for blitting
        self.tiles_drawn_to_cairo_surface = False # Flag to indicate if cairo surface has up-to-date tiles
        self.cached_adjust = None
        self.cached_tile_color = None

    def draw_tiles(self, init: bool, tile_color: str = render_utils.TILECOLOR) -> pygame.Surface:
        """Draws all tiles onto a surface, using a cache for performance.

        Args:
            init: Boolean indicating if this is the first draw call or if a full redraw is forced.
            tile_color: The color to use for the tiles.

        Returns:
            A pygame.Surface with the rendered tiles.
        """
        current_screen_width = self.screen.get_width()
        current_screen_height = self.screen.get_height()

        needs_rerender = (
            init or
            self.cached_tile_pygame_surface is None or
            self.cached_adjust != self.adjust or
            self.cached_tile_color != tile_color or
            (self.cached_tile_pygame_surface.get_width() != current_screen_width or
             self.cached_tile_pygame_surface.get_height() != current_screen_height)
        )

        if needs_rerender:
            if (self.render_surface is None or
                self.cached_adjust != self.adjust or 
                self.render_surface.get_width() != current_screen_width or
                self.render_surface.get_height() != current_screen_height):
                
                self.render_surface = cairo.ImageSurface(
                    cairo.Format.ARGB32, current_screen_width, current_screen_height
                )
                self.render_context = cairo.Context(self.render_surface)

            self.render_context.set_operator(cairo.Operator.CLEAR)
            self.render_context.paint()
            self.render_context.set_operator(cairo.Operator.SOURCE)

            tilesize = 24 * self.adjust

            if tile_color == render_utils.TILECOLOR:
                self.render_context.set_source_rgb(*render_utils.TILECOLOR_RGB)
            else:
                self.render_context.set_source_rgb(*render_utils.hex2float(tile_color))

            tile_groups = {}
            for coords, tile_val in self.sim.tile_dic.items():
                if tile_val != 0:
                    if tile_val not in tile_groups:
                        tile_groups[tile_val] = []
                    tile_groups[tile_val].append(coords)
            
            for tile_type, coords_list in tile_groups.items():
                if tile_type == 1 or tile_type > 33:
                    for x, y in coords_list:
                        self.render_context.rectangle(x * tilesize, y * tilesize,
                                                      tilesize, tilesize)
                    self.render_context.fill()
                elif tile_type < 6:
                    dx = tilesize / 2 if tile_type == 3 else 0
                    dy = tilesize / 2 if tile_type == 4 else 0
                    w = tilesize if tile_type % 2 == 0 else tilesize / 2
                    h = tilesize / 2 if tile_type % 2 == 0 else tilesize
                    for x, y in coords_list:
                        self.render_context.rectangle(x * tilesize + dx,
                                                      y * tilesize + dy, w, h)
                    self.render_context.fill() # Fill per group
                else:
                    for x, y in coords_list: # Complex tiles fill individually
                        self._draw_complex_tile(tile_type, x, y, tilesize)

            buffer = self.render_surface.get_data()
            self.cached_tile_pygame_surface = pygame.image.frombuffer(
                buffer, (self.render_surface.get_width(), self.render_surface.get_height()), "BGRA"
            )
            
            self.cached_adjust = self.adjust
            self.cached_tile_color = tile_color

        return self.cached_tile_pygame_surface

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