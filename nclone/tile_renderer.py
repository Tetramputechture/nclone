import cairo
import pygame
from . import render_utils
from .shared_tile_renderer import group_tiles_by_type, render_tile_group_to_cairo


class TileRenderer:
    def __init__(self, sim, screen, adjust):
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.render_surface = None  # Cairo surface for drawing
        self.render_context = None
        self.cached_tile_pygame_surface = None  # Pygame surface for blitting
        self.tiles_drawn_to_cairo_surface = (
            False  # Flag to indicate if cairo surface has up-to-date tiles
        )
        self.cached_adjust = None
        self.cached_tile_color = None

    def draw_tiles(
        self, init: bool, tile_color: str = render_utils.TILECOLOR
    ) -> pygame.Surface:
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
            init
            or self.cached_tile_pygame_surface is None
            or self.cached_adjust != self.adjust
            or self.cached_tile_color != tile_color
            or (
                self.cached_tile_pygame_surface.get_width() != current_screen_width
                or self.cached_tile_pygame_surface.get_height() != current_screen_height
            )
        )

        if needs_rerender:
            if (
                self.render_surface is None
                or self.cached_adjust != self.adjust
                or self.render_surface.get_width() != current_screen_width
                or self.render_surface.get_height() != current_screen_height
            ):
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

            # Use shared tile rendering logic
            tile_groups = group_tiles_by_type(self.sim.tile_dic)

            for tile_type, coords_list in tile_groups.items():
                render_tile_group_to_cairo(
                    self.render_context, tile_type, coords_list, tilesize
                )

            buffer = self.render_surface.get_data()
            self.cached_tile_pygame_surface = pygame.image.frombuffer(
                buffer,
                (self.render_surface.get_width(), self.render_surface.get_height()),
                "BGRA",
            )

            self.cached_adjust = self.adjust
            self.cached_tile_color = tile_color

        return self.cached_tile_pygame_surface
