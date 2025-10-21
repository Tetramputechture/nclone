import pygame
from typing import Optional
from . import render_utils
from .tile_renderer import TileRenderer
from .entity_renderer import EntityRenderer
from .debug_overlay_renderer import DebugOverlayRenderer


class NSimRenderer:
    def __init__(
        self,
        sim,
        render_mode: str = "grayscale_array",
        enable_debug_overlay: bool = False,
        grayscale: bool = False,
    ):
        self.sim = sim
        self.grayscale = grayscale

        if render_mode == "human":
            # RGB for human viewing
            self.screen = pygame.display.set_mode(
                (render_utils.SRCWIDTH, render_utils.SRCHEIGHT), pygame.RESIZABLE
            )
            pygame.display.set_caption("NSim Renderer")
        else:
            # OPTIMIZATION: Create grayscale surface (8-bit) for faster processing
            # This saves ~0.2s per 60 frames by avoiding RGB->grayscale conversion
            if grayscale:
                self.screen = pygame.Surface(
                    (render_utils.SRCWIDTH, render_utils.SRCHEIGHT), depth=8
                )
                # Set up grayscale palette (0-255 mapping to gray shades)
                palette = [(i, i, i) for i in range(256)]
                self.screen.set_palette(palette)
            else:
                self.screen = pygame.Surface(
                    (render_utils.SRCWIDTH, render_utils.SRCHEIGHT)
                )

        if not pygame.font.get_init():
            pygame.font.init()

        self.render_mode = render_mode
        self.adjust = 1.0
        self.width = float(render_utils.SRCWIDTH)
        self.height = float(render_utils.SRCHEIGHT)
        self.tile_x_offset = 0.0
        self.tile_y_offset = 0.0
        self.enable_debug_overlay = enable_debug_overlay
        self.tile_rendering_enabled = True  # Toggle for tile rendering

        self.tile_paths = {}

        self.tile_renderer = TileRenderer(self.sim, self.screen, self.adjust)
        self.entity_renderer = EntityRenderer(
            self.sim, self.screen, self.adjust, self.width, self.height
        )
        self.debug_overlay_renderer = DebugOverlayRenderer(
            self.sim, self.screen, self.adjust, self.tile_x_offset, self.tile_y_offset
        )

    def draw(self, init: bool, debug_info: Optional[dict] = None) -> pygame.Surface:
        self._update_screen_size_and_offsets()

        # Fill the main screen with the general background color
        if self.grayscale:
            # For grayscale: convert RGB to grayscale value (Y = 0.299R + 0.587G + 0.114B)
            r, g, b = render_utils.BGCOLOR_RGB
            gray_value = int((0.299 * r + 0.587 * g + 0.114 * b) * 255)
            self.screen.fill(gray_value)
        else:
            # Convert 0-1 float RGB to 0-255 int RGB for Pygame fill
            pygame_bgcolor = tuple(int(c * 255) for c in render_utils.BGCOLOR_RGB)
            self.screen.fill(pygame_bgcolor)

        # Draw entities first
        entities_surface = self.entity_renderer.draw_entities(init)
        self.screen.blit(entities_surface, (self.tile_x_offset, self.tile_y_offset))

        # Draw tiles on top of entities (if enabled)
        if self.tile_rendering_enabled:
            tiles_surface = self.tile_renderer.draw_tiles(init)
            self.screen.blit(tiles_surface, (self.tile_x_offset, self.tile_y_offset))

        if self.enable_debug_overlay:
            overlay_surface = self.debug_overlay_renderer.draw_debug_overlay(debug_info)
            self.screen.blit(overlay_surface, (0, 0))

        if self.render_mode == "human":
            pygame.display.flip()
        return self.screen

    def draw_collision_map(self, init: bool) -> pygame.Surface:
        self._update_screen_size_and_offsets()
        self.screen.fill(render_utils.hex2float(render_utils.BASIC_BG_COLOR))
        self.screen.blit(
            self.tile_renderer.draw_tiles(
                init, tile_color=render_utils.BASIC_TILE_COLOR
            ),
            (self.tile_x_offset, self.tile_y_offset),
        )
        return self.screen

    def _update_screen_size_and_offsets(self):
        screen_width = float(self.screen.get_width())
        screen_height = float(self.screen.get_height())

        self.adjust = min(
            screen_width / render_utils.SRCWIDTH, screen_height / render_utils.SRCHEIGHT
        )

        self.width = render_utils.SRCWIDTH * self.adjust
        self.height = render_utils.SRCHEIGHT * self.adjust

        self.tile_x_offset = (screen_width - self.width) / 2.0
        self.tile_y_offset = (screen_height - self.height) / 2.0

        self.tile_renderer.adjust = self.adjust
        self.entity_renderer.update_dimensions(self.adjust, self.width, self.height)
        self.debug_overlay_renderer.update_params(
            self.adjust, self.tile_x_offset, self.tile_y_offset
        )
