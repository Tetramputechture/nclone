import pygame
from typing import Optional
from . import render_utils
from .tile_renderer import TileRenderer
from .entity_renderer import EntityRenderer
from .debug_overlay_renderer import DebugOverlayRenderer

SRCWIDTH = 1056
SRCHEIGHT = 600

BGCOLOR = "cbcad0"
TILECOLOR = "797988"
NINJACOLOR = "000000"
ENTITYCOLORS = {1: "9E2126", 2: "DBE149", 3: "838384", 4: "6D97C3", 5: "000000", 6: "000000",
                7: "000000", 8: "000000", 9: "000000", 10: "868793", 11: "666666", 12: "000000",
                13: "000000", 14: "6EC9E0", 15: "6EC9E0", 16: "000000", 17: "E3E3E5", 18: "000000",
                19: "000000", 20: "838384", 21: "CE4146", 22: "000000", 23: "000000", 24: "666666",
                25: "15A7BD", 26: "6EC9E0", 27: "000000", 28: "6EC9E0"}

BASIC_BG_COLOR = "ffffff"
BASIC_TILE_COLOR = "000000"

SEGMENTWIDTH = 1
NINJAWIDTH = 1.25
DOORWIDTH = 2
PLATFORMWIDTH = 3

LIMBS = ((0, 12), (1, 12), (2, 8), (3, 9), (4, 10),
         (5, 11), (6, 7), (8, 0), (9, 0), (10, 1), (11, 1))

# Pre-calculate color values
BGCOLOR_RGB = tuple(
    int(c, 16)/255 for c in (BGCOLOR[0:2], BGCOLOR[2:4], BGCOLOR[4:6]))
TILECOLOR_RGB = tuple(
    int(c, 16)/255 for c in (TILECOLOR[0:2], TILECOLOR[2:4], TILECOLOR[4:6]))
NINJACOLOR_RGB = tuple(
    int(c, 16)/255 for c in (NINJACOLOR[0:2], NINJACOLOR[2:4], NINJACOLOR[4:6]))
ENTITYCOLORS_RGB = {k: tuple(int(
    c, 16)/255 for c in (v[0:2], v[2:4], v[4:6])) for k, v in ENTITYCOLORS.items()}

# Colors for exploration grid visualization
EXPLORATION_COLORS = {
    'cell': (0, 0, 0, 0),  # Transparent for unvisited cells
    # Bright green with 75% opacity for visited cells
    'cell_visited': (0, 255, 0, 192),
    'grid_cell': (255, 255, 255, 64)  # White with 25% opacity for cell grid
}

# Base colors for each area type
AREA_BASE_COLORS = {
    '4x4': (255, 50, 50),  # Base red for 4x4
    '8x8': (50, 50, 255),  # Base blue for 8x8
    '16x16': (128, 128, 128)  # Base grey for 16x16
}


def hex2float(string):
    """Convert hex color to RGB floats. This is now only used for dynamic colors not in the cache."""
    value = int(string, 16)
    red = ((value & 0xFF0000) >> 16) / 255
    green = ((value & 0x00FF00) >> 8) / 255
    blue = (value & 0x0000FF) / 255
    return red, green, blue


class NSimRenderer:
    def __init__(self, sim, render_mode: str = 'rgb_array', enable_debug_overlay: bool = False):
        self.sim = sim
        if render_mode == 'human':
            self.screen = pygame.display.set_mode(
                (render_utils.SRCWIDTH, render_utils.SRCHEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption("NSim Renderer")
        else:
            self.screen = pygame.Surface((render_utils.SRCWIDTH, render_utils.SRCHEIGHT))
        
        if not pygame.font.get_init():
            pygame.font.init()

        self.render_mode = render_mode
        self.adjust = 1.0
        self.width = float(render_utils.SRCWIDTH)
        self.height = float(render_utils.SRCHEIGHT)
        self.tile_x_offset = 0.0
        self.tile_y_offset = 0.0
        self.enable_debug_overlay = enable_debug_overlay

        self.tile_paths = {}

        self.tile_renderer = TileRenderer(self.sim, self.screen, self.adjust)
        self.entity_renderer = EntityRenderer(self.sim, self.screen, self.adjust, self.width, self.height)
        self.debug_overlay_renderer = DebugOverlayRenderer(self.screen, self.adjust, self.tile_x_offset, self.tile_y_offset)

    def draw(self, init: bool, debug_info: Optional[dict] = None) -> pygame.Surface:
        self._update_screen_size_and_offsets()

        # Fill the main screen with the general background color
        # Convert 0-1 float RGB to 0-255 int RGB for Pygame fill
        pygame_bgcolor = tuple(int(c * 255) for c in render_utils.BGCOLOR_RGB)
        self.screen.fill(pygame_bgcolor)

        # Draw entities first
        entities_surface = self.entity_renderer.draw_entities(init)
        self.screen.blit(entities_surface, (self.tile_x_offset, self.tile_y_offset))

        # Draw tiles on top of entities
        tiles_surface = self.tile_renderer.draw_tiles(init)
        self.screen.blit(tiles_surface, (self.tile_x_offset, self.tile_y_offset))

        if self.enable_debug_overlay:
            overlay_surface = self.debug_overlay_renderer.draw_debug_overlay(debug_info)
            self.screen.blit(overlay_surface, (0, 0))

        if self.render_mode == 'human':
            pygame.display.flip()
        return self.screen

    def draw_collision_map(self, init: bool) -> pygame.Surface:
        self._update_screen_size_and_offsets()
        self.screen.fill(render_utils.hex2float(render_utils.BASIC_BG_COLOR))
        self.screen.blit(self.tile_renderer.draw_tiles(
            init, tile_color=render_utils.BASIC_TILE_COLOR), (self.tile_x_offset, self.tile_y_offset))
        return self.screen

    def _update_screen_size_and_offsets(self):
        screen_width = float(self.screen.get_width())
        screen_height = float(self.screen.get_height())
        
        self.adjust = min(screen_width / render_utils.SRCWIDTH, 
                          screen_height / render_utils.SRCHEIGHT)
        
        self.width = render_utils.SRCWIDTH * self.adjust
        self.height = render_utils.SRCHEIGHT * self.adjust
        
        self.tile_x_offset = (screen_width - self.width) / 2.0
        self.tile_y_offset = (screen_height - self.height) / 2.0
        
        self.tile_renderer.adjust = self.adjust
        self.entity_renderer.update_dimensions(self.adjust, self.width, self.height)
        self.debug_overlay_renderer.update_params(self.adjust, self.tile_x_offset, self.tile_y_offset)
