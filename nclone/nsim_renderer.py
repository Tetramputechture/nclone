import pygame
import numpy as np
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
        
        # Blit entities onto screen
        if self.grayscale:
            # For grayscale mode, manually composite to preserve pixel values
            self._blit_grayscale(entities_surface, (self.tile_x_offset, self.tile_y_offset))
        else:
            self.screen.blit(entities_surface, (self.tile_x_offset, self.tile_y_offset))

        # Draw tiles on top of entities (if enabled)
        if self.tile_rendering_enabled:
            tiles_surface = self.tile_renderer.draw_tiles(init)
            if self.grayscale:
                self._blit_grayscale(tiles_surface, (self.tile_x_offset, self.tile_y_offset))
            else:
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
    
    def _blit_grayscale(self, source_surface: pygame.Surface, offset: tuple):
        """
        Blit a colored surface (BGRA from Cairo) onto the grayscale screen,
        converting colors to grayscale and respecting alpha transparency.
        
        Args:
            source_surface: The BGRA surface to blit (typically from entity_renderer)
            offset: (x, y) offset for blitting
        """
        offset_x, offset_y = int(offset[0]), int(offset[1])
        
        try:
            # Get RGB and alpha data from source surface
            rgb_array = pygame.surfarray.array3d(source_surface)  # (W, H, 3)
            alpha_array = pygame.surfarray.array_alpha(source_surface)  # (W, H)
            
            # Convert to grayscale
            gray_wh = (
                0.2989 * rgb_array[:, :, 0] + 
                0.5870 * rgb_array[:, :, 1] + 
                0.1140 * rgb_array[:, :, 2]
            ).astype(np.uint8)
            
            # Get screen pixel array
            screen_pixels = pygame.surfarray.pixels2d(self.screen)
            
            src_width, src_height = source_surface.get_size()
            screen_width, screen_height = self.screen.get_size()
            
            # Calculate the region to copy
            # Handle cases where source might extend beyond screen boundaries
            src_x_start = max(0, -offset_x)
            src_y_start = max(0, -offset_y)
            src_x_end = min(src_width, screen_width - offset_x)
            src_y_end = min(src_height, screen_height - offset_y)
            
            dst_x_start = max(0, offset_x)
            dst_y_start = max(0, offset_y)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            
            if src_x_start < src_x_end and src_y_start < src_y_end:
                # Extract the relevant region
                src_gray_region = gray_wh[src_x_start:src_x_end, src_y_start:src_y_end]
                src_alpha_region = alpha_array[src_x_start:src_x_end, src_y_start:src_y_end]
                
                # Get the destination region
                dst_region = screen_pixels[dst_x_start:dst_x_end, dst_y_start:dst_y_end]
                
                # Alpha blend: only draw where alpha > 0
                # For alpha = 255, use entity color; for alpha = 0, keep background
                alpha_mask = src_alpha_region > 0
                alpha_normalized = src_alpha_region.astype(np.float32) / 255.0
                
                # Blend: result = src * alpha + dst * (1 - alpha)
                blended = (
                    src_gray_region * alpha_normalized + 
                    dst_region * (1 - alpha_normalized)
                ).astype(np.uint8)
                
                # Apply only where there's some alpha
                dst_region[:] = np.where(alpha_mask, blended, dst_region)
            
            del screen_pixels  # Unlock the surface
            
        except Exception as e:
            # Fallback: use normal blit (may not work well with grayscale)
            import sys
            print(f"WARNING: _blit_grayscale failed: {e}, falling back to normal blit", file=sys.stderr)
            self.screen.blit(source_surface, offset)
