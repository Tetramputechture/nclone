import cairo
import pyglet
from pyglet.image import ImageData
import math
from . import render_utils
from typing import Optional # For type hinting

class TileRenderer:
    def __init__(self, sim, window, adjust): # screen changed to window
        self.sim = sim
        self.window = window # Store pyglet window
        self.adjust = adjust
        self.render_surface: Optional[cairo.ImageSurface] = None # Cairo surface for drawing
        self.render_context: Optional[cairo.Context] = None
        self.cached_tile_sprite: Optional[pyglet.sprite.Sprite] = None
        self.cached_adjust = None
        self.cached_tile_color = None
        # Cache dimensions used to create the sprite to detect when it needs recreation
        self.cached_sprite_source_width = 0 
        self.cached_sprite_source_height = 0


    def update_params(self, adjust: float):
        if self.adjust != adjust:
            self.adjust = adjust
            # The draw_tiles method will pick up this change and rerender if needed
            # because self.cached_adjust will differ from self.adjust.
    # tile_color_override argument is used for tile_color_override from NSimRenderer.draw_collision_map
    def draw_tiles(self, init: bool, tile_color_override: Optional[str] = None): # Return type is None
        """Draws all tiles, caching them as a Pyglet sprite.
           The method now directly draws the sprite if it\'s up-to-date, or regenerates and then draws it.
        Args:
            init: Boolean indicating if this is the first draw call or if a full redraw is forced.
            tile_color_override: Optional color to use for the tiles, overriding default.
        """
        current_draw_color = tile_color_override if tile_color_override is not None else render_utils.TILECOLOR

        # Use window dimensions for the Cairo surface, matching original intent for surface size.
        target_cairo_surface_width = int(self.window.width)
        target_cairo_surface_height = int(self.window.height)
        
        needs_rerender = (
            init or
            self.cached_tile_sprite is None or
            self.cached_adjust != self.adjust or 
            self.cached_tile_color != current_draw_color or 
            self.cached_sprite_source_width != target_cairo_surface_width or  # Check against dimensions used for cairo surface
            self.cached_sprite_source_height != target_cairo_surface_height
        )

        if needs_rerender:
            # Create or resize Cairo surface if necessary
            if (self.render_surface is None or
                self.render_surface.get_width() != target_cairo_surface_width or
                self.render_surface.get_height() != target_cairo_surface_height):
                
                self.render_surface = cairo.ImageSurface(
                    cairo.Format.ARGB32, target_cairo_surface_width, target_cairo_surface_height
                )
                self.render_context = cairo.Context(self.render_surface)
            
            assert self.render_context is not None, "Cairo context not initialized"
            # Clear Cairo surface (it paints to transparent black by default with ARGB32)
            self.render_context.set_operator(cairo.Operator.CLEAR)
            self.render_context.paint() 
            # Set operator for drawing new content
            self.render_context.set_operator(cairo.Operator.OVER) # Use OVER for proper alpha blending if tiles have alpha

            tilesize = 24 * self.adjust

            # Set tile color for Cairo
            if current_draw_color == render_utils.TILECOLOR:
                r, g, b = render_utils.TILECOLOR_RGB
                self.render_context.set_source_rgb(r, g, b)
            else:
                r, g, b = render_utils.hex2float(current_draw_color)
                self.render_context.set_source_rgb(r, g, b)

            # Group tiles by type for efficient drawing where possible
            tile_groups = {}
            for coords, tile_val in self.sim.tile_dic.items():
                if tile_val != 0: # 0 is empty space, don't draw
                    if tile_val not in tile_groups:
                        tile_groups[tile_val] = []
                    tile_groups[tile_val].append(coords)
            
            # Draw tiles onto Cairo surface
            for tile_type, coords_list in tile_groups.items():
                if tile_type == 1 or tile_type > 33: # Simple square tiles
                    for x, y in coords_list:
                        self.render_context.rectangle(x * tilesize, y * tilesize,
                                                      tilesize, tilesize)
                    self.render_context.fill() # Fill after all rects for this type
                elif tile_type < 6: # Half tiles (original logic had specific dx,dy,w,h for each type)
                    for x_coord, y_coord in coords_list:
                        # Re-implementing half-tile logic based on original Pygame version more clearly
                        # Pygame rect was (x * tilesize + dx, y * tilesize + dy, w, h)
                        # dx, dy were offsets from top-left of the tile cell.
                        # w, h were width/height of the half-tile.
                        half_tilesize = tilesize / 2.0
                        rect_x, rect_y, rect_w, rect_h = 0.0, 0.0, 0.0, 0.0
                        if tile_type == 2: # Top half
                            rect_x, rect_y, rect_w, rect_h = x_coord * tilesize, y_coord * tilesize, tilesize, half_tilesize
                        elif tile_type == 3: # Right half
                            rect_x, rect_y, rect_w, rect_h = x_coord * tilesize + half_tilesize, y_coord * tilesize, half_tilesize, tilesize
                        elif tile_type == 4: # Bottom half
                            rect_x, rect_y, rect_w, rect_h = x_coord * tilesize, y_coord * tilesize + half_tilesize, tilesize, half_tilesize
                        elif tile_type == 5: # Left half
                            rect_x, rect_y, rect_w, rect_h = x_coord * tilesize, y_coord * tilesize, half_tilesize, tilesize
                        self.render_context.rectangle(rect_x, rect_y, rect_w, rect_h)
                    self.render_context.fill() # Fill after all rects for this type
                else: # Complex tiles (slopes, curves) are drawn individually
                    for x, y in coords_list:
                        self._draw_complex_tile(tile_type, x, y, tilesize)
            
            # Ensure all drawing to Cairo surface is finished
            self.render_surface.flush()
            buffer = self.render_surface.get_data() # This is a memoryview
            width = self.render_surface.get_width()
            height = self.render_surface.get_height()
            # Cairo stride is bytes per row. Pyglet ImageData takes pitch (can be negative for top-down)
            pitch = self.render_surface.get_stride()

            # Create Pyglet ImageData. Cairo ARGB32 is BGRA in memory on little-endian.
            # Negative pitch makes Pyglet interpret rows from top to bottom.
            image_data = ImageData(width, height, 'BGRA', bytes(buffer), -pitch)
            
            # Create or update the sprite
            if self.cached_tile_sprite is None:
                self.cached_tile_sprite = pyglet.sprite.Sprite(img=image_data)
            else:
                self.cached_tile_sprite.image = image_data # Update texture of existing sprite
            
            # Update cache variables
            self.cached_adjust = self.adjust
            self.cached_tile_color = current_draw_color
            self.cached_sprite_source_width = width
            self.cached_sprite_source_height = height

        # Draw the sprite if it exists
        if self.cached_tile_sprite:
            # Sprite position is (0,0) because NSimRenderer handles the translation
            self.cached_tile_sprite.x = 0 
            self.cached_tile_sprite.y = 0
            self.cached_tile_sprite.draw()
        # No return, drawing is done directly.

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