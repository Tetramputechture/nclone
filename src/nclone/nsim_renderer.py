import pyglet
from pyglet.gl import (
    GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
    glClearColor, glViewport,
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TEXTURE_2D # For Texture.create target
)
from pyglet.math import Mat4, Vec3 # Added for new projection
from pyglet import image as pyglet_image 
from pyglet.image.buffer import Framebuffer 
import numpy as np
from typing import Optional, Union, Tuple, Dict 
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
    def __init__(self, sim, 
                 render_mode: str = 'human', 
                 width: int = SRCWIDTH, 
                 height: int = SRCHEIGHT, 
                 window: Optional[pyglet.window.Window] = None, # User-provided window
                 enable_debug_overlay: bool = False):
        
        self.sim = sim
        self.render_mode = render_mode
        self.initial_render_width = width   # Requested render width for rgb_array or initial window
        self.initial_render_height = height # Requested render height
        self.enable_debug_overlay = enable_debug_overlay

        # Dynamic properties, updated by _update_screen_size_and_offsets or window events
        self.width: float = float(self.initial_render_width) 
        self.height: float = float(self.initial_render_height)
        self.adjust: float = 1.0
        self.tile_x_offset: float = 0.0
        self.tile_y_offset: float = 0.0
        
        self.user_provided_window: Optional[pyglet.window.Window] = window 
        self.context_window: pyglet.window.Window # The window whose GL context is used for rendering
        self.internal_offscreen_window: Optional[pyglet.window.Window] = None # Only for rgb_array if no window given
        
        self.fbo: Optional[Framebuffer] = None
        self.render_texture: Optional[pyglet_image.Texture] = None

        self.tile_renderer: Optional[TileRenderer] = None
        self.entity_renderer: Optional[EntityRenderer] = None
        self.debug_overlay_renderer: Optional[DebugOverlayRenderer] = None
        
        if self.render_mode == 'human':
            if self.user_provided_window is None:
                # Create a window if in human mode and none is provided.
                self.user_provided_window = pyglet.window.Window(self.initial_render_width, self.initial_render_height, "nclone", resizable=True)
            self.context_window = self.user_provided_window
            self.context_window.switch_to()
            r_bg, g_bg, b_bg = BGCOLOR_RGB # Assumes BGCOLOR_RGB is defined globally
            glClearColor(r_bg, g_bg, b_bg, 1.0)
        elif self.render_mode == 'rgb_array':
            if self.user_provided_window: # If a window is given for rgb_array, use its context
                self.context_window = self.user_provided_window
            else: # Otherwise, create an internal, invisible window for the context
                self.internal_offscreen_window = pyglet.window.Window(self.initial_render_width, self.initial_render_height, visible=False)
                self.context_window = self.internal_offscreen_window
            
            self.context_window.switch_to() # Activate the context

            # Create FBO and texture for offscreen rendering
            self.render_texture = pyglet_image.Texture.create(self.initial_render_width, self.initial_render_height, target=GL_TEXTURE_2D)
            self.fbo = Framebuffer()
            self.fbo.attach_texture(self.render_texture, attachment=GL_COLOR_ATTACHMENT0)
            
            if not self.fbo.is_complete: 
                status = self.fbo.get_status()
                print(f"Pyglet FBO creation warning: Framebuffer is not complete. Status: {status}")
            
            r_bg, g_bg, b_bg = BGCOLOR_RGB # Assumes BGCOLOR_RGB is defined globally
            glClearColor(r_bg, g_bg, b_bg, 1.0)
        else:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        self.tile_renderer = TileRenderer(self.sim, self.context_window, self.adjust)
        self.entity_renderer = EntityRenderer(self.sim, self.context_window, self.adjust, self.initial_render_width, self.initial_render_height)
        self.debug_overlay_renderer = DebugOverlayRenderer(self.context_window, self.adjust, self.tile_x_offset, self.tile_y_offset)
        
        self._update_screen_size_and_offsets() # Initial calculation based on context_window size

        # Set clear color (background color) once
        # BGCOLOR_RGB is (r,g,b) floats from 0-1
        if render_utils.BGCOLOR_RGB:
            glClearColor(render_utils.BGCOLOR_RGB[0], render_utils.BGCOLOR_RGB[1], render_utils.BGCOLOR_RGB[2], 1.0)


    def draw(self, init: bool = False, debug_info: Optional[dict] = None) -> Optional[np.ndarray]:
        """Main drawing method. Returns a numpy array if in rgb_array mode, otherwise None."""
        
        self.context_window.switch_to() # Ensure correct GL context is active

        # 1. Setup render target (screen or FBO)
        if self.render_mode == 'rgb_array':
            if not self.fbo or not self.render_texture:
                raise RuntimeError("FBO not initialized for rgb_array mode.")
            self.fbo.bind()
            # Viewport and projection for FBO (size of texture)
            glViewport(0, 0, self.initial_render_width, self.initial_render_height)
            # Set projection and view matrices on the context_window for Pyglet's default rendering pipeline
            self.context_window.projection = Mat4.orthogonal_projection(
                0, self.initial_render_width, 
                0, self.initial_render_height, 
                -1, 1  # z_near, z_far, simple for 2D
            )
            self.context_window.view = Mat4() # Identity view matrix
            # Clear the FBO (uses glClearColor set on the context)
            glClear(GL_COLOR_BUFFER_BIT) # Only color, assuming no depth for 2D
        else: # 'human' mode
            # For human mode, the main window is the target. 
            # nplay.py's on_draw calls window.clear() before this method.
            # Viewport and projection for the main window are typically set by pyglet's default
            # on_resize handler or can be customized if needed.
            # We assume the window is already set up correctly by nplay.py or pyglet defaults.
            pass


        # 2. Update dynamic rendering parameters (zoom, offset)
        # This uses self.context_window.width/height which is correct for both modes.
        self._update_screen_size_and_offsets() 
        
        # Update parameters for sub-renderers
        if self.tile_renderer:
            self.tile_renderer.update_params(self.adjust)
        if self.entity_renderer:
            # EntityRenderer's update_params was not defined, but its __init__ takes adjust.
            # If it needs dynamic width/height for cairo surface, it should get from its window.
            self.entity_renderer.adjust = self.adjust # Directly update if needed
        if self.debug_overlay_renderer: # Debug overlay uses tile_x/y_offset for grid
            self.debug_overlay_renderer.update_params(self.adjust, self.tile_x_offset, self.tile_y_offset)

        # 3. Core drawing logic (common for both modes, renders to active target)
        original_view_matrix = self.context_window.view
        # Apply translation based on tile offsets (centers the view)
        translation_matrix = Mat4.from_translation(Vec3(self.tile_x_offset, self.tile_y_offset, 0))
        # Assuming the original view was identity or should be overwritten for this scene's base view
        self.context_window.view = translation_matrix 
        if self.tile_renderer:
            self.tile_renderer.draw_tiles(init=init) 
        
        if self.entity_renderer:
            self.entity_renderer.draw_entities(init=init)

        self.context_window.view = original_view_matrix

        # Draw debug overlay (on top, without the main game world's translation)
        if self.enable_debug_overlay and self.debug_overlay_renderer and debug_info:
            self.debug_overlay_renderer.draw_debug_overlay(debug_info)
        
        # 4. Finalize and retrieve data if rgb_array
        if self.render_mode == 'rgb_array':
            if not self.fbo or not self.render_texture: 
                 raise RuntimeError("FBO not available for pixel retrieval.")
            
            pyglet.gl.glFlush() # Ensure all drawing commands are processed

            image_data = self.render_texture.get_image_data()
            self.fbo.unbind() # Unbind FBO

            buffer = image_data.get_data('RGBA', image_data.width * 4) 
            array = np.frombuffer(buffer, dtype=np.uint8).reshape((image_data.height, image_data.width, 4))
            
            return np.flipud(array) # Flip Y axis as OpenGL renders bottom-up into texture

        return None # For 'human' mode

        # pygame.display.flip() is not needed. Pyglet's app loop handles buffer swaps.
        
        # For 'human' mode, we don't return a surface/image.
        # If 'rgb_array' was implemented, here we would grab the buffer.
        if self.render_mode == 'rgb_array':
            # buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            # image_data = buffer.get_image_data()
            # data = np.frombuffer(image_data.get_data('RGB', image_data.width * 3), dtype=np.uint8)
            # data = data.reshape((image_data.height, image_data.width, 3))[::-1, :, :] # Flip Y
            # return data
            pass # Not implemented yet

        return None # Or indicate success/failure if needed

    def draw_collision_map(self, init: bool) -> Optional[np.ndarray]:
        if self.render_mode != 'rgb_array':
            print("Warning: draw_collision_map is intended for 'rgb_array' mode to return an array. Returning None.")
            return None

        self.context_window.switch_to()

        if not self.fbo or not self.render_texture:
            print("Error: FBO not initialized for draw_collision_map in rgb_array mode.")
            return None 

        self.fbo.bind()
        glViewport(0, 0, self.initial_render_width, self.initial_render_height)
        self.context_window.projection = Mat4.orthogonal_projection(0, self.initial_render_width, 0, self.initial_render_height, -1, 1)
        # self.context_window.view = Mat4() # Base view for FBO drawing, translation will be applied

        original_clear_color_values = pyglet.gl.glGetFloatv(pyglet.gl.GL_COLOR_CLEAR_VALUE)
        
        bg_r, bg_g, bg_b = render_utils.hex2float(render_utils.BASIC_BG_COLOR)
        glClearColor(bg_r, bg_g, bg_b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self._update_screen_size_and_offsets() 
        
        original_view_matrix = self.context_window.view 
        translation_matrix = Mat4.from_translation(Vec3(self.tile_x_offset, self.tile_y_offset, 0))
        self.context_window.view = translation_matrix
        
        if self.tile_renderer:
            self.tile_renderer.draw_tiles(init=init, tile_color_override=render_utils.BASIC_TILE_COLOR)
        
        self.context_window.view = original_view_matrix

        pyglet.gl.glFlush()

        image_data = self.render_texture.get_image_data()
        self.fbo.unbind()

        glClearColor(original_clear_color_values[0], original_clear_color_values[1], original_clear_color_values[2], original_clear_color_values[3])

        buffer = image_data.get_data('RGBA', image_data.width * 4) 
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((image_data.height, image_data.width, 4))
        
        return np.flipud(array)

    def _update_screen_size_and_offsets(self):
        # Uses self.context_window, which is a Pyglet window
        screen_width = float(self.context_window.width)
        screen_height = float(self.context_window.height)
        
        # render_utils.SRCWIDTH and SRCHEIGHT are the original design dimensions
        self.adjust = min(screen_width / render_utils.SRCWIDTH, 
                          screen_height / render_utils.SRCHEIGHT)
        
        self.width = render_utils.SRCWIDTH * self.adjust
        self.height = render_utils.SRCHEIGHT * self.adjust
        
        self.tile_x_offset = (screen_width - self.width) / 2.0
        self.tile_y_offset = (screen_height - self.height) / 2.0
        
        # Update sub-renderers with new dimensions/adjustments
        # Their update methods will also need to be checked/refactored.
        self.tile_renderer.update_params(self.adjust)

        self.entity_renderer.update_dimensions(self.adjust, self.width, self.height)
        self.debug_overlay_renderer.update_params(self.adjust, self.tile_x_offset, self.tile_y_offset)


    def close(self):
        """Cleans up resources, like FBO, textures, and internally created windows."""
        # Ensure context is active for GL deletions if needed, though pyglet might handle this.
        if self.context_window:
            try:
                self.context_window.switch_to()
            except Exception as e:
                print(f"Warning: Could not switch to context for NSimRenderer cleanup: {e}")

        if self.fbo:
            try:
                self.fbo.delete()
            except Exception as e:
                print(f"Error deleting FBO: {e}")
            self.fbo = None
        if self.render_texture:
            try:
                self.render_texture.delete()
            except Exception as e:
                print(f"Error deleting render texture: {e}")
            self.render_texture = None
        
        if self.internal_offscreen_window and self.context_window:
            try:
                self.context_window.close()
            except Exception as e:
                print(f"Error closing internal offscreen window: {e}")
            self.context_window = None # Nullify as we closed it
            self.internal_offscreen_window = False # Reset flag
        elif self.internal_offscreen_window: # If flag was true but context_window was already None
             self.internal_offscreen_window = False # Just reset flag
        
        # Call close on sub-renderers if they have such a method
        if hasattr(self.debug_overlay_renderer, 'close') and callable(getattr(self.debug_overlay_renderer, 'close')):
            self.debug_overlay_renderer.close()
        # TileRenderer and EntityRenderer currently don't have explicit close for their sprites/cairo, assumed GC.
