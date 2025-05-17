import pyglet
from pyglet import shapes
from pyglet import text
from pyglet.graphics import Batch
import numpy as np
from . import render_utils

class DebugOverlayRenderer:
    def __init__(self, window, adjust, tile_x_offset, tile_y_offset):
        self.window = window # Pyglet window
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset
        self.batch = Batch() # Batch for drawing shapes and labels
        self.labels = [] # To keep track of labels to clear/update
        self.shapes = [] # To keep track of shapes to clear/update

        # Pyglet font loading is usually done when creating a Label or by pyglet.font.add_file
        # No explicit global font init like pygame.font.init() is needed.

    def update_params(self, adjust, tile_x_offset, tile_y_offset):
        self.adjust = adjust
        self.tile_x_offset = tile_x_offset
        self.tile_y_offset = tile_y_offset

    def _get_area_color(self, base_color: tuple[int, int, int], index: int, max_index: int, opacity: int = 192) -> tuple[int, int, int, int]:
        """Calculate color based on area index, making it darker as index increases."""
        # Calculate brightness factor (0.3 to 1.0)
        brightness = 1.0 - (0.7 * index / max_index if max_index > 0 else 0)
        return (
            int(base_color[0] * brightness),
            int(base_color[1] * brightness),
            int(base_color[2] * brightness),
            opacity
        )

    def _draw_exploration_grid(self, debug_info: dict): # No return type, draws to batch
        """Draw the exploration grid overlay using Pyglet shapes."""
        if 'exploration' not in debug_info:
            return

        exploration = debug_info['exploration']
        visited_cells = exploration.get('visited_cells')
        visited_4x4 = exploration.get('visited_4x4')
        visited_8x8 = exploration.get('visited_8x8')
        visited_16x16 = exploration.get('visited_16x16')

        if not all(isinstance(arr, np.ndarray) and arr is not None 
                   for arr in [visited_cells, visited_4x4, visited_8x8, visited_16x16]):
            return

        cell_size = 24 * self.adjust
        quarter_size = cell_size / 2.0

        for y_idx in range(visited_cells.shape[0]):
            for x_idx in range(visited_cells.shape[1]):
                # Pygame/Cairo X,Y (top-left of cell)
                pygame_cell_top_left_x = x_idx * cell_size + self.tile_x_offset
                pygame_cell_top_left_y = y_idx * cell_size + self.tile_y_offset

                # Pyglet Y for bottom of cell (Pyglet is bottom-up)
                pyglet_cell_bottom_y = self.window.height - (pygame_cell_top_left_y + cell_size)

                if visited_cells[y_idx, x_idx]:
                    # Top-left quarter (in Pygame/Cairo sense)
                    self.shapes.append(shapes.Rectangle(
                        pygame_cell_top_left_x, pyglet_cell_bottom_y + quarter_size, 
                        quarter_size, quarter_size,
                        color=render_utils.EXPLORATION_COLORS['cell_visited'], batch=self.batch))

                    # Top-right quarter
                    area_4x4_x, area_4x4_y = x_idx // 4, y_idx // 4
                    if 0 <= area_4x4_y < visited_4x4.shape[0] and \
                       0 <= area_4x4_x < visited_4x4.shape[1] and \
                       visited_4x4[area_4x4_y, area_4x4_x]:
                        index_4x4 = area_4x4_y * visited_4x4.shape[1] + area_4x4_x
                        max_index_4x4 = visited_4x4.size -1 if visited_4x4.size > 0 else 0
                        color_4x4 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS['4x4'], index_4x4, max_index_4x4)
                        self.shapes.append(shapes.Rectangle(
                            pygame_cell_top_left_x + quarter_size, pyglet_cell_bottom_y + quarter_size,
                            quarter_size, quarter_size,
                            color=color_4x4, batch=self.batch))

                    # Bottom-left quarter
                    area_8x8_x, area_8x8_y = x_idx // 8, y_idx // 8
                    if 0 <= area_8x8_y < visited_8x8.shape[0] and \
                       0 <= area_8x8_x < visited_8x8.shape[1] and \
                       visited_8x8[area_8x8_y, area_8x8_x]:
                        index_8x8 = area_8x8_y * visited_8x8.shape[1] + area_8x8_x
                        max_index_8x8 = visited_8x8.size - 1 if visited_8x8.size > 0 else 0
                        color_8x8 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS['8x8'], index_8x8, max_index_8x8)
                        self.shapes.append(shapes.Rectangle(
                            pygame_cell_top_left_x, pyglet_cell_bottom_y,
                            quarter_size, quarter_size,
                            color=color_8x8, batch=self.batch))
                    
                    # Bottom-right quarter
                    area_16x16_x, area_16x16_y = x_idx // 16, y_idx // 16
                    if 0 <= area_16x16_y < visited_16x16.shape[0] and \
                       0 <= area_16x16_x < visited_16x16.shape[1] and \
                       visited_16x16[area_16x16_y, area_16x16_x]:
                        index_16x16 = area_16x16_y * visited_16x16.shape[1] + area_16x16_x
                        max_index_16x16 = visited_16x16.size - 1 if visited_16x16.size > 0 else 0
                        color_16x16 = self._get_area_color(
                            render_utils.AREA_BASE_COLORS['16x16'], index_16x16, max_index_16x16)
                        self.shapes.append(shapes.Rectangle(
                            pygame_cell_top_left_x + quarter_size, pyglet_cell_bottom_y,
                            quarter_size, quarter_size,
                            color=color_16x16, batch=self.batch))

                    # Draw cell grid outline (4 lines)
                    grid_line_color = render_utils.EXPLORATION_COLORS['grid_cell']
                    # Top line of cell
                    self.shapes.append(shapes.Line(pygame_cell_top_left_x, pyglet_cell_bottom_y + cell_size, 
                                                   pygame_cell_top_left_x + cell_size, pyglet_cell_bottom_y + cell_size, 
                                                   color=grid_line_color, batch=self.batch))
                    # Bottom line of cell
                    self.shapes.append(shapes.Line(pygame_cell_top_left_x, pyglet_cell_bottom_y, 
                                                   pygame_cell_top_left_x + cell_size, pyglet_cell_bottom_y, 
                                                   color=grid_line_color, batch=self.batch))
                    # Left line of cell
                    self.shapes.append(shapes.Line(pygame_cell_top_left_x, pyglet_cell_bottom_y, 
                                                   pygame_cell_top_left_x, pyglet_cell_bottom_y + cell_size, 
                                                   color=grid_line_color, batch=self.batch))
                    # Right line of cell
                    self.shapes.append(shapes.Line(pygame_cell_top_left_x + cell_size, pyglet_cell_bottom_y, 
                                                   pygame_cell_top_left_x + cell_size, pyglet_cell_bottom_y + cell_size, 
                                                   color=grid_line_color, batch=self.batch))

    def draw_debug_overlay(self, debug_info: dict = None): # No return type
        """Helper method to draw debug overlay with nested dictionary support using Pyglet."""
        
        # Clear previous shapes and labels from the batch by deleting them
        # This ensures they are removed from the GPU and the batch itself.
        for item in self.shapes + self.labels:
            item.delete() 
        self.shapes.clear()
        self.labels.clear()

        if not debug_info:
            # If there's an existing batch from previous frames that needs drawing (e.g. persistent elements not cleared here)
            # then self.batch.draw() might be needed. But if this method owns all batch content, 
            # and it's cleared, then drawing an empty batch is fine or just returning.
            # For now, assume if no debug_info, nothing new is added, so batch might be empty or hold other things.
            # Let's ensure the batch is drawn if it has content from other sources, or if it's just this overlay.
            # If this is the *only* drawing this renderer does, then an empty batch draw is harmless.
            self.batch.draw() 
            return

        # Draw exploration grid (adds shapes to self.batch)
        self._draw_exploration_grid(debug_info)

        # Font and text settings for Pyglet
        font_name = 'Arial' # A common default font
        font_size = 10 # Smaller font size for debug text
        line_height = 12 # Pixel height per line of text
        base_color = (255, 255, 255, 191)  # White with 75% opacity (RGBA format for Pyglet)

        # --- Text rendering logic ---
        def calc_text_block_height(d: dict, current_level: int = 0) -> int:
            """Calculate total pixel height needed for the text block."""
            h = 0
            for k, v in d.items():
                if k == 'exploration': # This is drawn graphically, not as text here
                    continue
                h += line_height
                if isinstance(v, dict):
                    h += calc_text_block_height(v, current_level + 1)
            return h

        text_block_total_height = line_height # Minimum one line for padding or initial entry
        if debug_info:
            text_block_total_height = calc_text_block_height(debug_info)

        # Pyglet Y coordinates are bottom-up. Text is positioned by its baseline (bottom of text).
        # We want the text block to appear in the bottom-right of the screen.
        # x_pos is the left edge of the text block.
        # y_start_pos is the y-coordinate for the *bottom* line of text if we were drawing top-down.
        # Since we draw bottom-up with Pyglet, this will be the y for the first (lowest) label.
        
        text_block_x_start = self.window.width - 250  # 250px wide block from right edge
        text_block_bottom_padding = 5 # Pixels from bottom of screen
        
        # y_pos_tracker will hold the current y baseline for the next label to be drawn.
        # We start at the bottom and add labels upwards.
        y_pos_tracker = [text_block_bottom_padding] 

        def format_value_for_pyglet(value):
            if isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.3f}"
            elif isinstance(value, tuple) and all(isinstance(x, (int, float, np.float32, np.float64)) for x in value):
                # Format tuple elements and join into a string
                return "(" + ", ".join(str(round(x, 2) if isinstance(x, (float, np.float32, np.float64)) else x) for x in value) + ")"
            elif isinstance(value, np.ndarray):
                return f"Array({value.shape})"
            return str(value) # Ensure all output is string

        def render_dict_recursively(data_dict: dict, indent_level: int = 0):
            """Renders dictionary content. Iterates normally, adds labels from bottom up."""
            indent_string = "  " * indent_level
            # Iterate in reverse to draw from bottom-most item upwards on screen
            for key, value in reversed(list(data_dict.items())):
                if key == 'exploration': # Already handled by _draw_exploration_grid
                    continue

                if isinstance(value, dict):
                    # For a dict, first recursively call to draw its content (which will be above it)
                    render_dict_recursively(value, indent_level + 1)
                    # Then add the label for the key of this dictionary
                    label_text = f"{indent_string}{key}:"
                    label = pyglet.text.Label(label_text, font_name=font_name, font_size=font_size,
                                              color=base_color, x=text_block_x_start, y=y_pos_tracker[0],
                                              batch=self.batch, anchor_y='bottom')
                    self.labels.append(label)
                    y_pos_tracker[0] += line_height # Move y up for the next label
                else:
                    # For a simple key-value pair
                    formatted_val = format_value_for_pyglet(value)
                    label_text = f"{indent_string}{key}: {formatted_val}"
                    label = pyglet.text.Label(label_text, font_name=font_name, font_size=font_size,
                                              color=base_color, x=text_block_x_start, y=y_pos_tracker[0],
                                              batch=self.batch, anchor_y='bottom')
                    self.labels.append(label)
                    y_pos_tracker[0] += line_height # Move y up for the next label
        
        if debug_info:
            render_dict_recursively(debug_info)

        self.batch.draw() # Draw all collected shapes and labels for this frame