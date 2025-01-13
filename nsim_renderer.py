import cairo
import math
import pygame
import numpy as np
from typing import Literal, Optional

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
                (SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE)
        else:
            self.screen = pygame.Surface((SRCWIDTH, SRCHEIGHT))

        self.render_mode = render_mode
        self.adjust = 1
        self.width = SRCWIDTH
        self.height = SRCHEIGHT
        self.tile_x_offset = 0
        self.tile_y_offset = 0
        self.render_surface = None
        self.render_context = None
        self.entitydraw_surface = None
        self.entitydraw_context = None
        self.enable_debug_overlay = enable_debug_overlay

        # Pre-calculate common values
        self.pi_div_2 = math.pi / 2
        self.tile_paths = {}  # Cache for tile rendering paths

    def draw(self, init: bool, debug_info: Optional[dict] = None) -> pygame.Surface:
        self._update_screen_size()
        self._update_tile_offsets()
        self.screen.fill("#"+TILECOLOR)
        self.screen.blit(self._draw_entities(
            init), (self.tile_x_offset, self.tile_y_offset))
        self.screen.blit(self._draw_tiles(
            init), (self.tile_x_offset, self.tile_y_offset))

        if self.enable_debug_overlay:
            self.screen.blit(self._draw_debug_overlay(debug_info), (0, 0))

        if self.render_mode == 'human':
            pygame.display.flip()
            # pygame.draw.rect(self.screen, "#"+TILECOLOR, (self.tile_x_offset,
            #                                               self.tile_y_offset, self.width, self.height), 24)
        return self.screen

    def draw_collision_map(self, init: bool):
        """Draws only the tiles, no entities. The background is white and the tiles are drawn in black."""
        self._update_screen_size()
        self._update_tile_offsets()
        self.screen.fill("#"+BASIC_BG_COLOR)
        self.screen.blit(self._draw_tiles(
            init, tile_color=BASIC_TILE_COLOR), (self.tile_x_offset, self.tile_y_offset))
        # pygame.display.flip()

        return self.screen

    def _update_screen_size(self):
        self.adjust = min(self.screen.get_width()/SRCWIDTH,
                          self.screen.get_height()/SRCHEIGHT)
        self.width = SRCWIDTH*self.adjust
        self.height = SRCHEIGHT*self.adjust

    def _update_tile_offsets(self):
        self.tile_x_offset = (self.screen.get_width() - self.width)/2
        self.tile_y_offset = (self.screen.get_height() - self.height)/2

    def _draw_tiles(self, init: bool, tile_color: str = TILECOLOR) -> pygame.Surface:
        if init:
            self.render_surface = cairo.ImageSurface(
                cairo.Format.RGB24, *self.screen.get_size())
            self.render_context = cairo.Context(self.render_surface)

        tilesize = 24*self.adjust

        # Clear the entire surface first
        self.render_context.set_operator(cairo.Operator.CLEAR)
        self.render_context.paint()  # Use paint() instead of rectangle + fill for full clear
        self.render_context.set_operator(cairo.Operator.SOURCE)

        # Use pre-calculated RGB values
        if tile_color == TILECOLOR:
            self.render_context.set_source_rgb(*TILECOLOR_RGB)
        else:
            self.render_context.set_source_rgb(*hex2float(tile_color))

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
                dx = tilesize/2 if tile_type == 3 else 0
                dy = tilesize/2 if tile_type == 4 else 0
                w = tilesize if tile_type % 2 == 0 else tilesize/2
                h = tilesize/2 if tile_type % 2 == 0 else tilesize
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
        """Draw a complex tile shape"""
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
            dy3 = tilesize/2
            self.render_context.move_to(x * tilesize + dx1, y * tilesize + dy1)
            self.render_context.line_to(x * tilesize + dx2, y * tilesize + dy2)
            self.render_context.line_to(x * tilesize + dx3, y * tilesize + dy3)
            self.render_context.close_path()
            self.render_context.fill()
        elif tile_type < 26:
            dx1 = 0
            dy1 = tilesize/2 if (tile_type == 23 or tile_type == 24) else 0
            dx2 = 0 if tile_type == 23 else tilesize
            dy2 = tilesize/2 if tile_type == 25 else 0
            dx3 = tilesize
            dy3 = (tilesize/2 if tile_type ==
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
            dx1 = tilesize/2
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
            dx1 = tilesize/2
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

    def _draw_entities(self, init):
        if init:
            self.entitydraw_surface = cairo.ImageSurface(
                cairo.Format.RGB24, *self.screen.get_size())
            self.entitydraw_context = cairo.Context(self.entitydraw_surface)
        context = self.entitydraw_context

        # Use pre-calculated RGB values
        context.set_source_rgb(*BGCOLOR_RGB)
        context.rectangle(0, 0, self.width, self.height)
        context.fill()

        # Batch render segments
        context.set_source_rgb(*TILECOLOR_RGB)
        context.set_line_width(DOORWIDTH*self.adjust)

        # Group segments for batch rendering
        active_segments = []
        for cell in self.sim.segment_dic.values():
            for segment in cell:
                if segment.active and segment.type == "linear" and not segment.oriented:
                    active_segments.append((segment.x1*self.adjust, segment.y1*self.adjust,
                                            segment.x2*self.adjust, segment.y2*self.adjust))

        # Batch draw segments
        if active_segments:
            for x1, y1, x2, y2 in active_segments:
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()

        # Group entities by type for batch rendering
        entity_groups = {}
        for entity in sum(self.sim.entity_dic.values(), []):
            if entity.active:
                if entity.type not in entity_groups:
                    entity_groups[entity.type] = []
                entity_groups[entity.type].append(entity)

        # Batch render entities by type
        context.set_line_width(PLATFORMWIDTH*self.adjust)
        for entity_type, entities in entity_groups.items():
            context.set_source_rgb(*ENTITYCOLORS_RGB[entity_type])
            for entity in entities:
                x = entity.xpos*self.adjust
                y = entity.ypos*self.adjust
                if hasattr(entity, "normal_x") and hasattr(entity, "normal_y"):
                    self._draw_oriented_entity(context, entity, x, y)
                elif entity.type != 23:
                    self._draw_physical_entity(context, entity, x, y)
                if entity.type == 23:
                    self._draw_type_23_entity(context, entity, x, y)

        # Draw ninja
        context.set_source_rgb(*NINJACOLOR_RGB)
        context.set_line_width(NINJAWIDTH*self.adjust)
        context.set_line_cap(cairo.LineCap.ROUND)
        self._draw_ninja(context)

        buffer = self.entitydraw_surface.get_data()
        return pygame.image.frombuffer(buffer, self.screen.get_size(), "BGRA")

    def _draw_oriented_entity(self, context, entity, x, y):
        """Helper method to draw oriented entities"""
        if hasattr(entity, "RADIUS"):
            radius = entity.RADIUS*self.adjust
        if hasattr(entity, "SEMI_SIDE"):
            radius = entity.SEMI_SIDE*self.adjust
        angle = math.atan2(entity.normal_x, entity.normal_y) + self.pi_div_2
        context.move_to(x + math.sin(angle) * radius,
                        y + math.cos(angle) * radius)
        context.line_to(x - math.sin(angle) * radius,
                        y - math.cos(angle) * radius)
        context.stroke()

    def _draw_physical_entity(self, context, entity, x, y):
        """Helper method to draw physical entities"""
        if hasattr(entity, "RADIUS"):
            radius = entity.RADIUS*self.adjust
            context.arc(x, y, radius, 0, 2 * math.pi)
            context.fill()
        elif hasattr(entity, "SEMI_SIDE"):
            radius = entity.SEMI_SIDE*self.adjust
            context.rectangle(x - radius, y - radius, radius * 2, radius * 2)
            context.fill()

    def _draw_type_23_entity(self, context, entity, x, y):
        """Helper method to draw type 23 entities"""
        context.set_line_width(1)
        context.move_to(x, y)
        context.line_to(entity.xend*self.adjust, entity.yend*self.adjust)
        context.stroke()

    def _draw_ninja(self, context):
        """Helper method to draw ninja"""
        radius = self.sim.ninja.RADIUS*self.adjust
        x = self.sim.ninja.xpos*self.adjust
        y = self.sim.ninja.ypos*self.adjust

        if self.sim.sim_config.enable_anim:
            bones = self.sim.ninja.bones
            segments = [[bones[limb[0]], bones[limb[1]]] for limb in LIMBS]

            # Batch draw ninja segments
            for segment in segments:
                x1 = segment[0][0]*2*radius + x
                y1 = segment[0][1]*2*radius + y
                x2 = segment[1][0]*2*radius + x
                y2 = segment[1][1]*2*radius + y
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()
        else:
            context.arc(x, y, radius, 0, 2 * math.pi)
            context.fill()

    def _draw_debug_overlay(self, debug_info: Optional[dict] = None):
        """Helper method to draw debug overlay with nested dictionary support.

        Args:
            debug_info: Optional dictionary containing debug information to display

        Returns:
            pygame.Surface: Surface containing the rendered debug text
        """
        # Create a surface for the debug overlay
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # Base font and settings
        font = pygame.font.Font(None, 20)  # Small font size
        line_height = 12  # Reduced from 20 to 16 for tighter spacing
        base_color = (255, 255, 255, 191)  # White with 75% opacity

        # Calculate total height needed for text
        def calc_text_height(d: dict, level: int = 0) -> int:
            height = 0
            for _, value in d.items():
                height += line_height
                if isinstance(value, dict):
                    height += calc_text_height(value, level + 1)
            return height

        total_height = line_height  # For frame number
        if debug_info:
            total_height += calc_text_height(debug_info)

        # Calculate starting position (bottom right)
        x_pos = self.screen.get_width() - 250  # Fixed width from right edge
        y_pos = self.screen.get_height() - total_height - 5  # 10px padding from bottom

        def format_value(value):
            """Format value with rounding for numbers."""
            if isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.3f}"
            elif isinstance(value, tuple) and all(isinstance(x, (int, float, np.float32, np.float64)) for x in value):
                return tuple(round(x, 2) if isinstance(x, (float, np.float32, np.float64)) else x for x in value)
            return value

        def render_dict(d: dict, indent_level: int = 0):
            nonlocal y_pos
            indent = "  " * indent_level

            for key, value in d.items():
                if isinstance(value, dict):
                    # Render dictionary key as a header
                    text = font.render(f"{indent}{key}:", True, base_color)
                    surface.blit(text, (x_pos, y_pos))
                    y_pos += line_height
                    # Recursively render nested dictionary
                    render_dict(value, indent_level + 1)
                else:
                    # Format and render key-value pair
                    formatted_value = format_value(value)
                    text = font.render(
                        f"{indent}{key}: {formatted_value}", True, base_color)
                    surface.blit(text, (x_pos, y_pos))
                    y_pos += line_height

        # Render debug info if provided
        if debug_info:
            render_dict(debug_info)

        return surface
