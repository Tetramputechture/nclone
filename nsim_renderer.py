import cairo
import math
import pygame
import numpy as np
from typing import Literal

SRCWIDTH = 1056
SRCHEIGHT = 600

BGCOLOR = "cbcad0"
TILECOLOR = "797988"
NINJACOLOR = "000000"
ENTITYCOLORS = {1: "9E2126", 2: "DBE149", 3: "838384", 4: "6D97C3", 5: "000000", 6: "000000",
                7: "000000", 8: "000000", 9: "000000", 10: "868793", 11: "666666", 12: "000000",
                13: "000000", 14: "6EC9E0", 15: "6EC9E0", 16: "000000", 17: "E3E3E5", 18: "000000",
                19: "000000", 20: "838384", 21: "9E2126", 22: "000000", 23: "000000", 24: "666666",
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
    def __init__(self, sim, render_mode: str = 'rgb_array'):
        self.sim = sim
        self.screen = pygame.display.set_mode(
            (SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE)

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

        # Pre-calculate common values
        self.pi_div_2 = math.pi / 2
        self.tile_paths = {}  # Cache for tile rendering paths

    def draw(self, init: bool) -> pygame.Surface:
        self._update_screen_size()
        self._update_tile_offsets()
        self.screen.fill("#"+TILECOLOR)
        self.screen.blit(self._draw_entities(
            init), (self.tile_x_offset, self.tile_y_offset))
        self.screen.blit(self._draw_tiles(
            init), (self.tile_x_offset, self.tile_y_offset))
        pygame.draw.rect(self.screen, "#"+TILECOLOR, (self.tile_x_offset,
                         self.tile_y_offset, self.width, self.height), 24)
        if self.render_mode == 'human':
            pygame.display.flip()

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

        self.render_context.set_operator(cairo.Operator.CLEAR)
        self.render_context.rectangle(0, 0, self.width, self.height)
        self.render_context.fill()
        self.render_context.set_operator(cairo.Operator.ADD)

        # Use pre-calculated RGB values
        if tile_color == TILECOLOR:
            self.render_context.set_source_rgb(*TILECOLOR_RGB)
        else:
            self.render_context.set_source_rgb(*hex2float(tile_color))

        # Group tiles by type for batch rendering
        tile_groups = {}
        for coords, tile in self.sim.tile_dic.items():
            if tile not in tile_groups:
                tile_groups[tile] = []
            tile_groups[tile].append(coords)

        # Batch render similar tiles
        for tile_type, coords_list in tile_groups.items():
            if tile_type == 1 or tile_type > 33:
                # Batch render rectangles
                for x, y in coords_list:
                    self.render_context.rectangle(x * tilesize, y * tilesize,
                                                  tilesize, tilesize)
                self.render_context.fill()
            else:
                self._draw_tile_group(tile_type, coords_list, tilesize)

        buffer = self.render_surface.get_data()
        return pygame.image.frombuffer(buffer, self.screen.get_size(), "BGRA")

    def _draw_tile_group(self, tile_type, coords_list, tilesize):
        """Helper method to batch render similar tiles"""
        if tile_type < 6:
            # Simple rectangles with offsets
            dx = tilesize/2 if tile_type == 3 else 0
            dy = tilesize/2 if tile_type == 4 else 0
            w = tilesize if tile_type % 2 == 0 else tilesize/2
            h = tilesize/2 if tile_type % 2 == 0 else tilesize
            for x, y in coords_list:
                self.render_context.rectangle(x * tilesize + dx,
                                              y * tilesize + dy, w, h)
            self.render_context.fill()
        else:
            # More complex shapes - process in batches
            path_key = f"tile_{tile_type}"
            if path_key not in self.tile_paths:
                self.tile_paths[path_key] = self._create_tile_path(
                    tile_type, tilesize)

            for x, y in coords_list:
                self.render_context.save()
                self.render_context.translate(x * tilesize, y * tilesize)
                self.render_context.append_path(self.tile_paths[path_key])
                self.render_context.fill()
                self.render_context.restore()

    def _create_tile_path(self, tile_type, tilesize):
        """Create and cache complex tile paths"""
        path = cairo.Path()
        # ... existing tile path creation logic ...
        return path

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
                elif not hasattr(entity, "orientation") or entity.is_physical_collidable:
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
        bones = self.sim.ninja.bones
        segments = [[bones[limb[0]], bones[limb[1]]] for limb in LIMBS]
        radius = self.sim.ninja.RADIUS*self.adjust
        x = self.sim.ninja.xpos*self.adjust
        y = self.sim.ninja.ypos*self.adjust

        # Batch draw ninja segments
        for segment in segments:
            x1 = segment[0][0]*2*radius + x
            y1 = segment[0][1]*2*radius + y
            x2 = segment[1][0]*2*radius + x
            y2 = segment[1][1]*2*radius + y
            context.move_to(x1, y1)
            context.line_to(x2, y2)
        context.stroke()
