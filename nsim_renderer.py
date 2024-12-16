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

SEGMENTWIDTH = 1
NINJAWIDTH = 1.25
DOORWIDTH = 2
PLATFORMWIDTH = 3

LIMBS = ((0, 12), (1, 12), (2, 8), (3, 9), (4, 10),
         (5, 11), (6, 7), (8, 0), (9, 0), (10, 1), (11, 1))


def hex2float(string):
    value = int(string, 16)
    red = ((value & 0xFF0000) >> 16) / 255
    green = ((value & 0x00FF00) >> 8) / 255
    blue = (value & 0x0000FF) / 255
    return red, green, blue


class NSimRenderer:
    def __init__(self, sim):
        self.sim = sim
        self.screen = pygame.display.set_mode(
            (SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE)

        self.adjust = 1
        self.width = SRCWIDTH
        self.height = SRCHEIGHT
        self.tile_x_offset = 0
        self.tile_y_offset = 0
        self.render_surface = None
        self.render_context = None
        self.entitydraw_surface = None
        self.entitydraw_context = None

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
        pygame.display.flip()

        return self.screen

    def _update_screen_size(self):
        self.adjust = min(self.screen.get_width()/SRCWIDTH,
                          self.screen.get_height()/SRCHEIGHT)
        self.width = SRCWIDTH*self.adjust
        self.height = SRCHEIGHT*self.adjust

    def _update_tile_offsets(self):
        self.tile_x_offset = (self.screen.get_width() - self.width)/2
        self.tile_y_offset = (self.screen.get_height() - self.height)/2

    def _draw_tiles(self, init: bool) -> pygame.Surface:
        if init:
            self.render_surface = cairo.ImageSurface(
                cairo.Format.RGB24, *self.screen.get_size())
            self.render_context = cairo.Context(self.render_surface)

        tilesize = 24*self.adjust

        self.render_context.set_operator(cairo.Operator.CLEAR)
        self.render_context.rectangle(0, 0, self.width, self.height)
        self.render_context.fill()
        self.render_context.set_operator(cairo.Operator.ADD)

        self.render_context.set_source_rgb(*hex2float(TILECOLOR))
        for coords, tile in self.sim.tile_dic.items():
            x, y = coords
            if tile == 1 or tile > 33:
                self.render_context.rectangle(x * tilesize, y * tilesize,
                                              tilesize, tilesize)
            elif tile > 1:
                if tile < 6:
                    dx = tilesize/2 if tile == 3 else 0
                    dy = tilesize/2 if tile == 4 else 0
                    w = tilesize if tile % 2 == 0 else tilesize/2
                    h = tilesize/2 if tile % 2 == 0 else tilesize
                    self.render_context.rectangle(x * tilesize + dx,
                                                  y * tilesize + dy, w, h)
                elif tile < 10:
                    dx1 = 0
                    dy1 = tilesize if tile == 8 else 0
                    dx2 = 0 if tile == 9 else tilesize
                    dy2 = tilesize if tile == 9 else 0
                    dx3 = 0 if tile == 6 else tilesize
                    dy3 = tilesize
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.line_to(
                        x * tilesize + dx2, y * tilesize + dy2)
                    self.render_context.line_to(
                        x * tilesize + dx3, y * tilesize + dy3)
                elif tile < 14:
                    dx = tilesize if (tile == 11 or tile == 12) else 0
                    dy = tilesize if (tile == 12 or tile == 13) else 0
                    a1 = (math.pi / 2) * (tile - 10)
                    a2 = (math.pi / 2) * (tile - 9)
                    self.render_context.move_to(
                        x * tilesize + dx, y * tilesize + dy)
                    self.render_context.arc(x * tilesize + dx, y *
                                            tilesize + dy, tilesize, a1, a2)
                    self.render_context.line_to(
                        x * tilesize + dx, y * tilesize + dy)
                elif tile < 18:
                    dx1 = tilesize if (tile == 15 or tile == 16) else 0
                    dy1 = tilesize if (tile == 16 or tile == 17) else 0
                    dx2 = tilesize if (tile == 14 or tile == 17) else 0
                    dy2 = tilesize if (tile == 14 or tile == 15) else 0
                    a1 = math.pi + (math.pi / 2) * (tile - 10)
                    a2 = math.pi + (math.pi / 2) * (tile - 9)
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.arc(x * tilesize + dx2, y *
                                            tilesize + dy2, tilesize, a1, a2)
                    self.render_context.line_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                elif tile < 22:
                    dx1 = 0
                    dy1 = tilesize if (tile == 20 or tile == 21) else 0
                    dx2 = tilesize
                    dy2 = tilesize if (tile == 20 or tile == 21) else 0
                    dx3 = tilesize if (tile == 19 or tile == 20) else 0
                    dy3 = tilesize/2
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.line_to(
                        x * tilesize + dx2, y * tilesize + dy2)
                    self.render_context.line_to(
                        x * tilesize + dx3, y * tilesize + dy3)
                elif tile < 26:
                    dx1 = 0
                    dy1 = tilesize/2 if (tile == 23 or tile == 24) else 0
                    dx2 = 0 if tile == 23 else tilesize
                    dy2 = tilesize/2 if tile == 25 else 0
                    dx3 = tilesize
                    dy3 = (tilesize/2 if tile ==
                           22 else 0) if tile < 24 else tilesize
                    dx4 = tilesize if tile == 23 else 0
                    dy4 = tilesize
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.line_to(
                        x * tilesize + dx2, y * tilesize + dy2)
                    self.render_context.line_to(
                        x * tilesize + dx3, y * tilesize + dy3)
                    self.render_context.line_to(
                        x * tilesize + dx4, y * tilesize + dy4)
                elif tile < 30:
                    dx1 = tilesize/2
                    dy1 = tilesize if (tile == 28 or tile == 29) else 0
                    dx2 = tilesize if (tile == 27 or tile == 28) else 0
                    dy2 = 0
                    dx3 = tilesize if (tile == 27 or tile == 28) else 0
                    dy3 = tilesize
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.line_to(
                        x * tilesize + dx2, y * tilesize + dy2)
                    self.render_context.line_to(
                        x * tilesize + dx3, y * tilesize + dy3)
                elif tile < 34:
                    dx1 = tilesize/2
                    dy1 = tilesize if (tile == 30 or tile == 31) else 0
                    dx2 = tilesize if (tile == 31 or tile == 33) else 0
                    dy2 = tilesize
                    dx3 = tilesize if (tile == 31 or tile == 32) else 0
                    dy3 = tilesize if (tile == 32 or tile == 33) else 0
                    dx4 = tilesize if (tile == 30 or tile == 32) else 0
                    dy4 = 0
                    self.render_context.move_to(
                        x * tilesize + dx1, y * tilesize + dy1)
                    self.render_context.line_to(
                        x * tilesize + dx2, y * tilesize + dy2)
                    self.render_context.line_to(
                        x * tilesize + dx3, y * tilesize + dy3)
                    self.render_context.line_to(
                        x * tilesize + dx4, y * tilesize + dy4)
            self.render_context.fill()

        buffer = self.render_surface.get_data()
        return pygame.image.frombuffer(buffer, self.screen.get_size(), "BGRA")

    def _draw_entities(self, init):
        if init:
            self.entitydraw_surface = cairo.ImageSurface(
                cairo.Format.RGB24, *self.screen.get_size())
            self.entitydraw_context = cairo.Context(self.entitydraw_surface)
        context = self.entitydraw_context

        context.set_source_rgb(*hex2float(BGCOLOR))
        context.rectangle(0, 0, self.width, self.height)
        context.fill()

        context.set_source_rgb(*hex2float(TILECOLOR))
        context.set_line_width(DOORWIDTH*self.adjust)
        for cell in self.sim.segment_dic.values():
            for segment in cell:
                if segment.active and segment.type == "linear" and not segment.oriented:
                    context.move_to(segment.x1*self.adjust,
                                    segment.y1*self.adjust)
                    context.line_to(segment.x2*self.adjust,
                                    segment.y2*self.adjust)
            context.stroke()

        context.set_line_width(PLATFORMWIDTH*self.adjust)
        for entity in sum(self.sim.entity_dic.values(), []):
            if entity.active:
                context.set_source_rgb(*hex2float(ENTITYCOLORS[entity.type]))
                x = entity.xpos*self.adjust
                y = entity.ypos*self.adjust
                if hasattr(entity, "normal_x") and hasattr(entity, "normal_y"):
                    if hasattr(entity, "RADIUS"):
                        radius = entity.RADIUS*self.adjust
                    if hasattr(entity, "SEMI_SIDE"):
                        radius = entity.SEMI_SIDE*self.adjust
                    angle = math.atan2(
                        entity.normal_x, entity.normal_y) + math.pi / 2
                    context.move_to(x + math.sin(angle) * radius,
                                    y + math.cos(angle) * radius)
                    context.line_to(x - math.sin(angle) * radius,
                                    y - math.cos(angle) * radius)
                    context.stroke()
                elif not hasattr(entity, "orientation") or entity.is_physical_collidable:
                    if hasattr(entity, "RADIUS"):
                        radius = entity.RADIUS*self.adjust
                        context.arc(x, y, radius, 0, 2 * math.pi)
                        context.fill()
                    elif hasattr(entity, "SEMI_SIDE"):
                        radius = entity.SEMI_SIDE*self.adjust
                        context.rectangle(x - radius, y - radius,
                                          radius * 2, radius * 2)
                        context.fill()
                if entity.type == 23:
                    context.set_line_width(1)
                    context.move_to(x, y)
                    context.line_to(entity.xend*self.adjust,
                                    entity.yend*self.adjust)
                    context.stroke()

        context.set_source_rgb(*hex2float(NINJACOLOR))
        context.set_line_width(NINJAWIDTH*self.adjust)
        context.set_line_cap(cairo.LineCap.ROUND)
        bones = self.sim.ninja.bones
        segments = [[bones[limb[0]], bones[limb[1]]] for limb in LIMBS]
        radius = self.sim.ninja.RADIUS*self.adjust
        x = self.sim.ninja.xpos*self.adjust
        y = self.sim.ninja.ypos*self.adjust
        for segment in segments:
            x1 = segment[0][0]*2*radius + x
            y1 = segment[0][1]*2*radius + y
            x2 = segment[1][0]*2*radius + x
            y2 = segment[1][1]*2*radius + y
            context.move_to(x1, y1)
            context.line_to(x2, y2)
            context.stroke()

        buffer = self.entitydraw_surface.get_data()
        return pygame.image.frombuffer(buffer, self.screen.get_size(), "BGRA")
