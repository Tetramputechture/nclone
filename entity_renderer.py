import cairo
import pygame
import math
from . import render_utils

class EntityRenderer:
    def __init__(self, sim, screen, adjust, width, height):
        self.sim = sim
        self.screen = screen
        self.adjust = adjust
        self.width = width
        self.height = height
        self.entitydraw_surface = None
        self.entitydraw_context = None
        self.cached_entity_adjust = None

    def update_dimensions(self, adjust, width, height):
        self.adjust = adjust
        self.width = width
        self.height = height

    def draw_entities(self, init: bool) -> pygame.Surface:
        """Draws all entities onto a surface, ensuring it's cleared each frame.

        Args:
            init: Boolean indicating if a full re-initialization of resources is needed.

        Returns:
            A pygame.Surface with the rendered entities.
        """
        current_screen_width, current_screen_height = self.screen.get_size()

        recreate_surface = (
            self.entitydraw_surface is None or
            self.entitydraw_surface.get_width() != current_screen_width or
            self.entitydraw_surface.get_height() != current_screen_height or
            init or
            self.cached_entity_adjust != self.adjust
        )

        if recreate_surface:
            self.entitydraw_surface = cairo.ImageSurface(
                cairo.Format.ARGB32, current_screen_width, current_screen_height
            )
            self.entitydraw_context = cairo.Context(self.entitydraw_surface)
            self.cached_entity_adjust = self.adjust

        self.entitydraw_context.set_operator(cairo.Operator.CLEAR)
        self.entitydraw_context.paint()
        self.entitydraw_context.set_operator(cairo.Operator.SOURCE)
        
        context = self.entitydraw_context

        context.set_source_rgb(*render_utils.TILECOLOR_RGB)
        context.set_line_width(render_utils.DOORWIDTH * self.adjust)

        active_segments = []
        for cell in self.sim.segment_dic.values():
            for segment in cell:
                if segment.active and segment.type == "linear" and not segment.oriented:
                    active_segments.append((segment.x1 * self.adjust, segment.y1 * self.adjust,
                                            segment.x2 * self.adjust, segment.y2 * self.adjust))
        if active_segments:
            for x1, y1, x2, y2 in active_segments:
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()

        entity_groups = {}
        for entity_list in self.sim.entity_dic.values():
            for entity in entity_list:
                if entity.active:
                    if entity.type not in entity_groups:
                        entity_groups[entity.type] = []
                    entity_groups[entity.type].append(entity)
        
        context.set_line_width(render_utils.PLATFORMWIDTH * self.adjust)
        for entity_type, entities in entity_groups.items():
            if entity_type in render_utils.ENTITYCOLORS_RGB:
                 context.set_source_rgb(*render_utils.ENTITYCOLORS_RGB[entity_type])
            else:
                 context.set_source_rgb(0,0,0) # Default black

            for entity in entities:
                x = entity.xpos * self.adjust
                y = entity.ypos * self.adjust
                if hasattr(entity, "normal_x") and hasattr(entity, "normal_y"):
                    self._draw_oriented_entity(context, entity, x, y)
                elif entity.type != 23:
                    self._draw_physical_entity(context, entity, x, y)
                
                if entity.type == 23:
                    self._draw_type_23_entity(context, entity, x, y)

        context.set_source_rgb(*render_utils.NINJACOLOR_RGB)
        context.set_line_width(render_utils.NINJAWIDTH * self.adjust)
        context.set_line_cap(cairo.LineCap.ROUND)
        self._draw_ninja(context)

        buffer = self.entitydraw_surface.get_data()
        return pygame.image.frombuffer(buffer, 
                                       (self.entitydraw_surface.get_width(), self.entitydraw_surface.get_height()), 
                                       "BGRA")

    def _draw_oriented_entity(self, context, entity, x, y):
        """Helper method to draw oriented entities"""
        radius = 5
        if hasattr(entity, "RADIUS"):
            radius = entity.RADIUS * self.adjust
        if hasattr(entity, "SEMI_SIDE"):
            radius = entity.SEMI_SIDE * self.adjust
        angle = math.atan2(entity.normal_x, entity.normal_y) + render_utils.PI_DIV_2
        context.move_to(x + math.sin(angle) * radius,
                        y + math.cos(angle) * radius)
        context.line_to(x - math.sin(angle) * radius,
                        y - math.cos(angle) * radius)
        context.stroke()

    def _draw_physical_entity(self, context, entity, x, y):
        """Helper method to draw physical entities"""
        if hasattr(entity, "RADIUS"):
            radius = entity.RADIUS * self.adjust
            context.arc(x, y, radius, 0, 2 * math.pi)
            context.fill()
        elif hasattr(entity, "SEMI_SIDE"):
            radius = entity.SEMI_SIDE * self.adjust
            context.rectangle(x - radius, y - radius, radius * 2, radius * 2)
            context.fill()

    def _draw_type_23_entity(self, context, entity, x, y):
        """Helper method to draw type 23 entities"""
        context.set_line_width(1)
        context.move_to(x, y)
        context.line_to(entity.xend * self.adjust, entity.yend * self.adjust)
        context.stroke()

    def _draw_ninja(self, context):
        """Helper method to draw ninja"""
        radius = self.sim.ninja.RADIUS * self.adjust
        x = self.sim.ninja.xpos * self.adjust
        y = self.sim.ninja.ypos * self.adjust

        if self.sim.sim_config.enable_anim:
            bones = self.sim.ninja.bones
            segments = [[bones[limb[0]], bones[limb[1]]] for limb in render_utils.LIMBS]

            # Batch draw ninja segments
            for segment_data in segments:
                x1 = segment_data[0][0] * 2 * radius + x
                y1 = segment_data[0][1] * 2 * radius + y
                x2 = segment_data[1][0] * 2 * radius + x
                y2 = segment_data[1][1] * 2 * radius + y
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()
        else:
            context.arc(x, y, radius, 0, 2 * math.pi)
            context.fill() 