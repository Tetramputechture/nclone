import cairo
import pyglet # Added
from pyglet.image import ImageData # Added
import math
from . import render_utils
from typing import Optional # Added for type hinting

class EntityRenderer:
    def __init__(self, sim, window, adjust, width, height): # screen -> window
        self.sim = sim
        self.window = window # Store pyglet window
        self.adjust = adjust
        self.width = width
        self.height = height
        self.entitydraw_surface: Optional[cairo.ImageSurface] = None
        self.entitydraw_context: Optional[cairo.Context] = None
        self.entity_sprite: Optional[pyglet.sprite.Sprite] = None # For Pyglet
        self.cached_entity_adjust = None

    def update_dimensions(self, adjust, width, height):
        self.adjust = adjust
        self.width = width
        self.height = height

    def draw_entities(self, init: bool): # Return type None
        """Draws all entities onto a Cairo surface, then to a Pyglet sprite.
        The Cairo surface is cleared and redrawn each frame.

        Args:
            init: Boolean indicating if a full re-initialization of resources is needed.
        """
        # Use window dimensions for the Cairo surface.
        target_cairo_width = int(self.window.width)
        target_cairo_height = int(self.window.height)

        recreate_cairo_surface = (
            self.entitydraw_surface is None or
            self.entitydraw_surface.get_width() != target_cairo_width or
            self.entitydraw_surface.get_height() != target_cairo_height or
            init or # If init is true, force recreate
            self.cached_entity_adjust != self.adjust # If zoom changed
        )

        if recreate_cairo_surface:
            self.entitydraw_surface = cairo.ImageSurface(
                cairo.Format.ARGB32, target_cairo_width, target_cairo_height
            )
            self.entitydraw_context = cairo.Context(self.entitydraw_surface)
            self.cached_entity_adjust = self.adjust # Cache adjust factor used for this surface
        
        assert self.entitydraw_context is not None, "Cairo context not initialized for entities"

        # Clear the Cairo surface (transparent)
        self.entitydraw_context.set_operator(cairo.Operator.CLEAR)
        self.entitydraw_context.paint()
        # Set operator for drawing new content
        self.entitydraw_context.set_operator(cairo.Operator.OVER) # Use OVER for alpha blending
        
        context = self.entitydraw_context # Use the instance variable directly

        # --- Start of existing Cairo drawing logic (mostly unchanged) ---
        context.set_source_rgb(*render_utils.TILECOLOR_RGB) # This seems to be for doors/segments
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
                 # Fallback color if not defined (e.g. black)
                 context.set_source_rgb(0.0, 0.0, 0.0) 

            for entity in entities:
                x = entity.xpos * self.adjust
                y = entity.ypos * self.adjust
                if hasattr(entity, "normal_x") and hasattr(entity, "normal_y"):
                    self._draw_oriented_entity(context, entity, x, y)
                elif entity.type != 23: # Type 23 has its own drawing logic
                    self._draw_physical_entity(context, entity, x, y)
                
                if entity.type == 23: # Specific drawing for type 23 (e.g. lasers)
                    self._draw_type_23_entity(context, entity, x, y)

        # Draw Ninja
        context.set_source_rgb(*render_utils.NINJACOLOR_RGB)
        context.set_line_width(render_utils.NINJAWIDTH * self.adjust)
        context.set_line_cap(cairo.LineCap.ROUND) # Keep ROUND for smoother limbs
        self._draw_ninja(context)
        # --- End of existing Cairo drawing logic ---

        # Convert Cairo surface to Pyglet sprite
        self.entitydraw_surface.flush() # Ensure all drawing is done
        buffer = self.entitydraw_surface.get_data()
        width = self.entitydraw_surface.get_width()
        height = self.entitydraw_surface.get_height()
        pitch = self.entitydraw_surface.get_stride()

        image_data = pyglet.image.ImageData(width, height, 'BGRA', bytes(buffer), -pitch)
        
        if self.entity_sprite is None:
            self.entity_sprite = pyglet.sprite.Sprite(img=image_data)
        else:
            self.entity_sprite.image = image_data
        
        # Draw the sprite
        if self.entity_sprite:
            self.entity_sprite.x = 0 # Positioned by NSimRenderer's glTranslate
            self.entity_sprite.y = 0
            self.entity_sprite.draw()
        # No return value

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