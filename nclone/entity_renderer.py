import cairo
import pygame
import math
from . import render_utils
from .constants.physics_constants import NINJA_RADIUS


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
            self.entitydraw_surface is None
            or self.entitydraw_surface.get_width() != current_screen_width
            or self.entitydraw_surface.get_height() != current_screen_height
            or init
            or self.cached_entity_adjust != self.adjust
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

        # --- Collect all active entities first ---
        active_entities_list = []
        for entity_list_for_type in self.sim.entity_dic.values():
            for entity in entity_list_for_type:
                if entity.active:
                    active_entities_list.append(entity)

        # --- Draw door segments from active door entities ---
        # Door entities are types 5 (Regular), 6 (Locked), 8 (Trap)
        # Their segments are drawn with TILECOLOR_RGB and DOORWIDTH
        context.set_source_rgb(*render_utils.TILECOLOR_RGB)
        context.set_line_width(render_utils.DOORWIDTH * self.adjust)

        door_segments_coords_to_draw = []
        # Draw door segments regardless of the entity's active flag. Some doors
        # (e.g., trap doors) deactivate their switch entity when toggled but the
        # blocking door segment should still be rendered when closed.
        for entity_list_for_type in self.sim.entity_dic.values():
            for entity in entity_list_for_type:
                if entity.type in [5, 6, 8] and hasattr(entity, "segment"):
                    segment = entity.segment
                    # Draw if the segment is active (door is closed) and it's a
                    # linear, non-oriented segment (criteria for door visuals)
                    if (
                        segment.active
                        and segment.type == "linear"
                        and not segment.oriented
                    ):
                        door_segments_coords_to_draw.append(
                            (
                                segment.x1 * self.adjust,
                                segment.y1 * self.adjust,
                                segment.x2 * self.adjust,
                                segment.y2 * self.adjust,
                            )
                        )

        if door_segments_coords_to_draw:
            for x1, y1, x2, y2 in door_segments_coords_to_draw:
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()

        # --- Group other active entities for drawing ---
        entity_groups = {}
        for entity in active_entities_list:
            # Skip regular doors (type 5) since their visual representation is only the segment.
            # For locked (6) and trap (8) doors we still draw the switch entity as a circle.
            if entity.type in [5]:
                continue

            # A mine that has gone off is spent -- the ninja is dead or a
            # ragdoll limb set it off. It stays in the sim (the flag is purely
            # cosmetic) but there is nothing left to draw.
            if getattr(entity, "detonated", False):
                continue

            if entity.type not in entity_groups:
                entity_groups[entity.type] = []
            entity_groups[entity.type].append(entity)

        # --- Draw grouped entities (excluding doors) ---
        context.set_line_width(
            render_utils.PLATFORMWIDTH * self.adjust
        )  # Default line width for other entities
        for entity_type, entities_in_group in entity_groups.items():
            # Set color based on entity_type
            if entity_type in render_utils.ENTITYCOLORS_RGB:
                context.set_source_rgb(*render_utils.ENTITYCOLORS_RGB[entity_type])
            else:
                context.set_source_rgb(
                    0, 0, 0
                )  # Default to black if color not specified

            for entity in entities_in_group:
                x = entity.xpos * self.adjust

                y = entity.ypos * self.adjust

                # Special coloring for EntityExit (type 3) when its switch is hit
                if (
                    entity.type == 3
                    and hasattr(entity, "switch_hit")
                    and entity.switch_hit
                ):
                    context.set_source_rgb(0, 0, 0.5)  # Dark Blue
                elif (
                    entity_type in render_utils.ENTITYCOLORS_RGB
                ):  # Reset to default color
                    context.set_source_rgb(*render_utils.ENTITYCOLORS_RGB[entity_type])
                else:  # Reset to default black for types not in ENTITYCOLORS_RGB
                    context.set_source_rgb(0, 0, 0)

                # Draw entity based on its properties/type
                if hasattr(entity, "normal_x") and hasattr(entity, "normal_y"):
                    self._draw_oriented_entity(context, entity, x, y)
                elif entity.type != 23:  # Type 23 is only its beam, no body
                    self._draw_physical_entity(context, entity, x, y)

                if self._beam_is_live(entity):
                    self._draw_beam(context, entity, x, y)

                # Neither of these has a RADIUS or a SEMI_SIDE -- both bodies are
                # harmless, so the sim never needed one -- which meant
                # `_draw_physical_entity` was a no-op and they were invisible.
                if entity.type == 19:
                    self._draw_turret(context, entity, x, y)
                elif entity.type == 18:
                    self._draw_rocket(context, entity, x, y)

        context.set_source_rgb(*render_utils.NINJACOLOR_RGB)
        context.set_line_width(render_utils.NINJAWIDTH * self.adjust)
        context.set_line_cap(cairo.LineCap.ROUND)
        self._draw_ninja(context)

        buffer = self.entitydraw_surface.get_data()
        return pygame.image.frombuffer(
            buffer,
            (self.entitydraw_surface.get_width(), self.entitydraw_surface.get_height()),
            "BGRA",
        )

    def _draw_oriented_entity(self, context, entity, x, y):
        """Helper method to draw oriented entities"""
        radius = 5
        if hasattr(entity, "RADIUS"):
            radius = entity.RADIUS * self.adjust
        if hasattr(entity, "SEMI_SIDE"):
            radius = entity.SEMI_SIDE * self.adjust
        angle = math.atan2(entity.normal_x, entity.normal_y) + render_utils.PI_DIV_2
        context.move_to(x + math.sin(angle) * radius, y + math.cos(angle) * radius)
        context.line_to(x - math.sin(angle) * radius, y - math.cos(angle) * radius)
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

    @staticmethod
    def _beam_is_live(entity):
        """Does this entity have a beam to draw from its body to `xend`/`yend`?

        The dual laser (23) always does -- the beam *is* its visual, which is why
        it draws no body at all. The laser drone (13) only has one during its
        discharge: `_discharge_tick` recasts `xend`/`yend` every frame of the 80
        it fires for, and leaves the last cast standing through the cooldown and
        the next charge, so drawing it unconditionally would leave a stale beam
        hanging off an idle drone for most of the level.

        The chaingun (12) shares the same fields and is deliberately excluded:
        its discharge is twenty discrete shots swept across an arc, not a
        sustained beam, so a line to `xend` would draw something the game never
        shows.
        """
        if entity.type == 23:
            return True
        if entity.type == 13:
            return getattr(entity, "state", None) == entity.FIRING
        return False

    def _draw_beam(self, context, entity, x, y):
        """Draw the beam from an entity's body to where its raycast stopped."""
        context.set_line_width(1)
        context.move_to(x, y)
        context.line_to(entity.xend * self.adjust, entity.yend * self.adjust)
        context.stroke()

    #: Body radii for the two entities whose bodies are harmless and therefore
    #: carry no sim radius at all. **Render-only** -- nothing reads these back,
    #: and the decompiled render functions are sprite-sheet UVs and rotations
    #: with no hitbox in them, so there is no "true" value to copy. Sized to sit
    #: with the drone's 7.5 in this schematic view.
    TURRET_DRAW_RADIUS = 8.0
    ROCKET_DRAW_RADIUS = 8.0
    #: The projectile's *collision* radius really is 0 -- the kill test uses the
    #: ninja's 10 px alone (`entity_projectile.py`) -- so it needs its own
    #: drawing size or it would be invisible.
    ROCKET_PROJECTILE_DRAW_RADIUS = 3.0
    TURRET_RETICULE_RADIUS = 4.0

    def _draw_turret(self, context, entity, x, y):
        """Body, aim reticule, and the shot during its muzzle window.

        The reticule is `aim_x`/`aim_y`, the turret's own eased aim point
        (`+0x80`/`+0x88`) -- the thing it converges onto the ninja over several
        frames and then FREEZES while the shot winds up. Drawing it is what makes
        a turret's behaviour legible: you can see it swing out from the barrel,
        settle, and stop moving a moment before it fires.

        The shot is drawn for the `MUZZLE_TIMER` frames the turret spends
        RECOVERING, which is what `timer` (`+0xF0`) counts down and is the same
        window the game shows its muzzle flash for. `xend`/`yend` are the traced
        beam `_fire` actually tested against the ninja, not a redraw of the aim.
        """
        r = self.TURRET_DRAW_RADIUS * self.adjust
        context.arc(x, y, r, 0, 2 * math.pi)
        context.fill()

        aim_x = getattr(entity, "aim_x", None)
        aim_y = getattr(entity, "aim_y", None)
        state = getattr(entity, "state", 0)
        if aim_x is not None and state != 0:  # 0 == SEARCHING, no target
            ax, ay = aim_x * self.adjust, aim_y * self.adjust
            rr = self.TURRET_RETICULE_RADIUS * self.adjust
            context.set_line_width(1)
            # Sight line, then the reticule ring and its four ticks.
            context.move_to(x, y)
            context.line_to(ax, ay)
            context.stroke()
            context.arc(ax, ay, rr, 0, 2 * math.pi)
            context.stroke()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                context.move_to(ax + dx * rr, ay + dy * rr)
                context.line_to(ax + dx * rr * 1.8, ay + dy * rr * 1.8)
                context.stroke()

        if getattr(entity, "timer", 0) > 0:
            context.set_line_width(2)
            context.move_to(x, y)
            context.line_to(entity.xend * self.adjust, entity.yend * self.adjust)
            context.stroke()

    def _draw_rocket(self, context, entity, x, y):
        """Launcher body, plus the in-flight projectile.

        The projectile lives inside the launcher struct rather than as its own
        entity (`+0x50`/`+0x58`), so it has no entry in `entity_dic` and nothing
        would ever have drawn it. It is only real while the launcher is FLYING;
        at every other moment those fields hold the last flight's wreckage.
        """
        r = self.ROCKET_DRAW_RADIUS * self.adjust
        context.set_line_width(2)
        context.arc(x, y, r, 0, 2 * math.pi)
        context.stroke()

        proj = getattr(entity, "projectile", None)
        if proj is None or getattr(entity, "state", 0) != 2:  # 2 == FLYING
            return
        px, py = proj.xpos * self.adjust, proj.ypos * self.adjust
        # A short tail along the heading, so the direction reads at a glance.
        context.set_line_width(2)
        context.move_to(px - proj.xdir * r, py - proj.ydir * r)
        context.line_to(px, py)
        context.stroke()
        context.arc(px, py, self.ROCKET_PROJECTILE_DRAW_RADIUS * self.adjust,
                    0, 2 * math.pi)
        context.fill()

    def _draw_ninja(self, context):
        """Helper method to draw ninja"""
        radius = NINJA_RADIUS * self.adjust
        x = self.sim.ninja.xpos * self.adjust
        y = self.sim.ninja.ypos * self.adjust

        if self.sim.sim_config.enable_anim:
            ragdoll = self._active_ragdoll()
            if ragdoll is not None:
                # The corpse owns its own world-space skeleton, so unlike the
                # live path there is nothing to scale or offset by the ninja.
                # Its `segs` are the game's stick table, which is
                # `render_utils.LIMBS` renumbered -- and after an explosion it
                # is the 17-bone table, so read it off the ragdoll rather than
                # assuming LIMBS.
                bones = ragdoll.bones_pos
                for a, b in ragdoll.segs:
                    context.move_to(bones[a][0] * self.adjust, bones[a][1] * self.adjust)
                    context.line_to(bones[b][0] * self.adjust, bones[b][1] * self.adjust)
                context.stroke()
                return

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

    def _active_ragdoll(self):
        """The ragdoll to draw instead of the live skeleton, or None.

        State 6 is the only state that runs ragdoll physics, so it is the only
        one where `bones_pos` holds anything meaningful.
        """
        ninja = self.sim.ninja
        if ninja.state != 6:
            return None
        ragdoll = getattr(ninja, "ragdoll", None)
        if ragdoll is None or not getattr(ragdoll, "active", False):
            return None
        return ragdoll
