import logging

from .sim_config import SimConfig
from .map_loader import MapLoader
from .physics import clear_caches
import os

CACHE_CLEAR_INTERVAL = 100

logger = logging.getLogger(__name__)

os.environ["SDL_AUDIODRIVER"] = "dsp"


class Simulator:
    """Main class that handles ninjas, entities and tile geometry for simulation."""

    def __init__(self, sc: SimConfig):
        """Initializes the simulator with a given SimConfig."""
        self.frame = 0
        if sc.debug:
            self.collisionlog = []
        else:
            self.collisionlog = None

        self.sim_config = sc
        self.ninja = None
        self.tile_dic = {}
        self.segment_dic = {}
        self.grid_entity = {}
        self.entity_dic = {}
        self.hor_grid_edge_dic = {}
        self.ver_grid_edge_dic = {}
        self.door_hor_grid_edge_dic = {}
        self.door_ver_grid_edge_dic = {}
        self.hor_segment_dic = {}
        self.ver_segment_dic = {}
        self.map_data = None
        self.map_loader = MapLoader(self)

        # Optional environment reference for cache invalidation
        # Set by gym environment to enable performance optimizations
        self.gym_env = None

        # Cache SimConfig flags to avoid repeated attribute lookups in tick()
        self._enable_anim = sc.enable_anim
        self._log_data = sc.log_data

        # Cache clear optimization - track next frame to clear instead of using modulo
        self._next_cache_clear = CACHE_CLEAR_INTERVAL

        # Action masking attributes (set by environment, used by ninja)
        self._enable_straightness_masking = True
        self._action_masking_mode = "hard"
        self._path_direction_data = None  # Dict with dx, dy, distance

        # OOB debug: suppress repeated warnings per episode; reset in fast_reset()
        self._oob_logged = False

    def load(self, map_data):
        """From the given map data, initiate the level geometry, the entities and the ninja."""
        self.map_data = map_data
        self.reset_map_tile_data()
        self.map_loader.load_map_tiles()
        self.reset()

    def load_from_created(self, created_map):
        """Load a map that was manually constructed using the Map class."""
        self.load(created_map.map_data())

    def get_tile_grid(self):
        """Return the raw 966-element tile data slice from map_data.

        Map data stores tiles at indices [184:1150] in column-major (Fortran)
        order: tile_data[x + y * 42] -> grid[x, y]. Callers reshape this to a
        (42, 23) grid via either `.reshape(42, 23, order='F')` or
        `np.asarray(...).reshape(23, 42).T`.
        """
        if self.map_data is None:
            return []
        return self.map_data[184:1150]

    def reset(self):
        """Reset the simulation to the initial state. Keeps the current map tiles, and resets the ninja,
        entities and the collision log."""
        self.frame = 0
        if self.sim_config.debug:
            self.collisionlog = []
        else:
            self.collisionlog = None
        self.ninja = None
        # Reset cache clear counter when frame is reset
        self._next_cache_clear = CACHE_CLEAR_INTERVAL
        # Reset action masking attributes
        self._path_direction_data = None
        # Reset OOB debug flag
        self._oob_logged = False
        self.reset_map_entity_data()
        self.map_loader.load_map_entities()

    def fast_reset(self):
        """Fast reset for same-level training. Resets entity state without recreation.

        This method is significantly faster than reset() because it:
        1. Resets ninja state in-place (no object allocation)
        2. Resets entity states in-place (no object allocation)
        3. Clears grid_entity lists (no dict recreation)
        4. Skips door segment cleanup (segments unchanged for same level)

        CRITICAL: Must be called ONLY when the map has not changed since the last
        reset. For new maps, use reset() instead.

        Curriculum compatibility: This method resets entities to their ORIGINAL positions.
        The environment must call goal_curriculum_manager.apply_to_simulator() after
        fast_reset() to move entities to curriculum positions.
        """
        # Reset frame counter and collision log
        self.frame = 0
        if self.sim_config.debug:
            if self.collisionlog is not None:
                self.collisionlog.clear()
            else:
                self.collisionlog = []
        else:
            self.collisionlog = None

        # Reset cache clear counter
        self._next_cache_clear = CACHE_CLEAR_INTERVAL

        # Reset action masking attributes
        self._path_direction_data = None

        # Reset OOB debug flag so the first OOB in each episode is logged
        self._oob_logged = False

        # Reset ninja state in-place (no new Ninja object)
        if self.ninja is not None:
            self.ninja.reset_state()
        else:
            # Fallback: If ninja doesn't exist yet, create it
            # This shouldn't happen in normal fast_reset usage
            self.map_loader.load_map_entities()
            return

        # Clear grid_entity lists (no dict recreation - major optimization)
        for cell_list in self.grid_entity.values():
            cell_list.clear()

        # Reset entities in-place and re-add to grid
        # CRITICAL: Exit doors (EntityExit) must NOT be added to grid_entity
        # They are only added when the exit switch is collected
        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                # Reset entity state to original position/state
                if hasattr(entity, "reset_state"):
                    entity.reset_state()
                else:
                    # Fallback for entities without reset_state() method
                    # Just reset active flag
                    if hasattr(entity, "active"):
                        entity.active = True

                # Add to grid_entity, but skip exit doors (type 3, EntityExit class)
                # Exit doors are only added to grid after switch collection
                if type(entity).__name__ != "EntityExit":
                    self.grid_entity[entity.cell].append(entity)

    def reset_map_entity_data(self):
        """Reset the map entity data. This is used when a new map is loaded or when the map is reset.

        IMPORTANT: This method must remove door segments from segment_dic before clearing entity_dic,
        otherwise door segments will persist and be duplicated when entities are reloaded. This is
        critical for curriculum learning where load_map_from_map_data() is followed by reset().
        """
        # Remove door segments from segment_dic before clearing entity_dic
        # Door entities are types 5 (Regular), 6 (Locked), 8 (Trap)
        door_types = [5, 6, 8]
        # Cache segment_dic.items() to avoid repeated dictionary access
        segment_dic_items = list(self.segment_dic.items())

        for door_type in door_types:
            if door_type in self.entity_dic:
                for door_entity in self.entity_dic[door_type]:
                    # Check if entity has a segment attribute (door entities should)
                    if hasattr(door_entity, "segment") and hasattr(
                        door_entity, "grid_edges"
                    ):
                        segment = door_entity.segment
                        grid_edges = door_entity.grid_edges
                        is_vertical = getattr(door_entity, "is_vertical", False)

                        # Find the cell containing this segment and remove it
                        # We need to find which cell contains the segment
                        # Segments are stored by their cell location
                        for cell_key, segments_list in segment_dic_items:
                            if segment in segments_list:
                                segments_list.remove(segment)
                                break

                        # Reset door grid edges that this door was using
                        for grid_edge in grid_edges:
                            if is_vertical:
                                if grid_edge in self.door_ver_grid_edge_dic:
                                    self.door_ver_grid_edge_dic[grid_edge] = max(
                                        0, self.door_ver_grid_edge_dic[grid_edge] - 1
                                    )
                            else:
                                if grid_edge in self.door_hor_grid_edge_dic:
                                    self.door_hor_grid_edge_dic[grid_edge] = max(
                                        0, self.door_hor_grid_edge_dic[grid_edge] - 1
                                    )

        # Now clear entity data structures
        # Use dict comprehension for grid_entity initialization (44x25 = 1,100 entries)
        self.grid_entity = {(x, y): [] for x in range(44) for y in range(25)}
        self.entity_dic = {
            i: [] for i in range(1, 29)
        }  # Initialize with expected entity types

    def reset_map_tile_data(self):
        """Reset the map cell data. This is used when a new map is loaded."""
        # Segment dictionary: 45x26 to match original (allows cells (0..44, 0..25))
        self.segment_dic = {(x, y): [] for x in range(45) for y in range(26)}

        # Initialize tile_dic with full tiles (type 1), matching original.
        # Interior tiles (1..42, 1..23) get overwritten from map data in load_map_tiles.
        # Border tiles remain as type 1 (solid walls at map edges).
        self.tile_dic = {(x, y): 1 for x in range(44) for y in range(25)}

        # Reset collision optimization structures (will be rebuilt after segments are populated)
        self.collision_data = None
        self.spatial_segment_index = None
        self.level_hash = None

        # Tile grid edge dictionaries: borders are solid (1), interior starts at 0.
        # Matches original tile_hor_grid_edge_dic / tile_ver_grid_edge_dic.
        self.hor_grid_edge_dic = {
            (x, y): (1 if y in (0, 50) else 0)
            for x in range(88) for y in range(51)
        }
        self.ver_grid_edge_dic = {
            (x, y): (1 if x in (0, 88) else 0)
            for x in range(89) for y in range(50)
        }

        # Separate door grid edge dictionaries (dynamic, modified when doors open/close)
        self.door_hor_grid_edge_dic = {
            (x, y): 0 for x in range(88) for y in range(51)
        }
        self.door_ver_grid_edge_dic = {
            (x, y): 0 for x in range(89) for y in range(50)
        }

        # Segment accumulator dictionaries: border values pre-initialized.
        # hor_segment_dic: y=0 → 1 (top border, segment pointing down),
        #                  y=50 → -1 (bottom border, segment pointing up)
        # ver_segment_dic: x=0 → 1 (left border, segment pointing right),
        #                  x=88 → -1 (right border, segment pointing left)
        self.hor_segment_dic = {}
        for x in range(88):
            for y in range(51):
                value = 0
                if y == 0:
                    value = 1
                if y == 50:
                    value = -1
                self.hor_segment_dic[(x, y)] = value
        self.ver_segment_dic = {}
        for x in range(89):
            for y in range(50):
                value = 0
                if x == 0:
                    value = 1
                if x == 88:
                    value = -1
                self.ver_segment_dic[(x, y)] = value

    def tick(self, hor_input, jump_input):
        """Gets called every frame to update the whole physics simulation."""
        # Increment the current frame
        self.frame += 1

        # Store inputs as ninja variables
        self.ninja.hor_input = hor_input
        self.ninja.jump_input = jump_input

        # Record position before physics execution for movement detection
        self.ninja.record_action_start_position()

        # Cache active entities to avoid repeated filtering
        # Single-pass classification (optimization - was 3 separate iterations)
        active_movable_entities = []
        active_thinkable_entities = []
        for entity_list in self.entity_dic.values():
            for e in entity_list:
                if e.active:
                    if e.is_movable:
                        active_movable_entities.append(e)
                    if e.is_thinkable:
                        active_thinkable_entities.append(e)

        # Move all movable entities
        for entity in active_movable_entities:
            entity.move()

        # Make all thinkable entities think
        for entity in active_thinkable_entities:
            entity.think()

        if self.ninja.state != 9:  # 9 typically means inactive or similar
            # if dead, apply physics to ragdoll instead.
            ninja_to_update = (
                self.ninja if self.ninja.state != 6 else self.ninja.ragdoll
            )  # 6 for dead state
            if ninja_to_update:  # Ensure there is a ninja or ragdoll to update
                ninja_to_update.integrate()  # Do preliminary speed and position updates.
                ninja_to_update.pre_collision()  # Do pre collision calculations.

                # Cache collision results
                for _ in range(4):  # Number of physics substeps
                    # Handle PHYSICAL collisions with entities.
                    ninja_to_update.collide_vs_objects()
                    # Handle physical collisions with tiles.
                    ninja_to_update.collide_vs_tiles()

                ninja_to_update.post_collision()  # Do post collision calculations.

            # OOB debug: log once per episode when ninja crosses level boundary.
            if not self._oob_logged and self.ninja.state < 6 and self.ninja.ypos > 600:
                from .physics import gather_segments_from_region
                rad = 10
                segs = gather_segments_from_region(
                    self,
                    self.ninja.xpos - rad, self.ninja.ypos - rad,
                    self.ninja.xpos + rad, self.ninja.ypos + rad,
                )
                logger.warning(
                    "OOB in tick: frame=%d pos=(%.2f,%.2f) old=(%.2f,%.2f) "
                    "vel=(%.4f,%.4f) state=%d floor_count=%d segments_nearby=%d",
                    self.frame,
                    self.ninja.xpos, self.ninja.ypos,
                    self.ninja.xpos_old, self.ninja.ypos_old,
                    self.ninja.xspeed, self.ninja.yspeed,
                    self.ninja.state,
                    self.ninja.floor_count,
                    len(segs),
                )
                self._oob_logged = True

            self.ninja.think()  # Make ninja think
            if self._enable_anim:
                self.ninja.update_graphics()  # Update limbs of ninja

        if self.ninja.state == 6 and self._enable_anim:  # Ragdoll state
            self.ninja.anim_frame = 105  # Specific animation frame for ragdoll
            self.ninja.anim_state = 7  # Specific animation state for ragdoll
            self.ninja.calc_ninja_position()

        if self._log_data:
            # Update all the logs for debugging purposes and for tracing the route.
            self.ninja.log()

            # Batch entity position logging
            for (
                entity
            ) in active_movable_entities:  # Log only active and movable entities
                entity.log_position()

        # Clear physics caches periodically to prevent memory bloat or stale data
        if self.frame >= self._next_cache_clear:
            clear_caches()
            self._next_cache_clear = self.frame + CACHE_CLEAR_INTERVAL

    def tick_physics_only(self, hor_input, jump_input):
        """Execute one physics frame WITHOUT entity AI (for value iteration).

        This method provides a lightweight physics simulation that skips
        entity AI (entity.move() and entity.think()) but keeps all ninja
        physics intact. Used by PhysicsStepWrapper for value function
        precomputation where entity AI doesn't affect physics outcomes.

        Skips:
            - entity.move() and entity.think() (entity AI)
            - graphics updates (ninja.update_graphics())
            - logging (ninja.log(), entity.log_position())
            - cache clearing (handled externally for batch operations)

        Keeps:
            - All ninja physics (integrate, collisions, post_collision, think)
            - Static entity collisions (collide_vs_objects for mines, doors, etc.)
            - Tile collisions (collide_vs_tiles)

        Args:
            hor_input: Horizontal input (-1, 0, or 1)
            jump_input: Jump input (0 or 1)
        """
        # Increment frame counter (needed for some physics calculations)
        self.frame += 1

        # Store inputs as ninja variables
        self.ninja.hor_input = hor_input
        self.ninja.jump_input = jump_input

        # Record position before physics execution
        self.ninja.record_action_start_position()

        # Execute ninja physics (same as tick() but without entity AI)
        if self.ninja.state != 9:  # 9 typically means inactive
            # Apply physics to ninja or ragdoll
            ninja_to_update = (
                self.ninja if self.ninja.state != 6 else self.ninja.ragdoll
            )  # 6 for dead state

            if ninja_to_update:
                ninja_to_update.integrate()  # Apply drag, gravity, update position
                ninja_to_update.pre_collision()  # Reset collision state

                # Physics substeps (same 4 substeps as normal tick)
                for _ in range(4):
                    # Collide with static entities (mines, doors, etc.)
                    ninja_to_update.collide_vs_objects()
                    # Collide with tile segments
                    ninja_to_update.collide_vs_tiles()

                ninja_to_update.post_collision()  # Death checks, state transitions

            # OOB debug: log once per episode when ninja crosses level boundary.
            if not self._oob_logged and self.ninja.state < 6 and self.ninja.ypos > 600:
                from .physics import gather_segments_from_region
                rad = 10
                segs = gather_segments_from_region(
                    self,
                    self.ninja.xpos - rad, self.ninja.ypos - rad,
                    self.ninja.xpos + rad, self.ninja.ypos + rad,
                )
                logger.warning(
                    "OOB in tick_physics_only: frame=%d pos=(%.2f,%.2f) old=(%.2f,%.2f) "
                    "vel=(%.4f,%.4f) state=%d floor_count=%d segments_nearby=%d",
                    self.frame,
                    self.ninja.xpos, self.ninja.ypos,
                    self.ninja.xpos_old, self.ninja.ypos_old,
                    self.ninja.xspeed, self.ninja.yspeed,
                    self.ninja.state,
                    self.ninja.floor_count,
                    len(segs),
                )
                self._oob_logged = True

            # Ninja think (handles jump logic, state machine)
            self.ninja.think()
