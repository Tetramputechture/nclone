import copy
import logging

from .sim_config import SimConfig
from .map_loader import MapLoader
from .physics import clear_caches
import os

CACHE_CLEAR_INTERVAL = 100

#: Length of the per-ninja position ring buffer that the evil ninja (type 22)
#: replays. The game keeps 8 ninjas x 18000 records x 0x78 bytes and wraps the
#: write index at 17999 -- 18000 frames is 5 minutes at 60Hz. See
#: docs/ENTITY_INTEGRATION_SPEC.md §8.10; the recorder lives in the frame loop,
#: not in the entity, and nothing else in the sim needs it.
NINJA_TAPE_LENGTH = 18000

#: Order the game's frame loop runs the THINK pass in (spec §4). It is **not**
#: ascending by type: type 1 (mine) is thought tenth, after type 20 and
#: immediately before type 21, which uses the same function.
#:
#: Both this simulator and `nclone-native` used to derive the movable and
#: thinkable lists from one ascending classification pass, so `move order ==
#: think order` by construction. The game uses two different orders -- movement
#: really is ascending, thinking is this.
#:
#: Observable when a hazard kills the ninja on the same frame a mine is
#: mid-toggle: `EntityToggleMine.think` reverts to untoggled if it sees a DEAD
#: ninja, so whether the killing think already ran decides the mine's state.
#: Types 18, 19 and 20 all think before type 1 in the game; before the rocket and
#: turret landed, only the thwump sat in that window.
THINK_ORDER = (5, 12, 13, 15, 16, 17, 18, 19, 20, 1, 21, 22, 23, 25, 27, 28, 29)
_THINK_RANK = {t: i for i, t in enumerate(THINK_ORDER)}

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

        # Ninja position recorder for the evil ninja (type 22). None until a map
        # that needs it is loaded; see _init_ninja_tape.
        self.ninja_tape = None
        self.ninja_tape_index = 0

        # Optional environment reference for cache invalidation
        # Set by gym environment to enable performance optimizations
        self.gym_env = None

        # Cache SimConfig flags to avoid repeated attribute lookups in tick().
        # NOT the animation flag: whether the skeleton is maintained is decided
        # per MAP (`MapLoader._map_has_hitscan`), so `tick` reads it off the
        # ninja rather than off the config.
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
        self._init_ninja_tape()

    def _init_ninja_tape(self):
        """Arm the ninja position recorder, but only if something replays it.

        The recorder is a per-frame write on every tick, so it is allocated only
        when the map actually carries an evil ninja (type 22) -- about 3% of
        official maps. `ninja_tape is None` is the "no recorder" state that
        `EntityEvilNinja.think` treats as a full reset, matching the binary's
        "buffer gone" branch.
        """
        from .constants.entity_types import EntityType

        if self.entity_dic.get(EntityType.EVIL_NINJA):
            self.ninja_tape = [(0.0, 0.0)] * NINJA_TAPE_LENGTH
            self.ninja_tape_index = 0
        else:
            self.ninja_tape = None
            self.ninja_tape_index = 0

    def _record_ninja_tape(self):
        """Snapshot the ninja's position at the END of the frame.

        Ordering is part of the contract: the game's snapshot loop is the last
        thing in the frame tick, so an evil ninja thinking on frame N can only
        ever read records written on frames 1..N-1. The wrap rule is the binary's
        `(idx + 1 > 17999) ? 0 : idx + 1`, not a modulo -- same result, kept
        literal so the index is comparable across ports.
        """
        if self.ninja_tape is None or self.ninja is None:
            return
        self.ninja_tape[self.ninja_tape_index] = (self.ninja.xpos, self.ninja.ypos)
        idx = self.ninja_tape_index + 1
        self.ninja_tape_index = 0 if idx > NINJA_TAPE_LENGTH - 1 else idx

    def fast_reset(self):
        """Fast reset for same-level training. Resets entity state without recreation.

        This method is significantly faster than reset() because it:
        1. Resets ninja state in-place (no object allocation)
        2. Resets entity states in-place (no object allocation)
        3. Clears grid_entity lists (no dict recreation)
        4. Skips door segment cleanup (segments unchanged for same level)

        CRITICAL: Must be called ONLY when the map has not changed since the last
        reset. For new maps, use reset() instead. Entities are reset to their
        ORIGINAL positions.
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

        # Rewind the ninja tape. Leaving stale positions in it would let an evil
        # ninja replay the PREVIOUS episode's path.
        if self.ninja_tape is not None:
            self.ninja_tape_index = 0
            for i in range(len(self.ninja_tape)):
                self.ninja_tape[i] = (0.0, 0.0)

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

    def get_full_state(self):
        """Capture the COMPLETE mutable simulation state for exact restoration.

        Returns a deep copy of the dynamic state — frame counter, the ninja, and
        every entity (with grid membership preserved) — so it can be restored
        later to resume the simulation bit-identically WITHOUT replaying actions.
        This backs Go-Explore checkpointing (no action replay -> no position
        drift). The blob is plain-picklable (it crosses the SubprocVecEnv
        boundary): we temporarily detach the ``sim`` back-references (which point
        at this Simulator and, through it, the gym env / renderer) before the
        deepcopy and reattach them afterwards, so only physics state is copied.

        Tiles/segments/grid-edges are immutable for a loaded map and are NOT
        copied; ``restore_full_state`` must be called on a Simulator that has the
        SAME map loaded.
        """
        ninja_sim = getattr(self.ninja, "sim", None)
        if self.ninja is not None:
            self.ninja.sim = None
        detached = []
        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                if hasattr(entity, "sim"):
                    detached.append((entity, entity.sim))
                    entity.sim = None
        try:
            blob = copy.deepcopy(
                {
                    "frame": self.frame,
                    "ninja": self.ninja,
                    "entity_dic": self.entity_dic,
                    "grid_entity": self.grid_entity,
                }
            )
        finally:
            if self.ninja is not None:
                self.ninja.sim = ninja_sim
            for entity, sim_ref in detached:
                entity.sim = sim_ref
        return blob

    def _entity_signature(self, entity_dic):
        """Cheap structural signature: sorted (entity_type, count) pairs."""
        return tuple(sorted((k, len(v)) for k, v in entity_dic.items()))

    def restore_full_state(self, blob):
        """Restore state captured by :meth:`get_full_state` (no action replay).

        Replaces the ninja, entity dict and entity grid wholesale with the
        snapshot's deep-copied objects and reattaches the ``sim`` back-reference
        so the restored objects query THIS Simulator. The map (tiles/segments)
        must be the same as when the snapshot was taken.

        Raises ValueError if the snapshot's entity structure does not match the
        currently-loaded level (e.g. a Go-Explore checkpoint from a different
        level), so the caller can fall back to action replay instead of silently
        loading another level's entities onto this level's tiles.
        """
        if self._entity_signature(blob["entity_dic"]) != self._entity_signature(
            self.entity_dic
        ):
            raise ValueError("snapshot incompatible with current level")
        # DEEP-COPY ON RESTORE. Installing blob["ninja"] / blob["entity_dic"]
        # directly makes the live sim ALIAS the snapshot: the very next tick
        # mutates the objects the blob holds, so the snapshot silently becomes a
        # record of wherever the sim went next. Every snapshot was therefore
        # single-use, and every later restore returned drifted state.
        #
        # That is not a corner case here -- the three heaviest users all restore
        # the SAME blob repeatedly:
        #   * is_current_state_doomed branches 6 inputs off one snapshot, so
        #     inputs 2-6 saw corrupted state AND the live episode was left drifted;
        #   * Go-Explore restores an archived checkpoint many times over a run,
        #     so a checkpoint "at frame N" degraded into wherever play last went;
        #   * demo/expert snapshots (EXECUTION_PRECISION_PLAN §7) stopped being
        #     expert states after their first use.
        #
        # Copying on restore (rather than on capture) keeps the blob pristine for
        # unlimited reuse, which is the contract every caller already assumed.
        restored = copy.deepcopy(
            {
                "ninja": blob["ninja"],
                "entity_dic": blob["entity_dic"],
                "grid_entity": blob["grid_entity"],
            }
        )
        self.frame = blob["frame"]
        self.ninja = restored["ninja"]
        self.ninja.sim = self
        self.entity_dic = restored["entity_dic"]
        self.grid_entity = restored["grid_entity"]
        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                if hasattr(entity, "sim"):
                    entity.sim = self

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
        # One list per entity type. Range goes to 29 inclusive because
        # EntityPlayerControlledRocket is type 29 (spec §2) -- it is not
        # map-placed, but a runtime spawn still needs a bucket to land in.
        self.entity_dic = {i: [] for i in range(1, 30)}
        # Per-type instance indices; must reset each map (see Entity.__init__ / map_loader)
        self.entity_counts = [0] * 40

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

        # The move pass is ascending by type, which is what iterating entity_dic
        # already gives; the think pass is not. Sorting is stable, so entities of
        # the same type keep map-load order, which is also part of the contract.
        if len(active_thinkable_entities) > 1:
            active_thinkable_entities.sort(
                key=lambda e: _THINK_RANK.get(e.type, len(THINK_ORDER))
            )

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
            if self.ninja.ninja_anim_mode:
                self.ninja.update_graphics()  # Update limbs of ninja

        if self.ninja.state == 6 and self.ninja.ninja_anim_mode:  # Ragdoll state
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

        # Snapshot the ninja's position for the evil ninja's tape. Last thing in
        # the frame, as in the game's tick.
        self._record_ninja_tape()

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
