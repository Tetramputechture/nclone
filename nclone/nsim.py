from .sim_config import SimConfig
from .map_loader import MapLoader

# Quadtree removed: grid cell-based collision queries are used instead.


class Simulator:
    """Main class that handles ninjas, entities and tile geometry for simulation."""

    def __init__(self, sc: SimConfig):
        """Initializes the simulator with a given SimConfig."""
        self.frame = 0
        self.collisionlog = []

        self.sim_config = sc
        self.ninja = None
        self.tile_dic = {}
        self.segment_dic = {}
        self.grid_entity = {}
        self.entity_dic = {}
        self.hor_grid_edge_dic = {}
        self.ver_grid_edge_dic = {}
        self.hor_segment_dic = {}
        self.ver_segment_dic = {}
        self.map_data = None
        self.map_loader = MapLoader(self)  # Initialize MapLoader

    def load(self, map_data):
        """From the given map data, initiate the level geometry, the entities and the ninja."""
        self.map_data = map_data
        self.reset_map_tile_data()  # Clears segment_dic
        self.map_loader.load_map_tiles()  # Loads segments into segment_dic and tiles into tile_dic

        self.reset()  # Resets entities and ninja

    def load_from_created(self, created_map):
        """Load a map that was manually constructed using the Map class."""
        self.load(created_map.map_data())

    def reset(self):
        """Reset the simulation to the initial state. Keeps the current map tiles, and resets the ninja,
        entities and the collision log."""
        self.frame = 0
        self.collisionlog = []
        self.ninja = None
        self.reset_map_entity_data()
        self.map_loader.load_map_entities()  # Use MapLoader

    def reset_map_entity_data(self):
        """Reset the map entity data. This is used when a new map is loaded or when the map is reset."""
        self.grid_entity = {}
        for x in range(44):
            for y in range(25):
                self.grid_entity[(x, y)] = []
        self.entity_dic = {
            i: [] for i in range(1, 29)
        }  # Initialize with expected entity types

    def reset_map_tile_data(self):
        """Reset the map cell data. This is used when a new map is loaded."""
        self.segment_dic = {}
        for x in range(45):  # map_width_tiles + a buffer often
            for y in range(26):  # map_height_tiles + a buffer often
                self.segment_dic[(x, y)] = []

        self.tile_dic = {}  # Clear tile_dic as well

        # Initialize horizontal grid edges, with outer edges set to 1 (solid)
        for x in range(89):
            for y in range(51):
                self.hor_grid_edge_dic[(x, y)] = 1 if y in (0, 50) else 0
        # Initialize vertical grid edges, with outer edges set to 1 (solid)
        for x in range(89):
            for y in range(51):
                self.ver_grid_edge_dic[(x, y)] = 1 if x in (0, 88) else 0
        # Initialize horizontal segment dictionary
        for x in range(89):
            for y in range(51):
                self.hor_segment_dic[(x, y)] = 0
        # Initialize vertical segment dictionary
        for x in range(89):
            for y in range(51):
                self.ver_segment_dic[(x, y)] = 0

    def tick(self, hor_input, jump_input):
        """Gets called every frame to update the whole physics simulation."""
        # Increment the current frame
        self.frame += 1

        # Store inputs as ninja variables
        self.ninja.hor_input = hor_input
        self.ninja.jump_input = jump_input

        # Cache active entities to avoid repeated filtering
        active_movable_entities = []
        active_thinkable_entities = []

        # Single pass to categorize entities
        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                if not entity.active:
                    continue
                if entity.is_movable:
                    active_movable_entities.append(entity)
                if entity.is_thinkable:
                    active_thinkable_entities.append(entity)

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
            self.ninja.think()  # Make ninja think
            if self.sim_config.enable_anim:
                self.ninja.update_graphics()  # Update limbs of ninja

        if self.ninja.state == 6 and self.sim_config.enable_anim:  # Ragdoll state
            self.ninja.anim_frame = 105  # Specific animation frame for ragdoll
            self.ninja.anim_state = 7  # Specific animation state for ragdoll
            self.ninja.calc_ninja_position()

        if self.sim_config.log_data:
            # Update all the logs for debugging purposes and for tracing the route.
            self.ninja.log()

            # Batch entity position logging
            for (
                entity
            ) in active_movable_entities:  # Log only active and movable entities
                entity.log_position()

        # Clear physics caches periodically to prevent memory bloat or stale data
        if self.frame % 100 == 0:  # Clear caches every 100 frames
            from .physics import (
                clear_caches,
            )  # Local import to avoid circular dependencies if physics imports Simulator

            clear_caches()  # This clears helper caches in physics.py
            # No quadtree clearing necessary
