from .ninja import Ninja

# Import base classes from entities.py module
from .entities import Entity
from .utils.entity_factory import create_entity_instance
from .utils.tile_segment_factory import TileSegmentFactory


class MapLoader:
    """Handles loading of map tiles and entities for the simulation."""

    def __init__(self, simulator):
        """Initializes the MapLoader with a reference to the main simulator."""
        self.sim = simulator

    def load_map_tiles(self):
        """Load the map tiles into the simulation. These shouldn't change during the simulation,
        only when a new map is loaded."""
        # extract tile data from map data
        tile_data = self.sim.map_data[184:1150]

        # map each tile to its cell
        for x_coord_tile in range(42):
            for y_coord_tile in range(23):
                self.sim.tile_dic[(x_coord_tile + 1, y_coord_tile + 1)] = tile_data[
                    x_coord_tile + y_coord_tile * 42
                ]

        # Set our outer edges to tile type 1 (full tile)
        for x_coord_tile in range(44):
            self.sim.tile_dic[(x_coord_tile, 0)] = 1
            self.sim.tile_dic[(x_coord_tile, 24)] = 1
        for y_coord_tile in range(25):
            self.sim.tile_dic[(0, y_coord_tile)] = 1
            self.sim.tile_dic[(43, y_coord_tile)] = 1

        # Use centralized tile segment factory to create all segments
        # This ensures consistency with PreciseCollision and eliminates code duplication
        self._populate_segment_dictionaries()

    def _populate_segment_dictionaries(self):
        """
        Populate segment dictionaries using the centralized TileSegmentFactory.

        This method uses the TileSegmentFactory with grid edge processing enabled
        to handle both grid edge creation (for MapLoader compatibility) and
        segment creation using consolidated logic.
        """
        # Use centralized factory for both grid edge and segment processing
        TileSegmentFactory.create_segments_for_simulator(
            self.sim, self.sim.tile_dic, process_grid_edges=True
        )

    def load_map_entities(self):
        """Load the map entities into the simulation. These should change during the simulation,
        and are reset when a new map is loaded."""

        # initiate player 1 instance of Ninja at spawn coordinates
        self.sim.ninja = Ninja(
            self.sim,
            ninja_anim_mode=(
                self.sim.sim_config.enable_anim and not self.sim.sim_config.basic_sim
            ),
        )

        if self.sim.map_data[1233] not in [-1, 1]:
            self.sim.map_data[1233] = -1

        # initiate each entity (other than ninjas)
        index = 1230
        exit_door_count = self.sim.map_data[1156]
        Entity.entity_counts = [0] * 40  # Reset global entity counts
        while index < len(self.sim.map_data):
            # Check if we have enough data for a complete entity (5 values needed)
            if index + 4 >= len(self.sim.map_data):
                # Incomplete entity data at end of map_data, skip it
                break

            entity_type = self.sim.map_data[index]
            xcoord = self.sim.map_data[index + 1]
            ycoord = self.sim.map_data[index + 2]
            orientation = self.sim.map_data[index + 3]
            mode = self.sim.map_data[index + 4]

            # Create entity using the entity factory utility
            entity = create_entity_instance(
                entity_type,
                self.sim,
                xcoord,
                ycoord,
                orientation,
                mode,
                map_data=self.sim.map_data,
                index=index,
                exit_door_count=exit_door_count,
            )

            if entity:
                # It's possible that the entity type does not yet exist in entity_dic if it's the first of its kind
                if entity_type not in self.sim.entity_dic:
                    self.sim.entity_dic[
                        entity_type
                    ] = []  # Initialize list for new entity type
                self.sim.entity_dic[entity_type].append(entity)
                self.sim.grid_entity[entity.cell].append(entity)
            index += 5

        for entity_list in self.sim.entity_dic.values():
            for entity_instance in entity_list:
                entity_instance.log_position()
