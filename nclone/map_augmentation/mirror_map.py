from ..map_generation.map import Map
from ..map_generation.constants import (
    NINJA_SPAWN_OFFSET_UNITS,
    SWITCH_OFFSET_UNITS,
    EXIT_DOOR_OFFSET_UNITS,
    GOLD_OFFSET_UNITS,
    GRID_SIZE_FACTOR,
)
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


def mirror_map_horizontally(original_map: Map) -> Map:
    """Creates a horizontally mirrored version of the input map.
    All tiles and entities are mirrored from left to right and rotated appropriately.

    Args:
        original_map: The Map object to mirror

    Returns:
        A new Map object containing the mirrored map
    """
    mirrored = Map()

    # Mirror tiles
    for y in range(MAP_TILE_HEIGHT):
        for x in range(MAP_TILE_WIDTH):
            original_tile = original_map.tile_data[x + y * MAP_TILE_WIDTH]
            if original_tile == 0:
                continue

            # Calculate mirrored x position
            mirrored_x = MAP_TILE_WIDTH - 1 - x

            # Mirror tile type based on orientation
            mirrored_tile = original_tile
            if original_tile in (2, 4):  # Half tiles horizontal
                mirrored_tile = original_tile
            elif original_tile in (3, 5):  # Half tiles vertical
                mirrored_tile = 5 if original_tile == 3 else 3
            elif original_tile in (6, 7, 8, 9):  # 45 degree slopes
                mirrored_tile = {6: 7, 7: 6, 8: 9, 9: 8}[original_tile]
            elif original_tile in (10, 11, 12, 13):  # Quarter moons
                mirrored_tile = {10: 11, 11: 10, 12: 13, 13: 12}[original_tile]
            elif original_tile in (14, 15, 16, 17):  # Quarter pipes
                mirrored_tile = {14: 16, 15: 17, 16: 14, 17: 15}[original_tile]
            elif original_tile in (18, 19, 20, 21):  # Short mild slopes
                mirrored_tile = {18: 19, 19: 18, 20: 21, 21: 20}[original_tile]
            elif original_tile in (22, 23, 24, 25):  # Raised mild slopes
                mirrored_tile = {22: 23, 23: 22, 24: 25, 25: 24}[original_tile]
            elif original_tile in (26, 27, 28, 29):  # Short steep slopes
                mirrored_tile = {26: 28, 27: 29, 28: 26, 29: 27}[original_tile]
            elif original_tile in (30, 31, 32, 33):  # Raised steep slopes
                mirrored_tile = {30: 32, 31: 33, 32: 30, 33: 31}[original_tile]

            mirrored.set_tile(mirrored_x, y, mirrored_tile)

    # Mirror ninja spawn
    mirrored.set_ninja_spawn(
        MAP_TILE_WIDTH
        - 1
        - (original_map.ninja_spawn_x - NINJA_SPAWN_OFFSET_UNITS) // GRID_SIZE_FACTOR,
        (original_map.ninja_spawn_y - NINJA_SPAWN_OFFSET_UNITS) // GRID_SIZE_FACTOR,
        -original_map.ninja_orientation,  # Flip orientation
    )

    # Mirror entities
    for i in range(0, len(original_map.entity_data), 5):
        entity_type = original_map.entity_data[i]
        x = original_map.entity_data[i + 1]
        y = original_map.entity_data[i + 2]
        orientation = original_map.entity_data[i + 3]
        mode = original_map.entity_data[i + 4]

        # Calculate mirrored x position (in screen coordinates)
        mirrored_x = MAP_TILE_WIDTH * GRID_SIZE_FACTOR - x

        # Mirror orientation for relevant entities
        mirrored_orientation = orientation
        if entity_type in (5, 6, 8, 10, 11, 14, 15, 20, 23, 26):
            if orientation in (0, 4):  # Vertical orientations stay the same
                pass
            else:  # Horizontal orientations flip
                mirrored_orientation = {1: 3, 2: 6, 3: 1, 5: 7, 6: 2, 7: 5}[orientation]

        # # Mirror mode for drone entities by swapping CW/CCW
        mirrored_mode = mode
        if entity_type in (14, 15, 26):  # Drone entity types
            if mode == 0:  # CW wall follow -> CCW wall follow
                mirrored_mode = 1
            elif mode == 1:  # CCW wall follow -> CW wall follow
                mirrored_mode = 0
            elif mode == 2:  # CW wander -> CCW wander
                mirrored_mode = 3
            elif mode == 3:  # CCW wander -> CW wander
                mirrored_mode = 2

        # Handle entity offset
        entity_offset = 0
        switch_offset = 0
        if entity_type == 3:
            entity_offset = EXIT_DOOR_OFFSET_UNITS
            switch_offset = SWITCH_OFFSET_UNITS
        elif entity_type == 2:
            entity_offset = GOLD_OFFSET_UNITS

        # Handle special cases for entities that need switch coordinates
        if entity_type in (3, 6, 8):
            switch_x = original_map.entity_data[i + 6]
            switch_y = original_map.entity_data[i + 7]
            mirrored_switch_x = MAP_TILE_WIDTH * GRID_SIZE_FACTOR - switch_x

            mirrored.add_entity(
                entity_type,
                (mirrored_x - entity_offset) / GRID_SIZE_FACTOR + 2,
                (y - entity_offset) / GRID_SIZE_FACTOR,
                mirrored_orientation,
                mode,
                (mirrored_switch_x - switch_offset) / GRID_SIZE_FACTOR + 2,
                (switch_y - switch_offset) / GRID_SIZE_FACTOR,
            )
        else:
            mirrored.add_entity(
                entity_type,
                (mirrored_x - entity_offset) / GRID_SIZE_FACTOR + 2,
                (y - entity_offset) / GRID_SIZE_FACTOR,
                mirrored_orientation,
                mirrored_mode,
            )

    return mirrored
