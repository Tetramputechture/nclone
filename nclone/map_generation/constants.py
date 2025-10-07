"""Constants used for map generation."""

VALID_TILE_TYPES = 33  # tiles 34-37 are not glitched tiles / not used

# Grid size factor. Multiply grid coordinates by this to get map data units.
# Map data units are then multiplied by 6 in the entity loading to get pixel coordinates.
GRID_SIZE_FACTOR = 4.0

# Offsets for entities in map data units (not pixels).
# These are applied during coordinate conversion and then multiplied by 6 during entity loading.
NINJA_SPAWN_OFFSET_UNITS = 6
EXIT_DOOR_OFFSET_UNITS = 6
SWITCH_OFFSET_UNITS = 6
GOLD_OFFSET_UNITS = -6
LOCKED_DOOR_OFFSET_UNITS = 6
