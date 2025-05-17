import math

SRCWIDTH = 1056
SRCHEIGHT = 600

BGCOLOR = "cbcad0"
TILECOLOR = "797988"
NINJACOLOR = "000000"
ENTITYCOLORS = {1: "9E2126", 2: "DBE149", 3: "838384", 4: "6D97C3", 5: "000000", 6: "000000",
                7: "000000", 8: "000000", 9: "000000", 10: "868793", 11: "666666", 12: "000000",
                13: "000000", 14: "6EC9E0", 15: "6EC9E0", 16: "000000", 17: "E3E3E5", 18: "000000",
                19: "000000", 20: "838384", 21: "CE4146", 22: "000000", 23: "000000", 24: "666666",
                25: "15A7BD", 26: "6EC9E0", 27: "000000", 28: "6EC9E0"}

BASIC_BG_COLOR = "ffffff"
BASIC_TILE_COLOR = "000000"

SEGMENTWIDTH = 1
NINJAWIDTH = 1.25
DOORWIDTH = 2
PLATFORMWIDTH = 3

LIMBS = ((0, 12), (1, 12), (2, 8), (3, 9), (4, 10),
         (5, 11), (6, 7), (8, 0), (9, 0), (10, 1), (11, 1))

# Pre-calculate color values
BGCOLOR_RGB = tuple(
    int(c, 16)/255 for c in (BGCOLOR[0:2], BGCOLOR[2:4], BGCOLOR[4:6]))
TILECOLOR_RGB = tuple(
    int(c, 16)/255 for c in (TILECOLOR[0:2], TILECOLOR[2:4], TILECOLOR[4:6]))
NINJACOLOR_RGB = tuple(
    int(c, 16)/255 for c in (NINJACOLOR[0:2], NINJACOLOR[2:4], NINJACOLOR[4:6]))
ENTITYCOLORS_RGB = {k: tuple(int(
    c, 16)/255 for c in (v[0:2], v[2:4], v[4:6])) for k, v in ENTITYCOLORS.items()}

# Colors for exploration grid visualization
EXPLORATION_COLORS = {
    'cell': (0, 0, 0, 0),  # Transparent for unvisited cells
    # Bright green with 75% opacity for visited cells
    'cell_visited': (0, 255, 0, 192),
    'grid_cell': (255, 255, 255, 64)  # White with 25% opacity for cell grid
}

# Base colors for each area type
AREA_BASE_COLORS = {
    '4x4': (255, 50, 50),  # Base red for 4x4
    '8x8': (50, 50, 255),  # Base blue for 8x8
    '16x16': (128, 128, 128)  # Base grey for 16x16
}

PI_DIV_2 = math.pi / 2

def hex2float(string):
    """Convert hex color to RGB floats. This is now only used for dynamic colors not in the cache."""
    value = int(string, 16)
    red = ((value & 0xFF0000) >> 16) / 255
    green = ((value & 0x00FF00) >> 8) / 255
    blue = (value & 0x0000FF) / 255
    return red, green, blue 