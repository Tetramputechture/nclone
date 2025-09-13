"""
Constants and type mappings for the N++ MCP server.

This module contains all entity and tile type mappings used throughout
the MCP server for consistent level creation.
"""

# Entity type mappings for user-friendly names
ENTITY_TYPE_MAPPING = {
    "exit_door": 3,
    "switch": 4,
    "gold": 2,
    "toggle_mine": 1,
    "toggle_mine_toggled": 21,
    "launch_pad": 10,
    "one_way_platform": 11,
    "drone_zap": 14,
    "bounce_block": 17,
    "thwump": 20,
    "boost_pad": 24,
    "death_ball": 25,
    "mini_drone": 26,
    "locked_door": 6,
    "trap_door": 8,
}

# Reverse mapping for display
ENTITY_NAME_MAPPING = {v: k for k, v in ENTITY_TYPE_MAPPING.items()}

# Tile type mappings with proper nclone terminology
TILE_TYPE_MAPPING = {
    # Basic tiles
    "empty": 0,
    "full": 1,
    "solid": 1,  # Alias for full
    # Half tiles
    "half_top": 2,
    "half_right": 3,
    "half_bottom": 4,
    "half_left": 5,
    # 45-degree slopes
    "slope_45_tl_br": 6,  # Top-left to bottom-right (\)
    "slope_45_tr_bl": 7,  # Top-right to bottom-left (/)
    "slope_45_bl_tr": 8,  # Bottom-left to top-right (/) inverted
    "slope_45_br_tl": 9,  # Bottom-right to top-left (\) inverted
    # Quarter moons (convex corners) - solid in corner
    "quarter_moon_br": 10,  # Bottom-right solid corner
    "quarter_moon_bl": 11,  # Bottom-left solid corner
    "quarter_moon_tl": 12,  # Top-left solid corner
    "quarter_moon_tr": 13,  # Top-right solid corner
    # Quarter pipes (concave corners) - hollow in corner
    "quarter_pipe_tl": 14,  # Hollow in top-left
    "quarter_pipe_tr": 15,  # Hollow in top-right
    "quarter_pipe_br": 16,  # Hollow in bottom-right
    "quarter_pipe_bl": 17,  # Hollow in bottom-left
    # Short mild slopes (shallow slopes, short rise)
    "slope_mild_up_left": 18,  # Mild slope up-left, short rise from left
    "slope_mild_up_right": 19,  # Mild slope up-right, short rise to right
    "slope_mild_down_right": 20,  # Mild slope down-right, short drop to right
    "slope_mild_down_left": 21,  # Mild slope down-left, short drop from left
    # Raised mild slopes (gentle slopes on raised platforms)
    "slope_mild_raised_left": 22,  # Platform on left with gentle rise
    "slope_mild_raised_right": 23,  # Platform on right with gentle rise
    "slope_mild_raised_drop_right": 24,  # Platform on right with gentle drop
    "slope_mild_raised_drop_left": 25,  # Platform on left with gentle drop
    # Short steep slopes (steep slopes, short rise)
    "slope_steep_up_left": 26,  # Steep slope up-left, sharp rise from left
    "slope_steep_up_right": 27,  # Steep slope up-right, sharp rise to right
    "slope_steep_down_right": 28,  # Steep slope down-right, sharp drop to right
    "slope_steep_down_left": 29,  # Steep slope down-left, sharp drop from left
    # Raised steep slopes (steep slopes on raised platforms)
    "slope_steep_raised_left": 30,  # Platform on left with sharp rise
    "slope_steep_raised_right": 31,  # Platform on right with sharp rise
    "slope_steep_raised_drop_right": 32,  # Platform on right with sharp drop
    "slope_steep_raised_drop_left": 33,  # Platform on left with sharp drop
    # Legacy aliases for backward compatibility
    "slope_tl_br": 6,
    "slope_tr_bl": 7,
    "slope_bl_tr": 8,
    "slope_br_tl": 9,
    "quarter_circle_br": 10,
    "quarter_circle_bl": 11,
    "quarter_circle_tl": 12,
    "quarter_circle_tr": 13,
}

TILE_NAME_MAPPING = {v: k for k, v in TILE_TYPE_MAPPING.items()}
