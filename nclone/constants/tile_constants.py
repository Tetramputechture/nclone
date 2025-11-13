"""
Tile classification constants for N++ physics simulation.

This module defines tile type classifications used for optimizing terminal
velocity precomputation and other physics-based predictions.

Terminal velocity deaths primarily occur on flat surfaces (floors/ceilings),
so we prioritize precomputation for those tile types while falling back to
runtime simulation for curved surfaces (which account for <1% of deaths).
"""

# High-priority tiles for terminal velocity precomputation (flat surfaces)
# These tiles have flat floor/ceiling surfaces where 99% of terminal deaths occur
FLAT_SURFACE_TILES = {1, 2, 3, 4, 5}  # Full solid and half tiles

# Medium-priority tiles (partial flat surfaces)
# Mild and steep slopes have some flat regions but less terminal impact risk
PARTIAL_FLAT_TILES = {
    18, 19, 20, 21,  # Short mild slopes
    22, 23, 24, 25,  # Raised mild slopes
    26, 27, 28, 29,  # Short steep slopes
    30, 31, 32, 33,  # Raised steep slopes
}

# Low-priority tiles (curved/diagonal surfaces - skip precomputation)
# Terminal impacts on these surfaces are rare (<1% of deaths)
# These will use runtime simulation instead of precomputed cache
CURVED_TILES = {
    6, 7, 8, 9,      # 45-degree diagonal slopes
    10, 11, 12, 13,  # Quarter circles (convex corners)
    14, 15, 16, 17,  # Quarter pipes (concave corners)
}

# Terminal velocity priority tiers for precomputation
TERMINAL_VELOCITY_PRIORITY_HIGH = FLAT_SURFACE_TILES
TERMINAL_VELOCITY_PRIORITY_MEDIUM = PARTIAL_FLAT_TILES
TERMINAL_VELOCITY_PRIORITY_SKIP = CURVED_TILES

# Empty tile (never causes terminal impacts)
EMPTY_TILE = 0

# All valid tile types (0-33, excluding glitched 34-37)
VALID_TILE_TYPES = set(range(34))

# Tile types that should be precomputed for terminal velocity
PRECOMPUTE_TILE_TYPES = FLAT_SURFACE_TILES | PARTIAL_FLAT_TILES

