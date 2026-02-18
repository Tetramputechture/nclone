"""
Landing point filtering against level geometry and deadly mines.

Validates analytical trajectory landing points before they become edges
in the reachability graph. Reuses the tile validity check from the
graph builder to ensure consistency with the existing node placement.
"""

from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    TOGGLE_MINE_RADII,
    TILE_PIXEL_SIZE,
)
from nclone.graph.reachability.graph_builder import _check_subnode_validity_simple


def is_position_in_traversable_tile(
    x: float,
    y: float,
    tile_dic: dict,
) -> bool:
    """Check if a world position is in traversable (non-solid) space.

    Converts world coordinates to tile + pixel-within-tile, then uses
    the same validity check the graph builder uses for sub-node placement.

    Args:
        x, y: World position in pixels.
        tile_dic: {(col, row): tile_type} from sim.tile_dic.

    Returns:
        True if position is in non-solid space.
    """
    # Clamp to map bounds
    if x < 0 or y < 0 or x >= 44 * TILE_PIXEL_SIZE or y >= 25 * TILE_PIXEL_SIZE:
        return False

    col = int(x // TILE_PIXEL_SIZE)
    row = int(y // TILE_PIXEL_SIZE)

    tile_type = tile_dic.get((col, row), 0)

    # Pixel position within the tile (0-23)
    pixel_x = int(x % TILE_PIXEL_SIZE)
    pixel_y = int(y % TILE_PIXEL_SIZE)

    return _check_subnode_validity_simple(tile_type, pixel_x, pixel_y)


def is_near_deadly_mine(
    x: float,
    y: float,
    deadly_mine_positions: list[tuple[float, float]],
    kill_radius: float = 0.0,
) -> bool:
    """Check if a position is within kill radius of any deadly mine.

    Args:
        x, y: World position in pixels.
        deadly_mine_positions: List of (mx, my) for deadly (state=0) mines.
        kill_radius: Distance threshold. Default: NINJA_RADIUS + mine_radius.

    Returns:
        True if position is dangerously close to a deadly mine.
    """
    if not deadly_mine_positions:
        return False

    kill_radius_sq = kill_radius * kill_radius

    for mx, my in deadly_mine_positions:
        dx = x - mx
        dy = y - my
        if dx * dx + dy * dy < kill_radius_sq:
            return True

    return False


def filter_landing_points(
    landing_points: list[dict],
    tile_dic: dict,
    deadly_mine_positions: list[tuple[float, float]],
    spatial_hash,
    node_physics: dict,
    match_radius: float = 12.0,
    mine_kill_radius: float = 0.0,
) -> list[dict]:
    """Filter landing points by geometry, mines, and surface existence.

    Each landing point is validated:
    1. Tile validity -- position must be in traversable space
    2. Mine proximity -- position must not be near a deadly mine
    3. Surface matching -- must match a grounded or walled node in node_physics

    Returns filtered list with additional keys:
        - 'matched_node': (x, y) of the matched graph node
        - 'node_grounded': bool
        - 'node_walled': bool

    Args:
        landing_points: List of dicts with 'x', 'y' keys (from ReachabilityScatter).
        tile_dic: {(col, row): tile_type} from simulator.
        deadly_mine_positions: List of (x, y) for mines with state=0.
        spatial_hash: SpatialHash from graph_data for node lookup.
        node_physics: {(x,y): {"grounded": bool, "walled": bool}} from graph_data.
        match_radius: Proximity threshold for surface node matching (px).
        mine_kill_radius: Rejection radius around deadly mines (px).
            Default 0.0 means auto-compute as NINJA_RADIUS + toggled mine radius.

    Returns:
        Filtered landing points with matched node and physics info.
    """
    if mine_kill_radius <= 0.0:
        mine_kill_radius = NINJA_RADIUS + TOGGLE_MINE_RADII[0]  # ~14 px

    if spatial_hash is None or not node_physics:
        return []

    filtered = []
    for lp in landing_points:
        lx, ly = lp["x"], lp["y"]

        # 1. Tile validity
        if not is_position_in_traversable_tile(lx, ly, tile_dic):
            continue

        # 2. Mine proximity
        if is_near_deadly_mine(lx, ly, deadly_mine_positions, mine_kill_radius):
            continue

        # 3. Surface matching -- find nearest graph node within radius
        nearby = spatial_hash.find_all_within_radius(lx, ly, match_radius)
        if not nearby:
            continue

        # Find the closest node that is grounded or walled
        matched_node = None
        matched_physics = None
        for node_pos, dist in nearby:
            physics = node_physics.get(node_pos)
            if physics and (physics["grounded"] or physics["walled"]):
                matched_node = node_pos
                matched_physics = physics
                break  # Already sorted by distance (closest first)

        if matched_node is None:
            continue

        # Annotate and keep
        lp_copy = dict(lp)
        lp_copy["matched_node"] = matched_node
        lp_copy["node_grounded"] = matched_physics["grounded"]
        lp_copy["node_walled"] = matched_physics["walled"]
        filtered.append(lp_copy)

    return filtered


def extract_deadly_mine_positions(level_data) -> list[tuple[float, float]]:
    """Extract positions of deadly (toggled, state=0) mines from level data.

    State semantics (from entity_toggle_mine.py):
        0 = toggled (deadly)
        1 = untoggled (safe)
        2 = toggling (transitioning)

    Args:
        level_data: LevelData instance with get_all_toggle_mines() method.

    Returns:
        List of (x, y) positions of deadly mines.
    """
    positions = []
    if not hasattr(level_data, "get_all_toggle_mines"):
        return positions

    for mine in level_data.get_all_toggle_mines():
        mine_state = mine.get("state", 0)
        if mine_state == 0:  # Toggled = deadly
            mx = mine.get("x", 0)
            my = mine.get("y", 0)
            # Skip invalid positions (extraction failures)
            if mx == 0 and my == 0:
                continue
            positions.append((mx, my))

    return positions
