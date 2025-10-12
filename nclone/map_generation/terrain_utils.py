"""Terrain generation utilities for N++ map creation.

This module provides functions for creating slopes, hills, and other terrain
features using the proper N++ tile types. These utilities can be used by any
map generator that needs to create terrain.

Slope Tile Reference:
---------------------
Mild Slopes (gentle, 2 tiles per height):
  - Ascending:  20, 24 (placed on same Y, then move up)
  - Descending: 21, 25 (placed with vertical offset)

Steep Slopes (1 tile per height):
  - Ascending (L→R):  28, 32 (top tile always 28)
    * Even height: 32, 28, 32, 28, ... (starts 32)
    * Odd height:  28, 32, 28, 32, ... (starts 28)
  - Descending (R→L): 29, 33 (top tile always 29)
    * Even height: 33, 29, 33, 29, ... (starts 33)
    * Odd height:  29, 33, 29, 33, ... (starts 29)

45-Degree Slopes (diagonal, 1 tile per height):
  - Ascending:  8
  - Descending: 9
"""

from ..constants import MAP_TILE_WIDTH

# Slope tile constants
MILD_SLOPE_UP_LEFT = 20
MILD_SLOPE_UP_RIGHT = 24
MILD_SLOPE_DOWN_LEFT = 21
MILD_SLOPE_DOWN_RIGHT = 25

STEEP_SLOPE_UP_STEP = 28  # Moves right and up
STEEP_SLOPE_UP_RISE = 32  # Moves up only
STEEP_SLOPE_DOWN_STEP = 33  # Moves right and down
STEEP_SLOPE_DOWN_DROP = 29  # Moves down only

DIAGONAL_UP = 8
DIAGONAL_DOWN = 9

SOLID_TILE = 1


def create_slope_up(
    map_instance,
    x: int,
    y: int,
    height_change: int,
    use_mild: bool = True,
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
) -> int:
    """Create an ascending slope starting at (x, y).

    Based on N++ tile definitions:
    - Mild slopes: alternating tiles 20, 24 on same Y, then move up (2 tiles per height)
    - Steep slopes: alternating tiles 28, 32 with specific placement pattern
    - 45-degree: tile 8 diagonal placement

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to rise (positive integer)
        use_mild: If True, use mild slopes; if False, use steep slopes
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check

    Returns:
        The x position where the slope ends
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x
    current_y = y
    remaining_height = height_change
    start_x = x

    while remaining_height > 0:
        # Check if we would exceed ceiling boundary
        if min_y is not None and current_y - 1 < min_y:
            break

        if use_mild:
            # Mild slope: tiles 20 and 24 on same Y level, then move up
            if remaining_height >= 1 and current_x >= min_x and current_x + 1 < max_x:
                map_instance.set_tile(current_x, current_y, MILD_SLOPE_UP_LEFT)
                map_instance.set_tile(current_x + 1, current_y, MILD_SLOPE_UP_RIGHT)
                # Fill below after first two positions
                if current_x > start_x + 1:
                    map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)
                if current_x + 1 > start_x + 1:
                    map_instance.set_tile(current_x + 1, current_y + 1, SOLID_TILE)
                current_x += 2
                current_y -= 1
                remaining_height -= 1
            else:
                break
        else:
            # Steep slope: alternating tiles 28 (step) and 32 (rise)
            # Top tile should always be 28 (ascending left->right)
            # If height is even: pattern is 32, 28, 32, 28, ... (ending with 28)
            # If height is odd: pattern is 28, 32, 28, 32, ... (ending with 28)
            if remaining_height >= 1 and current_x >= min_x and current_x < max_x:
                index = height_change - remaining_height
                if height_change % 2 == 0:  # Even height
                    tile_to_place = (
                        STEEP_SLOPE_UP_RISE if index % 2 == 0 else STEEP_SLOPE_UP_STEP
                    )
                else:  # Odd height
                    tile_to_place = (
                        STEEP_SLOPE_UP_STEP if index % 2 == 0 else STEEP_SLOPE_UP_RISE
                    )

                map_instance.set_tile(current_x, current_y, tile_to_place)

                # Fill below past leftmost tile
                if current_x > start_x:
                    map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)

                if tile_to_place == STEEP_SLOPE_UP_STEP:
                    # Move right and up
                    current_x += 1
                    current_y -= 1
                else:
                    # Just move up (tile 32)
                    current_y -= 1
                remaining_height -= 1
            else:
                break

    return current_x


def create_slope_down(
    map_instance,
    x: int,
    y: int,
    height_change: int,
    use_mild: bool = True,
    min_x: int = None,
    max_x: int = None,
    max_y: int = None,
) -> int:
    """Create a descending slope starting at (x, y).

    Based on N++ tile definitions:
    - Mild slopes: alternating tiles 21, 25 with specific placement pattern
    - Steep slopes: alternating tiles 33, 29
    - 45-degree: tile 9 diagonal placement

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to drop (positive integer)
        use_mild: If True, use mild slopes; if False, use steep slopes
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position where the slope ends
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x
    current_y = y
    remaining_height = height_change
    start_x = x
    start_with_raised = remaining_height + 1 % 2 == 1
    mild_tile_type_one = (
        MILD_SLOPE_DOWN_LEFT if start_with_raised else MILD_SLOPE_DOWN_RIGHT
    )
    mild_tile_type_two = (
        MILD_SLOPE_DOWN_RIGHT if start_with_raised else MILD_SLOPE_DOWN_LEFT
    )

    while remaining_height > 0:
        # Check if we would exceed floor boundary
        if max_y is not None and current_y + 1 > max_y:
            break

        if use_mild:
            # Mild slope down: tiles 21 and 25 with vertical offset
            if remaining_height >= 1 and current_x >= min_x and current_x + 1 < max_x:
                map_instance.set_tile(current_x, current_y, mild_tile_type_one)
                if start_with_raised:
                    current_y += 1
                map_instance.set_tile(current_x + 1, current_y, mild_tile_type_two)
                if not start_with_raised:
                    current_y += 1
                # Fill below after first two positions
                if current_x > start_x + 1:
                    map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)
                if current_x + 1 > start_x + 1:
                    map_instance.set_tile(current_x + 1, current_y + 1, SOLID_TILE)
                current_x += 2
                remaining_height -= 1
            else:
                break
        else:
            # Steep slope down: alternating tiles 33 (step) and 29 (drop)
            # Top tile (first placed in descent) should always be 29 (descending right->left)
            # If height is even: pattern is 33, 29, 33, 29, ... (starting with 33 at top)
            # If height is odd: pattern is 29, 33, 29, 33, ... (starting with 29 at top)
            if remaining_height >= 1 and current_x >= min_x and current_x < max_x:
                index = height_change - remaining_height
                if height_change % 2 == 0:  # Even height
                    tile_to_place = (
                        STEEP_SLOPE_DOWN_STEP
                        if index % 2 == 0
                        else STEEP_SLOPE_DOWN_DROP
                    )
                else:  # Odd height
                    tile_to_place = (
                        STEEP_SLOPE_DOWN_DROP
                        if index % 2 == 0
                        else STEEP_SLOPE_DOWN_STEP
                    )

                map_instance.set_tile(current_x, current_y, tile_to_place)

                # Fill below past leftmost tile
                if current_x > start_x:
                    map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)

                if tile_to_place == STEEP_SLOPE_DOWN_STEP:
                    # Move down and right
                    current_x += 1
                    current_y += 1
                else:
                    # Just move down (tile 29)
                    current_y += 1
                remaining_height -= 1
            else:
                break

    return current_x


def create_45_degree_slope_up(
    map_instance,
    x: int,
    y: int,
    height_change: int = 1,
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
) -> int:
    """Create a 45-degree ascending slope.

    Uses tile 8 with diagonal placement and fill below.

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to rise (positive integer)
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check

    Returns:
        The x position after the slope
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x
    current_y = y

    for i in range(height_change):
        if current_x < min_x or current_x >= max_x:
            break
        # Check ceiling boundary
        if min_y is not None and current_y < min_y:
            break
        map_instance.set_tile(current_x, current_y, DIAGONAL_UP)
        # Fill below
        if i > 0:
            map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)
        current_x += 1
        current_y -= 1

    return current_x


def create_45_degree_slope_down(
    map_instance,
    x: int,
    y: int,
    height_change: int = 1,
    min_x: int = None,
    max_x: int = None,
    max_y: int = None,
) -> int:
    """Create a 45-degree descending slope.

    Uses tile 9 with diagonal placement and fill below.

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to drop (positive integer)
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position after the slope
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x
    current_y = y

    for i in range(height_change):
        if current_x < min_x or current_x >= max_x:
            break
        # Check floor boundary
        if max_y is not None and current_y > max_y:
            break
        map_instance.set_tile(current_x, current_y, DIAGONAL_DOWN)
        # Fill below
        if i > 0:
            map_instance.set_tile(current_x, current_y + 1, SOLID_TILE)
        current_x += 1
        current_y += 1

    return current_x


def create_mild_hill(
    map_instance,
    x: int,
    y: int,
    height: int,
    width: int = None,
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
    max_y: int = None,
) -> int:
    """Create a gentle hill using mild slope tiles.

    Pattern: Uses tiles 20+24 for ascending, 21+25 for descending.

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Base ground level
        height: Height of the hill peak (in tiles)
        width: Optional width control (if None, auto-calculated from height)
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position after the hill
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x

    # Ascend with mild slopes
    current_x = create_slope_up(
        map_instance,
        current_x,
        y - 1,
        height,
        use_mild=True,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
    )

    # Optional plateau
    if width and width > height * 4:
        plateau_width = width - (height * 4)
        plateau_y = y - height
        for _ in range(plateau_width):
            if current_x >= max_x:
                break
            map_instance.set_tile(current_x, plateau_y, SOLID_TILE)
            current_x += 1

    # Descend with mild slopes
    descend_y = y - height
    current_x = create_slope_down(
        map_instance,
        current_x,
        descend_y + 1,
        height,
        use_mild=True,
        min_x=min_x,
        max_x=max_x + 1,
        max_y=max_y + 1,
    )

    return current_x


def create_steep_hill(
    map_instance,
    x: int,
    y: int,
    height: int,
    width: int = None,
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
    max_y: int = None,
) -> int:
    """Create a steep hill using steep slope tiles.

    Pattern: Uses tiles 28+32 for ascending, 33+29 for descending.

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Base ground level
        height: Height of the hill peak (in tiles)
        width: Optional width control (if None, auto-calculated from height)
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position after the hill
    """
    if min_x is None:
        min_x = 0
    if max_x is None:
        max_x = MAP_TILE_WIDTH

    current_x = x

    # Ascend with steep slopes
    current_x = create_slope_up(
        map_instance,
        current_x,
        y,
        height,
        use_mild=False,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
    )

    # Optional plateau
    if width and width > height * 2:
        plateau_width = width - (height * 2)
        plateau_y = y - height
        for _ in range(plateau_width):
            if current_x >= max_x:
                break
            map_instance.set_tile(current_x, plateau_y, SOLID_TILE)
            current_x += 1

    # Descend with steep slopes
    descend_y = y - height
    current_x = create_slope_down(
        map_instance,
        current_x,
        descend_y + 1,
        height,
        use_mild=False,
        min_x=min_x,
        max_x=max_x,
        max_y=max_y,
    )

    return current_x


def create_45_degree_hill(
    map_instance,
    x: int,
    y: int,
    height: int,
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
    max_y: int = None,
) -> int:
    """Create a sharp 45-degree hill.

    Pattern: Uses tile 8 for ascent, tile 9 for descent.
    Creates very sharp, angular hills.

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Base ground level
        height: Height of the hill peak (in tiles)
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position after the hill
    """
    current_x = x

    # Ascend with 45-degree slopes (tile 8)
    current_x = create_45_degree_slope_up(
        map_instance, current_x, y - 1, height, min_x=min_x, max_x=max_x, min_y=min_y
    )

    # Descend with 45-degree slopes (tile 9)
    descend_y = y - height
    current_x = create_45_degree_slope_down(
        map_instance,
        current_x,
        descend_y,
        height,
        min_x=min_x,
        max_x=max_x,
        max_y=max_y,
    )

    return current_x


def create_mixed_hill(
    map_instance,
    x: int,
    y: int,
    height: int,
    ascent_type: str = "mild",
    descent_type: str = "mild",
    min_x: int = None,
    max_x: int = None,
    min_y: int = None,
    max_y: int = None,
) -> int:
    """Create a hill with different slope types for ascent and descent.

    Allows mixing of slope types (e.g., steep ascent with mild descent).

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Base ground level
        height: Height of the hill peak (in tiles)
        ascent_type: Type of ascent ("mild", "steep", or "45")
        descent_type: Type of descent ("mild", "steep", or "45")
        min_x: Minimum x coordinate (left boundary). If None, starts from 0
        max_x: Maximum x coordinate (right boundary). If None, uses MAP_TILE_WIDTH
        min_y: Minimum y coordinate (ceiling boundary). If None, no ceiling check
        max_y: Maximum y coordinate (floor boundary). If None, no floor check

    Returns:
        The x position after the hill
    """
    current_x = x

    # Ascend based on type
    if ascent_type == "mild":
        current_x = create_slope_up(
            map_instance,
            current_x,
            y,
            height - 1,
            use_mild=True,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
        )
    elif ascent_type == "steep":
        current_x = create_slope_up(
            map_instance,
            current_x,
            y,
            height,
            use_mild=False,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
        )
    elif ascent_type == "45":
        current_x = create_45_degree_slope_up(
            map_instance, current_x, y, height, min_x=min_x, max_x=max_x, min_y=min_y
        )

    # Descend based on type
    descend_y = y - height
    if descent_type == "mild":
        current_x = create_slope_down(
            map_instance,
            current_x,
            descend_y + 1,
            height,
            use_mild=True,
            min_x=min_x,
            max_x=max_x,
            max_y=max_y,
        )
    elif descent_type == "steep":
        current_x = create_slope_down(
            map_instance,
            current_x,
            descend_y + 1,
            height,
            use_mild=False,
            min_x=min_x,
            max_x=max_x,
            max_y=max_y,
        )
    elif descent_type == "45":
        current_x = create_45_degree_slope_down(
            map_instance,
            current_x,
            descend_y,
            height,
            min_x=min_x,
            max_x=max_x,
            max_y=max_y,
        )

    return current_x
