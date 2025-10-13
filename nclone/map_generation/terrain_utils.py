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

from typing import Tuple
from ..constants import MAP_TILE_WIDTH
from .constants import VALID_TILE_TYPES

# Slope tile constants
MILD_SLOPE_UP_LEFT = 20
MILD_SLOPE_UP_RIGHT = 24
MILD_SLOPE_DOWN_LEFT = 25
MILD_SLOPE_DOWN_RIGHT = 21

STEEP_SLOPE_UP_STEP = 28  # Moves right and up
STEEP_SLOPE_UP_RISE = 32  # Moves up only
STEEP_SLOPE_DOWN_STEP = 33  # Moves right and down
STEEP_SLOPE_DOWN_DROP = 29  # Moves down only

DIAGONAL_UP = 8
DIAGONAL_DOWN = 9

SOLID_TILE = 1


def random_tile(rng) -> int:
    """Return a random tile type."""
    return rng.randint(1, VALID_TILE_TYPES)


# ============================================================================
# Boundary and Helper Functions
# ============================================================================


def _init_boundaries(
    min_x: int = None, max_x: int = None, min_y: int = None, max_y: int = None
) -> Tuple[int, int, int, int]:
    """Initialize boundary values with defaults."""
    return (
        min_x if min_x is not None else 0,
        max_x if max_x is not None else MAP_TILE_WIDTH,
        min_y,
        max_y,
    )


def _is_within_x_bounds(x: int, min_x: int, max_x: int) -> bool:
    """Check if x coordinate is within horizontal boundaries."""
    return min_x <= x < max_x


def _exceeds_ceiling(y: int, min_y: int) -> bool:
    """Check if y coordinate exceeds ceiling boundary."""
    return min_y is not None and y < min_y


def _exceeds_floor(y: int, max_y: int) -> bool:
    """Check if y coordinate exceeds floor boundary."""
    return max_y is not None and y > max_y


def _get_steep_up_tile(height_change: int, index: int) -> int:
    """Determine which steep upward tile to use based on position and total height."""
    is_even_height = height_change % 2 == 0
    is_even_index = index % 2 == 0

    if is_even_height:
        return STEEP_SLOPE_UP_RISE if is_even_index else STEEP_SLOPE_UP_STEP
    else:
        return STEEP_SLOPE_UP_STEP if is_even_index else STEEP_SLOPE_UP_RISE


def _get_steep_down_tile(height_change: int, index: int) -> int:
    """Determine which steep downward tile to use based on position and total height."""
    is_even_height = height_change % 2 == 0
    is_even_index = index % 2 == 0

    if is_even_height:
        return STEEP_SLOPE_DOWN_STEP if is_even_index else STEEP_SLOPE_DOWN_DROP
    else:
        return STEEP_SLOPE_DOWN_DROP if is_even_index else STEEP_SLOPE_DOWN_STEP


def _fill_below(map_instance, x: int, y: int, start_x: int, offset: int = 1):
    """Fill solid tile below the given position if past the starting point."""
    if x > start_x + offset:
        map_instance.set_tile(x, y + 1, random_tile(map_instance.rng))


# ============================================================================
# Mild Slope Functions
# ============================================================================


def _create_mild_slope_up_step(
    map_instance, x: int, y: int, start_x: int, min_x: int, max_x: int
) -> Tuple[int, int]:
    """Create one step of a mild upward slope. Returns (new_x, new_y)."""
    if not (
        _is_within_x_bounds(x, min_x, max_x)
        and _is_within_x_bounds(x + 1, min_x, max_x)
    ):
        return x, y

    map_instance.set_tile(x, y, MILD_SLOPE_UP_LEFT)
    map_instance.set_tile(x + 1, y, MILD_SLOPE_UP_RIGHT)

    # Fill below after first two positions
    _fill_below(map_instance, x, y, start_x, offset=0)
    _fill_below(map_instance, x + 1, y, start_x, offset=0)

    return x + 2, y - 1


def _create_mild_slope_down_step(
    map_instance, x: int, y: int, start_x: int, min_x: int, max_x: int
) -> Tuple[int, int]:
    """Create one step of a mild downward slope. Returns (new_x, new_y)."""
    if not (
        _is_within_x_bounds(x, min_x, max_x)
        and _is_within_x_bounds(x + 1, min_x, max_x)
    ):
        return x, y

    # Place LEFT tile higher, RIGHT tile lower (vertical offset for downward slope)
    map_instance.set_tile(x, y, MILD_SLOPE_DOWN_LEFT)
    map_instance.set_tile(x + 1, y, MILD_SLOPE_DOWN_RIGHT)

    # Fill below after first two positions
    _fill_below(map_instance, x + 1, y, start_x - 1, offset=0)
    _fill_below(map_instance, x, y, start_x - 1, offset=0)

    # Next step starts at the y level of the RIGHT tile
    return x + 2, y + 1


# ============================================================================
# Steep Slope Functions
# ============================================================================


def _create_steep_slope_up_step(
    map_instance,
    x: int,
    y: int,
    start_x: int,
    height_change: int,
    remaining: int,
    min_x: int,
    max_x: int,
) -> Tuple[int, int, bool]:
    """Create one step of a steep upward slope. Returns (new_x, new_y, success)."""
    if not _is_within_x_bounds(x, min_x, max_x):
        return x, y, False

    index = height_change - remaining
    tile = _get_steep_up_tile(height_change, index)
    map_instance.set_tile(x, y, tile)

    if x > start_x - 1:
        map_instance.set_tile(x + 1, y, random_tile(map_instance.rng))

    if tile == STEEP_SLOPE_UP_STEP:
        return x + 1, y - 1, True
    else:  # STEEP_SLOPE_UP_RISE
        return x, y - 1, True


def _create_steep_slope_down_step(
    map_instance,
    x: int,
    y: int,
    start_x: int,
    height_change: int,
    remaining: int,
    min_x: int,
    max_x: int,
) -> Tuple[int, int, bool]:
    """Create one step of a steep downward slope. Returns (new_x, new_y, success)."""
    if not _is_within_x_bounds(x, min_x, max_x):
        return x, y, False

    index = height_change - remaining
    tile = _get_steep_down_tile(height_change, index)
    map_instance.set_tile(x, y, tile)

    # Fill below past leftmost tile
    if x > start_x:
        map_instance.set_tile(x - 1, y, random_tile(map_instance.rng))

    if tile == STEEP_SLOPE_DOWN_STEP:
        return x + 1, y + 1, True
    else:  # STEEP_SLOPE_DOWN_DROP
        return x, y + 1, True


# ============================================================================
# Main Slope Creation Functions
# ============================================================================


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

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to rise (positive integer)
        use_mild: If True, use mild slopes; if False, use steep slopes
        min_x: Minimum x coordinate (left boundary)
        max_x: Maximum x coordinate (right boundary)
        min_y: Minimum y coordinate (ceiling boundary)

    Returns:
        The x position where the slope ends
    """
    min_x, max_x, min_y, _ = _init_boundaries(min_x, max_x, min_y, None)

    current_x, current_y = x, y
    remaining = height_change

    while remaining > 0:
        if _exceeds_ceiling(current_y - 1, min_y):
            break

        if use_mild:
            new_x, new_y = _create_mild_slope_up_step(
                map_instance, current_x, current_y, x, min_x, max_x
            )
            if new_x == current_x:  # No progress made
                break
            current_x, current_y = new_x, new_y
            remaining -= 1
        else:
            new_x, new_y, success = _create_steep_slope_up_step(
                map_instance,
                current_x,
                current_y,
                x,
                height_change,
                remaining,
                min_x,
                max_x,
            )
            if not success:
                break
            current_x, current_y = new_x, new_y
            remaining -= 1

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

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to drop (positive integer)
        use_mild: If True, use mild slopes; if False, use steep slopes
        min_x: Minimum x coordinate (left boundary)
        max_x: Maximum x coordinate (right boundary)
        max_y: Maximum y coordinate (floor boundary)

    Returns:
        The x position where the slope ends
    """
    min_x, max_x, _, max_y = _init_boundaries(min_x, max_x, None, max_y)

    current_x, current_y = x, y
    remaining = height_change

    while remaining > 0:
        if _exceeds_floor(current_y + 1, max_y):
            break

        if use_mild:
            new_x, new_y = _create_mild_slope_down_step(
                map_instance, current_x, current_y, x, min_x, max_x
            )
            if new_x == current_x:  # No progress made
                break
            current_x, current_y = new_x, new_y
            remaining -= 1
        else:
            new_x, new_y, success = _create_steep_slope_down_step(
                map_instance,
                current_x,
                current_y,
                x,
                height_change,
                remaining,
                min_x,
                max_x,
            )
            if not success:
                break
            current_x, current_y = new_x, new_y
            remaining -= 1

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

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to rise (positive integer)
        min_x: Minimum x coordinate (left boundary)
        max_x: Maximum x coordinate (right boundary)
        min_y: Minimum y coordinate (ceiling boundary)

    Returns:
        The x position after the slope
    """
    min_x, max_x, min_y, _ = _init_boundaries(min_x, max_x, min_y, None)
    current_x, current_y = x, y

    for i in range(height_change):
        if not _is_within_x_bounds(current_x, min_x, max_x) or _exceeds_ceiling(
            current_y, min_y
        ):
            break

        map_instance.set_tile(current_x, current_y, DIAGONAL_UP)
        if i > 0:
            map_instance.set_tile(
                current_x, current_y + 1, random_tile(map_instance.rng)
            )

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

    Args:
        map_instance: The Map instance to modify
        x: Starting x coordinate
        y: Starting y coordinate (ground level)
        height_change: How many tiles to drop (positive integer)
        min_x: Minimum x coordinate (left boundary)
        max_x: Maximum x coordinate (right boundary)
        max_y: Maximum y coordinate (floor boundary)

    Returns:
        The x position after the slope
    """
    min_x, max_x, _, max_y = _init_boundaries(min_x, max_x, None, max_y)
    current_x, current_y = x, y

    for i in range(height_change):
        if not _is_within_x_bounds(current_x, min_x, max_x) or _exceeds_floor(
            current_y, max_y
        ):
            break

        map_instance.set_tile(current_x, current_y, DIAGONAL_DOWN)
        if i > 0:
            map_instance.set_tile(
                current_x - 1, current_y, random_tile(map_instance.rng)
            )

        current_x += 1
        current_y += 1

    return current_x


# ============================================================================
# Hill Creation Functions
# ============================================================================


def _create_plateau(map_instance, x: int, y: int, width: int, max_x: int) -> int:
    """Create a flat plateau section. Returns the x position after the plateau."""
    current_x = x
    for _ in range(width):
        if current_x >= max_x:
            break
        map_instance.set_tile(current_x, y, SOLID_TILE)
        current_x += 1
    return current_x


def _create_hill(
    map_instance,
    x: int,
    y: int,
    height: int,
    width: int,
    ascent_func,
    descent_func,
    ascent_args: dict,
    descent_args: dict,
    width_multiplier: int,
    min_x: int,
    max_x: int,
) -> int:
    """Generic hill builder using provided ascent and descent functions."""
    current_x = ascent_func(map_instance, x, y, **ascent_args)

    # Optional plateau
    if width and width > height * width_multiplier:
        plateau_width = width - (height * width_multiplier)
        plateau_y = y - height
        current_x = _create_plateau(
            map_instance, current_x, plateau_y, plateau_width, max_x
        )

    current_x = descent_func(map_instance, current_x, **descent_args)
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
    """Create a gentle hill using mild slope tiles."""
    min_x, max_x, min_y, max_y = _init_boundaries(min_x, max_x, min_y, max_y)

    ascent_args = {
        "height_change": height + 1,
        "use_mild": True,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
    }

    descent_args = {
        "y": y - height,
        "height_change": height,
        "use_mild": True,
        "min_x": min_x,
        "max_x": max_x + 1,
        "max_y": max_y + 1 if max_y else None,
    }

    return _create_hill(
        map_instance,
        x,
        y,
        height,
        width,
        create_slope_up,
        create_slope_down,
        ascent_args,
        descent_args,
        4,
        min_x,
        max_x,
    )


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
    """Create a steep hill using steep slope tiles."""
    min_x, max_x, min_y, max_y = _init_boundaries(min_x, max_x, min_y, max_y)

    ascent_args = {
        "height_change": height,
        "use_mild": False,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
    }

    descent_args = {
        "y": y - height + 1,
        "height_change": height,
        "use_mild": False,
        "min_x": min_x,
        "max_x": max_x,
        "max_y": max_y,
    }

    return _create_hill(
        map_instance,
        x,
        y,
        height,
        width,
        create_slope_up,
        create_slope_down,
        ascent_args,
        descent_args,
        2,
        min_x,
        max_x,
    )


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
    """Create a sharp 45-degree hill."""
    min_x, max_x, min_y, max_y = _init_boundaries(min_x, max_x, min_y, max_y)

    current_x = create_45_degree_slope_up(
        map_instance, x, y - 1, height, min_x=min_x, max_x=max_x, min_y=min_y
    )

    current_x = create_45_degree_slope_down(
        map_instance,
        current_x,
        y - height,
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

    Args:
        ascent_type: Type of ascent ("mild", "steep", or "45")
        descent_type: Type of descent ("mild", "steep", or "45")
    """
    min_x, max_x, min_y, max_y = _init_boundaries(min_x, max_x, min_y, max_y)

    # Map slope types to functions, parameters, and actual climb heights
    # Each entry: (ascent_func, ascent_params, actual_ascent_height, descent_func, descent_params)
    slope_configs = {
        "mild": (
            create_slope_up,
            {"use_mild": True},
            height - 1,  # Mild slopes climb height - 1
            create_slope_down,
            {"use_mild": True},
        ),
        "steep": (
            create_slope_up,
            {"use_mild": False},
            height,  # Steep slopes climb full height
            create_slope_down,
            {"use_mild": False},
        ),
        "45": (
            create_45_degree_slope_up,
            {},
            height,  # 45-degree slopes climb full height
            create_45_degree_slope_down,
            {},
        ),
    }

    # Ascend
    ascent_func, ascent_params, ascent_height, descent_func, descent_params = (
        slope_configs[ascent_type]
    )
    current_x = ascent_func(
        map_instance,
        x,
        y,
        ascent_height,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        **ascent_params,
    )

    # Calculate where the ascent actually ended
    # All slope types end at starting_y - ascent_height
    actual_peak_y = y - ascent_height

    # Descend from the actual peak position
    current_x = descent_func(
        map_instance,
        current_x,
        actual_peak_y + 1,
        height,
        min_x=min_x,
        max_x=max_x,
        max_y=max_y,
        **descent_params,
    )

    return current_x
