"""Level geometry extraction for the optimal-route planner.

Parses ``map_data`` (the list-of-ints format produced by
:class:`nclone.map_generation.map.Map`) into a :class:`LevelGeometry` that
captures the minimal information the planners need: the rectangular chamber
containing the ninja, the spawn / switch / exit-door pixel positions, and
flags asserting that surfaces are flat and the chamber is mine-free.

Coordinate conventions
----------------------

* ``map_data`` stores tiles at indices ``[184:1150]`` in a 42x23 grid
  addressed column-first: ``map_data[184 + gx + gy * 42]``.
* The simulator loads tile ``(gx, gy)`` at sim-tile ``(gx + 1, gy + 1)``, so
  the pixel rectangle for map-grid tile ``(gx, gy)`` is

  .. code-block:: text

      x in [(gx + 1) * 24, (gx + 2) * 24]
      y in [(gy + 1) * 24, (gy + 2) * 24]

* Entities are stored in map-data units; pixel = ``units * 6``. The ninja
  spawn and exit door/switch all include an offset of ``6`` map-data units,
  so a grid ``(gx, gy)`` spawn ends up at sim pixel
  ``((gx + 1.5) * 24, (gy + 1.5) * 24)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ...constants.physics_constants import (
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    NINJA_RADIUS,
    TILE_PIXEL_SIZE,
)


TILE_DATA_START = 184
TILE_DATA_END = 1150
NINJA_SPAWN_X_IDX = 1231
NINJA_SPAWN_Y_IDX = 1232
NINJA_ORIENTATION_IDX = 1233
ENTITY_DATA_START = 1235

EXIT_DOOR_TYPE = 3
EXIT_SWITCH_TYPE = 4
TOGGLE_MINE_TYPE = 1
TOGGLE_MINE_TOGGLED_TYPE = 21
LOCKED_DOOR_TYPE = 6
TRAP_DOOR_TYPE = 8

TILE_EMPTY = 0
TILE_SOLID = 1
"""Tile type constants. :class:`MapWallJump` uses tile 1 (full solid) for
all boundary tiles. :class:`MapHorizontalCorridor` selects from curated
tile sets (see :func:`nclone.map_generation.map.Map.set_hollow_rectangle`)
whose interior-facing face is solid at the tile boundary. In both cases,
from the ninja's perspective the boundary collides at the tile grid line,
so the planner can treat the chamber as a flat axis-aligned rectangle.
"""

BOTTOM_FACE_FLAT_TILES = frozenset({1, 2, 6, 7, 10, 11, 14, 15, 18, 19, 30, 31})
"""Tile types placed in the floor row that present a flat solid top face
at the tile grid line (i.e. at ``(tile_y) * TILE_PIXEL_SIZE``). Mirrors
``BOTTOM_WALL_TILES`` in :func:`Map.set_hollow_rectangle`."""

TOP_FACE_FLAT_TILES = frozenset({1, 4, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 32, 33})
"""Tile types placed in the ceiling row that present a flat solid bottom
face at the tile grid line. Mirrors ``TOP_WALL_TILES`` in
:func:`Map.set_hollow_rectangle`."""

RIGHT_FACE_FLAT_TILES = frozenset({1, 3, 7, 8, 11, 15, 16, 23, 24, 27, 28, 31, 32})
"""Tile types placed in the left wall column that present a flat solid
right face. Mirrors ``LEFT_WALL_TILES`` in
:func:`Map.set_hollow_rectangle`."""

LEFT_FACE_FLAT_TILES = frozenset({1, 5, 6, 9, 10, 13, 14, 17, 22, 25, 26, 29, 30, 33})
"""Tile types placed in the right wall column that present a flat solid
left face. Mirrors ``RIGHT_WALL_TILES`` in
:func:`Map.set_hollow_rectangle`."""


@dataclass(frozen=True)
class Entity:
    """Minimal representation of a parsed entity."""

    type: int
    xpos_px: float
    ypos_px: float


@dataclass(frozen=True)
class LevelGeometry:
    """Rectangular chamber containing the ninja, flat-and-mine-free flagged.

    All ``*_px`` fields are in simulator pixel coordinates. The chamber is
    the largest axis-aligned rectangle of empty tiles enclosing the ninja's
    spawn tile, bounded on all four sides by solid tiles.

    The four surface attributes mark the pixel where the ninja's body
    *surface* would touch the corresponding wall:

    * ``floor_surface_y_px`` -- pixel y of the top of the floor solid (ninja
      bottom rests at this y, so ninja center = ``floor_surface_y_px - NINJA_RADIUS``).
    * ``ceiling_surface_y_px`` -- pixel y of the bottom of the ceiling solid
      (ninja top touches this y).
    * ``left_wall_surface_x_px`` -- pixel x of the right face of the left wall.
    * ``right_wall_surface_x_px`` -- pixel x of the left face of the right wall.
    """

    tile_grid_width: int
    tile_grid_height: int

    chamber_min_gx: int
    chamber_min_gy: int
    chamber_max_gx: int
    chamber_max_gy: int

    floor_surface_y_px: float
    ceiling_surface_y_px: float
    left_wall_surface_x_px: float
    right_wall_surface_x_px: float

    ninja_spawn_px: Tuple[float, float]
    ninja_orientation: int

    switch_px: Tuple[float, float]
    exit_door_px: Tuple[float, float]

    is_flat_mine_free: bool
    rejection_reason: Optional[str]

    @property
    def chamber_interior_width_px(self) -> float:
        """Usable horizontal pixel range for the ninja *center*."""
        return (self.right_wall_surface_x_px - NINJA_RADIUS) - (
            self.left_wall_surface_x_px + NINJA_RADIUS
        )

    @property
    def chamber_interior_height_px(self) -> float:
        """Usable vertical pixel range for the ninja *center*."""
        return (self.floor_surface_y_px - NINJA_RADIUS) - (
            self.ceiling_surface_y_px + NINJA_RADIUS
        )


def _read_tile(map_data: List[int], gx: int, gy: int) -> int:
    """Return the tile type at map-grid ``(gx, gy)``, or ``-1`` if OOB."""
    if not (0 <= gx < MAP_TILE_WIDTH and 0 <= gy < MAP_TILE_HEIGHT):
        return -1
    return map_data[TILE_DATA_START + gx + gy * MAP_TILE_WIDTH]


def _parse_entities(map_data: List[int]) -> List[Entity]:
    """Parse the entity list, handling the irregular door-extra-bytes format.

    Every entity record starts with a 5-int ``[type, x, y, orient, mode]``
    block. Locked (type 6) and trap (type 8) doors append an additional
    4-byte switch descriptor; an exit door (type 3) is followed by a full
    5-byte exit-switch (type 4) record. We advance accordingly.
    """
    entities: List[Entity] = []
    i = ENTITY_DATA_START
    n = len(map_data)
    while i + 4 < n:
        etype = map_data[i]
        if etype == 0 and map_data[i + 1] == 0 and map_data[i + 2] == 0:
            break
        xu = map_data[i + 1]
        yu = map_data[i + 2]
        entities.append(Entity(type=etype, xpos_px=xu * 6.0, ypos_px=yu * 6.0))
        i += 5
        if etype == LOCKED_DOOR_TYPE or etype == TRAP_DOOR_TYPE:
            i += 4
    return entities


def _find_chamber(
    map_data: List[int], spawn_gx: int, spawn_gy: int
) -> Optional[Tuple[int, int, int, int]]:
    """Find the rectangular empty chamber containing ``(spawn_gx, spawn_gy)``.

    The chamber is the maximal rectangle such that (a) every interior tile
    is empty (``TILE_EMPTY``) and (b) every boundary tile immediately
    outside the rectangle on each side is solid. Returns
    ``(min_gx, min_gy, max_gx, max_gy)`` or ``None`` if the spawn is not
    inside a proper closed rectangle.
    """
    if _read_tile(map_data, spawn_gx, spawn_gy) != TILE_EMPTY:
        return None

    gx = spawn_gx
    while _read_tile(map_data, gx - 1, spawn_gy) == TILE_EMPTY:
        gx -= 1
    left = gx
    gx = spawn_gx
    while _read_tile(map_data, gx + 1, spawn_gy) == TILE_EMPTY:
        gx += 1
    right = gx

    gy = spawn_gy
    while _read_tile(map_data, spawn_gx, gy - 1) == TILE_EMPTY:
        gy -= 1
    top = gy
    gy = spawn_gy
    while _read_tile(map_data, spawn_gx, gy + 1) == TILE_EMPTY:
        gy += 1
    bottom = gy

    return left, top, right, bottom


def _is_chamber_flat_and_mine_free(
    map_data: List[int],
    entities: List[Entity],
    min_gx: int,
    min_gy: int,
    max_gx: int,
    max_gy: int,
) -> Tuple[bool, Optional[str]]:
    """Check that the chamber's boundary collides flat and contains no hazards.

    Rejects if:

    * Any interior tile (inside the rectangle) is non-empty.
    * Any boundary tile is not in the set of tiles that expose a flat face
      at the tile grid line on the interior-facing side (see
      :data:`BOTTOM_FACE_FLAT_TILES` etc.). This is a slightly relaxed
      check compared to a strict tile-1 requirement because
      :class:`MapHorizontalCorridor` places varied-but-equivalent-face
      tiles on each boundary via ``use_random_tiles_type=True``.
    * Any entity of toggle-mine type lies inside the rectangle pixel bounds.
    * Any locked/trap door lies inside the rectangle pixel bounds.
    """
    for gy in range(min_gy, max_gy + 1):
        for gx in range(min_gx, max_gx + 1):
            if _read_tile(map_data, gx, gy) != TILE_EMPTY:
                return False, f"non-empty interior tile at ({gx},{gy})"

    for gx in range(min_gx, max_gx + 1):
        ceil_tile = _read_tile(map_data, gx, min_gy - 1)
        if ceil_tile not in TOP_FACE_FLAT_TILES:
            return False, f"ceiling tile {ceil_tile} at ({gx},{min_gy - 1}) is not flat-faced"
        floor_tile = _read_tile(map_data, gx, max_gy + 1)
        if floor_tile not in BOTTOM_FACE_FLAT_TILES:
            return False, f"floor tile {floor_tile} at ({gx},{max_gy + 1}) is not flat-faced"
    for gy in range(min_gy, max_gy + 1):
        lw_tile = _read_tile(map_data, min_gx - 1, gy)
        if lw_tile not in RIGHT_FACE_FLAT_TILES:
            return False, f"left wall tile {lw_tile} at ({min_gx - 1},{gy}) is not flat-faced"
        rw_tile = _read_tile(map_data, max_gx + 1, gy)
        if rw_tile not in LEFT_FACE_FLAT_TILES:
            return False, f"right wall tile {rw_tile} at ({max_gx + 1},{gy}) is not flat-faced"

    left_px = (min_gx + 1) * TILE_PIXEL_SIZE
    right_px = (max_gx + 2) * TILE_PIXEL_SIZE
    top_px = (min_gy + 1) * TILE_PIXEL_SIZE
    bottom_px = (max_gy + 2) * TILE_PIXEL_SIZE

    for ent in entities:
        if ent.type not in (
            TOGGLE_MINE_TYPE,
            TOGGLE_MINE_TOGGLED_TYPE,
            LOCKED_DOOR_TYPE,
            TRAP_DOOR_TYPE,
        ):
            continue
        if left_px <= ent.xpos_px <= right_px and top_px <= ent.ypos_px <= bottom_px:
            return False, f"blocking entity type {ent.type} inside chamber"

    return True, None


def extract_level_geometry(map_data: List[int]) -> Optional[LevelGeometry]:
    """Best-effort extraction of a :class:`LevelGeometry` from ``map_data``.

    Returns ``None`` if spawn/exit entities are missing or the chamber
    cannot be identified. A non-flat or mined chamber is *still* returned
    with ``is_flat_mine_free = False`` and ``rejection_reason`` set so that
    callers can distinguish "unsupported geometry" from "unparseable map".
    """
    if len(map_data) < TILE_DATA_END:
        return None

    spawn_px = (map_data[NINJA_SPAWN_X_IDX] * 6.0, map_data[NINJA_SPAWN_Y_IDX] * 6.0)
    orientation = int(map_data[NINJA_ORIENTATION_IDX]) if len(map_data) > NINJA_ORIENTATION_IDX else 1

    entities = _parse_entities(map_data)
    exit_door = next((e for e in entities if e.type == EXIT_DOOR_TYPE), None)
    exit_switch = next((e for e in entities if e.type == EXIT_SWITCH_TYPE), None)
    if exit_door is None or exit_switch is None:
        return None

    spawn_gx = int(spawn_px[0] // TILE_PIXEL_SIZE) - 1
    spawn_gy = int(spawn_px[1] // TILE_PIXEL_SIZE) - 1
    chamber = _find_chamber(map_data, spawn_gx, spawn_gy)
    if chamber is None:
        return None
    min_gx, min_gy, max_gx, max_gy = chamber

    flat, reason = _is_chamber_flat_and_mine_free(
        map_data, entities, min_gx, min_gy, max_gx, max_gy
    )

    return LevelGeometry(
        tile_grid_width=MAP_TILE_WIDTH,
        tile_grid_height=MAP_TILE_HEIGHT,
        chamber_min_gx=min_gx,
        chamber_min_gy=min_gy,
        chamber_max_gx=max_gx,
        chamber_max_gy=max_gy,
        floor_surface_y_px=float((max_gy + 2) * TILE_PIXEL_SIZE),
        ceiling_surface_y_px=float((min_gy + 1) * TILE_PIXEL_SIZE),
        left_wall_surface_x_px=float((min_gx + 1) * TILE_PIXEL_SIZE),
        right_wall_surface_x_px=float((max_gx + 2) * TILE_PIXEL_SIZE),
        ninja_spawn_px=spawn_px,
        ninja_orientation=orientation if orientation in (-1, 1) else 1,
        switch_px=(exit_switch.xpos_px, exit_switch.ypos_px),
        exit_door_px=(exit_door.xpos_px, exit_door.ypos_px),
        is_flat_mine_free=flat,
        rejection_reason=reason,
    )


TOPOLOGY_WALL_JUMP_SHAFT = "wall_jump_shaft"
TOPOLOGY_FLAT_RUN_CORRIDOR = "flat_run_corridor"


def classify(geom: LevelGeometry) -> Optional[str]:
    """Classify ``geom`` into one of the supported topologies or ``None``.

    Heuristics match the dimensional ranges of :class:`MapWallJump` and
    :class:`MapHorizontalCorridor`:

    * *Wall-jump shaft*: *exactly one tile wide* (``chamber_tile_width == 1``)
      and tall (``chamber_tile_height >= 6``), with the exit door vertically
      separated from the spawn by more than half the chamber height. Wider
      shafts are not handled by :class:`WallJumpShaftPlanner` and classify as
      ``None`` so callers fall back to generic planning.
    * *Flat-run corridor*: wide (``chamber_tile_width >= 3``) chamber that
      is *exactly one tile tall* (``chamber_tile_height == 1``), with spawn
      and exit on the same floor (|dy| < one tile). Taller chambers are
      rejected because they admit elevated switches / doors that the
      ground-only strategies in
      :mod:`nclone.planning.optimal_route.flat_run_corridor` cannot reach;
      see the planner docstring for the rationale.

    Returns ``None`` if the chamber does not satisfy either rule, or if
    :attr:`is_flat_mine_free` is ``False``.
    """
    if not geom.is_flat_mine_free:
        return None

    chamber_tile_width = geom.chamber_max_gx - geom.chamber_min_gx + 1
    chamber_tile_height = geom.chamber_max_gy - geom.chamber_min_gy + 1

    spawn_x, spawn_y = geom.ninja_spawn_px
    door_x, door_y = geom.exit_door_px

    if (
        chamber_tile_width == 1
        and chamber_tile_height >= 6
        and abs(door_y - spawn_y) > 0.5 * chamber_tile_height * TILE_PIXEL_SIZE
    ):
        return TOPOLOGY_WALL_JUMP_SHAFT

    if (
        chamber_tile_width >= 3
        and chamber_tile_height == 1
        and abs(door_y - spawn_y) <= TILE_PIXEL_SIZE
        and abs(door_x - spawn_x) >= 0.5 * TILE_PIXEL_SIZE
    ):
        return TOPOLOGY_FLAT_RUN_CORRIDOR

    return None
