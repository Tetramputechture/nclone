"""Physics-preserving *translation* augmentation on raw ``map_data``.

N++ physics is position-invariant: gravity, collision response, and every game
mechanic depend only on the ninja's position *relative* to the surrounding
geometry, never on absolute coordinates. Shifting the entire playable region
(tiles + entities + ninja spawn) by an integer number of tiles therefore yields
a level whose original winning input sequence still succeeds -- and, crucially,
whose *difficulty is unchanged* while its absolute playspace position differs.
This is the spatial-variety augmentation that forces a CoordConv policy to
generalize over playspace position instead of memorizing it.

This is the raw-``map_data`` analogue of
:class:`nclone.map_augmentation.attract.reachable_area_translator.ReachableAreaTranslator`
(which operates on the decoded ``AttractData`` dict). It reuses the same
reachability-graph bounding-box logic but emits/consumes the simulator
``map_data`` integer-list format produced by :meth:`Map.map_data` and read from
official ``.map`` binaries -- the format the benchmark dataset stores.

Coordinate conventions (see :mod:`nclone.map_augmentation.transforms`):

* Tiles live in ``map_data[184:1150]`` -- the 42x23 *interior* grid, row-major
  (``index = col + row * 42``).
* The ninja spawn (``map_data[1231:1233]``) and every entity record's ``x, y``
  (5-int records from ``map_data[1235]``) live in the *absolute* map-data frame
  (1 tile = ``GRID_SIZE_FACTOR`` = 4 units; the absolute frame includes the
  1-tile border, so a tile shift of ``dx`` columns is a coordinate shift of
  ``dx * 4`` units).

A translation by ``(dx, dy)`` tiles shifts the interior tile grid by ``dx``
columns / ``dy`` rows and every absolute coordinate by ``dx * 4`` / ``dy * 4``
units. Vacated interior tiles are left SOLID (type 1) -- the ninja can never
reach them, and a solid fill (unlike random fill) keeps the variant trivially
byte-stable and never introduces a stray reachable pocket.
"""
from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..map_generation.constants import GRID_SIZE_FACTOR

_TILE_START = 184
_TILE_END = 1150
_NINJA_X = 1231
_NINJA_Y = 1232
_ENTITY_START = 1235
_ENTITY_STRIDE = 5
_UNITS_PER_TILE = int(GRID_SIZE_FACTOR)
_W = MAP_TILE_WIDTH
_H = MAP_TILE_HEIGHT
_SOLID_TILE = 1


def reachable_tile_bbox(
    map_data: List[int], buffer: int
) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(min_col, min_row, max_col, max_row)`` of the protected region.

    The protected region is the set of tiles the ninja can reach from spawn
    (via the jump-aware reachability graph) expanded by ``buffer`` tiles in
    every direction -- the geometry that physically interacts with the ninja and
    must move together. Returns ``None`` if reachability can't be computed
    (caller should then skip translating that level).
    """
    from ..map_generation.solvability import _build_level_data, _reachable_set
    from ..graph.reachability.graph_builder import CELL_SIZE

    level_data = _build_level_data(list(map_data))
    reachable, _adjacency = _reachable_set(level_data)
    if not reachable:
        return None
    tiles: Set[Tuple[int, int]] = set()
    for px, py in reachable:
        col = px // CELL_SIZE
        row = py // CELL_SIZE
        for dc in range(-buffer, buffer + 1):
            for dr in range(-buffer, buffer + 1):
                nc = col + dc
                nr = row + dr
                if not (0 <= nc < _W):
                    continue
                if not (0 <= nr < _H):
                    continue
                tiles.add((nc, nr))
    if not tiles:
        return None
    cols = [c for c, _ in tiles]
    rows = [r for _, r in tiles]
    return (min(cols), min(rows), max(cols), max(rows))


def translate_map_data(map_data: List[int], dx: int, dy: int) -> List[int]:
    """Return a copy of ``map_data`` with everything shifted by ``(dx, dy)`` tiles.

    The interior tile grid is shifted by ``dx`` columns / ``dy`` rows (vacated
    cells filled SOLID); the ninja spawn and every entity coordinate are shifted
    by ``dx * 4`` / ``dy * 4`` absolute units. Pure: input is not modified.
    """
    out = list(map_data)
    src = map_data[_TILE_START:_TILE_END]
    new_tiles = [_SOLID_TILE] * len(src)
    for row in range(_H):
        nr = row + dy
        if not (0 <= nr < _H):
            continue
        for col in range(_W):
            nc = col + dx
            if not (0 <= nc < _W):
                continue
            new_tiles[nc + nr * _W] = src[col + row * _W]
    out[_TILE_START:_TILE_END] = new_tiles
    x_shift = dx * _UNITS_PER_TILE
    y_shift = dy * _UNITS_PER_TILE
    out[_NINJA_X] = map_data[_NINJA_X] + x_shift
    out[_NINJA_Y] = map_data[_NINJA_Y] + y_shift
    i = _ENTITY_START
    n = len(map_data)
    while i + 4 < n:
        out[i + 1] = map_data[i + 1] + x_shift
        out[i + 2] = map_data[i + 2] + y_shift
        i += _ENTITY_STRIDE
    return out


def _feasible_offsets(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Return ``(max_left, max_right, max_up, max_down)`` for a bbox.

    The max shift in each direction that keeps the bounding box inside the
    interior 42x23 grid.
    """
    (min_col, min_row, max_col, max_row) = bbox
    max_left = min_col
    max_right = _W - 1 - max_col
    max_up = min_row
    max_down = _H - 1 - max_row
    return (max_left, max_right, max_up, max_down)


def sample_translations(
    map_data: List[int], k: int, *, seed: Optional[int] = None, buffer: int = 1
) -> List[Tuple[Tuple[int, int], List[int]]]:
    """Return up to ``k`` distinct non-identity translated variants.

    Each element is ``((dx, dy), translated_map_data)``. Returns ``[]`` when the
    reachable bounding box already fills the grid (no room to translate) or when
    reachability could not be computed. Offsets are sampled deterministically
    from ``seed`` and de-duplicated; fewer than ``k`` may be returned if the
    feasible offset set is smaller than ``k``.
    """
    bbox = reachable_tile_bbox(map_data, buffer=buffer)
    if bbox is None:
        return []
    max_left, max_right, max_up, max_down = _feasible_offsets(bbox)
    n_x = max_left + max_right + 1
    n_y = max_up + max_down + 1
    n_feasible = n_x * n_y - 1
    if n_feasible <= 0:
        return []
    rng = random.Random(seed)
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[Tuple[int, int], List[int]]] = []
    target = min(k, n_feasible)
    attempts = 0
    max_attempts = target * 50 + 50
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        dx = rng.randint(-max_left, max_right)
        dy = rng.randint(-max_up, max_down)
        if (dx, dy) == (0, 0) or (dx, dy) in seen:
            continue
        seen.add((dx, dy))
        out.append(((dx, dy), translate_map_data(map_data, dx, dy)))
    return out
