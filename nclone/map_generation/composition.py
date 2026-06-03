"""Segment-stitching engine for large composite N++ levels.

A *segment* carves one self-contained piece of terrain on a continuous ground
baseline, left-to-right, placing NO entities. ``MapComposite`` walks a sequence
of segments to build a large level that chains several mechanics (flat run,
gap jump, slope, curve, multi-jump staircase, mines) end-to-end across the
42-wide canvas.

Floor convention (shared with ``curve_utils`` / ``terrain_utils``): a flat
floor is a solid tile at ``floor_row``; the ninja walks on its top edge (the
empty cell at ``floor_row - 1``). Segments thread a running ``floor_row`` so
slopes/staircases can raise or lower the ground while keeping the surface
continuous. Every segment fills solid support beneath its terrain down to
``base_row``.

Each builder returns a :class:`SegmentResult` carrying the new x cursor, the
new floor row, and any candidate mine positions (empty cells just above the
walkable surface) -- the host generator decides how many mines to actually
place, so mine purity / budget stays centralized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import curve_utils

SOLID = 1
EMPTY = 0
DIAG_UP = 8     # "/" ascending floor, left->right
DIAG_DOWN = 9   # "\" descending floor, left->right


@dataclass
class SegmentResult:
    exit_x: int
    floor_row: int
    mine_candidates: List[Tuple[int, int]] = field(default_factory=list)
    name: str = ""
    entry_x: int = -1


# A Segment builder: (map, entry_x, floor_row, base_row, width, rng) -> SegmentResult
Segment = Callable[..., SegmentResult]


def _fill_support(m, x0: int, x1: int, top_row: int, base_row: int) -> None:
    """Fill solid from ``top_row``..``base_row`` across columns [x0, x1)."""
    for x in range(x0, x1):
        for y in range(top_row, min(base_row, MAP_TILE_HEIGHT - 1) + 1):
            if m.get_tile(x, y) == EMPTY:
                m.set_tile(x, y, SOLID)


def seg_flat(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """Flat solid floor run."""
    for xx in range(x, x + width):
        m.set_tile(xx, floor_row, SOLID)
    _fill_support(m, x, x + width, floor_row + 1, base_row)
    cands = [(xx, floor_row - 1) for xx in range(x + 1, x + width - 1)]
    return SegmentResult(x + width, floor_row, cands, "flat")


def seg_gap(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """Flat floor with a 2-3 wide pit in the middle the ninja must jump."""
    width = max(width, 6)
    for xx in range(x, x + width):
        m.set_tile(xx, floor_row, SOLID)
    _fill_support(m, x, x + width, floor_row + 1, base_row)
    gap_w = min(3, max(2, width - 4))
    gap_start = x + (width - gap_w) // 2
    for xx in range(gap_start, gap_start + gap_w):
        for y in range(floor_row, base_row + 1):
            m.set_tile(xx, y, EMPTY)
    return SegmentResult(x + width, floor_row, [], "gap")


def seg_slope_up(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """45-degree ascending ramp (clamped so it stays on the map)."""
    rise = min(width, floor_row - 2)  # don't climb past row 2
    cur_x, cur_row = x, floor_row
    for _ in range(max(1, rise)):
        m.set_tile(cur_x, cur_row, DIAG_UP)
        _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
        cur_x += 1
        cur_row -= 1
    # short flat landing so the next segment connects cleanly
    m.set_tile(cur_x, cur_row, SOLID)
    _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
    cur_x += 1
    return SegmentResult(cur_x, cur_row, [], "slope_up")


def seg_slope_down(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """45-degree descending ramp (clamped so it stays on the map)."""
    drop = min(width, base_row - floor_row - 1)
    cur_x, cur_row = x, floor_row
    for _ in range(max(1, drop)):
        m.set_tile(cur_x, cur_row, DIAG_DOWN)
        _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
        cur_x += 1
        cur_row += 1
    m.set_tile(cur_x, cur_row, SOLID)
    _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
    cur_x += 1
    return SegmentResult(cur_x, cur_row, [], "slope_down")


def seg_bowl(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """Concave bowl (ride through)."""
    nx = curve_utils.create_concave_bowl(
        m, x, floor_row, base_row, bottom_width=max(0, min(2, width - 2))
    )
    # flank with a little flat floor so it stitches cleanly
    m.set_tile(nx, floor_row, SOLID)
    _fill_support(m, nx, nx + 1, floor_row + 1, base_row)
    return SegmentResult(nx + 1, floor_row, [], "bowl")


def seg_hump(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """Convex hump (ride over)."""
    nx = curve_utils.create_convex_hump(
        m, x, floor_row, base_row, top_width=max(0, min(2, width - 2))
    )
    m.set_tile(nx, floor_row, SOLID)
    _fill_support(m, nx, nx + 1, floor_row + 1, base_row)
    return SegmentResult(nx + 1, floor_row, [], "hump")


def seg_staircase(m, x, floor_row, base_row, width, rng) -> SegmentResult:
    """A run of 1-tile step-ups requiring repeated small jumps (multi-jump)."""
    cur_x, cur_row = x, floor_row
    steps = max(2, min(width // 2, floor_row - 3))
    for _ in range(steps):
        # flat tread (2 wide), then step up one tile
        for _t in range(2):
            m.set_tile(cur_x, cur_row, SOLID)
            _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
            cur_x += 1
        cur_row -= 1  # next tread one tile higher -> must jump up
    # final landing
    m.set_tile(cur_x, cur_row, SOLID)
    _fill_support(m, cur_x, cur_x + 1, cur_row + 1, base_row)
    cur_x += 1
    return SegmentResult(cur_x, cur_row, [], "staircase")


# Registry of segment builders keyed by mechanic name.
SEGMENT_BUILDERS: dict[str, Segment] = {
    "flat": seg_flat,
    "gap": seg_gap,
    "slope_up": seg_slope_up,
    "slope_down": seg_slope_down,
    "bowl": seg_bowl,
    "hump": seg_hump,
    "staircase": seg_staircase,
}

# Mechanic segments (excludes plain flat) used to compose variety.
MECHANIC_SEGMENTS: Tuple[str, ...] = (
    "gap",
    "slope_up",
    "slope_down",
    "bowl",
    "hump",
    "staircase",
)


def assemble_segments(
    m,
    plan: List[Tuple[str, int]],
    floor_row: int,
    base_row: int,
    rng,
    start_x: int = 1,
) -> List[SegmentResult]:
    """Walk a ``[(segment_name, width), ...]`` plan left-to-right.

    Threads the x cursor and the running ``floor_row`` (clamped to the map)
    through each segment builder. Returns the per-segment results in order; the
    floor row carried into segment ``i+1`` is the exit floor of segment ``i``.

    The plan should begin and end with a ``"flat"`` segment so the ninja spawn
    and the switch/exit have clean ground to stand on.
    """
    results: List[SegmentResult] = []
    x = start_x
    cur_floor = floor_row
    for name, width in plan:
        if x >= MAP_TILE_WIDTH - 2:
            break
        cur_floor = max(2, min(MAP_TILE_HEIGHT - 3, cur_floor))
        width = min(width, MAP_TILE_WIDTH - 2 - x)
        if width < 2:
            break
        builder = SEGMENT_BUILDERS[name]
        res = builder(m, x, cur_floor, base_row, width, rng)
        res.entry_x = x
        results.append(res)
        x = res.exit_x
        cur_floor = res.floor_row
    return results


def finalize_floor_level(m, results: List[SegmentResult], max_mines: int, rng):
    """Place the ninja spawn (left), exit switch + door (right) and mines.

    Spawn sits on the first segment's floor; switch and exit door on the last
    segment's floor (which must be flat). Mines are sampled from the flat
    segments' candidate cells, never on the spawn/switch/door columns.

    Returns ``True`` on success, ``False`` if the layout is too degenerate to
    place the required entities (caller should reseed).
    """
    if len(results) < 2:
        return False
    first, last = results[0], results[-1]

    spawn_x = first.entry_x + 1
    spawn_row = first.floor_row - 1

    door_x = last.exit_x - 2
    switch_x = last.exit_x - 4
    exit_row = last.floor_row - 1
    if switch_x <= spawn_x + 1 or door_x <= switch_x:
        return False

    reserved = {spawn_x, switch_x, door_x}
    m.set_ninja_spawn(spawn_x, spawn_row, orientation=1)
    m.add_entity(3, door_x, exit_row, 0, 0, switch_x, exit_row)

    if max_mines > 0:
        candidates = [
            (cx, cy)
            for res in results
            for (cx, cy) in res.mine_candidates
            if cx not in reserved and spawn_x + 1 < cx < door_x - 1
        ]
        rng.shuffle(candidates)
        for mx, my in candidates[:max_mines]:
            m.add_entity(1, mx, my)
    return True
