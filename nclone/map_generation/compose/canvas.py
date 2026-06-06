"""Canvas / MapBuilder: the region grid, coordinate translation, and the
guaranteed on-path corridor carving that lets us defer physics solvability.

The Canvas wraps a :class:`Map`, partitions the interior into a region grid, and
threads a single seeded RNG (``Map.rng``) so generation is bit-reproducible. It
exposes a shared horizontal traversal baseline (``path_floor_row``) and bottom
support floor (``base_row``) -- the same contract the 1D segment engine uses --
so consecutive path cells share a continuous floor BY CONSTRUCTION; carving only
has real work to do when a template deliberately steps floor heights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..map import Map
from . import region as region_mod
from . import connectors
from .component import Component, ComponentContext, StampResult, SOLID, EMPTY

PIT_DEPTH = 5  # tiles of empty space below the baseline so gaps are real pits
HEAD_CLEAR = 4  # empty rows kept above the floor at cell boundaries


class Canvas:
    def __init__(self, seed: Optional[int] = None, rows: int = 2, cols: int = 3,
                 jitter: bool = False):
        self.map = Map(seed)
        self.rng = self.map.rng
        self.rows = rows
        self.cols = cols
        self.base_row = MAP_TILE_HEIGHT - 2  # 21
        self.path_floor_row = self.base_row - PIT_DEPTH  # common walking baseline
        jr = self.rng if jitter else None
        self.grid: List[List[region_mod.Region]] = region_mod.partition(rows, cols, jitter_rng=jr)
        self._stamps: Dict[Tuple[int, int], StampResult] = {}

    # --- grid access ----------------------------------------------------------
    def region(self, row: int, col: int) -> region_mod.Region:
        return self.grid[row][col]

    def context_for(self, region: region_mod.Region, params: Dict,
                    floor_row: Optional[int] = None) -> ComponentContext:
        fr = self.path_floor_row if floor_row is None else floor_row
        return ComponentContext(self, region, params, self.rng, fr, self.base_row)

    def stamp_component(self, region: region_mod.Region, component: Component,
                        floor_row: Optional[int] = None) -> StampResult:
        params = component.sample_params(self.rng)
        ctx = self.context_for(region, params, floor_row)
        res = component.stamp(ctx)
        # merge the component's declared tags/difficulty with what stamp realized
        res.tags = set(res.tags) | set(component.tags)
        for k, v in (component.difficulty or {}).items():
            res.difficulty[k] = max(res.difficulty.get(k, 0), v)
        self._stamps[(region.row, region.col)] = res
        return res

    # --- structure ------------------------------------------------------------
    def border(self) -> None:
        """1-tile solid frame around the canvas."""
        m = self.map
        for x in range(MAP_TILE_WIDTH):
            m.set_tile(x, 0, SOLID)
            m.set_tile(x, MAP_TILE_HEIGHT - 1, SOLID)
        for y in range(MAP_TILE_HEIGHT):
            m.set_tile(0, y, SOLID)
            m.set_tile(MAP_TILE_WIDTH - 1, y, SOLID)

    # --- the main technical burden: guaranteed on-path connectivity -----------
    def carve_corridor(self, a: region_mod.Region, b: region_mod.Region) -> None:
        """Guarantee a passable connection between consecutive on-path cells ``a``->``b``."""
        if a.row == b.row and abs(a.col - b.col) == 1:
            self._carve_horizontal(a, b)
        elif a.col == b.col and abs(a.row - b.row) == 1:
            self._carve_vertical(a, b)
        # diagonal steps are not produced by the monotone-walk path policy

    def _edge_row(self, region: region_mod.Region, edge: str) -> int:
        res = self._stamps.get((region.row, region.col))
        if res and edge in res.edge_rows:
            return res.edge_rows[edge]
        return self.path_floor_row

    def _carve_horizontal(self, a: region_mod.Region, b: region_mod.Region) -> None:
        left, right = (a, b) if a.col < b.col else (b, a)
        boundary = right.x  # right cell's first column == left.x1 + 1 (contiguous)
        floor_l = self._edge_row(left, "E")
        floor_r = self._edge_row(right, "W")
        # ensure a solid, continuous floor across the seam at the lower of the two
        seam_floor = max(floor_l, floor_r)
        for gx in (boundary - 1, boundary):
            self.map.set_tile(gx, seam_floor, SOLID)
            for gy in range(seam_floor + 1, self.base_row + 1):
                if self.map.get_tile(gx, gy) == EMPTY:
                    self.map.set_tile(gx, gy, SOLID)
            # head clearance above the seam
            for gy in range(max(1, seam_floor - HEAD_CLEAR), seam_floor):
                self.map.set_tile(gx, gy, EMPTY)
        # stepped floors: drop a 45 connector ramp across the seam
        if abs(floor_l - floor_r) > 1:
            lo, hi = min(floor_l, floor_r), max(floor_l, floor_r)
            if floor_l < floor_r:  # descending into the right cell
                connectors.slope_down(self.map, boundary - 1, lo, hi - lo,
                                      kind="45", base_row=self.base_row)
            else:
                connectors.slope_up(self.map, boundary - 1, hi, hi - lo,
                                    kind="45", base_row=self.base_row)

    def _carve_vertical(self, a: region_mod.Region, b: region_mod.Region) -> None:
        top, bottom = (a, b) if a.row < b.row else (b, a)
        # open a 2-wide empty shaft at a shared column across the boundary
        col = top.x + top.w // 2
        y_from = top.y + 1
        y_to = bottom.y1 - 1
        for gx in (col, min(col + 1, top.x1)):
            for gy in range(y_from, y_to + 1):
                self.map.set_tile(gx, gy, EMPTY)

    # --- output ---------------------------------------------------------------
    def to_map(self) -> Map:
        return self.map

    def map_data(self):
        return self.map.map_data()

    def to_ascii(self, detail: str = "compact", grid: bool = True,
                 path: bool = True, cell=None) -> str:
        from .visualize import to_compact_ascii
        path_cells = getattr(self, "_path_cells", None) if path else None
        grid_obj = self.grid if grid else None
        return to_compact_ascii(self.map.map_data(), grid=grid_obj, path=path_cells)
