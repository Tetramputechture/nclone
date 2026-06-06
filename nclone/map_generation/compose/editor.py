"""Free-form map editor for building a level from scratch.

Where the template system composes labelled fragments for curriculum data, this
is the *direct* authoring surface an agent uses to draw an arbitrary level --
including pictorial ones ("a level that looks like a smiley face") -- with single
and batch tile/entity tools, then validate that it is actually playable via
:meth:`MapEditor.check`.

Coordinates are integer grid tiles (x in [0, 41], y in [0, 22]); (0,0) is top-left,
y increases DOWN, matching the ASCII view and the physics. All ops clamp/skip
out-of-bounds writes. Entity placement is idempotent (the editor owns its entity
lists and synthesizes a ``Map`` on demand), so re-calling ``place_player`` /
``place_exit`` overwrites rather than duplicating.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..map import Map
from .coords import PX_PER_TILE  # 24px per tile

SOLID = 1
EMPTY = 0

# Entities are stored in integer map-data units where 1 unit = 6px = a quarter
# tile (PX_PER_TILE / 4). That is the finest byte-safe placement step -- NOT 1px
# -- but still 4x finer per axis than the tile grid. ``add_mine``/``add_gold``
# accept floats and snap to this lattice; by default they bias to the tile CENTRE
# (anchor="center") because a bare integer grid coord lands an entity on the tile
# *corner* (the seam between four tiles), which reads as up-and-left of intent.
SUBTILE = 0.25  # quarter-tile = 1 map-unit = 6px


def _snap_entity(x: float, y: float, anchor: str) -> Tuple[float, float]:
    """Snap a desired entity position to the 6px lattice. ``anchor="center"``
    biases by +0.5 tile so an integer ``(x, y)`` means "centre of tile (x, y)";
    ``anchor="corner"`` treats ``(x, y)`` as an absolute sub-tile coordinate."""
    bias = 0.5 if anchor == "center" else 0.0
    return (round((x + bias) / SUBTILE) * SUBTILE,
            round((y + bias) / SUBTILE) * SUBTILE)


# --- spatial anchors (grid coords; the playable interior is inside the border) -
# Map is MAP_TILE_WIDTH x MAP_TILE_HEIGHT tiles (42 x 23). These let an agent say
# "in the middle" / "near the top" without re-deriving the geometry every time.
MIDDLE = (MAP_TILE_WIDTH / 2.0, MAP_TILE_HEIGHT / 2.0)          # (21.0, 11.5)
CENTER = MIDDLE
TOP_MIDDLE = (MAP_TILE_WIDTH / 2.0, 2.0)
BOTTOM_MIDDLE = (MAP_TILE_WIDTH / 2.0, MAP_TILE_HEIGHT - 2.0)
LEFT_MIDDLE = (2.0, MAP_TILE_HEIGHT / 2.0)
RIGHT_MIDDLE = (MAP_TILE_WIDTH - 2.0, MAP_TILE_HEIGHT / 2.0)
TOP_LEFT = (2.0, 2.0)
TOP_RIGHT = (MAP_TILE_WIDTH - 2.0, 2.0)
BOTTOM_LEFT = (2.0, MAP_TILE_HEIGHT - 2.0)
BOTTOM_RIGHT = (MAP_TILE_WIDTH - 2.0, MAP_TILE_HEIGHT - 2.0)


def above(p: Tuple[float, float], n: float = 1.0) -> Tuple[float, float]:
    return (p[0], p[1] - n)


def below(p: Tuple[float, float], n: float = 1.0) -> Tuple[float, float]:
    return (p[0], p[1] + n)


def left_of(p: Tuple[float, float], n: float = 1.0) -> Tuple[float, float]:
    return (p[0] - n, p[1])


def right_of(p: Tuple[float, float], n: float = 1.0) -> Tuple[float, float]:
    return (p[0] + n, p[1])

# friendly tile-name aliases for ASCII legends (raw ints also accepted)
TILE_NAMES: Dict[str, int] = {
    "empty": 0, "air": 0, " ": 0, ".": 0,
    "solid": 1, "wall": 1, "block": 1, "#": 1,
    "slope_45_up": 8, "slope_45_down": 9,
    "hump_l": 12, "hump_r": 13, "bowl_l": 17, "bowl_r": 16,
}

# special legend tokens that place ENTITIES instead of tiles
_ENTITY_TOKENS = {"player", "ninja", "mine", "gold", "exit_door", "door",
                  "exit_switch", "switch"}


class MapEditor:
    def __init__(self, seed: int = 0, fill: str = "empty"):
        self._map = Map(seed)
        self.rng = self._map.rng
        if fill in ("solid", "wall"):
            self._map.tile_data = [SOLID] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        self._mines: List[Tuple[int, int]] = []
        self._gold: List[Tuple[int, int]] = []
        self._player: Optional[Tuple[int, int, int]] = None
        self._exit: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    # --- bounds ---------------------------------------------------------------
    @staticmethod
    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT

    # --- single tile ----------------------------------------------------------
    def set_tile(self, x: int, y: int, tile: int = SOLID) -> "MapEditor":
        if self.in_bounds(int(x), int(y)):
            self._map.set_tile(int(x), int(y), int(tile))
        return self

    def get_tile(self, x: int, y: int) -> int:
        return self._map.get_tile(int(x), int(y))

    def erase(self, x: int, y: int) -> "MapEditor":
        return self.set_tile(x, y, EMPTY)

    # --- single entity (idempotent) ------------------------------------------
    def place_player(self, x: int, y: int, facing: str = "right") -> "MapEditor":
        self._player = (int(x), int(y), 1 if facing == "right" else -1)
        return self

    def place_exit(self, switch: Tuple[int, int], door: Tuple[int, int]) -> "MapEditor":
        """Set the (single) required objective pair: exit switch + exit door."""
        self._exit = ((int(switch[0]), int(switch[1])), (int(door[0]), int(door[1])))
        return self

    def add_mine(self, x: float, y: float, anchor: str = "center") -> "MapEditor":
        """Place a toggle mine. Accepts floats; snaps to the 6px lattice. Default
        ``anchor="center"`` centres it in tile ``(int(x), int(y))``; use
        ``anchor="corner"`` to treat ``(x, y)`` as an absolute sub-tile point."""
        gx, gy = _snap_entity(x, y, anchor)
        if self.in_bounds(int(gx), int(gy)):
            self._mines.append((gx, gy))
        return self

    def add_gold(self, x: float, y: float, anchor: str = "center") -> "MapEditor":
        gx, gy = _snap_entity(x, y, anchor)
        if self.in_bounds(int(gx), int(gy)):
            self._gold.append((gx, gy))
        return self

    # --- batch tiles ----------------------------------------------------------
    def fill_rect(self, x0, y0, x1, y1, tile: int = SOLID) -> "MapEditor":
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_tile(x, y, tile)
        return self

    def clear_rect(self, x0, y0, x1, y1) -> "MapEditor":
        return self.fill_rect(x0, y0, x1, y1, EMPTY)

    def rect_outline(self, x0, y0, x1, y1, tile: int = SOLID) -> "MapEditor":
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        for x in range(x0, x1 + 1):
            self.set_tile(x, y0, tile)
            self.set_tile(x, y1, tile)
        for y in range(y0, y1 + 1):
            self.set_tile(x0, y, tile)
            self.set_tile(x1, y, tile)
        return self

    def border(self, tile: int = SOLID) -> "MapEditor":
        return self.rect_outline(0, 0, MAP_TILE_WIDTH - 1, MAP_TILE_HEIGHT - 1, tile)

    def hline(self, y, x0, x1, tile: int = SOLID) -> "MapEditor":
        for x in range(min(int(x0), int(x1)), max(int(x0), int(x1)) + 1):
            self.set_tile(x, y, tile)
        return self

    def vline(self, x, y0, y1, tile: int = SOLID) -> "MapEditor":
        for y in range(min(int(y0), int(y1)), max(int(y0), int(y1)) + 1):
            self.set_tile(x, y, tile)
        return self

    def line(self, x0, y0, x1, y1, tile: int = SOLID) -> "MapEditor":
        """Bresenham line between two tiles."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.set_tile(x0, y0, tile)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return self

    def disk(self, cx, cy, r, tile: int = SOLID) -> "MapEditor":
        """Filled circle of radius ``r`` centred at (cx, cy)."""
        cx, cy, r = int(cx), int(cy), int(r)
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r + r:
                    self.set_tile(x, y, tile)
        return self

    def ring(self, cx, cy, r, tile: int = SOLID, thickness: float = 1.0) -> "MapEditor":
        """Circle outline of radius ``r`` (annulus of given thickness)."""
        cx, cy, r = int(cx), int(cy), int(r)
        outer2 = (r + 0.5) ** 2
        inner2 = (r - thickness + 0.5) ** 2
        for y in range(cy - r - 1, cy + r + 2):
            for x in range(cx - r - 1, cx + r + 2):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if inner2 <= d2 <= outer2:
                    self.set_tile(x, y, tile)
        return self

    def arc(self, cx, cy, r, a0_deg: float, a1_deg: float, tile: int = SOLID,
            step_deg: float = 4.0) -> "MapEditor":
        """Arc of radius ``r`` from ``a0`` to ``a1`` degrees (0=east, CW since y is down)."""
        cx, cy = int(cx), int(cy)
        a = a0_deg
        while a <= a1_deg:
            rad = math.radians(a)
            self.set_tile(round(cx + r * math.cos(rad)), round(cy + r * math.sin(rad)), tile)
            a += step_deg
        return self

    def set_tiles(self, coords: Iterable[Tuple[int, int]], tile: int = SOLID) -> "MapEditor":
        for x, y in coords:
            self.set_tile(x, y, tile)
        return self

    # --- batch entities -------------------------------------------------------
    def add_mines(self, coords: Iterable[Tuple[float, float]],
                  anchor: str = "center") -> "MapEditor":
        for x, y in coords:
            self.add_mine(x, y, anchor=anchor)
        return self

    def fill_mines(self, x0, y0, x1, y1, spacing_px: float = 12.0,
                   mask=None, on_empty_only: bool = True) -> "MapEditor":
        """Tile a region with mines on a sub-tile lattice so a *shape* reads as a
        solid cluster instead of a few lonely dots. ``spacing_px`` is the gap
        between mines in pixels (12 = every half-tile, 6 = every quarter-tile).
        ``mask(gx, gy) -> bool`` restricts placement to an arbitrary shape; the
        default fills the whole bounding box. Mines that would land in a solid
        tile are skipped when ``on_empty_only`` (the default)."""
        step = spacing_px / PX_PER_TILE  # tiles per step (12px -> 0.5 tile)
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        gy = y0
        while gy <= y1 + 1e-9:
            gx = x0
            while gx <= x1 + 1e-9:
                if mask is None or mask(gx, gy):
                    if not on_empty_only or self.get_tile(int(gx), int(gy)) == EMPTY:
                        # gx, gy are absolute sub-tile points -> place verbatim
                        self.add_mine(gx, gy, anchor="corner")
                gx += step
            gy += step
        return self

    def mine_disk(self, cx, cy, r, spacing_px: float = 12.0,
                  on_empty_only: bool = True) -> "MapEditor":
        """Fill a disk of radius ``r`` tiles (float ok) with a mine cluster."""
        r2 = float(r) * float(r)
        return self.fill_mines(
            cx - r, cy - r, cx + r, cy + r, spacing_px=spacing_px,
            mask=lambda gx, gy: (gx - cx) ** 2 + (gy - cy) ** 2 <= r2,
            on_empty_only=on_empty_only,
        )

    def scatter_mines(self, x0, y0, x1, y1, n: int, on_empty_only: bool = True) -> "MapEditor":
        """Place ``n`` mines at random EMPTY tiles inside a rectangle (seeded)."""
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        cells = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                 if (not on_empty_only or self.get_tile(x, y) == EMPTY)]
        self.rng.shuffle(cells)
        for x, y in cells[:n]:
            self.add_mine(x, y)
        return self

    # --- ASCII-art drawing (the pictorial entry point) ------------------------
    def draw_ascii(self, art: str, legend: Dict[str, object], origin=(0, 0),
                   default: Optional[object] = None) -> "MapEditor":
        """Paint a multi-line string. ``legend`` maps each char to a tile id, a
        tile name (see :data:`TILE_NAMES`), or an entity token
        (``player``/``mine``/``gold``/``exit_door``/``exit_switch``).

        A char absent from the legend uses ``default`` (``None`` = leave as-is).
        The exit door + switch are paired automatically if both tokens appear.
        """
        ox, oy = origin
        door_pos = switch_pos = None
        lines = art.strip("\n").splitlines()
        for j, row in enumerate(lines):
            for i, ch in enumerate(row):
                spec = legend.get(ch, default)
                if spec is None:
                    continue
                x, y = ox + i, oy + j
                tok = spec.lower() if isinstance(spec, str) else spec
                if tok in ("player", "ninja"):
                    self.place_player(x, y)
                elif tok == "mine":
                    self.add_mine(x, y)
                elif tok == "gold":
                    self.add_gold(x, y)
                elif tok in ("exit_door", "door"):
                    door_pos = (x, y)
                elif tok in ("exit_switch", "switch"):
                    switch_pos = (x, y)
                else:
                    self.set_tile(x, y, self._resolve_tile(spec))
        if door_pos and switch_pos:
            self.place_exit(switch_pos, door_pos)
        return self

    @staticmethod
    def _resolve_tile(spec) -> int:
        if isinstance(spec, int):
            return spec
        if isinstance(spec, str) and spec in TILE_NAMES:
            return TILE_NAMES[spec]
        raise ValueError(f"unknown tile spec {spec!r} (use an int id or a TILE_NAMES key)")

    # --- output ---------------------------------------------------------------
    def _build_map(self) -> Map:
        m = self._map
        m.entity_data = []
        m.entity_counts = {"exit_door": 0, "gold": 0, "death_ball": 0}
        if self._player is not None:
            m.set_ninja_spawn(self._player[0], self._player[1], self._player[2])
        if self._exit is not None:
            (sx, sy), (dx, dy) = self._exit
            m.add_entity(3, dx, dy, 0, 0, sx, sy)
        for x, y in self._mines:
            m.add_entity(1, x, y)
        for x, y in self._gold:
            m.add_entity(2, x, y)
        return m

    def map_data(self) -> List[int]:
        return self._build_map().map_data()

    def to_map(self) -> Map:
        return self._build_map()

    def to_ascii(self) -> str:
        from .visualize import to_compact_ascii

        return to_compact_ascii(self.map_data())

    def describe(self) -> Dict:
        from .visualize import describe_level

        return describe_level(self.map_data())

    def render(self, out_path: str, backend: str = "schematic", title: str = "level") -> str:
        from .render_image import render_schematic, render_ground_truth

        md = self.map_data()
        if backend == "ground_truth":
            return render_ground_truth(md, out_path)
        return render_schematic(md, out_path, title=title)

    def check(self) -> Dict:
        """Structural + solvability report. See :func:`validate.check_level`."""
        from .validate import check_level

        return check_level(self.map_data())

    def save(self, path: str, fmt: Optional[str] = None,
             metadata: Optional[Dict] = None, **extra) -> str:
        """Save the level to disk (binary ``.map`` / ``.json`` / ``.pkl``).

        Binary is directly loadable by the simulator / RL env; see
        :mod:`nclone.map_generation.compose.mapio`. Returns ``path``.
        """
        from .mapio import save_map

        return save_map(self.map_data(), path, fmt=fmt, metadata=metadata, **extra)
