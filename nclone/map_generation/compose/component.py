"""Component base, ComponentContext, and parameter-range types.

A *component* is the unit an agent authors: declarative metadata (tags, params,
ports, difficulty) plus one imperative ``stamp(ctx)`` method that paints terrain
and entities into ONE region. The :class:`ComponentContext` it receives works in
GLOBAL grid coordinates but **clamps every write to the component's column band**
so a component physically cannot scribble into a neighbouring cell. It threads a
shared ``floor_row`` baseline and ``base_row`` support floor (the same contract
the 1D ``composition`` segment engine uses), so a cell is effectively a bounded
1D strip and the existing segment builders can fill it directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from ...constants import MAP_TILE_HEIGHT
from .region import Region

SOLID = 1
EMPTY = 0


# --- parameter ranges (the framework samples these from the seeded RNG) -------
class ParamRange:
    kind = "param"

    def sample(self, rng: random.Random) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": self.kind}


class IntRange(ParamRange):
    kind = "int"

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi

    def sample(self, rng):
        return rng.randint(self.lo, self.hi)

    def describe(self):
        return {"type": "int", "lo": self.lo, "hi": self.hi}


class FloatRange(ParamRange):
    kind = "float"

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def sample(self, rng):
        return rng.uniform(self.lo, self.hi)

    def describe(self):
        return {"type": "float", "lo": self.lo, "hi": self.hi}


class Choice(ParamRange):
    kind = "choice"

    def __init__(self, options: List[Any]):
        self.options = list(options)

    def sample(self, rng):
        return rng.choice(self.options)

    def describe(self):
        return {"type": "choice", "options": list(self.options)}


@dataclass
class StampResult:
    """What a component realized: edge floor rows, entities, tags, difficulty."""

    edge_rows: Dict[str, int]  # {"W": global_floor_row, "E": ...}
    entities: List[Tuple] = field(default_factory=list)  # (type, gx, gy, ...)
    tags: set = field(default_factory=set)
    difficulty: Dict[str, int] = field(default_factory=dict)
    placed_spawn: bool = False
    placed_goal: bool = False


class ComponentContext:
    """Region-clamped, global-coordinate drawing surface handed to ``stamp``."""

    def __init__(
        self,
        canvas,
        region: Region,
        params: Dict[str, Any],
        rng: random.Random,
        floor_row: int,
        base_row: int,
    ):
        self.canvas = canvas
        self.map = canvas.map
        self.region = region
        self.rng = rng
        self.p = SimpleNamespace(**params)
        self.floor_row = int(floor_row)  # global baseline the ninja walks on
        self.base_row = int(base_row)  # global bottom of solid support
        # column band this component owns (hard horizontal clamp):
        self.x0 = region.x
        self.x1 = region.x1
        # bookkeeping
        self._entities: List[Tuple] = []
        self._tags: set = set()
        self._difficulty: Dict[str, int] = {}
        self._edge_rows: Dict[str, int] = {"W": self.floor_row, "E": self.floor_row}
        self._placed_spawn = False
        self._placed_goal = False

    # --- geometry helpers (all clamp x to the owned band) ---------------------
    @property
    def width(self) -> int:
        return self.region.w

    def _in_band(self, gx: int, gy: int) -> bool:
        return self.x0 <= gx <= self.x1 and 1 <= gy <= MAP_TILE_HEIGHT - 2

    def set_tile(self, gx: int, gy: int, tile_id: int) -> None:
        if self._in_band(int(gx), int(gy)):
            self.map.set_tile(int(gx), int(gy), tile_id)

    def get_tile(self, gx: int, gy: int) -> int:
        return self.map.get_tile(int(gx), int(gy))

    def fill_support(self, gx: int, top: int, base: Optional[int] = None) -> None:
        """Fill solid from ``top`` down to ``base`` (default ``base_row``) in column gx."""
        base = self.base_row if base is None else base
        for gy in range(int(top), int(base) + 1):
            if self.get_tile(gx, gy) == EMPTY:
                self.set_tile(gx, gy, SOLID)

    def floor(self, row: Optional[int] = None, x0: Optional[int] = None,
              x1: Optional[int] = None) -> None:
        """Solid floor run at ``row`` (default baseline) with support below it."""
        row = self.floor_row if row is None else int(row)
        x0 = self.x0 if x0 is None else int(x0)
        x1 = self.x1 if x1 is None else int(x1)
        for gx in range(x0, x1 + 1):
            self.set_tile(gx, row, SOLID)
            self.fill_support(gx, row + 1)

    def clear_column(self, gx: int, top: int, bottom: int) -> None:
        for gy in range(int(top), int(bottom) + 1):
            self.set_tile(gx, gy, EMPTY)

    def spread_x(self, n: int, margin: int = 1) -> List[int]:
        """``n`` evenly-spaced integer columns inside the band (seeded order kept)."""
        lo = self.x0 + margin
        hi = self.x1 - margin
        if hi <= lo or n <= 0:
            return []
        if n == 1:
            return [(lo + hi) // 2]
        step = (hi - lo) / (n - 1)
        return [int(round(lo + i * step)) for i in range(n)]

    # --- entity helpers -------------------------------------------------------
    def add_mine(self, gx: int, gy: int) -> None:
        self.map.add_entity(1, int(gx), int(gy))
        self._entities.append((1, int(gx), int(gy)))

    def add_gold(self, gx: int, gy: int) -> None:
        self.map.add_entity(2, int(gx), int(gy))
        self._entities.append((2, int(gx), int(gy)))

    def place_spawn(self, gx: int, gy: int, orient: int = 1) -> None:
        self.map.set_ninja_spawn(int(gx), int(gy), orient)
        self._placed_spawn = True

    def place_switch_door(self, sx: int, sy: int, dx: int, dy: int) -> None:
        self.map.add_entity(3, int(dx), int(dy), 0, 0, int(sx), int(sy))
        self._entities.append((3, int(dx), int(dy)))
        self._entities.append((4, int(sx), int(sy)))
        self._placed_goal = True

    # --- connector / segment delegation (see connectors.py) -------------------
    def slope_up(self, gx: int, gy: int, height: int, kind: str = "45") -> int:
        from . import connectors
        nx, surf, tag = connectors.slope_up(self.map, gx, gy, height, kind,
                                             base_row=self.base_row,
                                             x_max=self.x1)
        self._tags.add(tag)
        self._edge_rows["E"] = surf
        return nx

    def slope_down(self, gx: int, gy: int, height: int, kind: str = "45") -> int:
        from . import connectors
        nx, surf, tag = connectors.slope_down(self.map, gx, gy, height, kind,
                                               base_row=self.base_row,
                                               x_max=self.x1)
        self._tags.add(tag)
        self._edge_rows["E"] = surf
        return nx

    def hump(self, gx: int, top_width: int = 0) -> int:
        from . import connectors
        nx, tag = connectors.convex_hump(self.map, gx, self.floor_row,
                                         self.base_row, top_width, x_max=self.x1)
        self._tags.add(tag)
        return nx

    def bowl(self, gx: int, bottom_width: int = 0) -> int:
        from . import connectors
        nx, tag = connectors.concave_bowl(self.map, gx, self.floor_row,
                                          self.base_row, bottom_width, x_max=self.x1)
        self._tags.add(tag)
        return nx

    def fill_cell_with_segment(self, seg_name: str, width: Optional[int] = None):
        """Fill this cell's band using a 1D ``composition`` segment builder."""
        from ..composition import SEGMENT_BUILDERS
        w = self.width if width is None else min(width, self.width)
        res = SEGMENT_BUILDERS[seg_name](
            self.map, self.x0, self.floor_row, self.base_row, w, self.rng
        )
        return res

    # --- tagging --------------------------------------------------------------
    def tag(self, *tags: str) -> None:
        self._tags.update(tags)

    def set_edge(self, edge: str, row: int) -> None:
        self._edge_rows[edge] = int(row)

    def result(self) -> StampResult:
        return StampResult(
            edge_rows=dict(self._edge_rows),
            entities=list(self._entities),
            tags=set(self._tags),
            difficulty=dict(self._difficulty),
            placed_spawn=self._placed_spawn,
            placed_goal=self._placed_goal,
        )


class Component:
    """Base class for authored components.

    Subclasses are usually created via the ``@register_component`` decorator,
    which sets ``name``/``tags``/``param_spec``/``ports``/``difficulty`` and
    registers the instance. Authors implement only :meth:`stamp`.
    """

    name: str = "component"
    tags: set = set()
    param_spec: Dict[str, ParamRange] = {}
    ports: Dict[str, str] = {"entry": "W@floor", "exit": "E@floor"}
    difficulty: Dict[str, int] = {}
    min_w: int = 3
    min_h: int = 2

    def fits(self, region: Region) -> bool:
        return region.w >= self.min_w and region.h >= self.min_h

    def sample_params(self, rng: random.Random) -> Dict[str, Any]:
        return {k: spec.sample(rng) for k, spec in self.param_spec.items()}

    def stamp(self, ctx: ComponentContext) -> StampResult:  # pragma: no cover
        raise NotImplementedError
