"""Dedicated minimal generator for the SLOPES mechanic bucket.

Levels isolate slope locomotion: single ramps, symmetric hills, asymmetric
hills, and (optionally) valley layouts. Optional toggle mines (T3/T4) are
the only other entities besides the exit switch + exit door pair.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .map import Map
from ..constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH
from . import terrain_utils

VALID_MAP_SLOPES_VARIANTS: Tuple[str, ...] = (
    "single_slope_up",
    "single_slope_down",
    "hill_symmetric",
    "hill_asymmetric",
    "valley",
)


class MapSlopes(Map):
    """Minimal SLOPES generator: single-chamber terrain with slope tiles."""

    MIN_WIDTH = 12
    MAX_WIDTH = 32
    MIN_HEIGHT = 6
    MAX_HEIGHT = 16

    MIN_SLOPE_HEIGHT = 2
    MAX_SLOPE_HEIGHT = 4

    # Optional secondary mechanic (toggle mines) — T3/T4 presets only
    MAX_MINES = 0
    MIN_MINE_SPACING = 2

    VARIANT = "single_slope_up"
    SLOPE_TYPE_POOL: Tuple[str, ...] = ("mild",)

    def _pick_chamber(self) -> Tuple[int, int, int, int]:
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        max_start_x = MAP_TILE_WIDTH - width - 2
        max_start_y = MAP_TILE_HEIGHT - height - 2
        x1 = self.rng.randint(2, max(2, max_start_x))
        y1 = self.rng.randint(2, max(2, max_start_y))
        x2 = x1 + width - 1
        y2 = y1 + height - 1
        return x1, y1, x2, y2

    def _carve_chamber(self, x1: int, y1: int, x2: int, y2: int) -> None:
        total = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
        self.set_tiles_bulk([1] * total)
        self.set_empty_rectangle(x1, y1, x2, y2)

    def _find_walkable_y(
        self, x: int, floor_y: int, ceiling_y: int
    ) -> int:
        """Standing row in column ``x``: the empty tile directly above the
        highest solid tile (floor or slope surface).

        BUGFIX: the previous implementation scanned top-down from ``ceiling_y``
        and returned the first empty tile — which is the chamber top, right
        under the ceiling. Static entities (exit switch / exit door) therefore
        spawned on the ceiling, unreachable. We now walk UP from the floor
        through the solid slope/floor substrate and return the first empty cell,
        which rests on the ground/slope surface.
        """
        x = max(0, min(x, MAP_TILE_WIDTH - 1))
        bottom = min(floor_y, MAP_TILE_HEIGHT - 1)
        y = bottom
        # Decrement through solid (floor + slope) tiles; stop at the first empty
        # cell, i.e. the standing position on top of the surface. ``ceiling_y``
        # (the carved chamber top) is the clamp / fallback and is itself empty.
        while y > ceiling_y and self.get_tile(x, y) != 0:
            y -= 1
        return y

    def _fill_floor_substrate(
        self, x_lo: int, x_hi: int, floor_y: int, depth: int = 2
    ) -> None:
        for x in range(x_lo, x_hi + 1):
            for d in range(1, depth + 1):
                fy = floor_y + d
                if fy < MAP_TILE_HEIGHT:
                    self.set_tile(x, fy, 1)

    def _ascent(
        self,
        x: int,
        y: int,
        height: int,
        kind: str,
        min_x: int,
        max_x: int,
        min_y: int,
    ) -> int:
        if kind == "mild":
            return terrain_utils.create_slope_up(
                self, x, y, height + 1, use_mild=True, min_x=min_x, max_x=max_x, min_y=min_y
            )
        if kind == "steep":
            return terrain_utils.create_slope_up(
                self, x, y, height, use_mild=False, min_x=min_x, max_x=max_x, min_y=min_y
            )
        if kind == "45":
            return terrain_utils.create_45_degree_slope_up(
                self, x, y - 1, height, min_x=min_x, max_x=max_x, min_y=min_y
            )
        raise ValueError(f"Unknown slope kind {kind!r}")

    def _descent(
        self,
        x: int,
        y: int,
        height: int,
        kind: str,
        min_x: int,
        max_x: int,
        max_y: int,
    ) -> int:
        if kind == "mild":
            return terrain_utils.create_slope_down(
                self, x, y, height, use_mild=True, min_x=min_x, max_x=max_x, max_y=max_y
            )
        if kind == "steep":
            return terrain_utils.create_slope_down(
                self, x, y, height, use_mild=False, min_x=min_x, max_x=max_x, max_y=max_y
            )
        if kind == "45":
            return terrain_utils.create_45_degree_slope_down(
                self, x, y, height, min_x=min_x, max_x=max_x, max_y=max_y
            )
        raise ValueError(f"Unknown slope kind {kind!r}")

    def _apply_single_slope_up(
        self, x1: int, y1: int, x2: int, y2: int, floor_y: int, h: int, kind: str
    ) -> None:
        min_x, max_x = x1 - 1, x2 + 1
        start_x = x1 + 2
        self._fill_floor_substrate(x1, x2, floor_y)
        self._ascent(start_x, floor_y, h, kind, min_x, max_x, min_y=y1)

    def _apply_single_slope_down(
        self, x1: int, y1: int, x2: int, y2: int, floor_y: int, h: int, kind: str
    ) -> None:
        """Left side high, slope descends to the right toward ``floor_y``."""
        min_x, max_x = x1 - 1, x2 + 1
        max_y = min(floor_y + 2, MAP_TILE_HEIGHT - 1)
        self._fill_floor_substrate(x1, x2, floor_y)
        start_x = x1 + 2
        left_surface_y = max(y1, floor_y - h)
        self._descent(start_x, left_surface_y, h, kind, min_x, max_x, max_y)

    def _apply_hill_symmetric(
        self, x1: int, y1: int, x2: int, y2: int, floor_y: int, h: int, kind: str
    ) -> None:
        min_x, max_x, min_y, max_y = x1 - 1, x2 + 1, y1, y2 + 2
        self._fill_floor_substrate(x1, x2, floor_y)
        start_x = x1 + 2
        if kind == "mild":
            terrain_utils.create_mild_hill(
                self, start_x, floor_y, h, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
            )
        elif kind == "steep":
            terrain_utils.create_steep_hill(
                self, start_x, floor_y, h, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
            )
        elif kind == "45":
            terrain_utils.create_45_degree_hill(
                self, start_x, floor_y, h, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
            )
        else:
            raise ValueError(kind)

    def _apply_hill_asymmetric(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        floor_y: int,
        h: int,
        ascent_kind: str,
        descent_kind: str,
    ) -> None:
        min_x, max_x, min_y, max_y = x1 - 1, x2 + 1, y1, y2 + 2
        self._fill_floor_substrate(x1, x2, floor_y)
        start_x = x1 + 2
        terrain_utils.create_mixed_hill(
            self,
            start_x,
            floor_y,
            h,
            ascent_type=ascent_kind,
            descent_type=descent_kind,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

    def _apply_valley(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        floor_y: int,
        h: int,
        down_kind: str,
        up_kind: str,
    ) -> None:
        min_x, max_x, min_y = x1 - 1, x2 + 1, y1
        max_y = min(floor_y + 2, MAP_TILE_HEIGHT - 1)
        self._fill_floor_substrate(x1, x2, floor_y)
        start_x = x1 + 2
        left_rim_y = max(y1, floor_y - h)
        x_at_basin = self._descent(
            start_x, left_rim_y, h, down_kind, min_x, max_x, max_y
        )
        basin_w = self.rng.randint(1, 2)
        bx = x_at_basin
        for _ in range(basin_w):
            if bx < max_x - 1:
                self.set_tile(bx, floor_y, 1)
                bx += 1
        self._ascent(
            bx, floor_y, h, up_kind, min_x, max_x, min_y=min_y
        )

    def _pick_height(self) -> int:
        return self.rng.randint(
            min(self.MIN_SLOPE_HEIGHT, self.MAX_SLOPE_HEIGHT),
            self.MAX_SLOPE_HEIGHT,
        )

    def _pick_kind(self) -> str:
        pool = [k for k in self.SLOPE_TYPE_POOL if k in ("mild", "steep", "45")]
        if not pool:
            return "mild"
        return self.rng.choice(pool)

    def _place_mines(
        self,
        x1: int,
        x2: int,
        floor_y: int,
        ceiling_y: int,
        reserved: Set[int],
        count: int,
    ) -> None:
        if count <= 0:
            return
        candidates: List[int] = []
        for x in range(x1 + 1, x2):
            if x in reserved:
                continue
            candidates.append(x)
        self.rng.shuffle(candidates)
        placed: List[int] = []
        for x in candidates:
            if len(placed) >= count:
                break
            if all(abs(x - px) >= self.MIN_MINE_SPACING for px in placed):
                placed.append(x)
        for x in placed:
            my = self._find_walkable_y(x, floor_y, ceiling_y)
            self.add_entity(1, x, my)

    def _set_entities(
        self,
        ninja_x: int,
        ninja_y: int,
        switch_x: int,
        switch_y: int,
        door_x: int,
        door_y: int,
    ) -> None:
        if switch_x == ninja_x:
            orientation = 1 if door_x > ninja_x else -1
        else:
            orientation = 1 if switch_x > ninja_x else -1
        self.set_ninja_spawn(ninja_x, ninja_y, orientation=orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

    def _layout_entities(
        self, variant: str, x1: int, x2: int
    ) -> Tuple[int, int, int]:
        """Return ``(ninja_x, switch_x, door_x)`` tile columns."""
        mid = (x1 + x2) // 2
        lo = x1 + 1
        hi = x2 - 1
        if variant == "single_slope_up":
            return lo, mid, hi
        if variant == "single_slope_down":
            return lo, (lo + mid) // 2, hi
        if variant in ("hill_symmetric", "hill_asymmetric"):
            n_left = self.rng.choice([True, False])
            if n_left:
                return lo, mid, hi
            return hi, mid, lo
        if variant == "valley":
            q1 = x1 + (x2 - x1) // 3
            q2 = x1 + 2 * (x2 - x1) // 3
            return lo, (q1 + q2) // 2, hi
        return lo, mid, hi

    def generate(self, seed: Optional[int] = None) -> "MapSlopes":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        variant = self.VARIANT
        if variant not in VALID_MAP_SLOPES_VARIANTS:
            raise ValueError(
                f"Unknown VARIANT {variant!r}. Valid: {VALID_MAP_SLOPES_VARIANTS}"
            )

        x1, y1, x2, y2 = self._pick_chamber()
        self._carve_chamber(x1, y1, x2, y2)
        floor_y = y2 + 1
        h = self._pick_height()
        h = min(h, max(1, floor_y - y1 - 1))

        if variant == "single_slope_up":
            self._apply_single_slope_up(
                x1, y1, x2, y2, floor_y, h, self._pick_kind()
            )
        elif variant == "single_slope_down":
            self._apply_single_slope_down(
                x1, y1, x2, y2, floor_y, h, self._pick_kind()
            )
        elif variant == "hill_symmetric":
            self._apply_hill_symmetric(
                x1, y1, x2, y2, floor_y, h, self._pick_kind()
            )
        elif variant == "hill_asymmetric":
            a, b = self._pick_kind(), self._pick_kind()
            n_try = 0
            while a == b and len({k for k in self.SLOPE_TYPE_POOL if k in ("mild", "steep", "45")}) > 1 and n_try < 8:
                b = self._pick_kind()
                n_try += 1
            self._apply_hill_asymmetric(x1, y1, x2, y2, floor_y, h, a, b)
        elif variant == "valley":
            self._apply_valley(
                x1, y1, x2, y2, floor_y, h, self._pick_kind(), self._pick_kind()
            )
        else:
            raise ValueError(variant)

        ninja_x, switch_x, door_x = self._layout_entities(variant, x1, x2)
        if switch_x == door_x and x2 - x1 > 4:
            door_x = min(switch_x + 1, x2 - 1)
            if door_x == switch_x:
                door_x = switch_x - 1

        ninja_y = self._find_walkable_y(ninja_x, floor_y, y1)
        switch_y = self._find_walkable_y(switch_x, floor_y, y1)
        door_y = self._find_walkable_y(door_x, floor_y, y1)

        reserved: Set[int] = {ninja_x, switch_x, door_x}
        mine_n = int(self.MAX_MINES)
        self._place_mines(x1, x2, floor_y, y1, reserved, mine_n)

        self._set_entities(
            ninja_x, ninja_y, switch_x, switch_y, door_x, door_y
        )
        return self
