"""Dedicated minimal generator for the SLOPES mechanic bucket.

Levels isolate slope locomotion: the ninja must traverse one or more
DUAL-SIDED hills (every rise has a matching descent back to the floor) to
reach the exit switch + exit door.

Design contract (so every level is solvable by construction):

* The chamber has a single CONTINUOUS flat floor. Hills live only in the
  interior; both ends keep a flat MARGIN.
* Hill height is capped to the interior width so each hill always descends
  back to the floor (no single-sided ramp that "stops after the first hill").
* The ninja stands on one flat margin; the exit switch + exit door stand on
  the OTHER flat margin, beyond the hills. All three are snapped exactly onto
  the floor surface (empty tile with a solid tile directly below) so nothing
  spawns buried or floating.
* Optional toggle mines (T3/T4) are placed only on flat floor away from the
  entity columns; a jump-aware reachability gate then verifies they never
  block the route.
* ``generate()`` validates grounding + co-reachability and retries; if the
  random terrain cannot be made valid it falls back to a flat (slope-free)
  but always-solvable layout.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .map import Map
from ..constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH
from . import terrain_utils

VALID_MAP_SLOPES_VARIANTS: Tuple[str, ...] = (
    "hill_symmetric",
    "hill_asymmetric",
    "multi_hill",
    # Legacy names kept for the mechanic registry (SLOPE_CLIMB / SLOPE_JUMP /
    # valley). They are now all routed to DUAL-SIDED hills -- a hill still has an
    # up-slope to climb and a down-slope to jump off -- so no level can spawn a
    # single-sided ramp that strands the ninja.
    "single_slope_up",
    "single_slope_down",
    "valley",
)

# Flat tiles reserved at each chamber end for the entities + run-up. The hill(s)
# occupy only the interior between the two margins.
FLAT_MARGIN = 3


class MapSlopes(Map):
    """Minimal SLOPES generator: a flat chamber with dual-sided hill(s)."""

    MIN_WIDTH = 12
    MAX_WIDTH = 32
    MIN_HEIGHT = 6
    MAX_HEIGHT = 16

    MIN_SLOPE_HEIGHT = 2
    MAX_SLOPE_HEIGHT = 4

    # Optional secondary mechanic (toggle mines) — T3/T4 presets only
    MAX_MINES = 0
    MIN_MINE_SPACING = 2

    VARIANT = "hill_symmetric"
    SLOPE_TYPE_POOL: Tuple[str, ...] = ("mild",)

    # How many (terrain + placement) attempts before the flat fallback.
    MAX_ATTEMPTS = 12

    # ------------------------------------------------------------------ chamber
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

    def _fill_floor_substrate(
        self, x_lo: int, x_hi: int, floor_y: int, depth: int = 2
    ) -> None:
        for x in range(x_lo, x_hi + 1):
            for d in range(1, depth + 1):
                fy = floor_y + d
                if fy < MAP_TILE_HEIGHT:
                    self.set_tile(x, fy, 1)

    # ------------------------------------------------------------------ surface
    def _find_walkable_y(self, x: int, floor_y: int, ceiling_y: int) -> int:
        """Standing row in column ``x``: the empty tile directly above the
        highest solid tile (floor or slope surface).

        Walks UP from the floor through the solid substrate and returns the
        first empty cell, i.e. the standing position on the surface. ``ceiling_y``
        (the carved chamber top) is the clamp / fallback and is itself empty.
        """
        x = max(0, min(x, MAP_TILE_WIDTH - 1))
        y = min(floor_y, MAP_TILE_HEIGHT - 1)
        while y > ceiling_y and self.get_tile(x, y) != 0:
            y -= 1
        return y

    def _is_grounded(self, x: int, y: int) -> bool:
        """True iff (x, y) is empty and the tile directly below is solid."""
        if not (0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT):
            return False
        if self.get_tile(x, y) != 0:
            return False
        below = y + 1
        if below >= MAP_TILE_HEIGHT:
            return False
        return self.get_tile(x, below) != 0

    # ------------------------------------------------------------------- hills
    def _margin(self, x1: int, x2: int) -> int:
        """Flat tiles reserved at each end (adaptive: shrinks on narrow chambers
        so a dual-sided hill still fits, but never below 2 so the switch+door
        pair and the ninja always land on flat ground)."""
        width = x2 - x1 + 1
        return max(2, min(FLAT_MARGIN, (width - 1) // 5))

    def _max_hill_height(self, hx1: int, hx2: int, floor_y: int, y1: int) -> int:
        """Cap height so the hill both fits the interior width AND clears the
        ceiling -- guaranteeing it descends back to the floor instead of running
        out of room."""
        by_width = max(1, (hx2 - hx1) // 3)
        by_ceiling = max(1, floor_y - y1 - 1)
        return min(by_width, by_ceiling)

    def _hill(
        self, start_x: int, floor_y: int, h: int, kind: str,
        min_x: int, max_x: int, min_y: int, max_y: int,
    ) -> int:
        """Build one dual-sided hill; return the x just past its descent."""
        if kind == "steep":
            return terrain_utils.create_steep_hill(
                self, start_x, floor_y, h,
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            )
        if kind == "45":
            return terrain_utils.create_45_degree_hill(
                self, start_x, floor_y, h,
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            )
        return terrain_utils.create_mild_hill(
            self, start_x, floor_y, h,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        )

    def _mixed_hill(
        self, start_x: int, floor_y: int, h: int, ascent: str, descent: str,
        min_x: int, max_x: int, min_y: int, max_y: int,
    ) -> int:
        return terrain_utils.create_mixed_hill(
            self, start_x, floor_y, h, ascent_type=ascent, descent_type=descent,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        )

    def _build_terrain(
        self, variant: str, x1: int, y1: int, x2: int, y2: int, floor_y: int
    ) -> None:
        """Lay a continuous floor + dual-sided hill(s) confined to the interior,
        leaving FLAT_MARGIN flat tiles at each end for the entities."""
        self._fill_floor_substrate(x1, x2, floor_y)
        m = self._margin(x1, x2)
        hx1, hx2 = x1 + m, x2 - m
        if hx2 - hx1 < 3:  # chamber too narrow for a hill -> stays flat
            return
        min_y = y1
        max_y = min(floor_y, MAP_TILE_HEIGHT - 1)
        h = min(self._pick_height(), self._max_hill_height(hx1, hx2, floor_y, y1))
        start_x = hx1 + 1

        if variant == "multi_hill":
            n_hills = 2
            seg = (hx2 - hx1) // n_hills
            hh = min(h, max(1, (seg - 1) // 4))
            cx = start_x
            for _ in range(n_hills):
                seg_max = min(hx2, cx + seg - 1)
                end = self._hill(cx, floor_y, hh, self._pick_kind(),
                                 hx1, seg_max, min_y, max_y)
                cx = max(end + 1, cx + 4)
                if cx >= hx2 - 2:
                    break
        elif variant == "hill_asymmetric":
            a = self._pick_kind()
            b = self._pick_kind()
            self._mixed_hill(start_x, floor_y, h, a, b, hx1, hx2, min_y, max_y)
        else:  # hill_symmetric (default) + any legacy single variants
            self._hill(start_x, floor_y, h, self._pick_kind(),
                       hx1, hx2, min_y, max_y)

    # ----------------------------------------------------------------- entities
    def _pick_height(self) -> int:
        return self.rng.randint(
            min(self.MIN_SLOPE_HEIGHT, self.MAX_SLOPE_HEIGHT),
            self.MAX_SLOPE_HEIGHT,
        )

    def _pick_kind(self) -> str:
        pool = [k for k in self.SLOPE_TYPE_POOL if k in ("mild", "steep", "45")]
        return self.rng.choice(pool) if pool else "mild"

    def _entity_columns(self, x1: int, x2: int) -> Tuple[int, int, int]:
        """``(ninja_x, switch_x, door_x)``: ninja on one flat margin, switch +
        door together on the OTHER flat margin (beyond the hills). Columns are
        taken from the reserved flat margins so all three sit on flat ground."""
        m = self._margin(x1, x2)
        left_inner, left_outer = x1 + m - 1, x1 + max(0, m - 2)
        right_inner, right_outer = x2 - m + 1, x2 - max(0, m - 2)
        if self.rng.choice([True, False]):  # ninja left, goals right
            return left_inner, right_inner, right_outer
        return right_inner, left_inner, left_outer  # ninja right, goals left

    def _place_mines(
        self, x1: int, x2: int, floor_y: int, ceiling_y: int,
        reserved: Set[int], count: int,
    ) -> None:
        if count <= 0:
            return
        # Only grounded floor columns in the interior, never within
        # MIN_MINE_SPACING of an entity or another mine. The reachability gate
        # verifies they do not block.
        m = self._margin(x1, x2)
        candidates = [
            x for x in range(x1 + m, x2 - m + 1)
            if all(abs(x - r) >= self.MIN_MINE_SPACING for r in reserved)
            and self._is_grounded(x, self._find_walkable_y(x, floor_y, ceiling_y))
        ]
        self.rng.shuffle(candidates)
        placed: List[int] = []
        for x in candidates:
            if len(placed) >= count:
                break
            if all(abs(x - px) >= self.MIN_MINE_SPACING for px in placed):
                placed.append(x)
        for x in placed:
            self.add_entity(1, x, self._find_walkable_y(x, floor_y, ceiling_y))

    def _set_entities(
        self, ninja_x, ninja_y, switch_x, switch_y, door_x, door_y
    ) -> None:
        ref_x = door_x if switch_x == ninja_x else switch_x
        orientation = 1 if ref_x > ninja_x else -1
        self.set_ninja_spawn(ninja_x, ninja_y, orientation=orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

    # ----------------------------------------------------------------- assembly
    def _attempt(self, variant: str) -> Optional[dict]:
        """One full build. Returns placement dict on success, else None."""
        x1, y1, x2, y2 = self._pick_chamber()
        if x2 - x1 < 2 * FLAT_MARGIN + 1:
            return None
        self._carve_chamber(x1, y1, x2, y2)
        floor_y = y2 + 1
        self._build_terrain(variant, x1, y1, x2, y2, floor_y)

        ninja_x, switch_x, door_x = self._entity_columns(x1, x2)
        ninja_y = self._find_walkable_y(ninja_x, floor_y, y1)
        switch_y = self._find_walkable_y(switch_x, floor_y, y1)
        door_y = self._find_walkable_y(door_x, floor_y, y1)

        if not (self._is_grounded(ninja_x, ninja_y)
                and self._is_grounded(switch_x, switch_y)
                and self._is_grounded(door_x, door_y)):
            return None

        self._place_mines(
            x1, x2, floor_y, y1, {ninja_x, switch_x, door_x}, int(self.MAX_MINES)
        )
        self._set_entities(ninja_x, ninja_y, switch_x, switch_y, door_x, door_y)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _build_flat_fallback(self) -> None:
        """Guaranteed-solvable slope-free layout (final safety net)."""
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        x1 = self.rng.randint(2, max(2, MAP_TILE_WIDTH - width - 2))
        y1 = MAP_TILE_HEIGHT // 2 - 2
        x2, y2 = x1 + width - 1, y1 + 4
        self._carve_chamber(x1, y1, x2, y2)
        floor_y = y2 + 1
        self._fill_floor_substrate(x1, x2, floor_y)
        ninja_x, switch_x, door_x = x1 + 1, x2 - 2, x2 - 1
        sy = self._find_walkable_y(switch_x, floor_y, y1)
        self._set_entities(
            ninja_x, self._find_walkable_y(ninja_x, floor_y, y1),
            switch_x, sy, door_x, self._find_walkable_y(door_x, floor_y, y1),
        )

    def _is_solvable(self) -> bool:
        try:
            from .solvability import is_map_solvable

            return is_map_solvable(self.map_data())
        except Exception:
            return False

    def generate(self, seed: Optional[int] = None) -> "MapSlopes":
        if seed is not None:
            self.rng.seed(seed)

        variant = self.VARIANT
        if variant not in VALID_MAP_SLOPES_VARIANTS:
            raise ValueError(
                f"Unknown VARIANT {variant!r}. Valid: {VALID_MAP_SLOPES_VARIANTS}"
            )

        for _ in range(self.MAX_ATTEMPTS):
            self.reset()
            if self._attempt(variant) is not None and self._is_solvable():
                return self

        # All random attempts failed validation -> guaranteed-solvable flat level.
        self.reset()
        self._build_flat_fallback()
        return self
