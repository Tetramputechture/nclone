"""Dedicated minimal generator for the BASIC_JUMP mechanic bucket.

Produces small, clean single-chamber levels that isolate the basic-jump
mechanic: a single required gap, drop, or step-up between ninja spawn
and the switch/exit pair. No mines, no slopes, no wall-jump segments --
everything outside the target mechanic is deliberately omitted so these
levels serve as a mechanic-purity baseline for the benchmark.

Variants are selected via the ``VARIANT`` class attribute (driven by
``generator_configs.py`` presets):

- ``single_gap_flat``: flat floor with one gap in the middle that must
  be jumped across.
- ``single_drop``: ninja side is higher than switch/exit side; a drop
  (optionally jumpable back) separates them.
- ``single_step_up``: ninja side is lower; a one- or two-tile step must
  be jumped up onto.
- ``double_gap``: two gaps with a narrow mid-platform between them.
"""

from typing import Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


VALID_BASIC_JUMP_VARIANTS: Tuple[str, ...] = (
    "single_gap_flat",
    "single_drop",
    "single_step_up",
    "double_gap",
)


class MapBasicJump(Map):
    """Minimal BASIC_JUMP generator: single-chamber, one-mechanic-only."""

    MIN_WIDTH = 12
    MAX_WIDTH = 20
    MIN_HEIGHT = 5
    MAX_HEIGHT = 8

    MIN_GAP_WIDTH = 2
    MAX_GAP_WIDTH = 3

    MIN_DROP_HEIGHT = 2
    MAX_DROP_HEIGHT = 3

    MIN_STEP_HEIGHT = 1
    MAX_STEP_HEIGHT = 2

    VARIANT = "single_gap_flat"

    def _pick_chamber(self) -> Tuple[int, int, int, int]:
        """Choose a chamber rectangle ``(x1, y1, x2, y2)`` inside the map.

        The chamber interior (all empty tiles) spans ``[x1..x2] x [y1..y2]``
        inclusive. The surrounding hollow rectangle (floor / ceiling /
        walls) lives at ``x1-1``, ``x2+1``, ``y1-1``, ``y2+1``.
        """
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
        """Fill the map with solid tiles and carve an empty chamber interior."""
        total = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
        self.set_tiles_bulk([1] * total)
        self.set_empty_rectangle(x1, y1, x2, y2)

    def _place_floor_segment(
        self, x_start: int, x_end_inclusive: int, floor_y: int
    ) -> None:
        """Fill the floor row (solid) across ``[x_start, x_end_inclusive]``."""
        for x in range(x_start, x_end_inclusive + 1):
            self.set_tile(x, floor_y, 1)

    def _clear_column_above(self, x: int, y_top: int, y_bottom: int) -> None:
        """Ensure the column at ``x`` is empty from ``y_top..y_bottom`` inclusive."""
        if y_top > y_bottom:
            y_top, y_bottom = y_bottom, y_top
        for y in range(y_top, y_bottom + 1):
            self.set_tile(x, y, 0)

    def _choose_entity_placement(
        self, x1: int, x2: int, obstacle_center_x: int, floor_y: int
    ) -> Tuple[int, int, int, int, int, int]:
        """Return ``(ninja_x, ninja_y, switch_x, switch_y, door_x, door_y)``.

        Ninja is placed on one side of the obstacle center, switch + exit
        door on the other. Orientation is implied by caller.
        """
        ninja_on_left = self.rng.choice([True, False])
        if ninja_on_left:
            ninja_x = self.rng.randint(x1, max(x1, obstacle_center_x - 2))
            rhs_lo = min(x2, obstacle_center_x + 2)
            switch_x = self.rng.randint(rhs_lo, x2)
            door_candidates = [x for x in range(rhs_lo, x2 + 1) if x != switch_x]
            door_x = self.rng.choice(door_candidates) if door_candidates else switch_x
        else:
            ninja_x = self.rng.randint(min(x2, obstacle_center_x + 2), x2)
            lhs_hi = max(x1, obstacle_center_x - 2)
            switch_x = self.rng.randint(x1, lhs_hi)
            door_candidates = [x for x in range(x1, lhs_hi + 1) if x != switch_x]
            door_x = self.rng.choice(door_candidates) if door_candidates else switch_x
        return ninja_x, floor_y, switch_x, floor_y, door_x, floor_y

    def _set_entities(
        self,
        ninja_x: int,
        ninja_y: int,
        switch_x: int,
        switch_y: int,
        door_x: int,
        door_y: int,
    ) -> None:
        orientation = 1 if switch_x > ninja_x else -1
        self.set_ninja_spawn(ninja_x, ninja_y, orientation=orientation)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

    def _apply_single_gap_flat(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int]:
        """Dig one gap of ``[MIN_GAP_WIDTH..MAX_GAP_WIDTH]`` through the floor."""
        floor_y = y2 + 1
        width = x2 - x1 + 1
        max_gap = max(self.MIN_GAP_WIDTH, min(self.MAX_GAP_WIDTH, width - 6))
        gap_w = self.rng.randint(self.MIN_GAP_WIDTH, max_gap)
        gap_center = x1 + width // 2
        gap_start = max(x1 + 2, gap_center - gap_w // 2)
        gap_end = min(x2 - 2, gap_start + gap_w - 1)
        for x in range(gap_start, gap_end + 1):
            self.set_tile(x, floor_y, 0)
            if floor_y + 1 < MAP_TILE_HEIGHT:
                self.set_tile(x, floor_y + 1, 0)
        return (gap_start + gap_end) // 2, y2

    def _apply_single_drop(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int]:
        """Raise one side by ``[MIN_DROP_HEIGHT..MAX_DROP_HEIGHT]`` tiles.

        The ninja starts on the high side; switch/exit on the low side.
        The drop edge is a clean vertical cliff (no intermediate steps).
        """
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        drop_h = self.rng.randint(
            self.MIN_DROP_HEIGHT, min(self.MAX_DROP_HEIGHT, max(self.MIN_DROP_HEIGHT, height - 2))
        )
        edge_x = x1 + width // 2 + self.rng.randint(-1, 1)
        raise_left = self.rng.choice([True, False])

        if raise_left:
            fill_range = range(x1, edge_x)
        else:
            fill_range = range(edge_x + 1, x2 + 1)

        raised_top_y = y2 - drop_h + 1
        for x in fill_range:
            for y in range(raised_top_y, y2 + 1):
                self.set_tile(x, y, 1)

        obstacle_center_x = edge_x
        return obstacle_center_x, y2

    def _apply_single_step_up(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int]:
        """Raise one side by a 1-2 tile step the ninja must jump up onto."""
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        step_h = self.rng.randint(
            self.MIN_STEP_HEIGHT,
            min(self.MAX_STEP_HEIGHT, max(self.MIN_STEP_HEIGHT, height - 2)),
        )
        edge_x = x1 + width // 2 + self.rng.randint(-1, 1)
        raise_left = self.rng.choice([True, False])

        if raise_left:
            fill_range = range(x1, edge_x)
        else:
            fill_range = range(edge_x + 1, x2 + 1)

        raised_top_y = y2 - step_h + 1
        for x in fill_range:
            for y in range(raised_top_y, y2 + 1):
                self.set_tile(x, y, 1)

        return edge_x, y2

    def _apply_double_gap(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int]:
        """Two gaps with a mid platform at least 2 tiles wide between them."""
        floor_y = y2 + 1
        width = x2 - x1 + 1
        gap_w = self.rng.randint(self.MIN_GAP_WIDTH, self.MAX_GAP_WIDTH)

        third = width // 3
        first_center = x1 + third
        second_center = x2 - third

        g1_start = max(x1 + 1, first_center - gap_w // 2)
        g1_end = min(x2 - 1, g1_start + gap_w - 1)
        g2_start = max(g1_end + 2, second_center - gap_w // 2)
        g2_end = min(x2 - 1, g2_start + gap_w - 1)

        for gs, ge in ((g1_start, g1_end), (g2_start, g2_end)):
            for x in range(gs, ge + 1):
                self.set_tile(x, floor_y, 0)
                if floor_y + 1 < MAP_TILE_HEIGHT:
                    self.set_tile(x, floor_y + 1, 0)

        return (g1_end + g2_start) // 2, y2

    def generate(self, seed: Optional[int] = None) -> "MapBasicJump":
        """Generate one BASIC_JUMP level for the configured variant/seed."""
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        variant = self.VARIANT
        if variant not in VALID_BASIC_JUMP_VARIANTS:
            raise ValueError(
                f"Unknown VARIANT {variant!r}. Valid: {VALID_BASIC_JUMP_VARIANTS}"
            )

        x1, y1, x2, y2 = self._pick_chamber()
        self._carve_chamber(x1, y1, x2, y2)

        if variant == "single_gap_flat":
            obstacle_center_x, floor_y = self._apply_single_gap_flat(x1, y1, x2, y2)
        elif variant == "single_drop":
            obstacle_center_x, floor_y = self._apply_single_drop(x1, y1, x2, y2)
        elif variant == "single_step_up":
            obstacle_center_x, floor_y = self._apply_single_step_up(x1, y1, x2, y2)
        else:
            obstacle_center_x, floor_y = self._apply_double_gap(x1, y1, x2, y2)

        ninja_x, ninja_y, switch_x, switch_y, door_x, door_y = (
            self._choose_entity_placement(x1, x2, obstacle_center_x, floor_y)
        )

        ninja_y = self._ground_y_for(ninja_x, x1, x2, y1, y2)
        switch_y = self._ground_y_for(switch_x, x1, x2, y1, y2)
        door_y = self._ground_y_for(door_x, x1, x2, y1, y2)

        self._set_entities(ninja_x, ninja_y, switch_x, switch_y, door_x, door_y)
        return self

    def _ground_y_for(
        self, x: int, x1: int, x2: int, y1: int, y2: int
    ) -> int:
        """Find the first empty tile at column ``x`` scanning upward from the floor.

        For `single_drop` / `single_step_up` variants the floor height differs
        between sides of the obstacle, so ninja/switch/exit Y must be resolved
        column-wise rather than assumed to be ``y2``.
        """
        if not (x1 <= x <= x2):
            return y2
        for y in range(y2, y1 - 1, -1):
            if self.get_tile(x, y) == 0:
                return y
        return y2
