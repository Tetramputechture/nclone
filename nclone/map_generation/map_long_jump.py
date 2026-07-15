"""Dedicated minimal generator for the LONG_JUMP mechanic.

Isolates the *build-speed-then-clear-a-wide-gap* movement skill: a flat **runway**
the ninja accelerates along, a single **wide gap** (a floorless pit) that is too
wide to clear from a standstill, and a flat **landing** platform at the same
height carrying the switch + exit. The ninja always spawns at the far start of
the runway, facing the gap, so a full running start is guaranteed (this is the
one thing the generic ``basic_jump`` generator does not promise).

Like the other dedicated minimal generators (``basic_jump`` / ``mine_path`` /
``slopes``) the background is clean solid fill (no random-tile augmentation), so
the level is mechanic-pure and its silhouette is unambiguous.

Config knobs (see ``LongJumpConfig`` / driven by ``generator_configs`` presets):

- ``min/max_gap_width``: width of the gap to clear (tiles). The difficulty driver.
- ``min/max_runway``:    flat run-up length before the gap (tiles). Kept long so
  the ninja can always reach speed.
- ``min/max_landing``:   flat landing length after the gap (tiles). Narrow
  landings (1-2 tiles) turn the jump into a no-overshoot precision landing.
- ``min/max_height``:    interior chamber height.
- ``max_mines``:         optional toggle mines on runway/landing (Layer-1 only).
- ``edge_mines``:        toggle mines on the FIRST landing columns (the gap's
  landing edge), forcing a deep landing past the mine(s). Requires a landing
  wide enough to keep the switch/door columns mine-free (landing >= 4 for one
  edge mine, >= 5 for two); excess edge mines are dropped, never overlapped.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapLongJump(Map):
    """Minimal LONG_JUMP generator: runway -> wide gap -> landing."""

    MIN_HEIGHT = 5
    MAX_HEIGHT = 8

    MIN_GAP_WIDTH = 4
    MAX_GAP_WIDTH = 8

    MIN_RUNWAY = 12
    MAX_RUNWAY = 16

    MIN_LANDING = 4
    MAX_LANDING = 6

    MAX_MINES = 0
    # Mines on the first landing columns just past the gap edge (0 = none).
    EDGE_MINES = 0
    # How deep to carve the pit under the gap (tiles). Deep enough that a missed
    # jump drops the ninja into an inescapable pit (-> no route -> faithful fail).
    PIT_DEPTH = 4

    def _pick_dims(self) -> Tuple[int, int, int, int]:
        """Return ``(runway, gap, landing, height)`` honoring the map width cap."""
        gap = self.rng.randint(self.MIN_GAP_WIDTH, self.MAX_GAP_WIDTH)
        runway = self.rng.randint(self.MIN_RUNWAY, self.MAX_RUNWAY)
        landing = self.rng.randint(self.MIN_LANDING, self.MAX_LANDING)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        # Clamp total interior width to fit inside the playfield (1-tile border).
        max_total = MAP_TILE_WIDTH - 4
        total = runway + gap + landing
        if total > max_total:
            # Trim the runway first (keep the gap = the mechanic intact).
            overflow = total - max_total
            runway = max(6, runway - overflow)
        return runway, gap, landing, height

    def _ground_y_for(self, x: int, x1: int, x2: int, y1: int, y2: int) -> int:
        """Topmost empty cell in column ``x`` (the standing surface)."""
        if not (x1 <= x <= x2):
            return y2
        for y in range(y2, y1 - 1, -1):
            if self.get_tile(x, y) == 0:
                return y
        return y2

    def _place_optional_mines(
        self, x1: int, x2: int, y2: int, gap_start: int, gap_end: int,
        reserved: set,
    ) -> None:
        n = int(self.MAX_MINES)
        if n <= 0:
            return
        # Candidate floor columns on runway/landing (never over the gap).
        cands: List[int] = [
            x for x in range(x1 + 2, x2 - 1)
            if (x < gap_start - 1 or x > gap_end + 1) and x not in reserved
        ]
        self.rng.shuffle(cands)
        placed: List[int] = []
        for x in cands:
            if len(placed) >= n:
                break
            if all(abs(x - px) >= 2 for px in placed):
                placed.append(x)
        for mx in placed:
            # Border-corrected floor placement (+1.5/+2.0 like the edge mines
            # below): raw tile coords float the mine 48px above the floor.
            self.add_entity(
                1, mx + 1.5, self._ground_y_for(mx, x1, x2, 0, y2) + 2.0
            )

    def _place_edge_mines(
        self, gap_end: int, x1: int, x2: int, y2: int, reserved: set
    ) -> None:
        """Mines on the first landing columns just past the gap edge.

        Placed left-to-right from ``gap_end + 1`` so a short jump that clips the
        landing edge dies; the ninja must land DEEP past the mine(s). Stops
        rather than overlapping the switch/door columns, so presets must keep
        the landing wide enough (>= 4 tiles for one mine, >= 5 for two).
        """
        n = int(getattr(self, "EDGE_MINES", 0))
        placed = 0
        mx = gap_end + 1
        while placed < n and mx <= x2:
            if mx in reserved:
                break
            # Mine entities carry NO center offset in ``add_entity`` (1.5-tile
            # mismatch vs the ninja frame -- see _place_corridor_floor_mines);
            # +1.5/+2.0 centers the mine on landing column ``mx`` sitting at
            # the floor's surface plane, so a short landing on it is lethal.
            gy = self._ground_y_for(mx, x1, x2, 0, y2)
            self.add_entity(1, mx + 1.5, gy + 2.0)
            placed += 1
            mx += 1

    def generate(self, seed: Optional[int] = None) -> "MapLongJump":
        """Generate one LONG_JUMP level."""
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        runway, gap, landing, height = self._pick_dims()
        total = runway + gap + landing

        # Place the chamber at a random valid origin.
        x1 = self.rng.randint(2, max(2, MAP_TILE_WIDTH - total - 2))
        y2 = self.rng.randint(height + 1, MAP_TILE_HEIGHT - 3)
        y1 = y2 - height + 1
        x2 = x1 + total - 1

        # Solid everywhere, then carve the chamber interior empty.
        self.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
        self.set_empty_rectangle(x1, y1, x2, y2)

        # Carve the gap: remove the floor (y2+1) and PIT_DEPTH rows below it.
        gap_start = x1 + runway
        gap_end = gap_start + gap - 1
        for gx in range(gap_start, gap_end + 1):
            for gy in range(y2 + 1, min(y2 + 1 + self.PIT_DEPTH, MAP_TILE_HEIGHT)):
                self.set_tile(gx, gy, 0)

        # Ninja at the start of the runway, facing the gap.
        ninja_x = x1 + 1
        # Switch + exit on the landing platform (right of the gap). A 1-tile
        # landing (the hardest precision variant) stacks both on the single
        # landing column -- the legacy layout would put the door OVER the gap.
        if landing <= 1:
            door_x = x2
            switch_x = x2
        else:
            door_x = x2 - 1
            switch_x = max(gap_end + 2, x2 - 2)
            if switch_x == door_x:
                switch_x = door_x - 1

        ninja_y = self._ground_y_for(ninja_x, x1, x2, y1, y2)
        switch_y = self._ground_y_for(switch_x, x1, x2, y1, y2)
        door_y = self._ground_y_for(door_x, x1, x2, y1, y2)

        self.set_ninja_spawn(ninja_x, ninja_y, orientation=1)
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)
        self._place_optional_mines(
            x1, x2, y2, gap_start, gap_end, {ninja_x, switch_x, door_x}
        )
        self._place_edge_mines(gap_end, x1, x2, y2, {switch_x, door_x})
        return self
