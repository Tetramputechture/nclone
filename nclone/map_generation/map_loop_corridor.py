"""Dedicated generator for the loop-corridor archetype (``circle-corridor``).

Carves a rectangular ring corridor out of solid rock around a central pillar,
then drops a short solid stub from the pillar to the bottom wall so the ring
becomes a horseshoe (C-shaped) corridor with two bottom tips. The ninja spawns
at one tip and must traverse the whole loop -- walking the top/bottom arms and
wall-jumping the (2-wide, wall-jumpable) vertical side arms -- to reach the exit
switch at the far tip, then back to the exit door. Toggle mines are scattered
along the *horizontal* arms only (the bottom floor and the pillar-top walkway),
never the vertical climb arms where a deadly mine would be undodgeable.

The corners of both the outer ring and the inner pillar are optionally rounded
with 45-degree tiles (the four diagonal orientations the ``circle-corridor``
reference uses), giving the smooth "circular" corridor look.

The mechanic is compositional -- sustained corridor navigation combining
walking, wall-jumping and mine avoidance -- so LOOP_CORRIDOR is scheduled after
the platforming/jump buckets in the curriculum.

Config-driven class attributes (set by ``GeneratorFactory`` from
``LoopCorridorConfig``): MIN/MAX_WIDTH + MIN/MAX_HEIGHT (outer ring extent, the
size curriculum's knobs), CORRIDOR_WIDTH (corridor thickness; 2 == the
wall-jumpable shaft width -- 3+ is too wide to wall-jump-chain and leaves the
loop unsolvable), MAX_MINES (toggle mines on the horizontal arms),
ROUND_CORNERS (whether to round the ring/pillar corners).
"""

from typing import List, Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


# 45-degree diagonal tiles, indexed by the corner they round. Orientations are
# taken directly from the ``circle-corridor`` reference map.
_OUTER_CORNER = {"TL": 6, "TR": 7, "BL": 9, "BR": 8}
_INNER_CORNER = {"TL": 8, "TR": 9, "BL": 7, "BR": 6}


class MapLoopCorridor(Map):
    """Horseshoe ring corridor with rounded corners and corridor mines."""

    MIN_WIDTH = 14
    MAX_WIDTH = 20
    MIN_HEIGHT = 12
    MAX_HEIGHT = 18

    # Corridor thickness. 2 == the canonical wall-jumpable shaft width: the
    # vertical side arms are climbed by chaining wall-jumps between the outer
    # wall and the central pillar, and a 2-wide (48px) gap is the width a ninja
    # (and the d*-guided executor) can actually climb. cw=3 (72px) is too wide
    # to chain, which left every multi-tile loop physically unsolvable.
    CORRIDOR_WIDTH = 2
    MAX_MINES = 4
    ROUND_CORNERS = True

    def _place_ring(
        self, cx1: int, cy1: int, cx2: int, cy2: int, cw: int
    ) -> Tuple[int, int, int, int]:
        """Carve the ring corridor and refill the central pillar.

        Returns the pillar rectangle ``(px1, py1, px2, py2)``.
        """
        # Carve the whole outer rectangle empty...
        for y in range(cy1, cy2 + 1):
            for x in range(cx1, cx2 + 1):
                self.set_tile(x, y, 0)
        # ...then refill the central pillar, leaving a ring of width ``cw``.
        px1, py1 = cx1 + cw, cy1 + cw
        px2, py2 = cx2 - cw, cy2 - cw
        for y in range(py1, py2 + 1):
            for x in range(px1, px2 + 1):
                self.set_tile(x, y, 1)
        return px1, py1, px2, py2

    def _round_corners(
        self, cx1, cy1, cx2, cy2, px1, py1, px2, py2
    ) -> None:
        self.set_tile(cx1, cy1, _OUTER_CORNER["TL"])
        self.set_tile(cx2, cy1, _OUTER_CORNER["TR"])
        self.set_tile(cx1, cy2, _OUTER_CORNER["BL"])
        self.set_tile(cx2, cy2, _OUTER_CORNER["BR"])
        self.set_tile(px1, py1, _INNER_CORNER["TL"])
        self.set_tile(px2, py1, _INNER_CORNER["TR"])
        self.set_tile(px1, py2, _INNER_CORNER["BL"])
        self.set_tile(px2, py2, _INNER_CORNER["BR"])

    def _corridor_ring_cells(
        self, cx1, cy1, cx2, cy2, cw
    ) -> List[Tuple[int, int]]:
        """Mine-candidate cells on the *horizontal* arms only.

        Mines go where the ninja can dodge them by jumping while running: the
        bottom-arm floor (lowest empty row, deep headroom above) and the
        top-arm walkway atop the central pillar (one row above the pillar). The
        vertical side arms are deliberately excluded -- they are wall-jumped, and
        a deadly toggle mine mid-climb is effectively undodgeable (it makes the
        loop unsolvable; see the cw=2 fix that restored physics solvability).
        """
        px1, px2 = cx1 + cw, cx2 - cw
        py1 = cy1 + cw
        cells: List[Tuple[int, int]] = []
        bot_y = cy2  # bottom-arm floor (ninja walks the lowest empty row)
        top_walk_y = py1 - 1  # walkway atop the pillar (the top-arm traversal)
        for x in range(cx1 + 1, cx2):  # bottom arm, tip to tip (corners trimmed)
            cells.append((x, bot_y))
        for x in range(px1, px2 + 1):  # top arm: only above the solid pillar
            cells.append((x, top_walk_y))
        return cells

    def generate(self, seed: Optional[int] = None) -> "MapLoopCorridor":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        cw = max(2, int(self.CORRIDOR_WIDTH))

        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        # Ring must fit a central pillar: each dimension > 2*cw + 1.
        width = max(width, 2 * cw + 3)
        height = max(height, 2 * cw + 3)
        width = min(width, MAP_TILE_WIDTH - 4)
        height = min(height, MAP_TILE_HEIGHT - 4)

        cx1 = self.rng.randint(2, max(2, MAP_TILE_WIDTH - 2 - width))
        cy1 = self.rng.randint(2, max(2, MAP_TILE_HEIGHT - 2 - height))
        cx2, cy2 = cx1 + width - 1, cy1 + height - 1

        # Solid rock everywhere, then carve the ring.
        self.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
        px1, py1, px2, py2 = self._place_ring(cx1, cy1, cx2, cy2, cw)

        # Bottom stub: drop a 2-wide solid column from the pillar bottom to the
        # bottom wall, splitting the bottom arm so the ring becomes a horseshoe.
        sc = (cx1 + cx2) // 2
        stub_cols = (sc, sc + 1) if px1 <= sc and sc + 1 <= px2 else (sc,)
        for x in stub_cols:
            for y in range(py2 + 1, cy2 + 1):
                self.set_tile(x, y, 1)

        if self.ROUND_CORNERS:
            self._round_corners(cx1, cy1, cx2, cy2, px1, py1, px2, py2)

        # Spawn + door at one bottom tip, switch at the other -- forcing a full
        # loop traversal. The bottom-arm floor is the lowest empty row (cy2),
        # supported by the bottom wall below it.
        spawn_left = self.rng.choice([True, False])
        left_tip_x = cx1 + 1
        right_tip_x = cx2 - 1
        if spawn_left:
            ninja_x, door_x, switch_x = left_tip_x, left_tip_x + 1, right_tip_x
            orientation = 1
        else:
            ninja_x, door_x, switch_x = right_tip_x, right_tip_x - 1, left_tip_x
            orientation = -1

        self.set_ninja_spawn(ninja_x, cy2, orientation)
        self.add_entity(3, door_x, cy2, 0, 0, switch_x, cy2)

        # Mines along the corridor centerline, kept clear of spawn/switch/door.
        reserved = {
            (ninja_x, cy2),
            (door_x, cy2),
            (switch_x, cy2),
        }

        def _too_close(cell: Tuple[int, int]) -> bool:
            for rx, ry in reserved:
                if abs(cell[0] - rx) <= 1 and abs(cell[1] - ry) <= 1:
                    return True
            return False

        candidates = [
            c
            for c in self._corridor_ring_cells(cx1, cy1, cx2, cy2, cw)
            if self.get_tile(c[0], c[1]) == 0 and not _too_close(c)
        ]
        # De-duplicate while preserving order (corners appear in two arms).
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        n_mines = min(int(self.MAX_MINES), len(unique))
        if n_mines > 0:
            step = len(unique) / n_mines
            picks = {int(i * step) for i in range(n_mines)}
            for idx in sorted(picks):
                mx, my = unique[idx]
                # Ground-level placement (+1.5/+2.0 border-corrected floor
                # convention, see map_mine_path): (mx, my) is the EMPTY
                # walkway cell; raw tile coords floated the mine 48px up —
                # flush with the 2-tile arm's ceiling, killing hoppers instead
                # of forcing the intended hop-over. On the floor line the mine
                # is tap-hoppable under the 48px headroom.
                self.add_entity(1, mx + 1.5, my + 2.0, 0, 1)

        return self
