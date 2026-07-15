"""CURVES bucket generator: traverse curved terrain (convex/concave).

Builds single-mechanic levels whose ground is shaped by curved tiles -- convex
humps the ninja rides over and concave bowls it rides through (see
:mod:`curve_utils` for the verified tile orientations). No slopes, gaps, or
wall-jump segments: the only mechanic is momentum control over/through curves.

Entity scope: player spawn, exactly one exit switch + exit door, and (only at
the higher tiers) toggle mines. Nothing else.

Config-driven class attributes (set by ``GeneratorFactory`` from
``CurvesConfig``): MIN/MAX_WIDTH, MIN/MAX_HEIGHT, MIN/MAX_CURVES, MAX_MINES,
VARIANT (``"bowl"`` | ``"hump"`` | ``"mixed"`` | ``"undulation"``).
"""

from typing import Optional

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import composition as comp


class MapCurves(Map):
    MIN_WIDTH = 22
    MAX_WIDTH = 40
    MIN_HEIGHT = 8
    MAX_HEIGHT = 14
    MIN_CURVES = 1
    MAX_CURVES = 2
    MAX_MINES = 0
    VARIANT = "mixed"

    def _curve_segment_name(self) -> str:
        if self.VARIANT == "bowl":
            return "bowl"
        if self.VARIANT == "hump":
            return "hump"
        return self.rng.choice(["bowl", "hump"])

    def generate(self, seed: Optional[int] = None) -> "MapCurves":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
        base_row = MAP_TILE_HEIGHT - 2
        floor_row = min(base_row - 1, max(5, height))

        n_curves = self.rng.randint(self.MIN_CURVES, self.MAX_CURVES)

        # Plan: spawn flat, [flat, curve]*, exit flat. Widths sized to fit.
        plan = [("flat", 6)]
        if self.VARIANT == "undulation":
            # one long alternating run of bowls and humps
            for i in range(max(2, n_curves)):
                plan.append(("bowl" if i % 2 == 0 else "hump", 3))
        else:
            for _ in range(n_curves):
                plan.append(("flat", self.rng.randint(2, 4)))
                plan.append((self._curve_segment_name(), self.rng.randint(3, 4)))
        plan.append(("flat", 7))

        results = comp.assemble_segments(self, plan, floor_row, base_row, self.rng)
        if not comp.finalize_floor_level(self, results, int(self.MAX_MINES), self.rng):
            # Degenerate fit: fall back to a minimal single-curve level.
            self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
            results = comp.assemble_segments(
                self,
                [("flat", 6), (self._curve_segment_name(), 3), ("flat", 8)],
                floor_row,
                base_row,
                self.rng,
            )
            comp.finalize_floor_level(self, results, 0, self.rng)
        # Seal the play area: make a continuous solid base under EVERY column
        # (and a flat floor wherever the segments left a gap) so the agent can
        # never run/fall off the curve platform into an empty pit.
        self._seal_play_area(floor_row, base_row)
        return self

    def _seal_play_area(self, floor_row: int, base_row: int) -> None:
        """Fill solid from each column's surface tile down to the BOTTOM row and
        add a flat floor at ``floor_row`` in any column with no surface, so the
        bottom of the playfield is one continuous solid mass (curve shape kept on
        top, no internal pits or fall-off-the-edge gaps).

        Covers the FULL inner width (cols 0..W-1) and down to the bottom inner row
        (W-1 height). Previously it stopped at cols 1..W-2 and ``base_row``, which
        left an empty 'moat' around the block — the two edge columns and the bottom
        row — so the ninja could run off the surface edge, drop into that moat, and
        get stuck UNDER the block. Sealing the full perimeter closes that trap.
        """
        for x in range(0, MAP_TILE_WIDTH):
            surface = None
            for y in range(MAP_TILE_HEIGHT):
                if self.get_tile(x, y) != 0:
                    surface = y
                    break
            if surface is None:
                surface = floor_row  # empty column -> extend the flat floor here
            for y in range(surface, MAP_TILE_HEIGHT):
                if self.get_tile(x, y) == 0:
                    self.set_tile(x, y, 1)
