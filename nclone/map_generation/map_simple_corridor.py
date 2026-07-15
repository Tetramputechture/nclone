"""Extremely-simple 1-tile-wide loop-corridor generator (the FAR_OBJECTIVE on-ramp).

The simplest possible "traverse around a bend to the switch, then back to the
exit" mechanic: an otherwise-all-solid map with exactly **5** empty tiles carved
into one of two 1-tile-wide corridor shapes (no slopes, square 0/1 tiles only,
no mines, no rounded corners). The ninja, the exit door, and the exit switch sit
at the two ends of the corridor, so completing the level requires the full
round-trip (spawn -> around the bend to the switch -> back to the exit door).

Two ORIENTATIONS (set by ``MapSimpleCorridor.ORIENTATION``; y increases
DOWNWARD, gravity pulls +y, "up" == -y):

``horizontal`` -- an upside-down U (``∩``). The agent jumps up one tile, walks
across the 3-tile top, and drops down the far leg::

    top  (gy)   : (gx,gy) (gx+1,gy) (gx+2,gy)      [3 empties]
    legs (gy+1) : (gx,gy+1)         (gx+2,gy+1)    [2 empties; (gx+1,gy+1) SOLID]
    floor(gy+2) : solid

  spawn + exit door at the bottom-left leg ``(gx, gy+1)``; switch at the
  bottom-right leg ``(gx+2, gy+1)``.

``vertical`` -- a sideways C opening right. The agent walks left along the
TOP, DESCENDS the 1-tile spine, then walks right along the bottom to the
switch::

    top    (gy)  : (gx,gy) (gx+1,gy)               [2 empties]
    spine  (gy+1): (gx,gy+1)                       [1 empty;  (gx+1,gy+1) SOLID]
    bottom (gy+2): (gx,gy+2) (gx+1,gy+2)           [2 empties]

  spawn + exit door at the TOP-right open end ``(gx+1, gy)``; switch at the
  BOTTOM-right open end ``(gx+1, gy+2)``.

  NB: spawn/switch are placed so the spine is DESCENDED, not climbed. The
  originally-specified climbing orientation (spawn bottom, switch top) is NOT
  physically solvable -- the ninja cannot scale a 1-tile-wide spine to reach a
  switch one tile up and over (verified via StartStateCurriculum: min
  cost-to-go to the switch plateaus at ~7.8px and never reaches 0). The
  descending orientation reaches the switch (min progress 0.0), so it is the
  one used here. The same applies to the mirrored "backwards-C".

The base ``(gx, gy)`` is randomized within the playable interior (away from the
1-tile border, with a solid floor below the carved region) and is deterministic
when a seed is supplied.
"""

from typing import Optional

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapSimpleCorridor(Map):
    """Extremely-simple 1-tile-wide ∩ / C loop corridor (5 empty tiles)."""

    # "horizontal" == upside-down-U (∩); "vertical" == sideways-C.
    ORIENTATION = "horizontal"

    # Footprint of the carved region (the 3x3 bounding box anchored at base).
    _FOOTPRINT_W = 3
    _FOOTPRINT_H = 3

    def _carve_horizontal(self, gx: int, gy: int) -> None:
        """Upside-down U (∩): 3-tile top + 2 legs, inside stays solid."""
        # Top corridor (3).
        self.set_tile(gx, gy, 0)
        self.set_tile(gx + 1, gy, 0)
        self.set_tile(gx + 2, gy, 0)
        # Two legs (1 each); (gx+1, gy+1) is left SOLID (the inside of the ∩).
        self.set_tile(gx, gy + 1, 0)
        self.set_tile(gx + 2, gy + 1, 0)

        # Spawn + exit door at the bottom-left leg; switch at the bottom-right leg.
        self.set_ninja_spawn(gx, gy + 1, 1)  # face right (toward the climb)
        self.add_entity(3, gx, gy + 1, 0, 0, gx + 2, gy + 1)

    def _carve_vertical(self, gx: int, gy: int) -> None:
        """Sideways C opening right: top (2) + spine mid (1) + bottom (2)."""
        # Top (2).
        self.set_tile(gx, gy, 0)
        self.set_tile(gx + 1, gy, 0)
        # Spine mid (1); (gx+1, gy+1) is left SOLID (closes the C).
        self.set_tile(gx, gy + 1, 0)
        # Bottom (2).
        self.set_tile(gx, gy + 2, 0)
        self.set_tile(gx + 1, gy + 2, 0)

        # Spawn + exit door at the TOP-right open end; switch at the BOTTOM-right
        # open end -- so the spine is DESCENDED (solvable), not climbed.
        self.set_ninja_spawn(gx + 1, gy, 1)  # face right initially
        self.add_entity(3, gx + 1, gy, 0, 0, gx + 1, gy + 2)

    def generate(self, seed: Optional[int] = None) -> "MapSimpleCorridor":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        orientation = str(self.ORIENTATION).lower()
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"MapSimpleCorridor.ORIENTATION must be 'horizontal' or "
                f"'vertical', got {self.ORIENTATION!r}"
            )

        # Solid rock everywhere; carve the 5-tile corridor.
        self.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        # Pick a base (gx, gy) in the playable interior. Keep the whole 3x3
        # footprint (and a solid floor row below it) clear of the 1-tile border.
        # gx range: [1 .. W-1-FOOTPRINT_W]; gy range: [1 .. H-2-FOOTPRINT_H]
        # (the extra -1 on gy leaves a solid floor row beneath the footprint).
        gx_lo, gx_hi = 1, MAP_TILE_WIDTH - 1 - self._FOOTPRINT_W
        gy_lo, gy_hi = 1, MAP_TILE_HEIGHT - 2 - self._FOOTPRINT_H
        gx = self.rng.randint(gx_lo, gx_hi)
        gy = self.rng.randint(gy_lo, gy_hi)

        if orientation == "horizontal":
            self._carve_horizontal(gx, gy)
        else:
            self._carve_vertical(gx, gy)

        return self
