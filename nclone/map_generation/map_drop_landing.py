"""Dedicated minimal generator for the DROP_LANDING mechanic.

Isolates the *controlled vertical descent* movement skill: the ninja spawns on
a short platform high up in a tall carved chamber and must FALL ``depth`` tiles
onto a narrow **landing ledge** at the bottom, steering laterally by ``offset``
tiles mid-fall. The ledge sits flush against the chamber's far wall, so the
wall is always available as a brake (press into it -> wall-slide friction).

PHYSICS FACT (measured in the sim, see ``tests/test_precision_route_generators``):
an uncontrolled walk-off free-fall is survivable up to **10 tiles**; from 11
tiles the flat-floor impact (``MAX_SURVIVABLE_IMPACT``) kills. Tiers at depth
>= 12 therefore REQUIRE braking -- the wall-slide (or a jump-hold launch) must
be learned, which is exactly the skill this bucket exists to teach. Depth 11 is
deliberately skipped by the presets (both naive strategies die right at the
boundary there).

A missed landing falls past the ledge into an inescapable pit (-> no route ->
faithful fail), mirroring ``map_long_jump``. Optional toggle mines hang in the
pit column(s) beside the ledge at surface height, so clipping the ledge edge
dies instead of merely dropping.

Like the other dedicated minimal generators (``basic_jump`` / ``long_jump`` /
``mine_path``) the background is clean solid fill, so the level is
mechanic-pure and its silhouette unambiguous.

Config knobs (see ``DropLandingConfig`` / ``generator_configs`` presets):

- ``min/max_depth``:  fall distance platform-surface -> ledge-surface (tiles).
  The difficulty driver; >= 12 crosses the braking boundary.
- ``min/max_ledge``:  landing ledge width (tiles); 1 = hardest precision.
- ``min/max_offset``: lateral gap between the platform edge and the ledge
  (tiles); the ninja must steer this far during the fall.
- ``max_mines``:      toggle mines in the pit beside the ledge edge (0-2).
- ``direction``:      ``"left" | "right" | "random"`` -- which way the drop
  chamber opens (layout mirrored for ``left``).
"""

from __future__ import annotations

from typing import Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapDropLanding(Map):
    """Minimal DROP_LANDING generator: high platform -> deep fall -> narrow ledge."""

    MIN_DEPTH = 3
    MAX_DEPTH = 5

    MIN_LEDGE = 4
    MAX_LEDGE = 5

    MIN_OFFSET = 0
    MAX_OFFSET = 0

    MAX_MINES = 0
    DIRECTION = "random"

    # Spawn platform length (tiles). Short: the drop is the mechanic.
    PLATFORM_LEN = 3
    # Pit depth below the ledge surface for the miss columns (inescapable fail).
    PIT_DEPTH = 4

    def _pick_dims(self) -> Tuple[int, int, int]:
        """Return ``(depth, ledge, offset)`` honoring the map height cap."""
        depth = self.rng.randint(self.MIN_DEPTH, self.MAX_DEPTH)
        ledge = self.rng.randint(self.MIN_LEDGE, self.MAX_LEDGE)
        offset = self.rng.randint(self.MIN_OFFSET, self.MAX_OFFSET)
        # Platform surface needs >= 2 rows of headroom below the border and the
        # ledge floor must stay above the bottom border: py >= 3, fy <= H - 3.
        max_depth = (MAP_TILE_HEIGHT - 3) - 3
        depth = min(depth, max_depth)
        return depth, ledge, offset

    def generate(self, seed: Optional[int] = None) -> "MapDropLanding":
        """Generate one DROP_LANDING level."""
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        depth, ledge, offset = self._pick_dims()
        plat = int(self.PLATFORM_LEN)
        total_w = plat + offset + ledge

        direction = self.DIRECTION
        if direction not in ("left", "right"):
            direction = self.rng.choice(["left", "right"])

        # Rows: platform surface at ``py`` (ninja stands on py - 1), landing
        # ledge floor at ``fy``. Random vertical origin within the valid band.
        py = self.rng.randint(3, (MAP_TILE_HEIGHT - 3) - depth)
        fy = py + depth
        pit_bottom = min(fy + int(self.PIT_DEPTH), MAP_TILE_HEIGHT - 2)

        # Horizontal origin (computed in RIGHTWARD frame, mirrored for left).
        x1 = self.rng.randint(2, max(2, MAP_TILE_WIDTH - total_w - 2))
        x2 = x1 + total_w - 1

        def X(x: int) -> int:
            """Mirror a column for the leftward variant (tiles are symmetric)."""
            return x if direction == "right" else MAP_TILE_WIDTH - 1 - x

        # Solid everywhere, then carve the drop chamber: from 2 rows above the
        # platform surface down to the pit bottom.
        self.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
        for x in range(x1, x2 + 1):
            for y in range(py - 2, pit_bottom + 1):
                self.set_tile(X(x), y, 0)

        # Spawn platform: a 1-tile-thick slab attached to the near wall at py.
        # Below it is pit (an undershoot falls past the platform's underside).
        for x in range(x1, x1 + plat):
            self.set_tile(X(x), py, 1)

        # Landing ledge: solid from the ledge surface down to the map bottom,
        # flush against the far wall (the brake wall).
        ledge_x0 = x1 + plat + offset
        for x in range(ledge_x0, x2 + 1):
            for y in range(fy, MAP_TILE_HEIGHT - 1):
                self.set_tile(X(x), y, 1)

        # Ninja on the platform's near end, facing the drop.
        ninja_x = x1
        self.set_ninja_spawn(X(ninja_x), py - 1, orientation=1 if direction == "right" else -1)

        # Switch + door on the ledge surface (door flush against the far wall;
        # a 1-tile ledge stacks both on the single column).
        door_x = x2
        switch_x = max(ledge_x0, x2 - 1)
        self.add_entity(3, X(door_x), fy - 1, 0, 0, X(switch_x), fy - 1)

        # Optional mines beside the ledge edge, hanging in the pit column(s) at
        # ledge-surface height: clipping the edge on a near-miss dies instead of
        # dropping. Never placed past the chamber's near end. NB: mine entities
        # carry NO center offset in ``add_entity`` (a 1.5-tile mismatch vs the
        # ninja-spawn frame -- see ``Map._place_corridor_floor_mines``), so the
        # +1.5/+1.0 below centers the mine in map column ``X(mx)`` exactly at
        # the ledge's surface plane. A raw anchor drifts a full tile into the
        # wall-slide landing path under the leftward mirror (verified fatal).
        n_mines = int(self.MAX_MINES)
        mx = ledge_x0 - 1
        while n_mines > 0 and mx > x1:
            self.add_entity(1, X(mx) + 1.5, fy + 1.0)
            n_mines -= 1
            mx -= 1

        return self
