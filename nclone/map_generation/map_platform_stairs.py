"""Dedicated generator for the ascending-platform-staircase archetype
(``open-platform-stairs``).

Builds an open arena containing a staircase of floating single-tile (or short)
platforms that ascend diagonally in a single direction. The ninja spawns on the
floor at the base, hops platform-to-platform up the staircase, hits the exit
switch on an intermediate platform and reaches the exit door on the top
platform. Toggle mines fill the floor under the gaps to punish missed jumps.

Unlike the ``jump_platforms`` generator (three randomly-offset platforms in a
mined chamber, the PLATFORMING bucket's anchor), this generalises a *monotonic
ascending staircase*: a clean, ordered sequence of upward hops. It is its own
PLATFORM_STAIRS bucket, scheduled after PLATFORMING.

Config-driven class attributes (set by ``GeneratorFactory`` from
``PlatformStairsConfig``): MIN/MAX_WIDTH + MIN/MAX_HEIGHT (size-curriculum
parity knobs), MIN/MAX_STEPS (number of platforms), MIN/MAX_STEP_DX +
MIN/MAX_STEP_DY (horizontal / vertical spacing between consecutive steps),
PLATFORM_WIDTH (tiles per platform), MAX_MINES, DIRECTION
(``"left"`` | ``"right"`` | ``"random"`` -- which way the staircase climbs),
PROFILE (``"ascend"`` -- spawn on the floor, hop UP the staircase to the door
on the top platform -- or ``"descend"`` -- spawn ON the top platform and hop
DOWN, door on the base platform; every hop is a drop onto a narrow target, so
the descent variant trains controlled falling instead of upward jumps).
"""

from typing import List, Optional, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapPlatformStairs(Map):
    """Open arena: hop an ascending staircase of floating platforms."""

    MIN_WIDTH = 24
    MAX_WIDTH = 38
    MIN_HEIGHT = 12
    MAX_HEIGHT = 20

    MIN_STEPS = 3
    MAX_STEPS = 5
    MIN_STEP_DX = 3
    MAX_STEP_DX = 5
    MIN_STEP_DY = 2
    MAX_STEP_DY = 3
    PLATFORM_WIDTH = 1
    MAX_MINES = 4
    DIRECTION = "random"
    PROFILE = "ascend"  # "ascend" (legacy) | "descend"

    def _plan_steps(self, floor_row: int) -> List[Tuple[int, int]]:
        """Return the (x, y) top-left cell of each platform, base -> top."""
        direction = self.DIRECTION
        if direction not in ("left", "right"):
            direction = self.rng.choice(["left", "right"])
        sign = 1 if direction == "right" else -1

        steps = self.rng.randint(self.MIN_STEPS, self.MAX_STEPS)
        dx = self.rng.randint(self.MIN_STEP_DX, self.MAX_STEP_DX)
        dy = self.rng.randint(self.MIN_STEP_DY, self.MAX_STEP_DY)
        pw = max(1, int(self.PLATFORM_WIDTH))

        # Base platform sits a short hop above the floor.
        base_y = floor_row - 2

        # Clamp the number of steps so the staircase stays inside the arena in
        # both axes (top row >= 1, x span within [1, W-2]).
        max_steps_y = (base_y - 1) // dy + 1
        span_budget = (MAP_TILE_WIDTH - 3) - pw
        max_steps_x = span_budget // dx + 1
        # Floor of 1 platform so the easiest tier can be a single atomic hop
        # (T1); entity placement below special-cases the single-platform case.
        steps = max(1, min(steps, max_steps_y, max_steps_x))

        # A SINGLE platform (T1) has no dx/dy chain to draw variety from, so a
        # fixed hop height collapses the unique-level count -> heavy dedup
        # collisions starve the tier (empirically only ~37 unique train / 0 test).
        # Randomize the single hop's HEIGHT over a small forgiving range (2-5
        # tiles) to restore variety without adding difficulty (still one mine-free
        # hop onto a wide ledge). Multi-step tiers keep the fixed base height.
        if steps == 1:
            max_up = min(5, max(2, base_y - 1))
            base_y = floor_row - self.rng.randint(2, max_up)

        total_dx = (steps - 1) * dx
        if sign > 0:
            base_x = self.rng.randint(1, max(1, (MAP_TILE_WIDTH - 2 - pw) - total_dx))
        else:
            base_x = self.rng.randint(
                1 + total_dx, max(1 + total_dx, MAP_TILE_WIDTH - 2 - pw)
            )

        return [(base_x + sign * i * dx, base_y - i * dy) for i in range(steps)]

    def generate(self, seed: Optional[int] = None) -> "MapPlatformStairs":
        if seed is not None:
            self.rng.seed(seed)
        self.reset()

        # Open arena; ninja stands on the implicit solid border.
        self.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

        floor_row = MAP_TILE_HEIGHT - 1
        pw = max(1, int(self.PLATFORM_WIDTH))

        platforms = self._plan_steps(floor_row)

        # Stamp each platform (a short solid ledge).
        for px, py in platforms:
            for x in range(px, px + pw):
                if 1 <= x <= MAP_TILE_WIDTH - 2 and 1 <= py <= MAP_TILE_HEIGHT - 2:
                    self.set_tile(x, py, 1)

        base_x, base_y = platforms[0]
        top_x, top_y = platforms[-1]
        ascending_right = top_x >= base_x
        descend = str(getattr(self, "PROFILE", "ascend")) == "descend"

        if descend:
            # DESCENT profile: spawn ON the top platform and hop DOWN the same
            # staircase -- every hop is a drop onto a (possibly single-tile)
            # target, so the route trains controlled falling. Switch on an
            # intermediate platform, door on the BASE (lowest) platform.
            ninja_x = top_x
            self.set_ninja_spawn(ninja_x, top_y - 1, -1 if ascending_right else 1)
            if len(platforms) == 1:
                # Degenerate single-platform descent (not used by the presets):
                # one drop from the spawn platform to the door on the floor.
                switch_x, switch_y = top_x, top_y - 1
                door_x = top_x + (-3 if ascending_right else 3)
                door_x = max(1, min(door_x, MAP_TILE_WIDTH - 2))
                door_y = floor_row
            else:
                mid = max(1, len(platforms) // 2)
                sw_x, sw_y = platforms[mid]
                switch_x, switch_y = sw_x, sw_y - 1
                door_x, door_y = base_x, base_y - 1
        else:
            # Ninja on the floor just behind the base platform, facing the climb.
            ninja_x = base_x - 1 if ascending_right else base_x + pw
            ninja_x = max(1, min(ninja_x, MAP_TILE_WIDTH - 2))
            self.set_ninja_spawn(ninja_x, floor_row, 1 if ascending_right else -1)

            # Switch on an intermediate platform, door on the top platform (each
            # sits on the platform's walking surface, one row above the ledge).
            # Single-platform (T1) special case: the staircase is one atomic hop, so
            # there is no intermediate platform — put the switch on the floor in the
            # mine-clear launch zone next to the spawn (a trivial walk) and the exit
            # door on the single platform, so the level is "touch switch, hop up to
            # the exit" (teaches the core floor->platform landing in isolation).
            if len(platforms) == 1:
                switch_x, switch_y = ninja_x, floor_row
                door_x, door_y = top_x, top_y - 1
            else:
                mid = max(1, len(platforms) // 2)
                sw_x, sw_y = platforms[mid]
                switch_x, switch_y = sw_x, sw_y - 1
                door_x, door_y = top_x, top_y - 1
        self.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)

        # Lethal floor under the staircase span: a missed hop lands here and
        # terminates the episode immediately (a clean negative signal) instead of
        # letting the ninja wander the open floor until timeout. Candidate slots
        # are spaced 12px (0.5 tile == 2 map-units) so a FULLY-dense floor leaves
        # no safe gap for a ninja (radius ~10px). A small launch zone around the
        # spawn stays clear so the first hop off the floor is possible.
        xs = [px for px, _ in platforms]
        lo, hi = min(xs), max(xs) + pw - 1
        lo = max(1, lo)
        hi = min(MAP_TILE_WIDTH - 2, hi)
        candidates = []
        x = float(lo)
        while x <= hi:
            # Ascent: keep the floor launch run-up around the spawn clear. The
            # descent profile spawns ON the top platform, so every floor touch
            # is a failed hop and the whole span may be mined.
            if descend or abs(x - ninja_x) > 2.0:
                candidates.append(x)
            x += 0.5  # 12px spacing

        # MAX_MINES controls the mine density so difficulty tiers can span from a
        # forgiving on-ramp to a zero-recovery floor:
        #   0            -> no mines (a missed hop is survivable; easiest tiers),
        #   >= #slots    -> fully-dense lethal floor (hardest tiers),
        #   otherwise    -> evenly subsampled to MAX_MINES (safe gaps between).
        cap = int(getattr(self, "MAX_MINES", len(candidates)))
        if cap <= 0:
            chosen = []
        elif cap >= len(candidates):
            chosen = candidates
        else:
            stride = len(candidates) / cap
            chosen = [candidates[int(i * stride)] for i in range(cap)]
        for mx in chosen:
            # Ground-level placement (+1.5/+2.0 border-corrected floor
            # convention, see map_mine_path): raw tile coords floated the
            # "mine floor" 48px above the actual floor — killing jump ARCS
            # instead of landings. On the floor line a missed jump that lands
            # in the pit dies (zero-recovery preserved) while clean platform
            # hops clear it.
            self.add_entity(1, mx + 1.5, floor_row + 2.0, 0, 1)

        return self
