"""Map generation for levels with jump platforms.

The level consists of a large chamber with scattered single-tile platforms.
The ninja must jump between platforms to reach the switch and then the exit door.

Key features:
- Large chamber (at least 30 tiles wide, 20 tiles high)
- Three main platforms: ninja spawn, exit switch, and exit door
- Platforms are spaced 8-12 tiles apart with random Y offsets
- Mines cover the floor at regular intervals
- Background can be solid, empty, or random tiles
"""

from .map import Map
from typing import Optional, Tuple
from .constants import VALID_TILE_TYPES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class MapJumpPlatforms(Map):
    """Generates N++ levels with jump platforms."""

    # Chamber size constants
    MIN_WIDTH = 30
    MAX_WIDTH = 40
    MIN_HEIGHT = 20
    MAX_HEIGHT = 24

    # Platform spacing constants
    MIN_PLATFORM_SPACING = 8
    MAX_PLATFORM_SPACING = 10
    MAX_Y_OFFSET = 3

    # Mine spacing in TILE units between consecutive floor mines. 0.5 tiles ==
    # 2 map-units == 12px, dense enough that a ninja (radius ~10px) cannot land
    # in a gap between mines. The catch-floor is ALWAYS mined: the ninja spawns
    # on an elevated platform and only ever touches the floor by falling, so a
    # missed hop dies instantly (a clean terminal) instead of wandering to a
    # timeout. Difficulty comes from the hops (spacing / width / height), not
    # from how sparse the floor is.
    MINE_SPACING = 0.5

    # Tiles per platform. Wider platforms = a bigger landing target = an easier
    # hop (single-tile platforms demand pixel-precise landings).
    PLATFORM_WIDTH = 3

    # OPT-IN. Mine the catch floor across the FULL map width instead of only the
    # chamber span, closing the unmined margin at each end of the line.
    #
    # WHY THIS EXISTS. ``place_floor_mines`` lays mines from ``chamber_x1 + 1`` to
    # ``chamber_x2 + 1`` on a 42-tile map, so a ninja that walks off the last
    # platform toward the chamber edge lands just past the end of the line and
    # SURVIVES. Measured on PLATFORMING/T2 level 131837852402414503: the line runs
    # x = 264..936 px with no internal gap wider than 12.5 px (exactly as dense as
    # advertised), but a ninja standing at (950, 470) is 17.2 px from the nearest
    # mine against a 14 px kill threshold -- it survived 5,000 frames, immobile.
    # That directly violates this class's own contract above ("a missed hop dies
    # instantly (a clean terminal) instead of wandering to a timeout"), and it is
    # the difference between a -3.5 death and a multi-thousand-frame stall.
    #
    # DEFAULT False, deliberately. Turning it on changes the levels, which
    # invalidates ``mechanic_benchmark_v7_official``, the 781 planner demos, the
    # 1,298 curricula, and every cross-run det-eval comparison in the run history.
    # It must be a dataset regeneration decision, not a silent behaviour change.
    #
    # Mines placed over solid background outside the chamber are inert -- nothing
    # can stand there -- so the only behavioural change is at the two ends of the
    # existing line, which is precisely the gap being closed.
    FULL_WIDTH_FLOOR_MINES = False

    def set_empty_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """Set a rectangular region of the map to empty space."""
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                self.set_tile(x, y, 0)

    def place_platform(self, x: int, y: int):
        """Place a solid platform PLATFORM_WIDTH tiles wide, centred on ``x``.

        Solid (type-1) tiles give a flat, forgiving landing surface; the width
        widens the landing target so the hop is achievable from-scratch.
        """
        pw = max(1, int(self.PLATFORM_WIDTH))
        half = pw // 2
        for xx in range(x - half, x - half + pw):
            if 1 <= xx <= MAP_TILE_WIDTH - 2:
                self.set_tile(xx, y, 1)

    def place_floor_mines(self, chamber_x1: int, chamber_x2: int, chamber_y2: int):
        """Lay a dense lethal mine line across the whole chamber floor.

        Spacing is ``MINE_SPACING`` tiles (default 0.5 == 12px). A fall off any
        platform lands here and terminates the episode immediately, so the agent
        gets a clean negative signal rather than wandering an open floor until
        timeout.

        Args:
            chamber_x1: Left edge of chamber
            chamber_x2: Right edge of chamber
            chamber_y2: Bottom edge of chamber (floor level)
        """
        # A large spacing remains a "no floor mines" escape hatch, but the
        # default tiers mine the floor densely.
        if self.MINE_SPACING >= 50:
            return
        step = max(0.5, float(self.MINE_SPACING))
        # See FULL_WIDTH_FLOOR_MINES: the chamber-only span leaves a survivable
        # margin past each end of the line, which is where measured missed hops
        # come to rest.
        if getattr(self, "FULL_WIDTH_FLOOR_MINES", False):
            x_start, x_end = 0.0, float(MAP_TILE_WIDTH)
        else:
            x_start, x_end = float(chamber_x1 + 1), float(chamber_x2 + 1)
        x = x_start
        while x <= x_end:
            self.add_entity(1, x, chamber_y2 + 2, 0, 1)  # Type 1 = toggle mine
            x += step

    def validate_platform_position(
        self,
        x: int,
        y: int,
        chamber_x1: int,
        chamber_y1: int,
        chamber_x2: int,
        chamber_y2: int,
    ) -> bool:
        """Check if a platform position is valid (at least 2 tiles from boundaries).

        Args:
            x: Platform X position
            y: Platform Y position
            chamber_x1: Left edge of chamber
            chamber_y1: Top edge of chamber
            chamber_x2: Right edge of chamber
            chamber_y2: Bottom edge of chamber

        Returns:
            True if position is valid, False otherwise
        """
        return (
            x >= chamber_x1 + 2
            and x <= chamber_x2 - 2
            and y >= chamber_y1 + 2
            and y <= chamber_y2 - 2
        )

    def place_main_platforms(
        self, chamber_x1: int, chamber_y1: int, chamber_x2: int, chamber_y2: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Place the three main platforms: ninja spawn, switch, and door.

        Args:
            chamber_x1: Left edge of chamber
            chamber_y1: Top edge of chamber
            chamber_x2: Right edge of chamber
            chamber_y2: Bottom edge of chamber

        Returns:
            Tuple of (ninja_pos, switch_pos, door_pos) where each is (x, y)
        """
        chamber_height = chamber_y2 - chamber_y1

        # Choose left or right side for first platform
        start_from_left = self.rng.choice([True, False])

        # Calculate midway height (at least midway through the chamber)
        base_y = chamber_y1 + chamber_height // 2

        if start_from_left:
            # First platform 1-3 tiles from left edge
            platform1_x = chamber_x1 + self.rng.randint(1, 3)
            platform1_y = base_y

            # Second platform 8-12 tiles away with Y offset
            y_offset1 = self.rng.randint(-self.MAX_Y_OFFSET, self.MAX_Y_OFFSET)
            platform2_x = platform1_x + self.rng.randint(
                self.MIN_PLATFORM_SPACING, self.MAX_PLATFORM_SPACING
            )
            platform2_y = base_y + y_offset1

            # Third platform 8-12 tiles away. Offset relative to platform2 (NOT
            # base_y) so the consecutive hop delta is bounded by MAX_Y_OFFSET;
            # offsetting both from base_y independently allowed a +MAX then -MAX
            # swing = 2x MAX_Y_OFFSET (frame-perfect-tier) between adjacent hops.
            y_offset2 = self.rng.randint(-self.MAX_Y_OFFSET, self.MAX_Y_OFFSET)
            platform3_x = platform2_x + self.rng.randint(
                self.MIN_PLATFORM_SPACING, self.MAX_PLATFORM_SPACING
            )
            platform3_y = platform2_y + y_offset2
        else:
            # First platform 1-3 tiles from right edge
            platform1_x = chamber_x2 - self.rng.randint(1, 3)
            platform1_y = base_y

            # Second platform 8-12 tiles away (to the left) with Y offset
            y_offset1 = self.rng.randint(-self.MAX_Y_OFFSET, self.MAX_Y_OFFSET)
            platform2_x = platform1_x - self.rng.randint(
                self.MIN_PLATFORM_SPACING, self.MAX_PLATFORM_SPACING
            )
            platform2_y = base_y + y_offset1

            # Third platform 8-12 tiles away. Offset relative to platform2 (NOT
            # base_y) so the consecutive hop delta is bounded by MAX_Y_OFFSET;
            # offsetting both from base_y independently allowed a +MAX then -MAX
            # swing = 2x MAX_Y_OFFSET (frame-perfect-tier) between adjacent hops.
            y_offset2 = self.rng.randint(-self.MAX_Y_OFFSET, self.MAX_Y_OFFSET)
            platform3_x = platform2_x - self.rng.randint(
                self.MIN_PLATFORM_SPACING, self.MAX_PLATFORM_SPACING
            )
            platform3_y = platform2_y + y_offset2

        # Ensure all platforms are at least 2 tiles from boundaries
        platform1_y = max(chamber_y1 + 2, min(platform1_y, chamber_y2 - 2))
        platform2_y = max(chamber_y1 + 2, min(platform2_y, chamber_y2 - 2))
        platform3_y = max(chamber_y1 + 2, min(platform3_y, chamber_y2 - 2))

        # Clamp X positions as well
        platform1_x = max(chamber_x1 + 2, min(platform1_x, chamber_x2 - 2))
        platform2_x = max(chamber_x1 + 2, min(platform2_x, chamber_x2 - 2))
        platform3_x = max(chamber_x1 + 2, min(platform3_x, chamber_x2 - 2))

        # Place the platforms
        self.place_platform(platform1_x, platform1_y)
        self.place_platform(platform2_x, platform2_y)
        self.place_platform(platform3_x, platform3_y)

        # Return positions: ninja, switch, door
        return (
            (platform1_x, platform1_y),
            (platform2_x, platform2_y),
            (platform3_x, platform3_y),
        )

    def generate(self, seed: Optional[int] = None) -> Map:
        """Generate a level with jump platforms.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            Map: A Map instance with the generated level
        """
        if seed is not None:
            self.rng.seed(seed)

        self.reset()

        # Generate random dimensions for chamber (at least 30x20)
        width = self.rng.randint(self.MIN_WIDTH, self.MAX_WIDTH)
        height = self.rng.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Calculate chamber position
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1

        chamber_x1 = self.rng.randint(2, max(3, max_start_x))
        chamber_y1 = self.rng.randint(2, max(3, max_start_y))
        chamber_x2 = min(chamber_x1 + width, MAP_TILE_WIDTH - 2)
        chamber_y2 = min(chamber_y1 + height, MAP_TILE_HEIGHT - 2)

        # Pre-generate background tiles
        # Choose if tiles will be random, solid, or empty
        choice = self.rng.randint(0, 2)
        if choice == 0:
            # Random tiles
            tile_types = [
                self.rng.randint(0, VALID_TILE_TYPES)
                for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
            ]
        elif choice == 1:
            # Solid walls
            tile_types = [1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        else:
            # Empty tiles
            tile_types = [0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        self.set_tiles_bulk(tile_types)

        # Create the empty chamber
        self.set_empty_rectangle(chamber_x1, chamber_y1, chamber_x2, chamber_y2)

        # Create boundary walls
        for x in range(chamber_x1, chamber_x2 + 1):
            self.set_tile(x, chamber_y2 + 1, 1)  # Bottom wall
            self.set_tile(x, chamber_y1 - 1, 1)  # Top wall

        for y in range(chamber_y1, chamber_y2 + 1):
            self.set_tile(chamber_x1 - 1, y, 1)  # Left wall
            self.set_tile(chamber_x2 + 1, y, 1)  # Right wall

        # Place the three main platforms and get their positions
        ninja_pos, switch_pos, door_pos = self.place_main_platforms(
            chamber_x1, chamber_y1, chamber_x2, chamber_y2
        )
        ninja_pos = ninja_pos[0], ninja_pos[1] - 1
        switch_pos = switch_pos[0], switch_pos[1] - 1
        door_pos = door_pos[0], door_pos[1] - 1

        # Place floor mines
        self.place_floor_mines(chamber_x1, chamber_x2, chamber_y2)

        # Determine ninja orientation based on which direction they need to go
        if switch_pos[0] > ninja_pos[0]:
            ninja_orientation = 1  # Face right
        else:
            ninja_orientation = -1  # Face left

        # Place entities
        self.set_ninja_spawn(ninja_pos[0], ninja_pos[1], ninja_orientation)
        self.add_entity(3, door_pos[0], door_pos[1], 0, 0, switch_pos[0], switch_pos[1])

        # Add random entities outside the playspace
        playspace = (chamber_x1 - 4, chamber_y1 - 4, chamber_x2 + 4, chamber_y2 + 4)
        self.add_random_entities_outside_playspace(
            playspace[0], playspace[1], playspace[2], playspace[3]
        )

        return self
