"""
Configuration system for procedural map generation.

This module centralizes all generator parameters, presets, and category
configurations for the test suite generation system.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MazeConfig:
    """Configuration for maze generator."""

    min_width: int = 6
    max_width: int = 20
    min_height: int = 6
    max_height: int = 10
    max_cell_size: int = 4
    use_random_wall_tiles: bool = False


@dataclass
class JumpRequiredConfig:
    """Configuration for jump-required levels."""

    min_width: int = 16
    max_width: int = 40
    min_height: int = 8
    max_height: int = 16
    min_pit_width: int = 3
    max_pit_width: int = 5
    max_mines_per_platform: int = 5


@dataclass
class HillsConfig:
    """Configuration for hills terrain generator."""

    min_width: int = 10
    max_width: int = 40
    min_height: int = 8
    max_height: int = 22
    min_hills: int = 1
    max_hills: int = 8
    min_height_change: int = 1
    max_height_change: int = 10


@dataclass
class VerticalCorridorConfig:
    """Configuration for vertical corridor generator."""

    min_width: int = 1
    max_width: int = 6
    min_height: int = 8
    max_height: int = 22
    min_mine_spacing: int = 2
    max_mine_spacing: int = 5
    add_platforms: bool = False
    add_mid_mines: bool = False
    add_chaotic_walls: bool = False
    add_wall_mines: bool = False
    add_boundary_mines: bool = False


@dataclass
class MineMazeConfig:
    """Configuration for mine maze generator."""

    min_width: int = 6
    max_width: int = 40
    min_height: int = 3
    max_height: int = 6
    min_skip_columns: int = 2
    max_skip_columns: int = 4
    min_mines_per_column: int = 1
    max_mines_per_column: int = 10


@dataclass
class IslandsConfig:
    """Configuration for islands generator."""

    min_width: int = 36
    max_width: int = 38
    min_height: int = 12
    max_height: int = 19
    min_island_spacing: int = 1
    max_island_spacing: int = 1
    min_island_size: int = 1
    max_island_size: int = 4
    border_distance: int = 1


@dataclass
class JumpPlatformsConfig:
    """Configuration for jump platforms generator."""

    min_width: int = 30
    max_width: int = 40
    min_height: int = 20
    max_height: int = 24
    min_platform_spacing: int = 8
    max_platform_spacing: int = 10
    max_y_offset: int = 3
    # 0.5 tiles == 12px == a dense lethal catch-floor (falls terminate cleanly).
    mine_spacing: float = 0.5
    # Tiles per platform: a wider landing target = an easier hop.
    platform_width: int = 3


@dataclass
class MultiChamberConfig:
    """Configuration for multi-chamber generator."""

    min_chambers: int = 2
    max_chambers: int = 4
    min_chamber_width: int = 4
    max_chamber_width: int = 10
    min_chamber_height: int = 3
    max_chamber_height: int = 6
    min_corridor_length: int = 1
    max_corridor_length: int = 5
    min_corridor_width: int = 1
    max_corridor_width: int = 3
    # Gold (score objective, docs/GOLD_AND_SCORE_OPTIMIZATION.md): CLUSTERED regime
    # — gold is grouped along the floors of the ninja/switch chambers (matching the
    # ~93%-clustered corpus majority), reachable by construction. 0 disables gold.
    min_gold_count: int = 4
    max_gold_count: int = 14


@dataclass
class SingleChamberConfig:
    """Configuration for single chamber generator."""

    min_width: int = 4
    max_width: int = 30
    min_height: int = 4
    max_height: int = 10
    global_max_up_deviation: int = 5
    global_max_down_deviation: int = 1
    add_center_obstacle: bool = False
    add_floor_mines: bool = False
    add_platform_gap: bool = False
    add_ceiling_obstacles: bool = False
    # Trivial on-ramp knob: when > 0, the switch is forced within this many
    # tiles of the exit door (and the ninja spawns just beyond the switch),
    # producing a near-trivial switch->door route on a small chamber. 0 keeps
    # the legacy independent random placement. The size curriculum interpolates
    # this field, so a tiny sep=1 floor preset anneals back to 0 (unconstrained)
    # as the bucket's size grows. See FLAT_RUN ``flat_run_onramp`` preset.
    switch_door_max_sep: int = 0
    # Gold (score objective, docs/GOLD_AND_SCORE_OPTIMIZATION.md): SPREAD regime —
    # gold scattered across the chamber floor so the score rewards sweeping the
    # whole arena (the large-open case). Reachable by construction. 0 disables gold.
    min_gold_count: int = 3
    max_gold_count: int = 12


@dataclass
class HorizontalCorridorConfig:
    """Configuration for horizontal corridor generator."""

    min_width: int = 3
    max_width: int = 23
    min_height: int = 1
    max_height: int = 5
    random_edge_tiles: bool = False
    fixed_height: int = None
    add_mines: bool = False
    max_mines: int = -1
    # When False, suppress the incidental locked-door injection that the
    # generator otherwise adds to height==1, width>=4 corridors. Locked doors
    # are out of scope for the Phase-1 entity subset ({mine, gold, exit door,
    # exit switch}), so this defaults to False -- a preset must explicitly opt
    # in with allow_locked_doors=True to emit them. (Previously defaulted True,
    # which silently leaked type-6 locked doors from the simplest/simple
    # presets that forgot to set it.)
    allow_locked_doors: bool = False


@dataclass
class CorridorsConfig:
    """Configuration for corridors generator."""

    min_corridors: int = 2
    max_corridors: int = 5
    min_horizontal_width: int = 8
    max_horizontal_width: int = 25
    min_horizontal_height: int = 1
    max_horizontal_height: int = 3
    min_vertical_width: int = 1
    max_vertical_width: int = 6
    min_vertical_height: int = 6
    max_vertical_height: int = 15
    add_zig_zag_drops: bool = False
    add_chaotic_walls: bool = False
    add_mines: bool = False
    max_mines: int = -1


@dataclass
class BasicJumpConfig:
    """Configuration for the dedicated BASIC_JUMP generator.

    Minimal single-chamber level with exactly one required jump-type
    obstacle (gap, drop, or step-up). No mines, no slopes, no
    wall-jump segments.
    """

    min_width: int = 12
    max_width: int = 20
    min_height: int = 5
    max_height: int = 8
    min_gap_width: int = 2
    max_gap_width: int = 3
    min_drop_height: int = 2
    max_drop_height: int = 3
    min_step_height: int = 1
    max_step_height: int = 2
    variant: str = "single_gap_flat"
    # Mixed tier 3+4: optional toggle mines (Layer 1 only)
    max_mines: int = 0


@dataclass
class WallJumpConfig:
    """Configuration for the dedicated WALL_JUMP generator.

    Clean narrow vertical shaft with smooth parallel walls. No mines,
    no chaotic wall tiles. Optional single middle ledge for higher
    tiers.
    """

    min_width: int = 2
    max_width: int = 3
    min_height: int = 8
    max_height: int = 18
    add_middle_ledge: bool = False
    goal_position: str = "top"


@dataclass
class MinePathConfig:
    """Configuration for the dedicated MINE_NAVIGATION generator.

    Flat horizontal corridor with N toggle mines on the floor at a
    parameterized minimum spacing. No walls, no slopes, no vertical
    mechanics.
    """

    min_width: int = 10
    max_width: int = 24
    min_height: int = 3
    max_height: int = 5
    min_mine_count: int = 1
    max_mine_count: int = 4
    min_mine_spacing: int = 2
    spawn_buffer_tiles: int = 2
    # One-tile step-up for mixed tier 3+4
    add_step: bool = False


@dataclass
class SlopesConfig:
    """Configuration for the dedicated SLOPES generator."""

    min_width: int = 12
    max_width: int = 32
    min_height: int = 6
    max_height: int = 16
    min_slope_height: int = 2
    max_slope_height: int = 4
    slope_type_pool: Tuple[str, ...] = ("mild",)
    variant: str = "single_slope_up"
    max_mines: int = 0
    min_mine_spacing: int = 2


@dataclass
class CurvesConfig:
    """Configuration for the dedicated CURVES generator.

    Traverse curved terrain -- convex humps (ride over) and concave bowls
    (ride through). No slopes/gaps/wall-jumps. ``min/max_curves`` scales the
    number of curved features with the tier (registered as an interpolatable
    count field in npp-rl's size curriculum).
    """

    min_width: int = 22
    max_width: int = 40
    min_height: int = 8
    max_height: int = 14
    min_curves: int = 1
    max_curves: int = 2
    max_mines: int = 0
    variant: str = "mixed"  # bowl | hump | mixed | undulation


@dataclass
class MultiJumpConfig:
    """Configuration for the dedicated MULTI_JUMP generator.

    Levels requiring >=2 sequenced jumps (gaps and/or step-up staircases).
    ``min/max_jumps`` scales the jump count with the tier.
    """

    min_width: int = 24
    max_width: int = 40
    min_height: int = 8
    max_height: int = 14
    min_jumps: int = 2
    max_jumps: int = 3
    max_mines: int = 0
    variant: str = "gaps"  # gaps | staircase | mixed
    max_gap_w: int = 3  # carved-gap width cap (2 == easy hop, 3 == hard long-jump)
    runup: int = 5  # initial flat run-up length (more = easier to clear a gap)


@dataclass
class CompositeConfig:
    """Configuration for the COMPOSITE generator (large multi-mechanic levels).

    ``min/max_segments`` scales how many mechanic segments are chained across
    the canvas with the tier.
    """

    min_width: int = 30
    max_width: int = 40
    min_height: int = 10
    max_height: int = 16
    min_segments: int = 2
    max_segments: int = 5
    max_mines: int = 0
    variant: str = "mixed"  # mixed | undulation | terrain


@dataclass
class PrecisionConfig:
    """Configuration for the PRECISION generator (narrow mined ledges).

    Chains of short ledges separated by gaps with toggle mines on the ledges --
    precise landing under a mine-avoidance constraint. Always carries >=1 mine.

    ``gap_width`` (default 2 == the legacy fixed hop) widens the carved gaps to
    3+ tiles on harder presets (a running long-jump between narrow ledges);
    ``ledge_width`` may now go down to 2 (the legacy clamp floor was 3). The
    defaults reproduce the legacy presets byte-for-byte.
    """

    min_width: int = 22
    max_width: int = 38
    min_height: int = 8
    max_height: int = 12
    min_jumps: int = 2
    max_jumps: int = 4
    max_mines: int = 2
    ledge_width: int = 3
    gap_width: int = 2


@dataclass
class OpenFieldConfig:
    """Configuration for the open flat-run generator (``flat-run-open``).

    Open arena, mine-free; the size knob is the run length the spawn/switch/door
    are spread across. Folded into FLAT_RUN as the open-space variety.
    """

    min_width: int = 18
    max_width: int = 38
    min_height: int = 10
    max_height: int = 20
    min_separation: int = 3
    max_separation: int = 8
    # Gold (score objective, docs/GOLD_AND_SCORE_OPTIMIZATION.md): SPREAD regime —
    # gold scattered across the full run width at/just above the floor so the score
    # rewards sweeping the whole open arena (the large-open case the doc targets).
    # Reachable by construction (on or a hop above the continuous floor). 0 disables.
    min_gold_count: int = 5
    max_gold_count: int = 16


@dataclass
class RoundCornerConfig:
    """Configuration for the rounded-corner wall-ascent generator
    (``open-round-corner``).

    Open arena, mine-free; a concave quarter-pipe eases the floor into a side
    wall the ninja climbs to a door floating ``door_height`` tiles up. Folded
    into CURVES as a curve-assisted wall-ascent variety.
    """

    min_width: int = 16
    max_width: int = 34
    min_height: int = 12
    max_height: int = 20
    min_door_height: int = 4
    max_door_height: int = 10
    min_corner_radius: int = 1
    max_corner_radius: int = 3
    side: str = "random"


@dataclass
class LoopCorridorConfig:
    """Configuration for the loop-corridor generator (``circle-corridor``).

    A horseshoe ring corridor carved from solid rock around a central pillar;
    the ninja traverses the full loop (walk + wall-jump) avoiding toggle mines.
    Anchor generator of the LOOP_CORRIDOR bucket.
    """

    min_width: int = 14
    max_width: int = 20
    min_height: int = 12
    max_height: int = 16
    # 2 == wall-jumpable shaft width (see MapLoopCorridor.CORRIDOR_WIDTH); cw=3
    # made the vertical-arm climb physically unsolvable.
    corridor_width: int = 2
    max_mines: int = 4
    round_corners: bool = True


@dataclass
class SimpleCorridorConfig:
    """Configuration for the extremely-simple 1-tile-wide corridor generator
    (``simple_corridor``).

    The minimal FAR_OBJECTIVE on-ramp: an all-solid map with exactly 5 empty
    tiles carved into an upside-down-U (``horizontal``) or a sideways-C
    (``vertical``) 1-tile corridor. Spawn/exit at one end, switch at the other,
    so the level demands the short round-trip. No mines, no slopes, no rounded
    corners. ``orientation`` selects the shape; there are no size knobs (it is
    always the smallest possible shape).
    """

    orientation: str = "horizontal"  # "horizontal" (∩) | "vertical" (C)


@dataclass
class PlatformStairsConfig:
    """Configuration for the ascending-platform-staircase generator
    (``open-platform-stairs``).

    A monotonic staircase of floating platforms in an open arena with mines on
    the floor under the gaps. Anchor generator of the PLATFORM_STAIRS bucket.
    """

    min_width: int = 24
    max_width: int = 38
    min_height: int = 12
    max_height: int = 20
    min_steps: int = 3
    max_steps: int = 5
    min_step_dx: int = 3
    max_step_dx: int = 5
    min_step_dy: int = 2
    max_step_dy: int = 3
    platform_width: int = 1
    max_mines: int = 4
    direction: str = "random"
    # "ascend" (legacy: spawn on the floor, hop UP to the door on the top
    # platform) or "descend" (spawn ON the top platform, hop DOWN to the door
    # on the base platform -- every hop is a controlled drop onto a narrow
    # target). See MapPlatformStairs.PROFILE.
    profile: str = "ascend"


@dataclass
class LongJumpConfig:
    """Configuration for the LONG_JUMP generator (build speed -> clear a wide gap).

    A flat runway, a single floorless gap too wide for a standing jump, and a flat
    landing carrying the switch + exit. The ninja spawns at the runway start, so a
    full running start is always available; ``min/max_gap_width`` is the difficulty
    driver and ``min/max_runway`` keeps the run-up long enough to reach speed.
    """

    min_height: int = 5
    max_height: int = 8
    min_gap_width: int = 4
    max_gap_width: int = 8
    min_runway: int = 12
    max_runway: int = 16
    min_landing: int = 4
    max_landing: int = 6
    max_mines: int = 0
    # Mines on the FIRST landing columns past the gap edge (forces a deep,
    # no-undershoot landing). Needs landing >= 4 for 1 mine, >= 5 for 2 (the
    # generator drops excess edge mines rather than overlap the switch/door).
    edge_mines: int = 0


@dataclass
class DropLandingConfig:
    """Configuration for the DROP_LANDING generator (controlled vertical descent).

    The ninja spawns on a high platform and must FALL ``depth`` tiles onto a
    ``ledge``-wide landing shelf offset ``offset`` tiles laterally, steering
    mid-fall. The shelf is flush against the chamber's far wall so wall-slide
    braking is always physically available. MEASURED physics boundary (see
    ``tests/test_precision_route_generators.py``): an uncontrolled walk-off
    free-fall survives up to 10 tiles and DIES from 11 -- presets at
    ``min_depth >= 12`` therefore force the braking skill; depth 11 itself is
    skipped (right at the boundary, both naive strategies die).
    """

    min_depth: int = 3
    max_depth: int = 5
    min_ledge: int = 4
    max_ledge: int = 5
    min_offset: int = 0
    max_offset: int = 0
    # Toggle mines hanging beside the ledge edge at surface height (a clipped
    # near-miss dies instead of dropping into the pit).
    max_mines: int = 0
    direction: str = "random"  # "left" | "right" | "random"


# Generator presets organized by difficulty
GENERATOR_PRESETS = {
    "maze": {
        # Micro maze: the easy on-ramp for the CHAMBER_MAZE size curriculum
        # (CHAMBER_MAZE was stuck ~7% at the old "tiny" floor). A 4-6 x 3-4 grid
        # is little more than a couple of branches to the exit.
        "micro": MazeConfig(
            min_width=4,
            max_width=6,
            min_height=3,
            max_height=4,
            max_cell_size=1,
        ),
        "tiny": MazeConfig(
            min_width=6,
            max_width=10,
            min_height=4,
            max_height=7,
            max_cell_size=1,
        ),
        "small": MazeConfig(
            min_width=14,
            max_width=30,
            min_height=8,
            max_height=16,
            max_cell_size=3,
        ),
        "large": MazeConfig(
            min_width=16,
            max_width=30,
            min_height=10,
            max_height=18,
            max_cell_size=4,
        ),
        "exploration": MazeConfig(
            min_width=15,
            max_width=40,
            min_height=10,
            max_height=20,
            max_cell_size=4,
        ),
    },
    "jump_required": {
        "simple": JumpRequiredConfig(
            min_width=12,
            max_width=20,
            min_height=8,
            max_height=10,
            min_pit_width=2,
            max_pit_width=3,
            max_mines_per_platform=2,
        ),
        "medium": JumpRequiredConfig(
            min_width=16,
            max_width=30,
            min_height=8,
            max_height=14,
            min_pit_width=3,
            max_pit_width=5,
            max_mines_per_platform=4,
        ),
        "complex": JumpRequiredConfig(
            min_width=25,
            max_width=42,
            min_height=12,
            max_height=18,
            min_pit_width=5,
            max_pit_width=7,
            max_mines_per_platform=6,
        ),
        "heavy_mine": JumpRequiredConfig(
            min_width=20,
            max_width=40,
            min_height=10,
            max_height=16,
            min_pit_width=4,
            max_pit_width=6,
            max_mines_per_platform=5,
        ),
    },
    "hills": {
        "simple": HillsConfig(
            min_width=10,
            max_width=20,
            min_height=8,
            max_height=12,
            min_hills=1,
            max_hills=4,
            min_height_change=1,
            max_height_change=3,
        ),
        "medium": HillsConfig(
            min_width=20,
            max_width=35,
            min_height=10,
            max_height=18,
            min_hills=4,
            max_hills=8,
            min_height_change=1,
            max_height_change=8,
        ),
        "complex": HillsConfig(
            min_width=30,
            max_width=42,
            min_height=16,
            max_height=22,
            min_hills=6,
            max_hills=12,
            min_height_change=2,
            max_height_change=8,
        ),
    },
    "vertical_corridor": {
        # Shortest, simplest shaft: a 3-4 tile drop/climb, no mines. Kept low so
        # the "simplest" category stays a genuine near-trivial on-ramp (a tall
        # 4-10 shaft is several wall-jumps, too hard for a from-scratch agent).
        "simplest": VerticalCorridorConfig(
            min_width=1,
            max_width=3,
            min_height=3,
            max_height=4,
            add_wall_mines=False,
            add_chaotic_walls=True,
        ),
        "simplest_few_mines": VerticalCorridorConfig(
            min_width=2,
            max_width=4,
            min_height=4,
            max_height=10,
            add_wall_mines=True,
            add_boundary_mines=False,
            add_chaotic_walls=True,
        ),
        "simplest_with_mines": VerticalCorridorConfig(
            min_width=2,
            max_width=4,
            min_height=4,
            max_height=10,
            add_wall_mines=True,
            add_boundary_mines=True,
            add_chaotic_walls=True,
        ),
        "simpler": VerticalCorridorConfig(
            min_width=1,
            max_width=5,
            min_height=4,
            max_height=12,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_chaotic_walls=True,
            add_wall_mines=False,
            add_boundary_mines=False,
        ),
        "simpler_with_mines": VerticalCorridorConfig(
            min_width=1,
            max_width=4,
            min_height=5,
            max_height=8,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_chaotic_walls=True,
            add_wall_mines=True,
            add_boundary_mines=True,
        ),
        "simple": VerticalCorridorConfig(
            min_width=1,
            max_width=4,
            min_height=8,
            max_height=12,
            min_mine_spacing=3,
            max_mine_spacing=6,
            add_platforms=True,
            add_chaotic_walls=True,
            add_wall_mines=True,
            add_boundary_mines=True,
        ),
        "medium": VerticalCorridorConfig(
            min_width=2,
            max_width=6,
            min_height=14,
            max_height=22,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_boundary_mines=True,
        ),
        "complex": VerticalCorridorConfig(
            min_width=3,
            max_width=8,
            min_height=20,
            max_height=22,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_boundary_mines=True,
        ),
        "platforms": VerticalCorridorConfig(
            min_width=2,
            max_width=5,
            min_height=12,
            max_height=18,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_platforms=True,
            add_boundary_mines=True,
        ),
        "mine_gauntlet": VerticalCorridorConfig(
            min_width=2,
            max_width=4,
            min_height=10,
            max_height=16,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_mid_mines=True,
            add_boundary_mines=True,
        ),
    },
    "mine_maze": {
        "medium": MineMazeConfig(
            min_width=12,
            max_width=25,
            min_height=4,
            max_height=7,
            min_skip_columns=3,
            max_skip_columns=6,
            min_mines_per_column=1,
            max_mines_per_column=4,
        ),
        "complex": MineMazeConfig(
            min_width=20,
            max_width=40,
            min_height=6,
            max_height=12,
            min_skip_columns=3,
            max_skip_columns=4,
            min_mines_per_column=4,
            max_mines_per_column=12,
        ),
        "heavy": MineMazeConfig(
            min_width=15,
            max_width=35,
            min_height=5,
            max_height=10,
            min_skip_columns=3,
            max_skip_columns=4,
            min_mines_per_column=3,
            max_mines_per_column=10,
        ),
    },
    "islands": {
        "medium": IslandsConfig(
            min_width=36,
            max_width=38,
            min_height=12,
            max_height=19,
            min_island_spacing=1,
            max_island_spacing=1,
            min_island_size=1,
            max_island_size=4,
            border_distance=1,
        ),
        "complex": IslandsConfig(
            min_width=37,
            max_width=38,
            min_height=16,
            max_height=19,
            min_island_spacing=1,
            max_island_spacing=1,
            min_island_size=1,
            max_island_size=4,
            border_distance=1,
        ),
    },
    "jump_platforms": {
        "medium": JumpPlatformsConfig(
            min_width=32,
            max_width=38,
            min_height=20,
            max_height=23,
            min_platform_spacing=8,
            max_platform_spacing=12,
            max_y_offset=3,
            mine_spacing=1.5,
        ),
        "complex": JumpPlatformsConfig(
            min_width=38,
            max_width=42,
            min_height=20,
            max_height=22,
            min_platform_spacing=8,
            max_platform_spacing=10,
            max_y_offset=3,
            mine_spacing=1.0,
        ),
    },
    "multi_chamber": {
        "medium_2": MultiChamberConfig(
            min_chambers=2,
            max_chambers=2,
            min_chamber_width=5,
            max_chamber_width=10,
            min_chamber_height=4,
            max_chamber_height=7,
            min_corridor_length=1,
            max_corridor_length=5,
            min_corridor_width=1,
            max_corridor_width=3,
        ),
        "complex": MultiChamberConfig(
            min_chambers=2,
            max_chambers=4,
            min_chamber_width=5,
            max_chamber_width=10,
            min_chamber_height=5,
            max_chamber_height=8,
            min_corridor_length=1,
            max_corridor_length=5,
            min_corridor_width=1,
            max_corridor_width=3,
        ),
        "exploration": MultiChamberConfig(
            min_chambers=2,
            max_chambers=3,
            min_chamber_width=4,
            max_chamber_width=9,
            min_chamber_height=5,
            max_chamber_height=9,
            min_corridor_length=3,
            max_corridor_length=9,
            min_corridor_width=1,
            max_corridor_width=4,
        ),
    },
    "single_chamber": {
        "simple": SingleChamberConfig(
            min_width=6,
            max_width=15,
            min_height=4,
            max_height=8,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
        ),
        "simple_deviation": SingleChamberConfig(
            min_width=6,
            max_width=15,
            min_height=4,
            max_height=8,
            global_max_up_deviation=3,
            global_max_down_deviation=1,
        ),
        "obstacle": SingleChamberConfig(
            min_width=8,
            max_width=18,
            min_height=4,
            max_height=8,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
            add_center_obstacle=True,
        ),
        "mines": SingleChamberConfig(
            min_width=10,
            max_width=20,
            min_height=4,
            max_height=8,
            global_max_up_deviation=1,
            global_max_down_deviation=0,
            add_floor_mines=True,
        ),
        "gap": SingleChamberConfig(
            min_width=12,
            max_width=25,
            min_height=4,
            max_height=8,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
            add_platform_gap=True,
        ),
        "ceiling_hazard": SingleChamberConfig(
            min_width=8,
            max_width=18,
            min_height=6,
            max_height=10,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
            add_ceiling_obstacles=True,
        ),
    },
    "horizontal_corridor": {
        "simplest": HorizontalCorridorConfig(
            min_width=4,
            max_width=10,
            min_height=1,
            max_height=2,
            random_edge_tiles=True,
            add_mines=False,
        ),
        "simplest_few_mines": HorizontalCorridorConfig(
            min_width=4,
            max_width=10,
            min_height=1,
            max_height=2,
            random_edge_tiles=False,
            add_mines=True,
            max_mines=8,
        ),
        "simplest_with_mines": HorizontalCorridorConfig(
            min_width=4,
            max_width=10,
            min_height=1,
            max_height=2,
            random_edge_tiles=False,
            add_mines=True,
        ),
        "simpler": HorizontalCorridorConfig(
            min_width=4,
            max_width=15,
            min_height=1,
            max_height=4,
            random_edge_tiles=True,
            add_mines=False,
        ),
        "simple": HorizontalCorridorConfig(
            min_width=4,
            max_width=30,
            min_height=1,
            max_height=5,
            random_edge_tiles=True,
            add_mines=True,
        ),
    },
    "corridors": {
        "simplest": CorridorsConfig(
            min_corridors=2,
            max_corridors=2,
            min_horizontal_width=4,
            max_horizontal_width=10,
            max_horizontal_height=1,
            min_vertical_width=1,
            max_vertical_width=1,
            min_vertical_height=2,
            max_vertical_height=5,
            add_zig_zag_drops=True,
            add_chaotic_walls=False,
            add_mines=False,
        ),
        "simplest_few_mines": CorridorsConfig(
            min_corridors=2,
            max_corridors=2,
            min_horizontal_width=4,
            max_horizontal_width=10,
            max_horizontal_height=1,
            min_vertical_width=1,
            max_vertical_width=1,
            min_vertical_height=2,
            max_vertical_height=5,
            add_zig_zag_drops=True,
            add_chaotic_walls=False,
            add_mines=True,
            max_mines=3,
        ),
        "simplest_with_mines": CorridorsConfig(
            min_corridors=2,
            max_corridors=3,
            min_horizontal_width=4,
            max_horizontal_width=20,
            max_horizontal_height=1,
            min_vertical_width=1,
            max_vertical_width=1,
            min_vertical_height=2,
            max_vertical_height=4,
            add_zig_zag_drops=True,
            add_chaotic_walls=False,
            add_mines=True,
            # Cap mines per corridor: uncapped (-1) packed floor+ceiling mines
            # ~1 tile apart, leaving no threadable path (levels were impossible;
            # guided-d* solved 0/N). 2/corridor stays a real mine challenge while
            # remaining solvable (verified). See ``corridors`` simple/medium/complex.
            max_mines=2,
        ),
        "simple": CorridorsConfig(
            min_corridors=2,
            max_corridors=4,
            min_horizontal_width=8,
            max_horizontal_width=15,
            max_horizontal_height=2,
            min_vertical_width=1,
            max_vertical_width=3,
            min_vertical_height=6,
            max_vertical_height=10,
            add_zig_zag_drops=True,
            add_chaotic_walls=True,
            add_mines=True,
            max_mines=2,  # see simplest_with_mines: cap dense floor+ceiling mines
        ),
        "medium": CorridorsConfig(
            min_corridors=3,
            max_corridors=5,
            min_horizontal_width=10,
            max_horizontal_width=20,
            max_horizontal_height=3,
            min_vertical_width=2,
            max_vertical_width=4,
            min_vertical_height=8,
            max_vertical_height=15,
            add_zig_zag_drops=True,
            add_chaotic_walls=True,
            add_mines=True,
            max_mines=2,  # see simplest_with_mines: cap dense floor+ceiling mines
        ),
        "complex": CorridorsConfig(
            min_corridors=4,
            max_corridors=5,
            min_horizontal_width=15,
            max_horizontal_width=25,
            max_horizontal_height=3,
            min_vertical_width=3,
            max_vertical_width=6,
            min_vertical_height=10,
            max_vertical_height=15,
            add_zig_zag_drops=True,
            add_chaotic_walls=True,
            add_mines=True,
            max_mines=2,  # see simplest_with_mines: cap dense floor+ceiling mines
        ),
    },
    # ---- Minimal per-mechanic presets (driven by BUCKET_CATALOG) --------
    "basic_jump": {
        # Tier 1: tiny chamber, smallest possible gap. Single-jump baseline.
        "single_gap_flat_t1": BasicJumpConfig(
            min_width=10,
            max_width=14,
            min_height=5,
            max_height=6,
            min_gap_width=2,
            max_gap_width=2,
            variant="single_gap_flat",
        ),
        # Tier 2: a one-tile step up and a short drop variant introduce
        # height change without additional mechanics.
        "single_step_up_t2": BasicJumpConfig(
            min_width=12,
            max_width=16,
            min_height=6,
            max_height=8,
            min_step_height=1,
            max_step_height=1,
            variant="single_step_up",
        ),
        "single_drop_t2": BasicJumpConfig(
            min_width=12,
            max_width=16,
            min_height=6,
            max_height=8,
            min_drop_height=2,
            max_drop_height=2,
            variant="single_drop",
        ),
        # Tier 3: wider chamber, bigger gap, two-tile step-ups.
        "single_gap_flat_t3": BasicJumpConfig(
            min_width=14,
            max_width=18,
            min_height=6,
            max_height=8,
            min_gap_width=3,
            max_gap_width=3,
            variant="single_gap_flat",
        ),
        "single_step_up_t3": BasicJumpConfig(
            min_width=14,
            max_width=18,
            min_height=6,
            max_height=8,
            min_step_height=2,
            max_step_height=2,
            variant="single_step_up",
        ),
        # Tier 4: double-gap forces two precisely-spaced jumps in a row.
        "double_gap_t4": BasicJumpConfig(
            min_width=18,
            max_width=24,
            min_height=7,
            max_height=9,
            min_gap_width=2,
            max_gap_width=3,
            variant="double_gap",
        ),
        # Mixed secondary: single gap + one toggle mine on the far platform.
        "single_gap_with_mine_t3": BasicJumpConfig(
            min_width=14,
            max_width=18,
            min_height=6,
            max_height=8,
            min_gap_width=2,
            max_gap_width=3,
            variant="single_gap_flat",
            max_mines=1,
        ),
        # Mixed secondary: double gap + one mine on the mid platform.
        "double_gap_mines_t4": BasicJumpConfig(
            min_width=18,
            max_width=24,
            min_height=7,
            max_height=9,
            min_gap_width=2,
            max_gap_width=3,
            variant="double_gap",
            max_mines=1,
        ),
    },
    "wall_jump": {
        # Tier 1: SINGLE wall-jump. Short 2-wide shaft just beyond a plain jump's
        # reach (~4 tiles), so one wall-jump completes it. The easy on-ramp so
        # the size curriculum can start WALL_JUMP trivial (was h8-10 ≈ 2 jumps,
        # which left the bucket stuck at 0% for an untrained agent).
        # Re-ramped 2026-06-22: the old T1(h2-3) -> T2(h7-9) jump was the steepest
        # step in the whole curriculum and left WALL_JUMP T2 stuck at 0% (the agent
        # solves a single hop but can't yet chain ~2-3 wall-jumps). Inserted an
        # INTERMEDIATE rung (h4-6) as the new T2 and shifted the taller shafts up,
        # so the climb grows gradually: 3 -> 6 -> 10 -> 18 tiles.
        # Tier 1: the tiniest wall-jump — a single short hop up a 2-3 tile gap.
        "shaft_short_t1": WallJumpConfig(
            min_width=2,
            max_width=2,
            min_height=2,
            max_height=3,
            add_middle_ledge=False,
            goal_position="top",
        ),
        # Tier 2: INTERMEDIATE — one full wall-jump up a 4-6 tile shaft (the new
        # bridging rung between the single hop and the multi-jump climb).
        "shaft_medium_t2": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=4,
            max_height=6,
            add_middle_ledge=False,
            goal_position="top",
        ),
        # Tier 3: DOUBLE/TRIPLE — ~2-3 chained wall-jumps (was the old T2/T3).
        "shaft_tall_t3": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=7,
            max_height=10,
            add_middle_ledge=False,
            goal_position="top",
        ),
        # Tier 4: TALL shaft with randomized goal + a ledge break to vary rhythm.
        "shaft_tall_ledge_t4": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=11,
            max_height=18,
            add_middle_ledge=True,
            goal_position="random",
        ),
    },
    "mine_path": {
        # Tier 1: short corridor, single mine.
        "flat_one_mine_t1": MinePathConfig(
            min_width=10,
            max_width=14,
            min_height=3,
            max_height=4,
            min_mine_count=1,
            max_mine_count=1,
            min_mine_spacing=2,
        ),
        # Tier 2: short corridor, 1-2 mines.
        "flat_few_mines_t2": MinePathConfig(
            min_width=12,
            max_width=16,
            min_height=3,
            max_height=4,
            min_mine_count=1,
            max_mine_count=2,
            min_mine_spacing=2,
        ),
        # Tier 3: medium corridor, 2-3 mines.
        "flat_mines_t3": MinePathConfig(
            min_width=16,
            max_width=20,
            min_height=3,
            max_height=5,
            min_mine_count=2,
            max_mine_count=3,
            min_mine_spacing=2,
        ),
        # Tier 4: long corridor, up to 4 mines, tighter spacing.
        "flat_mines_t4": MinePathConfig(
            min_width=20,
            max_width=24,
            min_height=3,
            max_height=5,
            min_mine_count=3,
            max_mine_count=4,
            min_mine_spacing=2,
        ),
        # Mixed secondary: same mine counts + one-tile step-up.
        "mines_with_step_t3": MinePathConfig(
            min_width=16,
            max_width=20,
            min_height=4,
            max_height=6,
            min_mine_count=2,
            max_mine_count=3,
            min_mine_spacing=2,
            add_step=True,
        ),
        "mines_with_step_t4": MinePathConfig(
            min_width=20,
            max_width=24,
            min_height=4,
            max_height=6,
            min_mine_count=3,
            max_mine_count=4,
            min_mine_spacing=2,
            add_step=True,
        ),
    },
    "slopes": {
        "single_slope_t1": SlopesConfig(
            min_width=12,
            max_width=18,
            min_height=6,
            max_height=10,
            min_slope_height=2,
            max_slope_height=3,
            slope_type_pool=("mild",),
            # Easiest tier: one small DUAL-SIDED mild hill (up and back down).
            variant="hill_symmetric",
            max_mines=0,
        ),
        "hill_symmetric_t2": SlopesConfig(
            min_width=16,
            max_width=24,
            min_height=8,
            max_height=12,
            min_slope_height=2,
            max_slope_height=4,
            slope_type_pool=("mild", "steep"),
            variant="hill_symmetric",
            max_mines=0,
        ),
        "hill_asymmetric_t3": SlopesConfig(
            min_width=16,
            max_width=28,
            min_height=8,
            max_height=14,
            min_slope_height=2,
            max_slope_height=4,
            slope_type_pool=("mild", "steep", "45"),
            variant="hill_asymmetric",
            max_mines=1,
        ),
        "hill_multi_t4": SlopesConfig(
            min_width=20,
            max_width=32,
            min_height=10,
            max_height=16,
            min_slope_height=2,
            max_slope_height=4,
            slope_type_pool=("mild", "steep", "45"),
            # Hardest tier: TWO dual-sided hills to cross, with mines.
            variant="multi_hill",
            max_mines=2,
        ),
    },
    "curves": {
        "single_curve_t1": CurvesConfig(
            min_width=20, max_width=26, min_height=8, max_height=10,
            min_curves=1, max_curves=1, max_mines=0, variant="mixed",
        ),
        "two_curves_t2": CurvesConfig(
            min_width=24, max_width=32, min_height=9, max_height=12,
            min_curves=2, max_curves=2, max_mines=0, variant="mixed",
        ),
        "curves_t3": CurvesConfig(
            min_width=28, max_width=36, min_height=10, max_height=13,
            min_curves=3, max_curves=3, max_mines=1, variant="mixed",
        ),
        "undulation_t4": CurvesConfig(
            min_width=32, max_width=40, min_height=11, max_height=14,
            min_curves=4, max_curves=5, max_mines=2, variant="undulation",
        ),
    },
    "multi_jump": {
        # Tier 1: a SINGLE jump over one small gap (was two chained jumps).
        # Wide width/height range so dedup leaves plenty of unique levels.
        # Tier 1: a SINGLE 2-tile gap (an easy hop) after a long flat run-up, so
        # the agent reliably builds speed and clears it (the old 2-3 tile mix
        # plateaued at ~0.25 because the 3-tile running long-jumps were too hard
        # from scratch). The lethal pit still punishes a genuine miss.
        "two_gaps_t1": MultiJumpConfig(
            min_width=18, max_width=28, min_height=6, max_height=11,
            min_jumps=1, max_jumps=1, max_mines=0, variant="gaps",
            max_gap_w=2, runup=8,
        ),
        "staircase_t2": MultiJumpConfig(
            min_width=24, max_width=30, min_height=9, max_height=12,
            min_jumps=2, max_jumps=3, max_mines=0, variant="staircase",
        ),
        "mixed_jumps_t3": MultiJumpConfig(
            min_width=28, max_width=36, min_height=10, max_height=13,
            min_jumps=3, max_jumps=3, max_mines=1, variant="mixed",
        ),
        "many_gaps_t4": MultiJumpConfig(
            min_width=32, max_width=40, min_height=10, max_height=14,
            min_jumps=4, max_jumps=4, max_mines=2, variant="mixed",
        ),
    },
    "composite": {
        "short_chain_t1": CompositeConfig(
            min_width=26, max_width=32, min_height=10, max_height=12,
            min_segments=2, max_segments=2, max_mines=0, variant="terrain",
        ),
        "chain_t2": CompositeConfig(
            min_width=30, max_width=36, min_height=11, max_height=14,
            min_segments=3, max_segments=3, max_mines=0, variant="mixed",
        ),
        "chain_t3": CompositeConfig(
            min_width=34, max_width=40, min_height=12, max_height=15,
            min_segments=4, max_segments=4, max_mines=1, variant="mixed",
        ),
        "long_chain_t4": CompositeConfig(
            min_width=36, max_width=40, min_height=12, max_height=16,
            min_segments=5, max_segments=6, max_mines=2, variant="mixed",
        ),
    },
    "precision": {
        "two_ledges_t1": PrecisionConfig(
            min_width=20, max_width=26, min_height=8, max_height=10,
            min_jumps=2, max_jumps=2, max_mines=1, ledge_width=4,
        ),
        "ledges_t2": PrecisionConfig(
            min_width=24, max_width=30, min_height=8, max_height=11,
            min_jumps=2, max_jumps=3, max_mines=1, ledge_width=3,
        ),
        "ledges_t3": PrecisionConfig(
            min_width=28, max_width=34, min_height=9, max_height=12,
            min_jumps=3, max_jumps=3, max_mines=2, ledge_width=3,
        ),
        "narrow_ledges_t4": PrecisionConfig(
            min_width=30, max_width=38, min_height=9, max_height=12,
            min_jumps=3, max_jumps=4, max_mines=2, ledge_width=3,
        ),
        # --- unclamped harder variants (audit P0-3c). The legacy presets above
        # are untouched (gap fixed at 2, ledge >= 3); these sweep the gap up to
        # 3 tiles (running long-jump between ledges) and the ledge down to 2.
        "wide_gap_ledges_t3": PrecisionConfig(
            min_width=28, max_width=36, min_height=9, max_height=12,
            min_jumps=2, max_jumps=3, max_mines=2, ledge_width=3, gap_width=3,
        ),
        "wide_gap_narrow_t4": PrecisionConfig(
            min_width=30, max_width=38, min_height=9, max_height=12,
            min_jumps=3, max_jumps=4, max_mines=2, ledge_width=2, gap_width=3,
        ),
    },
}


# -------------------------------------------------------------------------
# Minimal FLAT_RUN bucket presets on existing generators.
#
# The FLAT_RUN bucket reuses existing generators (horizontal_corridor and
# single_chamber) because they are already clean enough when the right
# knobs are cleared. We add explicit per-tier presets here to keep the
# bucket catalog declarative.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["horizontal_corridor"].update(
    {
        "flat_run_t1": HorizontalCorridorConfig(
            min_width=4,
            max_width=8,
            min_height=1,
            max_height=2,
            random_edge_tiles=False,
            add_mines=False,
            allow_locked_doors=False,
        ),
        "flat_run_t2": HorizontalCorridorConfig(
            min_width=6,
            max_width=12,
            min_height=1,
            max_height=3,
            random_edge_tiles=False,
            add_mines=False,
            allow_locked_doors=False,
        ),
        "flat_run_t3": HorizontalCorridorConfig(
            min_width=10,
            max_width=18,
            min_height=1,
            max_height=3,
            random_edge_tiles=False,
            add_mines=False,
            allow_locked_doors=False,
        ),
        "flat_run_t4": HorizontalCorridorConfig(
            min_width=14,
            max_width=23,
            min_height=1,
            max_height=4,
            random_edge_tiles=False,
            add_mines=False,
            allow_locked_doors=False,
        ),
    }
)
GENERATOR_PRESETS["single_chamber"].update(
    {
        # Trivial curriculum on-ramp: a tiny enclosed chamber with the switch
        # placed adjacent to the exit door and the ninja just beyond it, so the
        # required route is the shortest possible switch->door traversal. This is
        # the size-curriculum FLAT_RUN floor (size~0); ``switch_door_max_sep``
        # anneals to 0 as size grows (see _DIMENSION_FIELDS in the size source).
        "flat_run_onramp": SingleChamberConfig(
            min_width=4,
            max_width=6,
            min_height=3,
            max_height=4,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
            add_center_obstacle=False,
            add_floor_mines=False,
            add_platform_gap=False,
            add_ceiling_obstacles=False,
            switch_door_max_sep=1,
        ),
        "flat_run_t1": SingleChamberConfig(
            min_width=6,
            max_width=10,
            min_height=4,
            max_height=6,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
            add_center_obstacle=False,
            add_floor_mines=False,
            add_platform_gap=False,
            add_ceiling_obstacles=False,
        ),
        "flat_run_t2": SingleChamberConfig(
            min_width=8,
            max_width=14,
            min_height=4,
            max_height=6,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
        ),
        "flat_run_t3": SingleChamberConfig(
            min_width=12,
            max_width=18,
            min_height=4,
            max_height=6,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
        ),
        "flat_run_t4": SingleChamberConfig(
            min_width=16,
            max_width=24,
            min_height=4,
            max_height=7,
            global_max_up_deviation=0,
            global_max_down_deviation=0,
        ),
    }
)

# -------------------------------------------------------------------------
# VERTICAL_SHAFT bucket presets (anchor generator: vertical_corridor).
#
# vertical_corridor lacks a clean trivial preset (every legacy preset turns on
# chaotic walls and/or mines). These four presets give the bucket a genuine
# size-curriculum on-ramp: a tiny clean shaft at t1 that grows in height/width
# monotonically, with wall mines only introduced at t3+. min/max_height and
# min/max_width are interpolated by npp-rl's size curriculum; mine spacing is
# carried from the floor preset (the t1 floor has no mines, so density never
# ramps non-monotonically).
# -------------------------------------------------------------------------
GENERATOR_PRESETS["vertical_corridor"].update(
    {
        # Tiniest non-degenerate on-ramp: a 1-2 tile single jump up (was 4-6;
        # height<2 collapses spawn/exit to the same row = no jump). The agent's
        # first vertical movement is one short jump, not a multi-walljump climb.
        # Heights ramp smoothly from here.
        "shaft_t1": VerticalCorridorConfig(
            min_width=1,
            max_width=2,
            min_height=2,
            max_height=3,
            add_platforms=False,
            add_mid_mines=False,
            add_chaotic_walls=False,
            add_wall_mines=False,
            add_boundary_mines=False,
        ),
        "shaft_t2": VerticalCorridorConfig(
            min_width=2,
            max_width=3,
            min_height=4,
            max_height=8,
            add_platforms=False,
            add_mid_mines=False,
            add_chaotic_walls=False,
            add_wall_mines=False,
            add_boundary_mines=False,
        ),
        "shaft_t3": VerticalCorridorConfig(
            min_width=2,
            max_width=4,
            min_height=9,
            max_height=15,
            min_mine_spacing=3,
            max_mine_spacing=5,
            add_platforms=False,
            add_mid_mines=False,
            add_chaotic_walls=False,
            add_wall_mines=True,
            add_boundary_mines=False,
        ),
        "shaft_t4": VerticalCorridorConfig(
            min_width=2,
            max_width=5,
            min_height=14,
            max_height=22,
            min_mine_spacing=2,
            max_mine_spacing=5,
            add_platforms=False,
            add_mid_mines=False,
            add_chaotic_walls=False,
            add_wall_mines=True,
            add_boundary_mines=True,
        ),
    }
)

# -------------------------------------------------------------------------
# PLATFORMING bucket presets (anchor generator: jump_platforms).
#
# jump_platforms is intrinsically large (legacy presets start at 30x20 over a
# fully-mined floor). These presets author a small on-ramp: a shrunk chamber
# with widely-spaced (sparse) floor mines and short platform hops at t1, growing
# in chamber size + platform spacing while mine spacing tightens with the tier.
# min/max_width, min/max_height, and min/max_platform_spacing are interpolated
# by the size curriculum; mine_spacing is carried from the floor preset.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["jump_platforms"].update(
    {
        # Tier 1: two short, FLAT hops between WIDE (3-tile) platforms over a
        # dense lethal floor — a forgiving landing target with a small gap, so a
        # miss dies cleanly instead of wandering. Easiest real platforming hop.
        "platforms_t1": JumpPlatformsConfig(
            min_width=22,
            max_width=28,
            min_height=8,
            max_height=10,
            min_platform_spacing=6,
            max_platform_spacing=6,  # uniform 1-tile gap on BOTH hops
            max_y_offset=0,  # FLAT: the y-offset made the 2nd hop (switch->exit)
            platform_width=5,  # the wall (switch_reached 0.6 but exit_given 0.27);
            mine_spacing=0.5,  # flat + wide + 1-tile gap makes both hops trivial
        ),
        "platforms_t2": JumpPlatformsConfig(
            min_width=24,
            max_width=30,
            min_height=14,
            max_height=18,
            min_platform_spacing=6,
            max_platform_spacing=8,
            max_y_offset=2,
            platform_width=3,
            mine_spacing=0.5,
        ),
        "platforms_t3": JumpPlatformsConfig(
            min_width=30,
            max_width=38,
            min_height=18,
            max_height=22,
            min_platform_spacing=7,
            max_platform_spacing=9,
            max_y_offset=3,
            platform_width=2,
            mine_spacing=0.5,
        ),
        "platforms_t4": JumpPlatformsConfig(
            min_width=36,
            max_width=42,
            min_height=20,
            max_height=24,
            min_platform_spacing=8,
            max_platform_spacing=10,
            max_y_offset=3,
            platform_width=1,
            mine_spacing=0.5,
        ),
    }
)

# -------------------------------------------------------------------------
# OPEN_FIELD presets (generator: open_field). Open-arena flat run, mine-free.
# Folded into FLAT_RUN as variety; the run length grows with the tier so the
# size curriculum still sees a monotonic difficulty ramp.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["open_field"] = {
    "open_t1": OpenFieldConfig(min_width=12, max_width=18, min_separation=2, max_separation=5),
    "open_t2": OpenFieldConfig(min_width=18, max_width=26, min_separation=3, max_separation=7),
    "open_t3": OpenFieldConfig(min_width=26, max_width=34, min_separation=4, max_separation=9),
    "open_t4": OpenFieldConfig(min_width=34, max_width=40, min_separation=5, max_separation=11),
}

# -------------------------------------------------------------------------
# ROUND_CORNER presets (generator: round_corner). Curve-assisted wall ascent,
# mine-free. Folded into CURVES as variety. Difficulty = required ascent height
# (door_height) + corner radius, growing with the tier.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["round_corner"] = {
    "corner_t1": RoundCornerConfig(
        min_width=14, max_width=20, min_door_height=3, max_door_height=4,
        min_corner_radius=1, max_corner_radius=1,
    ),
    "corner_t2": RoundCornerConfig(
        min_width=18, max_width=26, min_door_height=4, max_door_height=6,
        min_corner_radius=1, max_corner_radius=2,
    ),
    "corner_t3": RoundCornerConfig(
        min_width=22, max_width=30, min_door_height=6, max_door_height=8,
        min_corner_radius=2, max_corner_radius=3,
    ),
    "corner_t4": RoundCornerConfig(
        min_width=26, max_width=34, min_door_height=8, max_door_height=11,
        min_corner_radius=2, max_corner_radius=3,
    ),
}

# -------------------------------------------------------------------------
# LOOP_CORRIDOR presets (anchor generator: loop_corridor). A horseshoe ring
# corridor that grows in outer extent and mine count with the tier; t1 is a
# small, mine-free on-ramp so the size curriculum has a near-trivial floor.
#
# corridor_width is fixed at 2 (the wall-jumpable shaft width) at every tier --
# cw=3 made the vertical-arm climb physically unsolvable (guided-d* solved 0/10
# at t2-t4). Difficulty therefore scales via WIDTH (longer top/bottom traversal)
# and mine count, while HEIGHT is bounded so the wall-jump climb (~height-4
# tiles) stays within what the ninja can actually chain (verified solvable by
# guided d*). Mines land only on the dodgeable horizontal arms (see
# MapLoopCorridor._corridor_ring_cells).
# -------------------------------------------------------------------------
GENERATOR_PRESETS["loop_corridor"] = {
    # loop_t0: the SMALLEST wall-jumpable loop -- the extremely-simple on-ramp.
    # Width/height sit at the generator floor (each must be >= 2*cw+3 == 7 for
    # cw=2), giving the shortest possible round-trip (climb one short side arm,
    # across a 1-cell top, down the other side to the switch, then back). This
    # teaches the "activate switch then return to exit" MECHANIC at minimal
    # scale before the size curriculum scales the loop up (t1..t4). Mine-free.
    "loop_t0": LoopCorridorConfig(
        min_width=8, max_width=9, min_height=7, max_height=8,
        corridor_width=2, max_mines=0,
    ),
    "loop_t1": LoopCorridorConfig(
        min_width=11, max_width=14, min_height=10, max_height=11,
        corridor_width=2, max_mines=0,
    ),
    # Tier 1.5: BRIDGE the T1->T2 cliff (v13 train diag: T1 0.898 -> T2 0.196 —
    # a ring-SIZE jump, the exact shape of WALL_JUMP's 0% shaft stall fixed by
    # an intermediate rung). Interpolates MIN/MAX_WIDTH/HEIGHT between loop_t1
    # and loop_t2 and introduces mines at the minimum count (1) so the
    # mined-walkway hop + return-leg re-cross is practised on a shorter ring
    # before the full T2 geometry. Blended into the T2 tier in BUCKET_CATALOG
    # (like stairs_t3_5 was for PLATFORM_STAIRS T4).
    "loop_t1_5": LoopCorridorConfig(
        min_width=12, max_width=17, min_height=10, max_height=11,
        corridor_width=2, max_mines=1,
    ),
    "loop_t2": LoopCorridorConfig(
        min_width=14, max_width=20, min_height=11, max_height=12,
        corridor_width=2, max_mines=2,
    ),
    "loop_t3": LoopCorridorConfig(
        min_width=20, max_width=28, min_height=12, max_height=13,
        corridor_width=2, max_mines=4,
    ),
    "loop_t4": LoopCorridorConfig(
        min_width=28, max_width=36, min_height=12, max_height=13,
        corridor_width=2, max_mines=5,
    ),
}

# -------------------------------------------------------------------------
# SIMPLE_CORRIDOR presets (generator: simple_corridor). The extremely-simple
# FAR_OBJECTIVE on-ramp: a 1-tile-wide ∩ (horizontal) or C (vertical) with
# exactly 5 empty tiles, spawn/exit at one end and switch at the other. No size
# knobs -- the shape is always minimal; only the orientation varies.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["simple_corridor"] = {
    "simple_corridor_h": SimpleCorridorConfig(orientation="horizontal"),
    "simple_corridor_v": SimpleCorridorConfig(orientation="vertical"),
}

# -------------------------------------------------------------------------
# PLATFORM_STAIRS presets (anchor generator: platform_stairs). An ascending
# staircase of floating platforms that grows in step count + spacing and mine
# density with the tier; t1 is a short, wide-platform, mine-free on-ramp.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["platform_stairs"] = {
    # PRIMARY difficulty axis = NUMBER OF PLATFORMS (step count):
    #   T1 = 1 platform, T2 = 2, T3 = 3-5, T4 = 6-7 (hardest, "5+").
    # The per-jump difficulty ALSO eases with the tier (platform_width 3->1,
    # mines 0->6, gaps/steepness grow) so the atomic landing skill is learnable
    # first and then chained — a tier is a coherent step up, but the ladder the
    # curriculum climbs is chain length. NB: horizontal extent (steps*step_dx)
    # and vertical drop (steps*step_dy) are kept within the map bounds (<=40
    # wide, <=22 tall). REGEN the benchmark dataset for these to take effect in
    # a packed-dataset run (the check_solvable gate reseeds per level).
    # Tier 1: a SINGLE platform — the atomic hop. Wide-ish + mine-free so the
    # landing is forgiving; this is where the atomic floor->platform landing is
    # first acquired. (Descent — controlled drops DOWN a staircase — is its own
    # profile: see the ``stairs_desc_t*`` presets below and DROP_LANDING.)
    "stairs_t1": PlatformStairsConfig(
        min_width=14, max_width=18, min_steps=1, max_steps=1,
        min_step_dx=1, max_step_dx=2, min_step_dy=1, max_step_dy=1,
        platform_width=3, max_mines=0,
    ),
    # Tier 2: TWO platforms — chain of two, slightly narrower, a few sparse mines.
    "stairs_t2": PlatformStairsConfig(
        min_width=20, max_width=26, min_steps=2, max_steps=2,
        min_step_dx=2, max_step_dx=4, min_step_dy=1, max_step_dy=2,
        platform_width=2, max_mines=3,
    ),
    # Tier 3: THREE-to-FIVE platforms — single-tile precision chain, moderate
    # mine density (subsampled -> some safe gaps, partial recovery).
    "stairs_t3": PlatformStairsConfig(
        min_width=28, max_width=36, min_steps=3, max_steps=5,
        min_step_dx=3, max_step_dx=5, min_step_dy=2, max_step_dy=3,
        platform_width=1, max_mines=10,
    ),
    # Tier 4: SIX-to-SEVEN platforms — the hardest long single-tile chain over a
    # FULLY-DENSE zero-recovery mine floor (max_mines >= #slots -> fill). dx/dy
    # tightened so 7 steps still fit the arena.
    "stairs_t4": PlatformStairsConfig(
        min_width=36, max_width=40, min_steps=6, max_steps=7,
        min_step_dx=3, max_step_dx=4, min_step_dy=2, max_step_dy=2,
        platform_width=1, max_mines=99,
    ),
    # Tier 3.5: BRIDGE the T3->T4 precision cliff (T4 stalled the generalist run at
    # ~0.33 == per-jump reliability^6). Keeps the single-tile platforms
    # (platform_width=1 -- the core precision skill is preserved, unlike t1/t2's
    # wide ledges) but caps chain length, gap width, steepness, and mine count
    # BELOW t4 so the per-jump reliability the agent must chain is achievable
    # (success sits between t3 ~0.65 and t4 ~0.33). Blended into the t4 bucket
    # below to give the policy a foothold instead of facing six frame-perfect jumps
    # over a fully-mined floor at once. See the PLATFORM_STAIRS T4 architecture
    # assessment.
    "stairs_t3_5": PlatformStairsConfig(
        min_width=30, max_width=38, min_steps=4, max_steps=5,
        min_step_dx=4, max_step_dx=5, min_step_dy=2, max_step_dy=3,
        platform_width=1, max_mines=5,
    ),
    # --- DESCENT profile (audit P0-3d): spawn ON the top platform and hop DOWN
    # the staircase to the door on the base platform — every hop is a drop onto
    # a (possibly single-tile) target over a mined floor, training controlled
    # falling. The legacy presets above are ascent-only and unchanged.
    "stairs_desc_t2": PlatformStairsConfig(
        min_width=20, max_width=26, min_steps=2, max_steps=2,
        min_step_dx=2, max_step_dx=4, min_step_dy=1, max_step_dy=2,
        platform_width=2, max_mines=3, profile="descend",
    ),
    "stairs_desc_t3": PlatformStairsConfig(
        min_width=28, max_width=36, min_steps=3, max_steps=5,
        min_step_dx=3, max_step_dx=5, min_step_dy=2, max_step_dy=3,
        platform_width=1, max_mines=10, profile="descend",
    ),
    "stairs_desc_t4": PlatformStairsConfig(
        min_width=36, max_width=40, min_steps=6, max_steps=7,
        min_step_dx=3, max_step_dx=4, min_step_dy=2, max_step_dy=2,
        platform_width=1, max_mines=99, profile="descend",
    ),
}

# -------------------------------------------------------------------------
# LONG_JUMP presets (generator: long_jump). Build speed on a guaranteed flat
# runway, then clear ONE wide floorless gap onto a landing platform (audit
# P0-3a — the generator existed but was wired into no preset/bucket/category).
# The gap sweeps 3 -> 6 tiles and the landing narrows 6 -> 2 across the tiers;
# ``edge_mines`` puts toggle mines on the first landing column(s) so the jump
# must land DEEP (the no-undershoot precision variant, folded into PRECISION).
#
# gap 7-8 is deliberately kept OUT of the tier presets: the reachability-graph
# solvability gate is optimistic near the jump envelope (see the loop_corridor
# corridor_width precedent above), and gaps that wide are unverified against
# real physics. They remain reachable via the ``max_gap_extreme`` preset for
# offline experimentation only.
# -------------------------------------------------------------------------
GENERATOR_PRESETS["long_jump"] = {
    # MULTI_JUMP folded variety: moderate gap, generous landing.
    "gap_t3": LongJumpConfig(
        min_gap_width=3, max_gap_width=4, min_runway=12, max_runway=16,
        min_landing=4, max_landing=6,
    ),
    # MULTI_JUMP T4: wide gap onto a short landing.
    "gap_t4": LongJumpConfig(
        min_gap_width=5, max_gap_width=6, min_runway=12, max_runway=16,
        min_landing=2, max_landing=3,
    ),
    # PRECISION folded variety: land past a mine on the landing edge.
    "edge_mine_t3": LongJumpConfig(
        min_gap_width=3, max_gap_width=4, min_runway=12, max_runway=16,
        min_landing=4, max_landing=6, edge_mines=1,
    ),
    # PRECISION T4: wider gap, TWO edge mines -> only the deep half of a 5-6
    # tile landing is safe.
    "edge_mine_t4": LongJumpConfig(
        min_gap_width=4, max_gap_width=6, min_runway=12, max_runway=16,
        min_landing=5, max_landing=6, edge_mines=2,
    ),
    # NOT wired into any bucket (see the header note): the physics-unverified
    # extreme of the sweep (gap 7-8, 1-2 tile landing).
    "max_gap_extreme": LongJumpConfig(
        min_gap_width=7, max_gap_width=8, min_runway=14, max_runway=16,
        min_landing=1, max_landing=2,
    ),
}

# -------------------------------------------------------------------------
# DROP_LANDING presets (anchor generator: drop_landing). A graded DROP: fall
# ``depth`` tiles onto a ``ledge``-wide shelf offset laterally by ``offset``
# tiles (steer mid-fall), optionally mined at the shelf edge (audit P0-3b).
#
# MEASURED braking boundary (real-sim, tests/test_precision_route_generators):
# uncontrolled walk-off free-fall survives <= 10 tiles and DIES from 11, while
# a wall-slide brake against the far wall (always flush with the ledge here)
# survives 12+. T3/T4 sit strictly ABOVE the boundary (>= 12) so the braking
# skill MUST be used; depth 11 itself is skipped (both naive strategies die
# right at the boundary -> an unlearnable cliff).
# -------------------------------------------------------------------------
GENERATOR_PRESETS["drop_landing"] = {
    # Tier 1: shallow drop, wide shelf directly below the platform edge.
    "drop_t1": DropLandingConfig(
        min_depth=3, max_depth=5, min_ledge=4, max_ledge=5,
        min_offset=0, max_offset=0, max_mines=0,
    ),
    # Tier 2: deeper (still inside the survivable free-fall band) + a small
    # lateral offset introduces mid-fall steering.
    "drop_t2": DropLandingConfig(
        min_depth=6, max_depth=10, min_ledge=3, max_ledge=4,
        min_offset=0, max_offset=2, max_mines=0,
    ),
    # Tier 3: past the braking boundary -- an unbraked fall is lethal.
    "drop_t3": DropLandingConfig(
        min_depth=12, max_depth=14, min_ledge=2, max_ledge=3,
        min_offset=1, max_offset=4, max_mines=1,
    ),
    # Tier 4: deep + narrow + offset + mined shelf edge.
    "drop_t4": DropLandingConfig(
        min_depth=15, max_depth=17, min_ledge=1, max_ledge=2,
        min_offset=3, max_offset=6, max_mines=2,
    ),
}


@dataclass
class CategoryConfig:
    """Configuration for a difficulty category."""

    name: str
    description: str
    ratio: float
    seed_base_test: int
    seed_base_train: int
    generators: List[Tuple[str, str]]


# =========================================================================
# CATEGORIES: difficulty-tiered taxonomy for general training.
#
# Each category maps to a list of ``(generator_type, preset)`` pairs that
# ``GeneratorRegistry`` samples from at runtime to pick a level. This is the
# legacy/general training taxonomy and is distinct from ``BUCKET_CATALOG``
# below (the mechanic-primary *benchmark* taxonomy) — both are live and serve
# different purposes; do not merge them.
#
# ``ratio`` sizes the *offline* static test-suite dataset
# (``generate_test_suite_maps.py``); the *runtime* sampling distribution lives
# separately in ``gym_environment/curriculum_config.py`` as ``default_weight``.
#
# Add new training categories here.
# =========================================================================
CATEGORIES = {
    # Easiest on-ramp: trivial, near-empty direct-route levels for a
    # from-scratch agent. Reduced parameters (tiny dimensions, short shafts)
    # and EXCLUDES every hard level type -- no mines, no mazes, no jumps, no
    # multi-chamber navigation. Only flat chambers / corridors, a 3-4 tile
    # shaft, and an open flat run. Sits below "simplest" in difficulty.
    "extremely_simple": CategoryConfig(
        name="extremely_simple",
        description="Trivial near-empty direct routes (no mines/jumps/mazes)",
        ratio=0.1,
        seed_base_test=50,
        seed_base_train=500,
        generators=[
            ("single_chamber", "flat_run_onramp"),
            ("single_chamber", "simple"),
            ("horizontal_corridor", "flat_run_t1"),
            ("vertical_corridor", "simplest"),
            ("open_field", "open_t1"),
        ],
    ),
    "simplest": CategoryConfig(
        name="simplest",
        description="Minimal direct paths (ninja -> switch -> door -> exit)",
        ratio=0.1,
        seed_base_test=100,
        seed_base_train=1000,
        generators=[
            ("vertical_corridor", "simplest"),
            ("horizontal_corridor", "simplest"),
            ("corridors", "simplest"),
        ],
    ),
    "simplest_few_mines": CategoryConfig(
        name="simplest_few_mines",
        description="Minimal direct paths with few mines",
        ratio=0.1,
        seed_base_test=200,
        seed_base_train=2000,
        generators=[
            ("vertical_corridor", "simplest_few_mines"),
            ("horizontal_corridor", "simplest_few_mines"),
            ("corridors", "simplest_few_mines"),
        ],
    ),
    "simplest_with_mines": CategoryConfig(
        name="simplest_with_mines",
        description="Minimal direct paths with mines (boundary_mines for vertical corridor)",
        ratio=0.1,
        seed_base_test=200,
        seed_base_train=2000,
        generators=[
            ("vertical_corridor", "simplest_with_mines"),
            ("horizontal_corridor", "simplest_with_mines"),
            ("corridors", "simplest_with_mines"),
        ],
    ),
    "simpler": CategoryConfig(
        name="simpler",
        description="Simple paths with slight layout variations",
        ratio=0.1,
        seed_base_test=500,
        seed_base_train=5000,
        generators=[
            ("vertical_corridor", "simpler"),
            ("horizontal_corridor", "simple"),
            ("vertical_corridor", "simpler_with_mines"),
            ("corridors", "simple"),
        ],
    ),
    "simple": CategoryConfig(
        name="simple",
        description="Basic platforming with small obstacles",
        ratio=0.2,
        seed_base_test=1000,
        seed_base_train=10000,
        generators=[
            ("maze", "tiny"),
            ("hills", "simple"),
            ("vertical_corridor", "simple"),
            ("jump_required", "simple"),
            ("single_chamber", "obstacle"),
            ("single_chamber", "gap"),
            ("vertical_corridor", "platforms"),
            ("corridors", "simple"),
        ],
    ),
    "medium": CategoryConfig(
        name="medium",
        description="Multi-chamber and medium mazes",
        ratio=0.4,
        seed_base_test=2000,
        seed_base_train=20000,
        generators=[
            ("maze", "small"),
            ("multi_chamber", "medium_2"),
            ("jump_required", "medium"),
            ("mine_maze", "medium"),
            ("hills", "medium"),
            ("vertical_corridor", "medium"),
            ("islands", "medium"),
            ("jump_platforms", "medium"),
            ("single_chamber", "mines"),
            ("single_chamber", "ceiling_hazard"),
            ("vertical_corridor", "mine_gauntlet"),
            ("corridors", "medium"),
        ],
    ),
    "complex": CategoryConfig(
        name="complex",
        description="Large mazes and advanced navigation",
        ratio=0.2,
        seed_base_test=4000,
        seed_base_train=40000,
        generators=[
            ("maze", "large"),
            ("multi_chamber", "complex"),
            ("jump_required", "complex"),
            ("mine_maze", "complex"),
            ("hills", "complex"),
            ("vertical_corridor", "complex"),
            ("islands", "complex"),
            ("jump_platforms", "complex"),
            ("corridors", "complex"),
        ],
    ),
    "mine_heavy": CategoryConfig(
        name="mine_heavy",
        description="High mine density requiring careful movement",
        ratio=0.1,
        seed_base_test=6000,
        seed_base_train=60000,
        generators=[
            ("mine_maze", "heavy"),
            ("jump_required", "heavy_mine"),
        ],
    ),
    "exploration": CategoryConfig(
        name="exploration",
        description="Large areas requiring extensive exploration",
        ratio=0.1,
        seed_base_test=7000,
        seed_base_train=70000,
        generators=[
            ("maze", "exploration"),
            ("multi_chamber", "exploration"),
        ],
    ),
}


def get_category_names() -> List[str]:
    """Get all category names in order."""
    return list(CATEGORIES.keys())


def get_category_config(name: str) -> CategoryConfig:
    """Get configuration for a specific category."""
    if name not in CATEGORIES:
        raise ValueError(f"Unknown category: {name}. Available: {get_category_names()}")
    return CATEGORIES[name]


# =========================================================================
# BUCKET_CATALOG: mechanic-primary procedural benchmark taxonomy.
#
# Maps each mechanic bucket to a dict of tier (1..4) -> list of
# ``(generator_type, preset, weight)`` entries. Weights are relative
# integers used for round-robin/weighted sampling inside a tier.
#
# The catalog drives ``scripts/generate_benchmark_dataset.py`` and is
# additive alongside the legacy ``CATEGORIES`` taxonomy. Empty dicts
# are placeholders for buckets that depend on entities outside Layer 1
# scope (see ``docs/MECHANIC_SUPPORT_MATRIX.md``); they stay empty
# until their generators come online.
# =========================================================================

BucketEntry = Tuple[str, str, int]
BUCKET_CATALOG: "dict[str, dict[int, List[BucketEntry]]]" = {
    "FLAT_RUN": {
        1: [
            # Trivial switch-adjacent on-ramp first: the size curriculum uses
            # entries[0] as the size~0 floor, so this is the agent's very first
            # (near-trivial) level. Listed first + weighted so the easiest tier
            # of the eval dataset also contains genuinely trivial routes.
            ("single_chamber", "flat_run_onramp", 2),
            ("horizontal_corridor", "flat_run_t1", 1),
            ("single_chamber", "flat_run_t1", 1),
        ],
        2: [
            ("horizontal_corridor", "flat_run_t2", 1),
            ("single_chamber", "flat_run_t2", 1),
            # Folded variety: the same flat run in wide-open space (no ceiling).
            ("open_field", "open_t2", 1),
        ],
        3: [
            ("horizontal_corridor", "flat_run_t3", 1),
            ("single_chamber", "flat_run_t3", 1),
            ("open_field", "open_t3", 1),
        ],
        4: [
            ("horizontal_corridor", "flat_run_t4", 1),
            ("single_chamber", "flat_run_t4", 1),
            ("open_field", "open_t4", 1),
        ],
    },
    "BASIC_JUMP": {
        1: [("basic_jump", "single_gap_flat_t1", 1)],
        2: [
            ("basic_jump", "single_drop_t2", 1),
            ("basic_jump", "single_step_up_t2", 1),
        ],
        3: [
            ("basic_jump", "single_gap_flat_t3", 1),
            ("basic_jump", "single_step_up_t3", 1),
            ("basic_jump", "single_gap_with_mine_t3", 1),
        ],
        4: [
            ("basic_jump", "double_gap_t4", 1),
            ("basic_jump", "double_gap_mines_t4", 1),
        ],
    },
    "WALL_JUMP": {
        1: [("wall_jump", "shaft_short_t1", 1)],
        2: [("wall_jump", "shaft_medium_t2", 1)],
        3: [("wall_jump", "shaft_tall_t3", 1)],
        4: [("wall_jump", "shaft_tall_ledge_t4", 1)],
    },
    # Climb a vertical shaft: wider than the WALL_JUMP shaft, grows in height and
    # introduces wall/boundary mines at higher tiers. Reuses the wall-jump skill
    # plus mine avoidance, so it is scheduled right after WALL_JUMP.
    "VERTICAL_SHAFT": {
        1: [("vertical_corridor", "shaft_t1", 1)],
        2: [("vertical_corridor", "shaft_t2", 1)],
        3: [("vertical_corridor", "shaft_t3", 1)],
        4: [("vertical_corridor", "shaft_t4", 1)],
    },
    "MINE_NAVIGATION": {
        1: [("mine_path", "flat_one_mine_t1", 1)],
        2: [("mine_path", "flat_few_mines_t2", 1)],
        3: [
            ("mine_path", "flat_mines_t3", 1),
            ("mine_path", "mines_with_step_t3", 1),
            # Folded variety: a horizontal mine field (always >=1 mine, so t3+).
            ("mine_maze", "medium", 1),
        ],
        4: [
            ("mine_path", "flat_mines_t4", 1),
            ("mine_path", "mines_with_step_t4", 1),
            ("mine_maze", "complex", 1),
            ("mine_maze", "heavy", 1),
            # Composed variety: thread a corridor of floor + ceiling mines + gaps.
            ("template_mine_gauntlet_2x3", "default", 1),
        ],
    },
    "SLOPES": {
        1: [("slopes", "single_slope_t1", 1)],
        2: [
            ("slopes", "hill_symmetric_t2", 1),
            # Folded variety: procedural rolling-hill terrain.
            ("hills", "simple", 1),
        ],
        3: [
            ("slopes", "hill_asymmetric_t3", 1),
            ("hills", "medium", 1),
        ],
        4: [
            ("slopes", "hill_multi_t4", 1),
            ("hills", "complex", 1),
        ],
    },
    # Chamber mazes: navigate a maze of connected chambers. Tiers map the maze
    # generator's presets tiny -> small -> large -> exploration (monotonic in
    # width/height/cell-size), so the size knob grows the maze smoothly.
    "CHAMBER_MAZE": {
        1: [("maze", "micro", 1)],
        2: [
            ("maze", "tiny", 1),
            # Folded variety: light zig-zag corridor navigation (no mines).
            ("corridors", "simplest", 1),
        ],
        3: [
            ("maze", "small", 1),
            ("corridors", "simple", 1),
            # Folded variety: connected multi-chamber layouts.
            ("multi_chamber", "medium_2", 1),
        ],
        4: [
            ("maze", "large", 1),
            ("multi_chamber", "complex", 1),
            ("multi_chamber", "exploration", 1),
        ],
    },
    # Curved-tile traversal: ride over convex humps / through concave bowls.
    "CURVES": {
        1: [("curves", "single_curve_t1", 1)],
        2: [("curves", "two_curves_t2", 1)],
        3: [
            ("curves", "curves_t3", 1),
            # Folded variety: a concave rounded corner ridden up into a
            # wall-jump ascent (curve-assisted wall ascent).
            ("round_corner", "corner_t3", 1),
        ],
        4: [
            ("curves", "undulation_t4", 1),
            ("round_corner", "corner_t4", 1),
        ],
    },
    # Sequenced jumps: >=2 gaps/step-ups in a row.
    "MULTI_JUMP": {
        1: [("multi_jump", "two_gaps_t1", 1)],
        2: [("multi_jump", "staircase_t2", 1)],
        3: [
            ("multi_jump", "mixed_jumps_t3", 1),
            # Folded variety: a required jump over a mined pit (always mined, t3+).
            ("jump_required", "simple", 1),
            # Folded variety: ONE wide running long-jump off a guaranteed runway
            # (audit P0-3a: gaps beyond the composed 2-3 tile cap).
            ("long_jump", "gap_t3", 1),
        ],
        4: [
            ("multi_jump", "many_gaps_t4", 1),
            ("jump_required", "medium", 1),
            ("jump_required", "heavy_mine", 1),
            # Composed variety: a 2x3 course of wide running long-jumps + steps.
            ("template_long_jump_course_2x3", "default", 1),
            # 5-6 tile running long-jump onto a 2-3 tile landing.
            ("long_jump", "gap_t4", 1),
        ],
    },
    # Large multi-mechanic courses chaining gaps/slopes/curves/staircases.
    "COMPOSITE": {
        1: [("composite", "short_chain_t1", 1)],
        2: [("composite", "chain_t2", 1)],
        3: [
            ("composite", "chain_t3", 1),
            # Composed variety: the full movement vocabulary in one 2x3 course.
            ("template_movement_mix_2x3", "default", 1),
        ],
        4: [
            ("composite", "long_chain_t4", 1),
            ("template_movement_mix_2x3", "default", 1),
            # Full-field serpentine: multi-row snake traversal with wall-jump
            # shaft turns + a full return trip (audit P0-3d).
            ("template_serpentine_full_field", "default", 1),
        ],
    },
    # Platform-to-platform hops over a mined floor (jump_platforms), with island
    # archipelagos as higher-tier variety. Reuses multi-jump + mine avoidance, so
    # it is scheduled after MULTI_JUMP. jump_platforms is the size-interpolation
    # anchor (entry[0] at every tier); islands appears only as t3/t4 variety.
    "PLATFORMING": {
        1: [("jump_platforms", "platforms_t1", 1)],
        2: [("jump_platforms", "platforms_t2", 1)],
        3: [
            ("jump_platforms", "platforms_t3", 2),
            ("islands", "medium", 1),
        ],
        4: [
            ("jump_platforms", "platforms_t4", 2),
            ("islands", "complex", 1),
        ],
    },
    # Ascending platform staircase: monotonic upward hops over a mined floor.
    # A structured sibling of PLATFORMING (ordered staircase vs scattered
    # platforms); scheduled right after it. platform_stairs is the size-
    # interpolation anchor at every tier.
    # Clean platform-COUNT ladder: T1=1 platform, T2=2, T3=3-5, T4=6-7. The count
    # ladder itself now smooths the old T3->T4 precision cliff (chain length grows
    # by <=2 per tier instead of 1->6), so T4 is pure genuine-difficulty rather
    # than a blend. stairs_t3_5 is retained as an optional bridge but unused here.
    # Regenerate the static benchmark dataset for this to take effect in a packed run.
    "PLATFORM_STAIRS": {
        1: [("platform_stairs", "stairs_t1", 1)],
        2: [
            ("platform_stairs", "stairs_t2", 1),
            # DESCENT profile (audit P0-3d): the same staircase hopped DOWNWARD
            # (spawn on the top platform, door on the base) — controlled drops
            # onto narrow targets instead of upward jumps.
            ("platform_stairs", "stairs_desc_t2", 1),
        ],
        3: [
            ("platform_stairs", "stairs_t3", 1),
            ("platform_stairs", "stairs_desc_t3", 1),
        ],
        4: [
            ("platform_stairs", "stairs_t4", 1),
            ("platform_stairs", "stairs_desc_t4", 1),
        ],
    },
    # Loop corridor: traverse a horseshoe ring corridor (walk the top/bottom
    # arms, wall-jump the side arms) while avoiding mines. Compositional, so it
    # is scheduled late. loop_corridor is the size-interpolation anchor.
    "LOOP_CORRIDOR": {
        1: [("loop_corridor", "loop_t1", 1)],
        # T1.5 rung blended into the T2 cell: the v13 T1->T2 cliff
        # (0.898 -> 0.196) is a ring-size jump; mid-size rings with a single
        # mine give the curriculum a graded step (see loop_t1_5 preset note).
        2: [
            ("loop_corridor", "loop_t1_5", 1),
            ("loop_corridor", "loop_t2", 1),
        ],
        3: [("loop_corridor", "loop_t3", 1)],
        4: [("loop_corridor", "loop_t4", 1)],
    },
    # Narrow mined ledges: precise landing + mine avoidance.
    "PRECISION": {
        1: [("precision", "two_ledges_t1", 1)],
        2: [("precision", "ledges_t2", 1)],
        3: [
            ("precision", "ledges_t3", 1),
            # Unclamped variety (audit P0-3c): 3-tile gaps between the ledges.
            ("precision", "wide_gap_ledges_t3", 1),
            # Folded variety: a running long-jump landing DEEP past an
            # edge mine on the landing platform.
            ("long_jump", "edge_mine_t3", 1),
        ],
        4: [
            ("precision", "narrow_ledges_t4", 1),
            # 3-tile gaps onto 2-tile ledges (ledge clamp unlocked to 2).
            ("precision", "wide_gap_narrow_t4", 1),
            ("long_jump", "edge_mine_t4", 1),
        ],
    },
    # Controlled vertical descent (audit P0-3b): fall a graded depth onto a
    # narrow landing shelf, steering laterally mid-fall. T3+ drops exceed the
    # measured survivable free-fall boundary (10 tiles; death from 11), so the
    # braking skill (wall-slide against the shelf-flush far wall / jump-hold
    # launch) MUST be used. drop_landing is the size-interpolation anchor.
    "DROP_LANDING": {
        1: [("drop_landing", "drop_t1", 1)],
        2: [
            ("drop_landing", "drop_t2", 1),
            # Composed variety: full-height cells chaining a sheer drop or a
            # 2-wide wall-jump slot climb.
            ("template_vertical_course_1x3", "default", 1),
        ],
        3: [("drop_landing", "drop_t3", 1)],
        4: [("drop_landing", "drop_t4", 1)],
    },
    # Far objective: ISOLATE long-horizon (switch+exit far from spawn) from
    # execution difficulty. Measured spawn cost-to-completion extends the
    # movement buckets' ~270 ceiling up to ~950 (free-space units), so the agent
    # actually practises distance. Lower tiers use loop_corridor (a single long
    # horseshoe path -- pure distance, no dead-ends); upper tiers add maze for
    # the longest routes a 42x23 map allows. All validated subset-clean
    # (mines + 1 exit + 1 switch) and free-space-reachable. See
    # SCALE_INVARIANT_PBRS_READINESS: adding these REQUIRES re-calibrating the
    # PBRS reference K to >= the new max cost (~1000), else far levels clamp.
    # Re-ramped 2026-06-22: was T1=loop_t2 (a 14-20-wide loop WITH 2 mines) ->
    # maze at T3/T4, which left the bucket stuck at 0%. Shifted the ramp down to
    # start at the EASY mine-free loop (loop_t1) and pushed the maze to the top
    # tier only, so the long-horizon objective grows gradually: small clean loop
    # -> wider loops + mines -> a single small maze.
    # 2026-06-23: made loop_t0 the EXTREMELY-SIMPLE on-ramp at tier 1 -- the
    # smallest wall-jumpable loop (~8-9 wide x 7-8 tall), a SHORT round-trip so
    # the agent learns the activate-switch-then-return mechanic at minimal scale
    # before the loop grows. The difficulty bands hard-map tier-number ->
    # band ("simplest"==1, "simple"==2, "medium"==3, "complex"==4; see
    # _band_spec + BUCKET_TIERS==[1,2,3,4]), so the on-ramp stays a 4-tier ladder
    # rather than a 5th tier (which BUCKET_TIERS would silently drop).
    #
    # The on-ramp now starts BELOW the smallest 2-wide loop: tier 1 is the
    # 1-tile-wide ``simple_corridor`` (horizontal ∩ + vertical C, 5 empty
    # tiles), the absolute-minimal "around the bend to the switch and back to
    # the exit" round-trip. The 2-wide loop ladder shifts up one tier:
    # tier 2 = the two smallest loops (loop_t0 + loop_t1), tier 3 = loop_t2,
    # tier 4 = the widest loops + a single small maze for the longest routes.
    "FAR_OBJECTIVE": {
        1: [
            ("simple_corridor", "simple_corridor_h", 1),
            ("simple_corridor", "simple_corridor_v", 1),
        ],
        2: [
            ("loop_corridor", "loop_t0", 1),
            ("loop_corridor", "loop_t1", 1),
        ],
        3: [("loop_corridor", "loop_t2", 1)],
        4: [
            ("loop_corridor", "loop_t3", 1),
            ("loop_corridor", "loop_t4", 1),
            ("maze", "small", 1),
            # Full-field boustrophedon (audit P0-3d): a 3-row snake route to the
            # switch in the far top corner and all the way back to the exit at
            # spawn -- the longest structured round-trip a 42x23 map allows.
            ("template_serpentine_full_field", "default", 1),
        ],
    },
    # DENSE_COMPOSITE (v22 bridge tier): the HARDEST in-scope mechanic bucket,
    # built to close the procgen->OFFICIAL distribution gap on the three measured
    # in-scope axes -- COMPOSITION (multi-mechanic in one level), DENSITY (more
    # mines, mixed states) and SIZE (2x3/3x3 templates toward OFFICIAL's byte
    # range) -- using EXISTING generators only (gold is deferred; see the v22
    # plan). It folds in the previously-unused hard compose templates
    # (grand_mix/slope_course/curve_course/tunnel_curves) + the 3x3
    # serpentine_full_field + the composite long-chain, and layers mine_maze for
    # density. NOT in CURRICULUM_SCHEDULE (so PHASE_1_BUCKETS/size-curriculum are
    # unchanged); it is generated explicitly and referenced by the P10a bridge
    # phase in the generalist_v5_official_attract preset. Ramp: T1 mixed course ->
    # T2 grand/slope composition -> T3 grand/curve/tunnel + mine density -> T4
    # grand + full-field serpentine + heavy mines + long chain.
    "DENSE_COMPOSITE": {
        1: [
            ("composite", "chain_t3", 1),
            ("template_movement_mix_2x3", "default", 1),
        ],
        2: [
            ("composite", "long_chain_t4", 1),
            ("template_grand_mix_2x3", "default", 1),
            ("template_slope_course_2x3", "default", 1),
        ],
        3: [
            ("template_grand_mix_2x3", "default", 2),
            ("template_curve_course_2x3", "default", 1),
            ("template_tunnel_curves_2x3", "default", 1),
            ("mine_maze", "complex", 1),
        ],
        4: [
            ("template_grand_mix_2x3", "default", 2),
            ("template_serpentine_full_field", "default", 1),
            ("mine_maze", "heavy", 1),
            ("composite", "long_chain_t4", 1),
        ],
    },
    # Out-of-scope buckets (empty until Layer 2+ generators exist).
    "SWITCH_PUZZLE": {},
    "LAUNCH_PAD": {},
    "THWUMP": {},
    "DRONE": {},
    "DEATH_BALL": {},
}


# =========================================================================
# CURRICULUM_SCHEDULE: the single canonical trivial -> complex bucket
# introduction order. This is the "introduce mechanics over time" spine and is
# the ONE place the ordered bucket list is authored. Everything else derives
# from it:
#   * ``PHASE_1_BUCKETS`` below (the in-scope, non-empty subset).
#   * the default ``--buckets`` for ``scripts/generate_benchmark_dataset.py``.
#   * npp-rl's ``DEFAULT_SIZE_BUCKETS`` (imported from here so the staged
#     ``ordered_unlock`` order can never drift from this file).
#
# Ordering rationale: a mechanic is introduced only once its prerequisite skill
# is unlocked -- WALL_JUMP before VERTICAL_SHAFT (the shaft reuses wall-jumps),
# MULTI_JUMP before PLATFORMING (platforming = sequenced jumps + mined floor +
# precise landing), PLATFORMING before PLATFORM_STAIRS (a structured staircase
# of the same hops), and the compositional buckets (CHAMBER_MAZE, LOOP_CORRIDOR,
# COMPOSITE) last. LOOP_CORRIDOR sits after CHAMBER_MAZE since it composes
# walking + wall-jumping + mine avoidance into one sustained navigation loop.
# =========================================================================
CURRICULUM_SCHEDULE: List[str] = [
    "FLAT_RUN",
    "BASIC_JUMP",
    "MINE_NAVIGATION",
    "WALL_JUMP",
    "VERTICAL_SHAFT",
    "SLOPES",
    "CURVES",
    "MULTI_JUMP",
    "PLATFORMING",
    "PLATFORM_STAIRS",
    # Controlled vertical descent: needs the platform-landing skill from
    # PLATFORMING/PLATFORM_STAIRS, adds mid-fall steering + wall-slide braking.
    "DROP_LANDING",
    "CHAMBER_MAZE",
    "LOOP_CORRIDOR",
    "PRECISION",
    "COMPOSITE",
]


# Phase-1 (toggle-mines only) in-scope buckets, in canonical schedule order.
# Derived from CURRICULUM_SCHEDULE so it can never drift: every scheduled bucket
# is in-scope for Phase 1 (all use only Layer-1 toggle-mine entities). The
# out-of-scope buckets (SWITCH_PUZZLE / LAUNCH_PAD / THWUMP / DRONE / DEATH_BALL)
# are empty placeholders and are excluded by the non-empty filter.
PHASE_1_BUCKETS: List[str] = [
    b for b in CURRICULUM_SCHEDULE if BUCKET_CATALOG.get(b)
]

BUCKET_TIERS: List[int] = [1, 2, 3, 4]


def get_bucket_names(include_empty: bool = True) -> List[str]:
    """Return all bucket names declared in ``BUCKET_CATALOG``."""
    if include_empty:
        return list(BUCKET_CATALOG.keys())
    return [b for b, tiers in BUCKET_CATALOG.items() if tiers]


def get_bucket_entries(bucket: str, tier: int) -> List[BucketEntry]:
    """Return catalog entries for a given bucket/tier (empty list if unset)."""
    return BUCKET_CATALOG.get(bucket, {}).get(tier, [])


def validate_bucket_catalog() -> List[str]:
    """Check every non-empty bucket entry points to a valid preset.

    Returns a list of human-readable error strings; empty if the catalog
    is consistent with ``GENERATOR_PRESETS``.
    """
    errors: List[str] = []
    for bucket, tiers in BUCKET_CATALOG.items():
        for tier, entries in tiers.items():
            for gen_type, preset, weight in entries:
                if weight <= 0:
                    errors.append(
                        f"{bucket}/T{tier}: entry {gen_type}:{preset} has non-positive weight {weight}"
                    )
                if gen_type not in GENERATOR_PRESETS:
                    errors.append(
                        f"{bucket}/T{tier}: generator {gen_type!r} not in GENERATOR_PRESETS"
                    )
                    continue
                if preset not in GENERATOR_PRESETS[gen_type]:
                    errors.append(
                        f"{bucket}/T{tier}: preset {gen_type}:{preset!r} not defined"
                    )
    return errors


# =========================================================================
# bucket_difficulty_view: a thin DERIVED projection of ``BUCKET_CATALOG`` onto
# the legacy difficulty-band names. This is what makes BUCKET_CATALOG the single
# source of truth: nclone's runtime ``GeneratorRegistry`` sources its per-band
# generator lists from here (instead of the hand-authored ``CATEGORIES.generators``
# lists), so every generator wired into a mechanic bucket -- now all 23 -- is
# reachable from the difficulty-banded interactive/test env path, and the two
# taxonomies can no longer drift.
#
# The projection maps each legacy band to one or more bucket tiers, optionally
# restricted to a subset of buckets so the band keeps its semantic meaning
# (e.g. the "*_mines" bands draw only from mine-bearing buckets, "exploration"
# only from navigation buckets). It is the ONE small hand-authored bridge between
# the mechanic taxonomy and the difficulty taxonomy.
# =========================================================================

# Bucket families used to keep the legacy band semantics meaningful.
_MINE_BUCKETS = ("MINE_NAVIGATION", "PRECISION", "PLATFORMING", "PLATFORM_STAIRS")
_NAV_BUCKETS = ("CHAMBER_MAZE", "COMPOSITE", "LOOP_CORRIDOR")
_NO_MINE_BUCKETS = (
    "FLAT_RUN",
    "BASIC_JUMP",
    "WALL_JUMP",
    "VERTICAL_SHAFT",
    "SLOPES",
    "CURVES",
    "MULTI_JUMP",
    # Mine-free at the T1/T2 tiers these bands draw from (T3+ adds edge mines,
    # same pattern as MULTI_JUMP).
    "DROP_LANDING",
)

def _band_spec() -> "dict[str, Tuple[List[int], Tuple[str, ...]]]":
    """Difficulty-band -> (tiers, allowed buckets). Built lazily to reference
    the bucket family tuples above without import-order issues."""
    all_phase1 = tuple(PHASE_1_BUCKETS)
    return {
        "simplest": ([1], _NO_MINE_BUCKETS),
        "simplest_few_mines": ([1, 2], _MINE_BUCKETS),
        "simplest_with_mines": ([2], _MINE_BUCKETS),
        "simpler": ([2], _NO_MINE_BUCKETS),
        "simple": ([2], all_phase1),
        "medium": ([3], all_phase1),
        "complex": ([4], all_phase1),
        "mine_heavy": ([4], _MINE_BUCKETS),
        "exploration": ([4], _NAV_BUCKETS),
    }


def bucket_difficulty_view() -> "dict[str, List[Tuple[str, str]]]":
    """Return ``{band_name: [(generator_type, preset), ...]}`` derived from
    ``BUCKET_CATALOG``.

    Generators are aggregated (and de-duplicated, preserving first-seen order)
    across the band's mapped tiers and allowed buckets. Bands that project to no
    catalog entries fall back to the legacy ``CATEGORIES[name].generators`` list
    so the interactive env always has something to generate.
    """
    view: "dict[str, List[Tuple[str, str]]]" = {}
    for band, (tiers, allowed) in _band_spec().items():
        seen: set = set()
        pairs: List[Tuple[str, str]] = []
        for bucket in allowed:
            for tier in tiers:
                for gen_type, preset, _weight in get_bucket_entries(bucket, tier):
                    key = (gen_type, preset)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
        if not pairs and band in CATEGORIES:
            pairs = list(CATEGORIES[band].generators)
        view[band] = pairs
    return view
