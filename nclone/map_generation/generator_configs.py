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
    mine_spacing: float = 1.0


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


# Generator presets organized by difficulty
GENERATOR_PRESETS = {
    "maze": {
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
        "simplest": VerticalCorridorConfig(
            min_width=1,
            max_width=3,
            min_height=4,
            max_height=10,
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
        # Tier 1: short shaft, 2-tile wide, goal always at the top.
        "shaft_short_t1": WallJumpConfig(
            min_width=2,
            max_width=2,
            min_height=8,
            max_height=10,
            add_middle_ledge=False,
            goal_position="top",
        ),
        # Tier 2: medium shaft; ninja still ascends.
        "shaft_medium_t2": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=10,
            max_height=14,
            add_middle_ledge=False,
            goal_position="top",
        ),
        # Tier 3: tall shaft, goal direction randomized per level.
        "shaft_tall_t3": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=14,
            max_height=18,
            add_middle_ledge=False,
            goal_position="random",
        ),
        # Tier 4: tall shaft with one ledge break to vary rhythm.
        "shaft_tall_ledge_t4": WallJumpConfig(
            min_width=2,
            max_width=3,
            min_height=14,
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
            variant="single_slope_up",
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
            variant="valley",
            max_mines=2,
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
        ),
        "flat_run_t2": HorizontalCorridorConfig(
            min_width=6,
            max_width=12,
            min_height=1,
            max_height=3,
            random_edge_tiles=False,
            add_mines=False,
        ),
        "flat_run_t3": HorizontalCorridorConfig(
            min_width=10,
            max_width=18,
            min_height=1,
            max_height=3,
            random_edge_tiles=False,
            add_mines=False,
        ),
        "flat_run_t4": HorizontalCorridorConfig(
            min_width=14,
            max_width=23,
            min_height=1,
            max_height=4,
            random_edge_tiles=False,
            add_mines=False,
        ),
    }
)
GENERATOR_PRESETS["single_chamber"].update(
    {
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


@dataclass
class CategoryConfig:
    """Configuration for a difficulty category."""

    name: str
    description: str
    ratio: float
    seed_base_test: int
    seed_base_train: int
    generators: List[Tuple[str, str]]


# Category configurations - add new categories here
CATEGORIES = {
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
            ("horizontal_corridor", "flat_run_t1", 1),
            ("single_chamber", "flat_run_t1", 1),
        ],
        2: [
            ("horizontal_corridor", "flat_run_t2", 1),
            ("single_chamber", "flat_run_t2", 1),
        ],
        3: [
            ("horizontal_corridor", "flat_run_t3", 1),
            ("single_chamber", "flat_run_t3", 1),
        ],
        4: [
            ("horizontal_corridor", "flat_run_t4", 1),
            ("single_chamber", "flat_run_t4", 1),
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
    "MINE_NAVIGATION": {
        1: [("mine_path", "flat_one_mine_t1", 1)],
        2: [("mine_path", "flat_few_mines_t2", 1)],
        3: [
            ("mine_path", "flat_mines_t3", 1),
            ("mine_path", "mines_with_step_t3", 1),
        ],
        4: [
            ("mine_path", "flat_mines_t4", 1),
            ("mine_path", "mines_with_step_t4", 1),
        ],
    },
    "SLOPES": {
        1: [("slopes", "single_slope_t1", 1)],
        2: [("slopes", "hill_symmetric_t2", 1)],
        3: [("slopes", "hill_asymmetric_t3", 1)],
        4: [("slopes", "hill_multi_t4", 1)],
    },
    # Out-of-scope buckets (empty until Layer 2+ generators exist).
    "SWITCH_PUZZLE": {},
    "LAUNCH_PAD": {},
    "THWUMP": {},
    "DRONE": {},
    "DEATH_BALL": {},
}


# Phase-1 (toggle-mines only) in-scope buckets, in canonical order.
PHASE_1_BUCKETS: List[str] = [
    "FLAT_RUN",
    "BASIC_JUMP",
    "WALL_JUMP",
    "MINE_NAVIGATION",
    "SLOPES",
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
