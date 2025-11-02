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
        "minimal": VerticalCorridorConfig(
            min_width=1,
            max_width=2,
            min_height=3,
            max_height=6,
            add_wall_mines=False,
        ),
        "minimal_with_mines": VerticalCorridorConfig(
            min_width=1,
            max_width=2,
            min_height=3,
            max_height=6,
            add_wall_mines=False,
            add_boundary_mines=True,
        ),
        "simpler": VerticalCorridorConfig(
            min_width=1,
            max_width=4,
            min_height=4,
            max_height=7,
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
        "minimal": HorizontalCorridorConfig(
            min_width=3,
            max_width=18,
            min_height=1,
            max_height=2,
            random_edge_tiles=False,
            add_mines=False,
        ),
        "minimal_with_mines": HorizontalCorridorConfig(
            min_width=3,
            max_width=23,
            min_height=1,
            max_height=2,
            random_edge_tiles=False,
            add_mines=True,
        ),
        "simple": HorizontalCorridorConfig(
            min_width=3,
            max_width=30,
            min_height=1,
            max_height=5,
            random_edge_tiles=True,
            fixed_height=None,
            add_mines=True,
        ),
    },
    "corridors": {
        "simplest": CorridorsConfig(
            min_corridors=2,
            max_corridors=2,
            min_horizontal_width=3,
            max_horizontal_width=8,
            max_horizontal_height=1,
            min_vertical_width=1,
            max_vertical_width=1,
            min_vertical_height=2,
            max_vertical_height=5,
            add_zig_zag_drops=True,
            add_chaotic_walls=False,
            add_mines=False,
        ),
        "simplest_with_mines": CorridorsConfig(
            min_corridors=2,
            max_corridors=3,
            min_horizontal_width=3,
            max_horizontal_width=8,
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


# Category configurations - add new categories here
CATEGORIES = {
    "simplest": CategoryConfig(
        name="simplest",
        description="Minimal direct paths (ninja -> switch -> door -> exit)",
        ratio=0.1,
        seed_base_test=100,
        seed_base_train=1000,
        generators=[
            ("vertical_corridor", "minimal"),
            ("horizontal_corridor", "minimal"),
            ("corridors", "simplest"),
        ],
    ),
    "simplest_with_mines": CategoryConfig(
        name="simplest_with_mines",
        description="Minimal direct paths with mines (boundary_mines for vertical corridor)",
        ratio=0.1,
        seed_base_test=200,
        seed_base_train=2000,
        generators=[
            ("vertical_corridor", "minimal_with_mines"),
            ("horizontal_corridor", "minimal_with_mines"),
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
