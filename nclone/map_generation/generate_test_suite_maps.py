"""
Generate comprehensive train and test suite maps for NPP-RL (Task 3.3).

This script creates deterministic datasets of N++ levels across 5 complexity categories:
- 50 simple levels (single switch, direct path, includes tiny mazes)
- 100 medium levels (includes medium-sized mazes and jump-required levels)
- 50 complex levels (multi-chamber, large mazes)
- 30 mine-heavy levels (significant mine obstacles)
- 20 movement based exploration levels

The script can generate:
- Training set (250 levels) with seeds 10000-99999
- Test set (250 levels) with seeds 1000-9999
- Both sets (500 unique levels total)

All maps are generated deterministically using fixed seeds to ensure reproducibility.
Levels are guaranteed to be unique across both train and test sets.

Usage:
    # Generate both train and test sets
    python -m nclone.map_generation.generate_test_suite_maps --mode both

    # Generate only training set
    python -m nclone.map_generation.generate_test_suite_maps --mode train

    # Generate only test set
    python -m nclone.map_generation.generate_test_suite_maps --mode test
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any
from .map import Map
from .map_single_chamber import SingleChamberGenerator
from .map_jump_required import MapJumpRequired
from .map_maze import MazeGenerator
from .map_multi_chamber import MultiChamberGenerator
from .map_mine_maze import MapMineMaze
from .map_islands import MapIslands
from .map_hills import MapHills
from .map_vertical_corridor import MapVerticalCorridor
from .map_jump_platforms import MapJumpPlatforms
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class TestSuiteGenerator:
    """Generator for comprehensive NPP-RL train and test suites."""

    # Base seeds for TEST set (1000-9999 range)
    TEST_SIMPLE_BASE_SEED = 1000
    TEST_MEDIUM_BASE_SEED = 2000
    TEST_COMPLEX_BASE_SEED = 3000
    TEST_MINE_HEAVY_BASE_SEED = 4000
    TEST_EXPLORATION_BASE_SEED = 5000

    # Base seeds for TRAIN set (10000-99999 range)
    TRAIN_SIMPLE_BASE_SEED = 10000
    TRAIN_MEDIUM_BASE_SEED = 20000
    TRAIN_COMPLEX_BASE_SEED = 30000
    TRAIN_MINE_HEAVY_BASE_SEED = 40000
    TRAIN_EXPLORATION_BASE_SEED = 50000

    # Maximum attempts to generate a unique map before giving up
    MAX_REGENERATION_ATTEMPTS = 1000

    def __init__(self, base_output_dir: str = "./datasets"):
        """Initialize the test suite generator.

        Args:
            base_output_dir: Base directory where train/test datasets will be saved
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Track generated levels for both train and test
        self.train_levels: Dict[str, List[Dict[str, Any]]] = {
            "simple": [],
            "medium": [],
            "complex": [],
            "mine_heavy": [],
            "exploration": [],
        }
        self.test_levels: Dict[str, List[Dict[str, Any]]] = {
            "simple": [],
            "medium": [],
            "complex": [],
            "mine_heavy": [],
            "exploration": [],
        }

        # Track unique maps by their data hash to prevent duplicates across both sets
        self.seen_maps: set = set()

        # Counter for generating new seeds when duplicates are found
        self.seed_offset = 200000

        # Track statistics
        self.duplicate_count = 0

    def _get_map_hash(self, map_data: List[int]) -> tuple:
        """Convert map data to a hashable tuple for duplicate detection.

        Args:
            map_data: List of integers representing the complete map state

        Returns:
            Tuple of map data that can be used as a set key
        """
        return tuple(map_data)

    def _is_unique_map(self, map_gen: Map) -> bool:
        """Check if a generated map is unique (not a duplicate).

        Args:
            map_gen: Generated map object

        Returns:
            True if the map is unique, False if it's a duplicate
        """
        map_data = map_gen.map_data()
        map_hash = self._get_map_hash(map_data)
        return map_hash not in self.seen_maps

    def _register_map(self, map_gen: Map) -> None:
        """Register a map as seen to prevent future duplicates.

        Args:
            map_gen: Generated map object to register
        """
        map_data = map_gen.map_data()
        map_hash = self._get_map_hash(map_data)
        self.seen_maps.add(map_hash)

    def _generate_unique_map(
        self, generator_func, base_seed: int, level_index: int, category: str
    ) -> Map:
        """Generate a unique map, retrying with different seeds if duplicates are found.

        Args:
            generator_func: Function that takes a seed and returns a Map object
            base_seed: Base seed for this level
            level_index: Index of the level being generated
            category: Category name for logging

        Returns:
            A unique Map object

        Raises:
            RuntimeError: If unable to generate a unique map after MAX_REGENERATION_ATTEMPTS
        """
        for attempt in range(self.MAX_REGENERATION_ATTEMPTS):
            # Try the base seed first, then use offset seeds for retries
            if attempt == 0:
                seed = base_seed
            else:
                seed = self.seed_offset + (level_index * 1000) + attempt

            map_gen = generator_func(seed)

            if self._is_unique_map(map_gen):
                self._register_map(map_gen)
                if attempt > 0:
                    self.duplicate_count += 1
                    print(
                        f"  ⚠ {category} level {level_index}: duplicate found, "
                        f"regenerated with seed {seed} (attempt {attempt + 1})"
                    )
                return map_gen

        # Should rarely happen, but provide a fallback
        raise RuntimeError(
            f"Failed to generate unique map for {category} level {level_index} "
            f"after {self.MAX_REGENERATION_ATTEMPTS} attempts"
        )

    def generate_simple_levels(self, count: int = 50, mode: str = "test") -> None:
        """Generate simple levels: single switch, direct path to exit.

        These levels test basic navigation and switch activation.
        Maps are evenly distributed across all available map types.

        Args:
            count: Number of simple levels to generate
            mode: 'train' or 'test' to determine seed range and output location
        """
        print(f"Generating {count} simple levels ({mode} set)...")

        base_seed_start = (
            self.TRAIN_SIMPLE_BASE_SEED
            if mode == "train"
            else self.TEST_SIMPLE_BASE_SEED
        )
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        # Define map types with their generator functions and descriptions
        # For minimal_simple_level, we use a special marker and handle it separately
        map_types = [
            (
                "minimal_simple_level",  # Special marker
                "Minimal chamber: ninja -> exit switch -> door (may have locked doors if 1-tile high)",
            ),
            (
                self._create_tiny_maze,
                "Tiny maze for basic navigation practice",
            ),
            (
                self._create_simple_hills_level,
                "Simple hills level with rolling terrain",
            ),
            (
                self._create_simple_vertical_corridor,
                "Simple vertical corridor: climb from bottom to exit at top",
            ),
            (
                self._create_simple_jump_level,
                "Simple jump required to reach switch or exit",
            ),
            (
                self._create_simple_jump_platforms,
                "Jump between platforms to reach switch and exit",
            ),
        ]

        # Calculate even distribution
        num_types = len(map_types)
        base_per_type = count // num_types
        remainder = count % num_types

        # Build distribution list: first 'remainder' types get one extra map
        type_counts = [
            base_per_type + (1 if i < remainder else 0) for i in range(num_types)
        ]

        # Generate levels with dynamic distribution
        current_index = 0
        for type_idx, (generator_func, description) in enumerate(map_types):
            type_count = type_counts[type_idx]

            for j in range(type_count):
                i = current_index + j
                base_seed = base_seed_start + i

                # Handle the special case for minimal_simple_level
                if generator_func == "minimal_simple_level":
                    # Create a closure that properly captures the current value of i
                    def make_generator(index):
                        return lambda seed: self._create_minimal_simple_level(
                            seed, index
                        )

                    actual_generator = make_generator(i)
                else:
                    actual_generator = generator_func

                map_gen = self._generate_unique_map(
                    actual_generator, base_seed, i, f"simple-{mode}"
                )
                actual_seed = base_seed  # The seed used (may differ if regenerated)

                # Calculate difficulty tier dynamically (4 tiers)
                difficulty_tier = min(4, (i * 4 // count) + 1)

                level_data = {
                    "level_id": f"simple_{i:03d}",
                    "seed": actual_seed,
                    "category": "simple",
                    "map_data": map_gen.map_data(),
                    "metadata": {
                        "description": description,
                        "difficulty_tier": difficulty_tier,
                        "split": mode,
                    },
                }
                levels_dict["simple"].append(level_data)

            current_index += type_count

        print(f"✓ Generated {count} simple levels ({mode} set)")
        # Print distribution summary
        dist_summary = ", ".join(
            f"{type_counts[i]} {map_types[i][1].split(':')[0].split('(')[0].strip()}"
            for i in range(num_types)
        )
        print(f"  Distribution: {dist_summary}")

    def _create_minimal_simple_level(self, seed: int, index: int) -> Map:
        """Create a minimal simple level (1-3 tiles high, 3-12 tiles wide).

        For 1-tile high and 5+ tiles wide levels, adds a locked door between ninja and exit switch.
        """
        map_gen = Map(seed=seed)
        rng = map_gen.rng

        # Very small dimensions for simplest levels
        max_width = 3 + (index % 20)  # 3-20 tiles wide
        max_height = 1 + (index % 5)  # 1-5 tiles high
        width = rng.randint(3, max_width)
        height = rng.randint(1, max_height)

        # Random offset for the chamber, ensuring walls fit within map bounds
        # Need space for: wall (-1), chamber (width/height), wall (+width/+height)
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1
        start_x = rng.randint(1, max_start_x)
        start_y = rng.randint(1, max_start_y)

        # Fill everything with walls first (using type 1 - full solid)
        should_fill_walls = rng.choice([True, False])
        if should_fill_walls:
            for y in range(MAP_TILE_HEIGHT):
                for x in range(MAP_TILE_WIDTH):
                    map_gen.set_tile(x, y, 1)

        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                map_gen.set_tile(x, y, 0)

        # Add decorative random walls on the chamber edges
        use_random_tiles_type = rng.choice([True, False])
        map_gen.set_hollow_rectangle(
            start_x - 1,
            start_y - 1,
            start_x + width,
            start_y + height,
            use_random_tiles_type=use_random_tiles_type,
        )

        # Randomly choose ninja starting side
        ninja_on_left = rng.choice([True, False])

        if ninja_on_left:
            ninja_x = start_x
            ninja_orientation = 1  # Facing right
        else:
            ninja_x = start_x + width - 1
            ninja_orientation = -1  # Facing left

        ninja_y = start_y + height - 1

        # Check if we should add a locked door (1 tile high, 4+ tiles wide)
        can_add_locked_door = height == 1 and width >= 4
        add_locked_door = can_add_locked_door and rng.choice([True, False])

        # Generate random positions with quarter-tile increments (0.25) for granular positioning
        num_positions = (width - 1) * 4  # Quadruple the positions with 0.25 increments
        available_positions = [start_x + i * 0.25 for i in range(num_positions)]

        # Remove positions too close to the edge of the playspace
        available_positions = [
            pos
            for pos in available_positions
            if pos > start_x + 0.25 and pos < start_x + width - 0.25
        ]

        # Remove positions within 1 tile of ninja spawn
        available_positions = [
            pos for pos in available_positions if abs(pos - ninja_x) >= 1
        ]

        # Sample positions based on layout complexity
        if add_locked_door:
            # Layout: Ninja -> Locked Door Switch -> Locked Door -> Exit Switch -> Exit Door
            positions = sorted(rng.sample(available_positions, k=4))

            # Reverse order if ninja is on the right so entities are still between ninja and exit
            if not ninja_on_left:
                positions = positions[::-1]

            locked_switch_x, locked_door_x, exit_switch_x, exit_door_x = positions

            entity_y = start_y
            map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)

            # Add locked door with its switch
            map_gen.add_entity(
                6, locked_door_x, entity_y, 4, 0, locked_switch_x, entity_y
            )

            # Add exit door with its switch
            map_gen.add_entity(3, exit_door_x, entity_y, 0, 0, exit_switch_x, entity_y)
        else:
            # Simple layout: Ninja -> Exit Switch -> Exit Door
            positions = sorted(rng.sample(available_positions, k=2))

            # Reverse order if ninja is on the right so entities are still between ninja and exit
            if not ninja_on_left:
                positions = positions[::-1]

            exit_switch_x, exit_door_x = positions

            # if height is more than 1, lets vary the height of the exit switch and exit door indepently by
            # a certain random height in 0.25
            exit_switch_y = start_y
            exit_door_y = start_y
            if height > 1:
                exit_switch_y = start_y + rng.randint(1, height - 1) * 0.25
                exit_door_y = start_y + rng.randint(1, height - 1) * 0.25

            entity_y = start_y + height - 1

            map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            map_gen.add_entity(
                3, exit_door_x, exit_door_y, 0, 0, exit_switch_x, exit_switch_y
            )

        return map_gen

    def _create_single_chamber_level(
        self, seed: int, with_deviation: bool = False
    ) -> Map:
        """Create a single chamber level with optional vertical deviation."""
        map_gen = SingleChamberGenerator(seed=seed)

        # Override some constants for controlled difficulty
        if not with_deviation:
            map_gen.GLOBAL_MAX_UP_DEVIATION = 0
            map_gen.GLOBAL_MAX_DOWN_DEVIATION = 0
        else:
            map_gen.GLOBAL_MAX_UP_DEVIATION = 3
            map_gen.GLOBAL_MAX_DOWN_DEVIATION = 1

        # Generate smaller chambers for simple levels
        map_gen.MIN_WIDTH = 6
        map_gen.MAX_WIDTH = 15
        map_gen.MIN_HEIGHT = 4
        map_gen.MAX_HEIGHT = 8

        map_gen.generate(seed=seed)
        return map_gen

    def _create_simple_jump_level(self, seed: int) -> Map:
        """Create a simple level that requires a jump."""
        map_gen = MapJumpRequired(seed=seed)

        # Make it simple: small pit, easy to navigate
        # Need enough height for pit + switch placement (minimum 8)
        map_gen.MIN_WIDTH = 12
        map_gen.MAX_WIDTH = 20
        map_gen.MIN_HEIGHT = 8  # Increased to ensure switch placement works
        map_gen.MAX_HEIGHT = 10
        map_gen.MIN_PIT_WIDTH = 2
        map_gen.MAX_PIT_WIDTH = 3
        map_gen.MAX_MINES_PER_PLATFORM = 2  # Very few mines

        map_gen.generate(seed=seed)
        return map_gen

    def _create_tiny_maze(self, seed: int) -> Map:
        """Create a tiny maze for simple levels."""
        map_gen = MazeGenerator(seed=seed)

        # Very small maze dimensions for simple difficulty
        map_gen.MIN_WIDTH = 6
        map_gen.MAX_WIDTH = 10
        map_gen.MIN_HEIGHT = 4
        map_gen.MAX_HEIGHT = 7
        map_gen.MAX_CELL_SIZE = 1

        map_gen.generate(seed=seed)
        return map_gen

    def _create_simple_hills_level(self, seed: int) -> Map:
        """Create a simple hills level with rolling terrain."""
        map_gen = MapHills(seed=seed)

        # Small dimensions for simple difficulty
        map_gen.MIN_WIDTH = 10
        map_gen.MAX_WIDTH = 20
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 12
        map_gen.MIN_HILLS = 1
        map_gen.MAX_HILLS = 4

        map_gen.generate(seed=seed)
        return map_gen

    def _create_simple_vertical_corridor(self, seed: int) -> Map:
        """Create a simple vertical corridor level."""
        map_gen = MapVerticalCorridor(seed=seed)

        # Small dimensions for simple difficulty
        map_gen.MIN_WIDTH = 1
        map_gen.MAX_WIDTH = 4
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 14

        map_gen.generate(seed=seed)
        return map_gen

    def _create_simple_jump_platforms(self, seed: int) -> Map:
        """Create a simple jump platforms level."""
        map_gen = MapJumpPlatforms(seed=seed)

        # Smaller dimensions for simple difficulty
        map_gen.MIN_WIDTH = 30
        map_gen.MAX_WIDTH = 32
        map_gen.MIN_HEIGHT = 20
        map_gen.MAX_HEIGHT = 21

        # Shorter platform spacing for easier jumps
        map_gen.MIN_PLATFORM_SPACING = 8
        map_gen.MAX_PLATFORM_SPACING = 10
        map_gen.MAX_Y_OFFSET = 2

        map_gen.generate(seed=seed)
        return map_gen

    def generate_medium_levels(self, count: int = 100, mode: str = "test") -> None:
        """Generate medium levels: 1-3 switches, simple dependencies.

        These levels test navigation with multiple objectives and basic planning.
        Maps are evenly distributed across all available map types.

        Args:
            count: Number of medium levels to generate
            mode: 'train' or 'test' to determine seed range and output location
        """
        print(f"Generating {count} medium levels ({mode} set)...")

        base_seed_start = (
            self.TRAIN_MEDIUM_BASE_SEED
            if mode == "train"
            else self.TEST_MEDIUM_BASE_SEED
        )
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        # Define map types with their generator functions and descriptions
        map_types = [
            (
                self._create_small_maze,
                "Medium-sized maze with navigation challenges",
            ),
            (
                lambda seed: self._create_medium_multi_chamber(seed, num_chambers=2),
                "2-chamber level with switch dependencies",
            ),
            (
                self._create_medium_jump_level,
                "Jump required with moderate mine obstacles",
            ),
            (
                self._create_medium_chamber_with_obstacles,
                "Medium chamber with obstacles",
            ),
            (
                self._create_medium_hills_level,
                "Medium hills level with rolling terrain and slopes",
            ),
            (
                self._create_medium_vertical_corridor,
                "Medium vertical corridor: climb from bottom to exit at top with wall mines",
            ),
            (
                self._create_islands_map,
                "Island-style level with 4x4 tile groups spread across empty space",
            ),
            (
                self._create_medium_jump_platforms,
                "Jump between platforms with mines on the floor",
            ),
        ]

        # Calculate even distribution
        num_types = len(map_types)
        base_per_type = count // num_types
        remainder = count % num_types

        # Build distribution list: first 'remainder' types get one extra map
        type_counts = [
            base_per_type + (1 if i < remainder else 0) for i in range(num_types)
        ]

        # Generate levels with dynamic distribution
        current_index = 0
        for type_idx, (generator_func, description) in enumerate(map_types):
            type_count = type_counts[type_idx]

            for j in range(type_count):
                i = current_index + j
                base_seed = base_seed_start + i

                map_gen = self._generate_unique_map(
                    generator_func, base_seed, i, f"medium-{mode}"
                )

                # Calculate difficulty tier dynamically (4 tiers)
                difficulty_tier = min(4, (i * 4 // count) + 1)

                level_data = {
                    "level_id": f"medium_{i:03d}",
                    "seed": base_seed,
                    "category": "medium",
                    "map_data": map_gen.map_data(),
                    "metadata": {
                        "description": description,
                        "difficulty_tier": difficulty_tier,
                        "split": mode,
                    },
                }
                levels_dict["medium"].append(level_data)

            current_index += type_count

        print(f"✓ Generated {count} medium levels ({mode} set)")
        # Print distribution summary
        dist_summary = ", ".join(
            f"{type_counts[i]} {map_types[i][1].split(':')[0].split('with')[0].strip()}"
            for i in range(num_types)
        )
        print(f"  Distribution: {dist_summary}")

    def _create_small_maze(self, seed: int) -> Map:
        """Create a medium-sized maze level."""
        map_gen = MazeGenerator(seed=seed)

        # Medium maze dimensions (larger than tiny mazes in simple levels)
        map_gen.MIN_WIDTH = 14
        map_gen.MAX_WIDTH = 30
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 16
        map_gen.MAX_CELL_SIZE = 3

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_multi_chamber(self, seed: int, num_chambers: int = 2) -> Map:
        """Create a multi-chamber level with specified number of chambers."""
        map_gen = MultiChamberGenerator(seed=seed)

        # Control chamber count
        map_gen.MIN_CHAMBERS = num_chambers
        map_gen.MAX_CHAMBERS = num_chambers

        # Medium-sized chambers
        map_gen.MIN_CHAMBER_WIDTH = 5
        map_gen.MAX_CHAMBER_WIDTH = 10
        map_gen.MIN_CHAMBER_HEIGHT = 4
        map_gen.MAX_CHAMBER_HEIGHT = 7

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_jump_level(self, seed: int) -> Map:
        """Create a medium difficulty jump level with mines."""
        map_gen = MapJumpRequired(seed=seed)

        # Medium difficulty
        map_gen.MIN_WIDTH = 16
        map_gen.MAX_WIDTH = 30
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 14
        map_gen.MIN_PIT_WIDTH = 3
        map_gen.MAX_PIT_WIDTH = 5
        map_gen.MAX_MINES_PER_PLATFORM = 4

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_chamber_with_obstacles(self, seed: int) -> Map:
        """Create a chamber with some mine obstacles."""
        map_gen = MapMineMaze(seed=seed)

        # Medium chamber with moderate mines
        map_gen.MIN_WIDTH = 12
        map_gen.MAX_WIDTH = 25
        map_gen.MIN_HEIGHT = 4
        map_gen.MAX_HEIGHT = 7

        # Fewer mines than mine-heavy levels
        map_gen.MIN_SKIP_COLUMNS = 3
        map_gen.MAX_SKIP_COLUMNS = 6
        map_gen.MIN_MINES_PER_COLUMN = 1
        map_gen.MAX_MINES_PER_COLUMN = 4

        map_gen.generate(seed=seed)
        return map_gen

    def _create_islands_map(self, seed: int) -> Map:
        """Create an islands map with 4x4 tile groups spread across empty space."""
        map_gen = MapIslands(seed=seed)

        # Use default constraints from MapIslands class
        # MIN_WIDTH = 36, MAX_WIDTH = MAP_TILE_WIDTH - 4
        # MIN_HEIGHT = 12, MAX_HEIGHT = MAP_TILE_HEIGHT - 4

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_hills_level(self, seed: int) -> Map:
        """Create a medium hills level with rolling terrain."""
        map_gen = MapHills(seed=seed)

        # Medium dimensions for moderate difficulty
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 35
        map_gen.MIN_HEIGHT = 10
        map_gen.MAX_HEIGHT = 18
        map_gen.MIN_HILLS = 4
        map_gen.MAX_HILLS = 8

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_vertical_corridor(self, seed: int) -> Map:
        """Create a medium vertical corridor level."""
        map_gen = MapVerticalCorridor(seed=seed)

        # Medium dimensions for moderate difficulty
        map_gen.MIN_WIDTH = 2
        map_gen.MAX_WIDTH = 6
        map_gen.MIN_HEIGHT = 14
        map_gen.MAX_HEIGHT = 22

        map_gen.generate(seed=seed)
        return map_gen

    def _create_medium_jump_platforms(self, seed: int) -> Map:
        """Create a medium difficulty jump platforms level."""
        map_gen = MapJumpPlatforms(seed=seed)

        # Medium dimensions for moderate difficulty
        map_gen.MIN_WIDTH = 32
        map_gen.MAX_WIDTH = 38
        map_gen.MIN_HEIGHT = 20
        map_gen.MAX_HEIGHT = 23

        # Full platform spacing range
        map_gen.MIN_PLATFORM_SPACING = 8
        map_gen.MAX_PLATFORM_SPACING = 12
        map_gen.MAX_Y_OFFSET = 3

        # Standard mine spacing
        map_gen.MINE_SPACING = 1.5

        map_gen.generate(seed=seed)
        return map_gen

    def generate_complex_levels(self, count: int = 50, mode: str = "test") -> None:
        """Generate complex levels: 4+ switches, complex dependencies.

        These levels test advanced planning and multi-step problem solving.
        Maps are evenly distributed across all available map types.

        Args:
            count: Number of complex levels to generate
            mode: 'train' or 'test' to determine seed range and output location
        """
        print(f"Generating {count} complex levels ({mode} set)...")

        base_seed_start = (
            self.TRAIN_COMPLEX_BASE_SEED
            if mode == "train"
            else self.TEST_COMPLEX_BASE_SEED
        )
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        # Define map types with their generator functions and descriptions
        map_types = [
            (
                self._create_large_maze,
                "Large maze requiring extensive navigation",
            ),
            (
                self._create_complex_multi_chamber,
                "3-4 chambers with complex switch dependencies",
            ),
            (
                self._create_complex_jump_level,
                "Complex jump sequence with heavy mine obstacles",
            ),
            (
                self._create_complex_mine_maze,
                "Complex maze with significant mine obstacles",
            ),
            (
                self._create_complex_hills_level,
                "Complex hills level with challenging terrain",
            ),
            (
                self._create_complex_vertical_corridor,
                "Complex vertical corridor with heavy mine obstacles",
            ),
            (
                self._create_complex_islands_map,
                "Complex island-style level with distant objectives",
            ),
            (
                self._create_complex_jump_platforms,
                "Complex jump platforms with challenging spacing and mines",
            ),
        ]

        # Calculate even distribution
        num_types = len(map_types)
        base_per_type = count // num_types
        remainder = count % num_types

        # Build distribution list: first 'remainder' types get one extra map
        type_counts = [
            base_per_type + (1 if i < remainder else 0) for i in range(num_types)
        ]

        # Generate levels with dynamic distribution
        current_index = 0
        for type_idx, (generator_func, description) in enumerate(map_types):
            type_count = type_counts[type_idx]

            for j in range(type_count):
                i = current_index + j
                base_seed = base_seed_start + i

                map_gen = self._generate_unique_map(
                    generator_func, base_seed, i, f"complex-{mode}"
                )

                # Calculate difficulty tier dynamically (3 tiers for complex)
                difficulty_tier = min(3, (i * 3 // count) + 1)

                level_data = {
                    "level_id": f"complex_{i:03d}",
                    "seed": base_seed,
                    "category": "complex",
                    "map_data": map_gen.map_data(),
                    "metadata": {
                        "description": description,
                        "difficulty_tier": difficulty_tier,
                        "split": mode,
                    },
                }
                levels_dict["complex"].append(level_data)

            current_index += type_count

        print(f"✓ Generated {count} complex levels ({mode} set)")
        # Print distribution summary
        dist_summary = ", ".join(
            f"{type_counts[i]} {map_types[i][1].split(':')[0].split('with')[0].strip()}"
            for i in range(num_types)
        )
        print(f"  Distribution: {dist_summary}")

    def _create_complex_multi_chamber(self, seed: int) -> Map:
        """Create a complex multi-chamber level."""
        map_gen = MultiChamberGenerator(seed=seed)

        # More chambers
        map_gen.MIN_CHAMBERS = 3
        map_gen.MAX_CHAMBERS = 4

        # Larger chambers
        map_gen.MIN_CHAMBER_WIDTH = 6
        map_gen.MAX_CHAMBER_WIDTH = 12
        map_gen.MIN_CHAMBER_HEIGHT = 5
        map_gen.MAX_CHAMBER_HEIGHT = 8

        map_gen.generate(seed=seed)
        return map_gen

    def _create_large_maze(self, seed: int) -> Map:
        """Create a large maze level."""
        map_gen = MazeGenerator(seed=seed)

        # Large maze dimensions
        map_gen.MIN_WIDTH = 16
        map_gen.MAX_WIDTH = 30
        map_gen.MIN_HEIGHT = 10
        map_gen.MAX_HEIGHT = 18
        map_gen.MAX_CELL_SIZE = 4

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_jump_level(self, seed: int) -> Map:
        """Create a complex jump level with heavy mines."""
        map_gen = MapJumpRequired(seed=seed)

        # Large and difficult
        map_gen.MIN_WIDTH = 25
        map_gen.MAX_WIDTH = 42
        map_gen.MIN_HEIGHT = 12
        map_gen.MAX_HEIGHT = 18
        map_gen.MIN_PIT_WIDTH = 5
        map_gen.MAX_PIT_WIDTH = 7
        map_gen.MAX_MINES_PER_PLATFORM = 6

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_mine_maze(self, seed: int) -> Map:
        """Create a complex mine maze with high mine density."""
        map_gen = MapMineMaze(seed=seed)

        # Large chamber with high mine density
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 40
        map_gen.MIN_HEIGHT = 6
        map_gen.MAX_HEIGHT = 12

        # High mine density
        map_gen.MIN_SKIP_COLUMNS = 3
        map_gen.MAX_SKIP_COLUMNS = 4
        map_gen.MIN_MINES_PER_COLUMN = 4
        map_gen.MAX_MINES_PER_COLUMN = 12

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_hills_level(self, seed: int) -> Map:
        """Create a complex hills level with challenging terrain."""
        map_gen = MapHills(seed=seed)

        # Large dimensions with many hills
        map_gen.MIN_WIDTH = 30
        map_gen.MAX_WIDTH = 42
        map_gen.MIN_HEIGHT = 16
        map_gen.MAX_HEIGHT = 22  # MAP_TILE_HEIGHT - 1 to stay within bounds
        map_gen.MIN_HILLS = 6
        map_gen.MAX_HILLS = 12

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_vertical_corridor(self, seed: int) -> Map:
        """Create a complex vertical corridor level."""
        map_gen = MapVerticalCorridor(seed=seed)

        # Large dimensions with more complexity
        map_gen.MIN_WIDTH = 3
        map_gen.MAX_WIDTH = 8
        map_gen.MIN_HEIGHT = 20
        map_gen.MAX_HEIGHT = 22  # MAP_TILE_HEIGHT - 1 to stay within bounds

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_islands_map(self, seed: int) -> Map:
        """Create a complex islands map with more spread out tile groups."""
        map_gen = MapIslands(seed=seed)

        # Use larger dimensions for complex difficulty
        # MapIslands defaults: MIN_WIDTH=36, MAX_WIDTH=38, MIN_HEIGHT=12, MAX_HEIGHT=19
        # For complex, we use the upper range
        map_gen.MIN_WIDTH = 37  # Near max for more challenging navigation
        map_gen.MAX_WIDTH = MAP_TILE_WIDTH - 4  # 38
        map_gen.MIN_HEIGHT = 16
        map_gen.MAX_HEIGHT = MAP_TILE_HEIGHT - 4  # 19

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_jump_platforms(self, seed: int) -> Map:
        """Create a complex jump platforms level."""
        map_gen = MapJumpPlatforms(seed=seed)

        # Large dimensions with challenging spacing
        map_gen.MIN_WIDTH = 38
        map_gen.MAX_WIDTH = 42
        map_gen.MIN_HEIGHT = 20
        map_gen.MAX_HEIGHT = 22  # MAP_TILE_HEIGHT - 1 to stay within bounds

        # Wider platform spacing for more challenging jumps
        map_gen.MIN_PLATFORM_SPACING = 10
        map_gen.MAX_PLATFORM_SPACING = 14
        map_gen.MAX_Y_OFFSET = 4

        # Tighter mine spacing for more obstacles
        map_gen.MINE_SPACING = 1.0

        map_gen.generate(seed=seed)
        return map_gen

    def generate_mine_heavy_levels(self, count: int = 30, mode: str = "test") -> None:
        """Generate mine-heavy levels: significant mine obstacles.

        These levels test hazard avoidance and precise navigation.
        Various configurations with high mine density.

        Args:
            count: Number of mine-heavy levels to generate
            mode: 'train' or 'test' to determine seed range and output location
        """
        print(f"Generating {count} mine-heavy levels ({mode} set)...")

        base_seed_start = (
            self.TRAIN_MINE_HEAVY_BASE_SEED
            if mode == "train"
            else self.TEST_MINE_HEAVY_BASE_SEED
        )
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        for i in range(count):
            base_seed = base_seed_start + i

            # Alternate between mine maze and jump with heavy mines
            if i % 2 == 0:

                def generator_func(seed):
                    return self._create_heavy_mine_maze(seed)

                desc = "Mine-heavy maze requiring careful navigation"
            else:

                def generator_func(seed):
                    return self._create_heavy_mine_jump(seed)

                desc = "Jump level with heavy mine obstacles"

            map_gen = self._generate_unique_map(
                generator_func, base_seed, i, f"mine_heavy-{mode}"
            )

            level_data = {
                "level_id": f"mine_heavy_{i:03d}",
                "seed": base_seed,
                "category": "mine_heavy",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 10) + 1,  # 3 tiers
                    "split": mode,
                },
            }
            levels_dict["mine_heavy"].append(level_data)

        print(f"✓ Generated {count} mine-heavy levels ({mode} set)")

    def _create_heavy_mine_maze(self, seed: int) -> Map:
        """Create a mine maze with heavy mine density."""
        map_gen = MapMineMaze(seed=seed)

        # Large chamber
        map_gen.MIN_WIDTH = 15
        map_gen.MAX_WIDTH = 35
        map_gen.MIN_HEIGHT = 5
        map_gen.MAX_HEIGHT = 10

        # High mine density
        map_gen.MIN_SKIP_COLUMNS = 3
        map_gen.MAX_SKIP_COLUMNS = 4
        map_gen.MIN_MINES_PER_COLUMN = 3
        map_gen.MAX_MINES_PER_COLUMN = 10

        map_gen.generate(seed=seed)
        return map_gen

    def _create_heavy_mine_jump(self, seed: int) -> Map:
        """Create a jump level with heavy mines."""
        map_gen = MapJumpRequired(seed=seed)

        # Large jump level
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 40
        map_gen.MIN_HEIGHT = 10
        map_gen.MAX_HEIGHT = 16
        map_gen.MIN_PIT_WIDTH = 4
        map_gen.MAX_PIT_WIDTH = 6
        map_gen.MAX_MINES_PER_PLATFORM = 5

        map_gen.generate(seed=seed)
        return map_gen

    def generate_exploration_levels(self, count: int = 20, mode: str = "test") -> None:
        """Generate exploration levels: hidden switches, extensive exploration.

        These levels test exploration strategies and discovery.
        Large mazes, multi-chamber with distant objectives.

        Args:
            count: Number of exploration levels to generate
            mode: 'train' or 'test' to determine seed range and output location
        """
        print(f"Generating {count} exploration levels ({mode} set)...")

        base_seed_start = (
            self.TRAIN_EXPLORATION_BASE_SEED
            if mode == "train"
            else self.TEST_EXPLORATION_BASE_SEED
        )
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        for i in range(count):
            base_seed = base_seed_start + i

            # Alternate between large mazes and sprawling multi-chamber
            if i % 2 == 0:

                def generator_func(seed):
                    return self._create_exploration_maze(seed)

                desc = "Large maze requiring extensive exploration"
            else:

                def generator_func(seed):
                    return self._create_exploration_multi_chamber(seed)

                desc = "Sprawling multi-chamber with distant objectives"

            map_gen = self._generate_unique_map(
                generator_func, base_seed, i, f"exploration-{mode}"
            )

            level_data = {
                "level_id": f"exploration_{i:03d}",
                "seed": base_seed,
                "category": "exploration",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 7) + 1,  # 3 tiers
                    "split": mode,
                },
            }
            levels_dict["exploration"].append(level_data)

        print(f"✓ Generated {count} exploration levels ({mode} set)")

    def _create_exploration_maze(self, seed: int) -> Map:
        """Create a large maze for exploration."""
        map_gen = MazeGenerator(seed=seed)

        # Maximum maze dimensions for exploration
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 40
        map_gen.MIN_HEIGHT = 12
        map_gen.MAX_HEIGHT = 20

        map_gen.generate(seed=seed)
        return map_gen

    def _create_exploration_multi_chamber(self, seed: int) -> Map:
        """Create a sprawling multi-chamber level."""
        map_gen = MultiChamberGenerator(seed=seed)

        # Maximum chambers
        map_gen.MIN_CHAMBERS = 4
        map_gen.MAX_CHAMBERS = 4

        # Larger chambers spread out
        map_gen.MIN_CHAMBER_WIDTH = 6
        map_gen.MAX_CHAMBER_WIDTH = 14
        map_gen.MIN_CHAMBER_HEIGHT = 5
        map_gen.MAX_CHAMBER_HEIGHT = 9

        # Long corridors for exploration feel
        map_gen.MIN_CORRIDOR_LENGTH = 3
        map_gen.MAX_CORRIDOR_LENGTH = 8

        map_gen.generate(seed=seed)
        return map_gen

    def save_dataset(self, mode: str = "both") -> None:
        """Save the generated datasets to disk.

        Args:
            mode: 'train', 'test', or 'both' to determine which datasets to save
        """
        print("\nSaving dataset...")

        datasets_to_save = []
        if mode in ["train", "both"]:
            datasets_to_save.append(("train", self.train_levels))
        if mode in ["test", "both"]:
            datasets_to_save.append(("test", self.test_levels))

        total_saved = 0
        for split_name, levels_dict in datasets_to_save:
            split_dir = self.base_output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Save each category separately
            split_total = 0
            for category, levels in levels_dict.items():
                if not levels:
                    continue

                category_dir = split_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)

                # Save each level as a separate file
                for level in levels:
                    level_file = category_dir / f"{level['level_id']}.pkl"
                    with open(level_file, "wb") as f:
                        pickle.dump(level, f)

                split_total += len(levels)
                print(f"  ✓ Saved {len(levels)} {category} levels to {category_dir}")

            # Save metadata summary for this split
            metadata = {
                "split": split_name,
                "total_levels": sum(len(levels) for levels in levels_dict.values()),
                "categories": {
                    category: {
                        "count": len(levels),
                        "level_ids": [level["level_id"] for level in levels],
                    }
                    for category, levels in levels_dict.items()
                },
                "generation_info": {
                    "script_version": "2.0",
                    "description": f"NPP-RL Task 3.3 {split_name} dataset",
                    "deterministic": True,
                },
            }

            metadata_file = split_dir / f"{split_name}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved {split_name} metadata to {metadata_file}")
            total_saved += split_total

        print("\n✓ Dataset generation complete!")
        print(f"  Total levels generated: {total_saved}")
        print(f"  Unique maps: {len(self.seen_maps)}")
        print(f"  Duplicates detected and regenerated: {self.duplicate_count}")
        print(f"  Output directory: {self.base_output_dir}")

    def generate_all(self, mode: str = "both") -> None:
        """Generate all dataset levels.

        Args:
            mode: 'train', 'test', or 'both' to determine which datasets to generate
        """
        print("=" * 70)
        print(f"NPP-RL Dataset Generation (Task 3.3) - Mode: {mode.upper()}")
        print("=" * 70)
        print()

        modes_to_generate = []
        if mode in ["train", "both"]:
            modes_to_generate.append("train")
        if mode in ["test", "both"]:
            modes_to_generate.append("test")

        for gen_mode in modes_to_generate:
            print(f"\n{'=' * 70}")
            print(f"Generating {gen_mode.upper()} dataset")
            print(f"{'=' * 70}\n")

            self.generate_simple_levels(50, mode=gen_mode)
            self.generate_medium_levels(100, mode=gen_mode)
            self.generate_complex_levels(50, mode=gen_mode)
            self.generate_mine_heavy_levels(30, mode=gen_mode)
            self.generate_exploration_levels(20, mode=gen_mode)

        self.save_dataset(mode)

        print()
        print("=" * 70)


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate NPP-RL train and test datasets (Task 3.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both train and test datasets (500 levels total)
  python -m nclone.map_generation.generate_test_suite_maps --mode both
  
  # Generate only training dataset (250 levels)
  python -m nclone.map_generation.generate_test_suite_maps --mode train
  
  # Generate only test dataset (250 levels)
  python -m nclone.map_generation.generate_test_suite_maps --mode test
  
  # Custom output directory
  python -m nclone.map_generation.generate_test_suite_maps --mode both --output_dir /custom/path
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which dataset(s) to generate: 'train', 'test', or 'both' (default: both)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Base output directory for datasets (default: ./datasets). Train and test subdirectories will be created automatically.",
    )

    args = parser.parse_args()

    # Generate the datasets
    generator = TestSuiteGenerator(args.output_dir)
    generator.generate_all(mode=args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
