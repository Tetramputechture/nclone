"""
Generate comprehensive test suite maps for NPP-RL evaluation (Task 3.3).

This script creates a deterministic dataset of 250 N++ levels across 5 complexity categories:
- 50 simple levels (single switch, direct path, includes tiny mazes)
- 100 medium levels (1-3 switches, simple dependencies, includes medium-sized mazes)
- 50 complex levels (4+ switches, complex dependencies, multi-chamber, large mazes)
- 30 mine-heavy levels (significant mine obstacles)
- 20 exploration levels (hidden switches, extensive exploration required)

All maps are generated deterministically using fixed seeds to ensure reproducibility.
The dataset can be used as a baseline for training and evaluating NPP-RL agents.

Usage:
    python -m nclone.map_generation.generate_test_suite_maps --output_dir /path/to/dataset
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
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class TestSuiteGenerator:
    """Generator for comprehensive NPP-RL test suite."""

    # Base seeds for each category to ensure deterministic generation
    SIMPLE_BASE_SEED = 1000
    MEDIUM_BASE_SEED = 2000
    COMPLEX_BASE_SEED = 3000
    MINE_HEAVY_BASE_SEED = 4000
    EXPLORATION_BASE_SEED = 5000

    # Maximum attempts to generate a unique map before giving up
    MAX_REGENERATION_ATTEMPTS = 1000

    def __init__(self, output_dir: str):
        """Initialize the test suite generator.

        Args:
            output_dir: Directory where the test suite will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track generated levels
        self.levels: Dict[str, List[Dict[str, Any]]] = {
            "simple": [],
            "medium": [],
            "complex": [],
            "mine_heavy": [],
            "exploration": [],
        }

        # Track unique maps by their data hash to prevent duplicates
        self.seen_maps: set = set()

        # Counter for generating new seeds when duplicates are found
        self.seed_offset = 100000

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

    def generate_simple_levels(self, count: int = 50) -> None:
        """Generate simple levels: single switch, direct path to exit.

        These levels test basic navigation and switch activation.
        Progression: flat surfaces -> locked doors -> small mazes -> require jump

        Args:
            count: Number of simple levels to generate
        """
        print(f"Generating {count} simple levels...")

        for i in range(count):
            base_seed = self.SIMPLE_BASE_SEED + i

            # First 25: very simple flat levels (minimal chamber with exit switch only)
            # Some of these will have locked doors (1-tile high, 5+ tiles wide)
            if i < 25:

                def generator_func(seed):
                    return self._create_minimal_simple_level(seed, i)
            # Next 15: small mazes for simple navigation practice
            elif i < 40:

                def generator_func(seed):
                    return self._create_tiny_maze(seed)
            # Last 10: require a jump
            else:

                def generator_func(seed):
                    return self._create_simple_jump_level(seed)

            map_gen = self._generate_unique_map(generator_func, base_seed, i, "simple")
            actual_seed = base_seed  # The seed used (may differ if regenerated)

            level_data = {
                "level_id": f"simple_{i:03d}",
                "seed": actual_seed,
                "category": "simple",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": self._get_simple_description(i),
                    "difficulty_tier": 1
                    if i < 15
                    else (2 if i < 25 else (3 if i < 40 else 4)),
                },
            }
            self.levels["simple"].append(level_data)

        print(f"✓ Generated {count} simple levels")

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

        # Center the chamber
        start_x = (MAP_TILE_WIDTH - width) // 2
        start_y = (MAP_TILE_HEIGHT - height) // 2

        # Fill everything with walls first
        for y in range(MAP_TILE_HEIGHT):
            for x in range(MAP_TILE_WIDTH):
                map_gen.set_tile(x, y, 1)

        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                map_gen.set_tile(x, y, 0)

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
        available_positions = [start_x + 1 + i * 0.25 for i in range(num_positions)]

        # Remove positions too close to the edge of the playspace
        available_positions = [
            pos
            for pos in available_positions
            if pos > start_x + 1 and pos < start_x + width - 1
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

            entity_y = start_y + height - 1

            map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            map_gen.add_entity(3, exit_door_x, entity_y, 0, 0, exit_switch_x, entity_y)

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

        map_gen.generate(seed=seed)
        return map_gen

    def _get_simple_description(self, index: int) -> str:
        """Get description for simple level based on index."""
        if index < 25:
            return "Minimal chamber: ninja -> exit switch -> door (may have locked doors if 1-tile high)"
        elif index < 40:
            return "Tiny maze for basic navigation practice"
        else:
            return "Simple jump required to reach switch or exit"

    def generate_medium_levels(self, count: int = 100) -> None:
        """Generate medium levels: 1-3 switches, simple dependencies.

        These levels test navigation with multiple objectives and basic planning.
        Mix of: small mazes, multi-switch chambers, jump-required with switches

        Args:
            count: Number of medium levels to generate
        """
        print(f"Generating {count} medium levels...")

        for i in range(count):
            base_seed = self.MEDIUM_BASE_SEED + i

            # Mix different types of medium levels
            level_type = i % 4

            if level_type == 0:
                # Medium-sized maze
                def generator_func(seed):
                    return self._create_small_maze(seed)

                desc = "Medium-sized maze with navigation challenges"
            elif level_type == 1:
                # Multi-chamber (2 chambers)
                def generator_func(seed):
                    return self._create_medium_multi_chamber(seed, num_chambers=2)

                desc = "2-chamber level with switch dependencies"
            elif level_type == 2:
                # Jump required with mines
                def generator_func(seed):
                    return self._create_medium_jump_level(seed)

                desc = "Jump required with moderate mine obstacles"
            else:
                # Larger single chamber with obstacles
                def generator_func(seed):
                    return self._create_medium_chamber_with_obstacles(seed)

                desc = "Medium chamber with and obstacles"

            map_gen = self._generate_unique_map(generator_func, base_seed, i, "medium")

            level_data = {
                "level_id": f"medium_{i:03d}",
                "seed": base_seed,
                "category": "medium",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 25) + 1,  # 4 tiers
                },
            }
            self.levels["medium"].append(level_data)

        print(f"✓ Generated {count} medium levels")

    def _create_small_maze(self, seed: int) -> Map:
        """Create a medium-sized maze level with optional locked doors."""
        map_gen = MazeGenerator(seed=seed)

        # Medium maze dimensions (larger than tiny mazes in simple levels)
        map_gen.MIN_WIDTH = 12
        map_gen.MAX_WIDTH = 22
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 14

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

    def generate_complex_levels(self, count: int = 50) -> None:
        """Generate complex levels: 4+ switches, complex dependencies.

        These levels test advanced planning and multi-step problem solving.
        Multi-chamber with many switches, large mazes, complex jump sequences.

        Args:
            count: Number of complex levels to generate
        """
        print(f"Generating {count} complex levels...")

        for i in range(count):
            base_seed = self.COMPLEX_BASE_SEED + i

            # Alternate between different complex level types
            level_type = i % 3

            if level_type == 0:
                # Large multi-chamber (3-4 chambers)
                def generator_func(seed):
                    return self._create_complex_multi_chamber(seed)

                desc = "3-4 chambers with complex switch dependencies"
            elif level_type == 1:
                # Large maze
                def generator_func(seed):
                    return self._create_large_maze(seed)

                desc = "Large maze requiring extensive navigation"
            else:
                # Complex jump sequence with multiple objectives
                def generator_func(seed):
                    return self._create_complex_jump_sequence(seed)

                desc = "Complex jump sequence with multiple switches"

            map_gen = self._generate_unique_map(generator_func, base_seed, i, "complex")

            level_data = {
                "level_id": f"complex_{i:03d}",
                "seed": base_seed,
                "category": "complex",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 17) + 1,  # 3 tiers
                },
            }
            self.levels["complex"].append(level_data)

        print(f"✓ Generated {count} complex levels")

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

        map_gen.generate(seed=seed)
        return map_gen

    def _create_complex_jump_sequence(self, seed: int) -> Map:
        """Create a complex jump level with multiple platforms."""
        # For complex jump sequences, we'll use multi-chamber with jump-like properties
        map_gen = MultiChamberGenerator(seed=seed)

        map_gen.MIN_CHAMBERS = 3
        map_gen.MAX_CHAMBERS = 4
        map_gen.MIN_CHAMBER_WIDTH = 5
        map_gen.MAX_CHAMBER_WIDTH = 10
        map_gen.MIN_CHAMBER_HEIGHT = 4
        map_gen.MAX_CHAMBER_HEIGHT = 7

        map_gen.generate(seed=seed)
        return map_gen

    def generate_mine_heavy_levels(self, count: int = 30) -> None:
        """Generate mine-heavy levels: significant mine obstacles.

        These levels test hazard avoidance and precise navigation.
        Various configurations with high mine density.

        Args:
            count: Number of mine-heavy levels to generate
        """
        print(f"Generating {count} mine-heavy levels...")

        for i in range(count):
            base_seed = self.MINE_HEAVY_BASE_SEED + i

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
                generator_func, base_seed, i, "mine_heavy"
            )

            level_data = {
                "level_id": f"mine_heavy_{i:03d}",
                "seed": base_seed,
                "category": "mine_heavy",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 10) + 1,  # 3 tiers
                },
            }
            self.levels["mine_heavy"].append(level_data)

        print(f"✓ Generated {count} mine-heavy levels")

    def _create_heavy_mine_maze(self, seed: int) -> Map:
        """Create a mine maze with heavy mine density."""
        map_gen = MapMineMaze(seed=seed)

        # Large chamber
        map_gen.MIN_WIDTH = 15
        map_gen.MAX_WIDTH = 35
        map_gen.MIN_HEIGHT = 5
        map_gen.MAX_HEIGHT = 10

        # High mine density
        map_gen.MIN_SKIP_COLUMNS = 2
        map_gen.MAX_SKIP_COLUMNS = 3
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

    def generate_exploration_levels(self, count: int = 20) -> None:
        """Generate exploration levels: hidden switches, extensive exploration.

        These levels test exploration strategies and discovery.
        Large mazes, multi-chamber with distant objectives.

        Args:
            count: Number of exploration levels to generate
        """
        print(f"Generating {count} exploration levels...")

        for i in range(count):
            base_seed = self.EXPLORATION_BASE_SEED + i

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
                generator_func, base_seed, i, "exploration"
            )

            level_data = {
                "level_id": f"exploration_{i:03d}",
                "seed": base_seed,
                "category": "exploration",
                "map_data": map_gen.map_data(),
                "metadata": {
                    "description": desc,
                    "difficulty_tier": (i // 7) + 1,  # 3 tiers
                },
            }
            self.levels["exploration"].append(level_data)

        print(f"✓ Generated {count} exploration levels")

    def _create_exploration_maze(self, seed: int) -> Map:
        """Create a large maze for exploration."""
        map_gen = MazeGenerator(seed=seed)

        # Maximum maze dimensions for exploration
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 35
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

    def save_dataset(self) -> None:
        """Save the generated test suite to disk."""
        print("\nSaving test suite dataset...")

        # Save each category separately
        for category, levels in self.levels.items():
            if not levels:
                continue

            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Save each level as a separate file
            for level in levels:
                level_file = category_dir / f"{level['level_id']}.pkl"
                with open(level_file, "wb") as f:
                    pickle.dump(level, f)

            print(f"  ✓ Saved {len(levels)} {category} levels to {category_dir}")

        # Save metadata summary
        metadata = {
            "total_levels": sum(len(levels) for levels in self.levels.values()),
            "categories": {
                category: {
                    "count": len(levels),
                    "level_ids": [level["level_id"] for level in levels],
                }
                for category, levels in self.levels.items()
            },
            "generation_info": {
                "script_version": "1.0",
                "description": "NPP-RL Task 3.3 comprehensive test suite",
                "deterministic": True,
            },
        }

        metadata_file = self.output_dir / "test_suite_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved metadata to {metadata_file}")
        print("\n✓ Test suite generation complete!")
        print(f"  Total levels: {metadata['total_levels']}")
        print(f"  Unique maps: {len(self.seen_maps)}")
        print(f"  Duplicates detected and regenerated: {self.duplicate_count}")
        print(f"  Output directory: {self.output_dir}")

    def generate_all(self) -> None:
        """Generate all test suite levels."""
        print("=" * 70)
        print("NPP-RL Test Suite Generation (Task 3.3)")
        print("=" * 70)
        print()

        self.generate_simple_levels(50)
        self.generate_medium_levels(100)
        self.generate_complex_levels(50)
        self.generate_mine_heavy_levels(30)
        self.generate_exploration_levels(20)

        self.save_dataset()

        print()
        print("=" * 70)


def main():
    """Main entry point for test suite generation."""
    parser = argparse.ArgumentParser(
        description="Generate NPP-RL test suite maps (Task 3.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full test suite (250 levels)
  python -m nclone.map_generation.generate_test_suite_maps --output_dir ./test_suite
  
  # Generate to custom location
  python -m nclone.map_generation.generate_test_suite_maps --output_dir /workspace/npp-rl/datasets/test_suite
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_suite",
        help="Output directory for generated test suite (default: ./test_suite)",
    )

    args = parser.parse_args()

    # Generate the test suite
    generator = TestSuiteGenerator(args.output_dir)
    generator.generate_all()

    return 0


if __name__ == "__main__":
    sys.exit(main())
