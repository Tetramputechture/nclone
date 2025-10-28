"""
Generate comprehensive train and test suite maps for NPP-RL.

This script creates deterministic datasets of N++ levels across difficulty categories
defined in generator_configs.py. All maps are generated deterministically using fixed
seeds to ensure reproducibility.

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
from typing import Dict, List, Any, Callable
from .map import Map
from .generator_factory import GeneratorFactory
from .generator_configs import CATEGORIES
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from .constants import VALID_TILE_TYPES


class TestSuiteGenerator:
    """Generator for comprehensive NPP-RL train and test suites.

    Uses configuration-driven generation from generator_configs.py.
    Adding new categories or generator types requires only configuration changes.
    """

    MAX_REGENERATION_ATTEMPTS = 1000

    def __init__(self, base_output_dir: str = "./datasets", map_count: int = 2000):
        """Initialize the test suite generator.

        Args:
            base_output_dir: Base directory where train/test datasets will be saved
            map_count: Total number of maps to generate per dataset
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Load category configs dynamically
        self.categories = CATEGORIES

        # Calculate counts per category based on ratios
        self.category_counts = {
            name: int(map_count * config.ratio)
            for name, config in self.categories.items()
        }

        # Track generated levels for both train and test
        self.train_levels: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.categories
        }
        self.test_levels: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.categories
        }

        # Track unique maps by their data hash to prevent duplicates
        self.seen_maps: set = set()

        # Counter for generating new seeds when duplicates are found
        self.seed_offset = 200000

        # Track statistics
        self.duplicate_count = 0

    def _get_map_hash(self, map_data: List[int]) -> tuple:
        """Convert map data to a hashable tuple for duplicate detection."""
        return tuple(map_data)

    def _is_unique_map(self, map_gen: Map) -> bool:
        """Check if a generated map is unique."""
        map_data = map_gen.map_data()
        map_hash = self._get_map_hash(map_data)
        return map_hash not in self.seen_maps

    def _register_map(self, map_gen: Map) -> None:
        """Register a map as seen to prevent future duplicates."""
        map_data = map_gen.map_data()
        map_hash = self._get_map_hash(map_data)
        self.seen_maps.add(map_hash)

    def _generate_unique_map(
        self,
        generator_func: Callable[[int], Map],
        base_seed: int,
        level_index: int,
        category: str,
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

        raise RuntimeError(
            f"Failed to generate unique map for {category} level {level_index} "
            f"after {self.MAX_REGENERATION_ATTEMPTS} attempts"
        )

    def _generate_level(
        self, generator_type: str, preset: str, seed: int, index: int = 0
    ) -> Map:
        """Generate a single level using generator type and preset.

        Args:
            generator_type: Type of generator (from GENERATOR_REGISTRY)
            preset: Preset name (from GENERATOR_PRESETS)
            seed: Random seed
            index: Level index (used for special generators like horizontal)

        Returns:
            Generated Map
        """
        # Special handling for horizontal generator (uses custom logic)
        if generator_type == "horizontal":
            if preset == "minimal":
                return self._create_minimal_simple_level_horizontal(seed, 8, height=1)
            else:
                return self._create_minimal_simple_level_horizontal(
                    seed, index, height=1
                )

        # Use factory for standard generators
        map_gen = GeneratorFactory.create_from_preset(generator_type, preset, seed)
        map_gen.generate(seed=seed)
        return map_gen

    def generate_category_levels(
        self, category: str, count: int, mode: str = "test"
    ) -> None:
        """Generic method to generate levels for any category.

        Args:
            category: Category name from CATEGORIES
            count: Number of levels to generate
            mode: 'train' or 'test'
        """
        print(f"Generating {count} {category} levels ({mode} set)...")

        config = self.categories[category]
        base_seed = config.seed_base_train if mode == "train" else config.seed_base_test
        levels_dict = self.train_levels if mode == "train" else self.test_levels

        # Distribute evenly across generators
        num_generators = len(config.generators)
        base_per_gen = count // num_generators
        remainder = count % num_generators

        # Build distribution list
        type_counts = [
            base_per_gen + (1 if i < remainder else 0) for i in range(num_generators)
        ]

        current_index = 0
        for gen_idx, (gen_type, preset) in enumerate(config.generators):
            gen_count = type_counts[gen_idx]

            for j in range(gen_count):
                i = current_index + j
                seed = base_seed + i

                # Generate with deduplication
                map_gen = self._generate_unique_map(
                    lambda s: self._generate_level(gen_type, preset, s, i),
                    seed,
                    i,
                    f"{category}-{mode}",
                )

                # Calculate difficulty tier dynamically
                difficulty_tier = min(4, (i * 4 // count) + 1)

                # Create level data
                level_data = {
                    "level_id": f"{category}_{i:03d}",
                    "seed": seed,
                    "category": category,
                    "map_data": map_gen.map_data(),
                    "metadata": {
                        "description": config.description,
                        "generator": f"{gen_type}:{preset}",
                        "difficulty_tier": difficulty_tier,
                        "split": mode,
                    },
                }
                levels_dict[category].append(level_data)

            current_index += gen_count

        print(f"✓ Generated {count} {category} levels ({mode} set)")

        # Print distribution summary
        dist_summary = ", ".join(
            f"{type_counts[i]} {config.generators[i][0]}:{config.generators[i][1]}"
            for i in range(num_generators)
        )
        print(f"  Distribution: {dist_summary}")

    def _create_minimal_simple_level_horizontal(
        self, seed: int, index: int, height: int = None
    ) -> Map:
        """Create a minimal horizontal level (special case for 'horizontal' generator).

        Args:
            seed: Random seed
            index: Level index for parameter variation
            height: Optional fixed height (default: random 1-5 tiles high)

        Returns:
            Generated Map
        """
        map_gen = Map(seed=seed)
        rng = map_gen.rng

        # Very small dimensions for simplest levels
        max_width = 3 + (index % 20)
        max_height = 1 + (index % 5)
        width = rng.randint(3, max_width)
        if height is None:
            height = rng.randint(1, max_height)

        # Random offset for the chamber
        max_start_x = MAP_TILE_WIDTH - width - 1
        max_start_y = MAP_TILE_HEIGHT - height - 1
        start_x = rng.randint(1, max_start_x)
        start_y = rng.randint(1, max_start_y)

        # Fill with random tiles
        tile_types = [
            rng.randint(0, VALID_TILE_TYPES)
            for _ in range(MAP_TILE_WIDTH * MAP_TILE_HEIGHT)
        ]
        map_gen.set_tiles_bulk(tile_types)

        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                map_gen.set_tile(x, y, 0)

        # Add decorative walls on chamber edges
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
            ninja_orientation = 1
        else:
            ninja_x = start_x + width - 1
            ninja_orientation = -1

        ninja_y = start_y + height - 1

        # Check if we should add a locked door
        can_add_locked_door = height == 1 and width >= 4
        add_locked_door = can_add_locked_door and rng.choice([True, False])

        # Generate positions with quarter-tile increments
        num_positions = (width - 1) * 4
        available_positions = [start_x + i * 0.25 for i in range(num_positions)]

        # Filter positions
        available_positions = [
            pos
            for pos in available_positions
            if pos > start_x + 0.25 and pos < start_x + width - 0.25
        ]
        available_positions = [
            pos for pos in available_positions if abs(pos - ninja_x) >= 1
        ]

        # For doors, filter to only integer positions (doors must be at 24-pixel tile boundaries)
        door_positions = [pos for pos in available_positions if pos == int(pos)]

        # Place entities based on layout complexity
        # Try to add locked door with proper switch placement
        locked_door_viable = False
        if (
            add_locked_door
            and len(door_positions) >= 2
            and len(available_positions) >= 4
        ):
            # Sample 2 door positions (must be integers for 24-pixel alignment)
            door_pos = sorted(rng.sample(door_positions, k=2))

            # For each door, find valid switch positions (between ninja and door)
            switch_available = [p for p in available_positions if p not in door_pos]

            # Check if we have enough switch positions between ninja and doors
            if ninja_on_left:
                locked_switch_candidates = [
                    p for p in switch_available if p < door_pos[0]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]
            else:
                locked_switch_candidates = [
                    p for p in switch_available if p > door_pos[1]
                ]
                exit_switch_candidates = [
                    p for p in switch_available if door_pos[0] < p < door_pos[1]
                ]

            # Only proceed if we have valid switch positions
            if locked_switch_candidates and exit_switch_candidates:
                locked_door_viable = True
                if ninja_on_left:
                    locked_switch_x = rng.choice(locked_switch_candidates)
                    exit_switch_x = rng.choice(exit_switch_candidates)
                    locked_door_x = door_pos[0]
                    exit_door_x = door_pos[1]
                else:
                    locked_switch_x = rng.choice(locked_switch_candidates)
                    exit_switch_x = rng.choice(exit_switch_candidates)
                    locked_door_x = door_pos[1]
                    exit_door_x = door_pos[0]
                entity_y = start_y

        if locked_door_viable:
            # Place locked door and exit door

            map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            # Doors must be integers (24-pixel boundaries)
            # Switches can be fractional - multiply by GRID_SIZE_FACTOR before int() to preserve fractional grid positions
            from nclone.map_generation.constants import GRID_SIZE_FACTOR

            # Convert switch positions preserving fractional coordinates
            # Multiply by GRID_SIZE_FACTOR, then int(), then divide back
            # This preserves fractional grid positions (e.g., 7.75 * 4 = 31, int(31) = 31, 31 / 4 = 7.75)
            locked_switch_grid = (
                int(locked_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR
            )
            exit_switch_grid = int(exit_switch_x * GRID_SIZE_FACTOR) / GRID_SIZE_FACTOR

            map_gen.add_entity(
                6,
                int(locked_door_x),
                entity_y,
                4,
                0,
                locked_switch_grid,
                entity_y,
            )
            map_gen.add_entity(
                3,
                int(exit_door_x),
                entity_y,
                0,
                0,
                exit_switch_grid,
                entity_y,
            )
        else:
            # Exit door only - use integer positions for door, can use any for switch
            if len(door_positions) >= 2:
                # Use integer positions for door entities
                positions = sorted(rng.sample(door_positions, k=2))
            else:
                # Fallback if not enough integer positions
                positions = sorted(rng.sample(available_positions, k=2))

            if not ninja_on_left:
                positions = positions[::-1]

            exit_switch_x, exit_door_x = positions
            exit_switch_y = start_y
            exit_door_y = start_y

            if height > 1:
                exit_switch_y = start_y + rng.randint(1, height - 1) * 0.25
                exit_door_y = start_y + rng.randint(1, height - 1) * 0.25

            map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=ninja_orientation)
            # Ensure positions are integers for door entities
            map_gen.add_entity(
                3,
                exit_door_x,
                exit_door_y,
                0,
                0,
                exit_switch_x,
                exit_switch_y,
            )

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

            # Save metadata summary
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
                    "script_version": "3.0",
                    "description": f"NPP-RL {split_name} dataset",
                    "deterministic": True,
                    "config_driven": True,
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
        """Generate all categories automatically from config.

        Args:
            mode: 'train', 'test', or 'both' to determine which datasets to generate
        """
        print("=" * 70)
        print(f"NPP-RL Dataset Generation - Mode: {mode.upper()}")
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

            # Generate all categories automatically
            for category_name in self.categories:
                count = self.category_counts[category_name]
                self.generate_category_levels(category_name, count, mode=gen_mode)

        self.save_dataset(mode)

        print()
        print("=" * 70)


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate NPP-RL train and test datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both train and test datasets
  python -m nclone.map_generation.generate_test_suite_maps --mode both
  
  # Generate only training dataset
  python -m nclone.map_generation.generate_test_suite_maps --mode train
  
  # Generate only test dataset
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
        help="Base output directory for datasets (default: ./datasets)",
    )

    parser.add_argument(
        "--map_count",
        type=int,
        default=2000,
        help="Total number of maps to generate per dataset (default: 2000)",
    )

    args = parser.parse_args()

    # Generate the datasets
    generator = TestSuiteGenerator(args.output_dir, args.map_count)
    generator.generate_all(mode=args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
