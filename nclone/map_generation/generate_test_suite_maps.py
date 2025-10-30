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

    def _generate_level(self, generator_type: str, preset: str, seed: int) -> Map:
        """Generate a single level using generator type and preset.

        Args:
            generator_type: Type of generator (from GENERATOR_REGISTRY)
            preset: Preset name (from GENERATOR_PRESETS)
            seed: Random seed

        Returns:
            Generated Map
        """
        # Use factory for all generators
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
                    lambda s: self._generate_level(gen_type, preset, s),
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
