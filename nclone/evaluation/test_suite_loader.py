"""
Test Suite Loader for N++ Evaluation (Task 3.3).

.. deprecated::
    This is the LEGACY evaluation loader. Its dataset (``datasets/test_suite/``)
    is NOT shipped in the repo — the directory is empty and must be regenerated
    with ``python -m nclone.map_generation.generate_test_suite_maps``. Code that
    points at a missing ``test_metadata.json`` falls back to random maps.

    The CURRENT, populated benchmark is the **mechanic-primary** dataset under
    ``datasets/mechanic_benchmark/`` (train/test split, buckets FLAT_RUN /
    BASIC_JUMP / WALL_JUMP / MINE_NAVIGATION / SLOPES), loaded via
    :class:`nclone.evaluation.mechanic_benchmark_loader.MechanicBenchmarkLoader`.
    New evaluation (the npp-rl trainer) uses that loader; prefer it.

This module provides utilities for loading and managing the comprehensive test suite
of 250 deterministic N++ levels across complexity categories.

The test suite dataset is expected at:
    datasets/test_suite/

Usage:
    from nclone.evaluation import TestSuiteLoader

    # Load test suite (use relative or absolute path)
    loader = TestSuiteLoader('datasets/test_suite')  # or absolute path

    # Get all simple levels
    simple_levels = loader.get_category('simple')

    # Get a specific level
    level = loader.get_level('simple_000')

    # Load level into environment (if using npp-rl)
    env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class TestSuiteLoader:
    """Loader for NPP-RL test suite maps."""

    CATEGORIES = [
        "simplest",
        "simplest_few_mines",
        "simplest_with_mines",
        "simpler",
        "simple",
        "medium",
        "complex",
        "mine_heavy",
        "exploration",
    ]

    def __init__(self, test_suite_dir: str):
        """Initialize the test suite loader.

        Args:
            test_suite_dir: Path to the test suite directory
        """
        self.test_suite_dir = Path(test_suite_dir)

        if not self.test_suite_dir.exists():
            raise ValueError(f"Test suite directory not found: {test_suite_dir}")

        # Load metadata
        metadata_file = self.test_suite_dir / "test_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Test suite metadata not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self._level_cache: Dict[str, Dict[str, Any]] = {}

        # Build a format-agnostic index. The canonical dataset is now the
        # mechanic-primary benchmark (``"buckets"`` schema, ``<BUCKET>/T<tier>/``
        # layout); the legacy difficulty-category schema (``"categories"``,
        # ``<category>/`` layout) is still read for backward compatibility.
        #   * ``self.categories``  -> ordered list of category/bucket names present
        #   * ``self._cat_ids``    -> {category: [level_id, ...]}
        #   * ``self._id_to_path`` -> {str(level_id): Path to its .pkl}
        self._cat_ids: Dict[str, List[str]] = {}
        self._id_to_path: Dict[str, Path] = {}
        self._build_index()
        self.categories: List[str] = list(self._cat_ids.keys())

    def _build_index(self) -> None:
        """Populate ``self._cat_ids`` / ``self._id_to_path`` from either schema."""
        if "buckets" in self.metadata:
            # Mechanic-primary benchmark: category == bucket; files live under
            # ``<bucket>/T<tier>/<level_id>.pkl``.
            for bucket, binfo in self.metadata["buckets"].items():
                ids: List[str] = []
                for tier_key, tinfo in binfo.get("tiers", {}).items():
                    tier_dir = self.test_suite_dir / bucket / f"T{tier_key}"
                    for level_id in tinfo.get("level_ids", []):
                        sid = str(level_id)
                        ids.append(sid)
                        self._id_to_path[sid] = tier_dir / f"{sid}.pkl"
                if ids:
                    self._cat_ids[bucket] = ids
        elif "categories" in self.metadata:
            # Legacy difficulty-category schema.
            for cat in self.CATEGORIES:
                cinfo = self.metadata["categories"].get(cat)
                if not cinfo:
                    continue
                ids = [str(i) for i in cinfo.get("level_ids", [])]
                if not ids:
                    continue
                self._cat_ids[cat] = ids
                cat_dir = self.test_suite_dir / cat
                for sid in ids:
                    self._id_to_path[sid] = cat_dir / f"{sid}.pkl"

    def get_category(self, category: str) -> List[Dict[str, Any]]:
        """Load all levels from a specific category.

        Args:
            category: Category name ('simple', 'medium', 'complex', 'mine_heavy', 'exploration')

        Returns:
            List of level dictionaries
        """
        if category not in self._cat_ids:
            raise ValueError(
                f"Unknown category: {category}. Must be one of {self.categories}"
            )

        return [self.get_level(level_id) for level_id in self._cat_ids[category]]

    def get_level(self, level_id: str) -> Dict[str, Any]:
        """Load a specific level by ID.

        Args:
            level_id: Level ID (legacy ``'simple_000'`` or a benchmark integer ID)

        Returns:
            Level dictionary with map_data and metadata
        """
        level_id = str(level_id)
        # Check cache first
        if level_id in self._level_cache:
            return self._level_cache[level_id]

        level_file = self._id_to_path.get(level_id)
        if level_file is None or not level_file.exists():
            raise ValueError(f"Level file not found for id: {level_id}")

        with open(level_file, "rb") as f:
            level = pickle.load(f)

        # Cache the level
        self._level_cache[level_id] = level

        return level

    def get_all_levels(self) -> List[Dict[str, Any]]:
        """Load all levels from all categories/buckets.

        Returns:
            List of all level dictionaries
        """
        all_levels = []
        for category in self.categories:
            all_levels.extend(self.get_category(category))
        return all_levels

    def get_metadata(self) -> Dict[str, Any]:
        """Get test suite metadata.

        Returns:
            Test suite metadata dictionary
        """
        return self.metadata

    def get_level_count(self, category: Optional[str] = None) -> int:
        """Get the number of levels in a category or total.

        Args:
            category: Optional category name. If None, returns total count.

        Returns:
            Number of levels
        """
        if category is None:
            return self.metadata.get(
                "total_levels", sum(len(v) for v in self._cat_ids.values())
            )

        if category not in self._cat_ids:
            raise ValueError(f"Unknown category: {category}")

        return len(self._cat_ids[category])

    def get_level_ids(self, category: Optional[str] = None) -> List[str]:
        """Get list of level IDs.

        Args:
            category: Optional category/bucket name. If None, returns all IDs in
                category order.

        Returns:
            List of level IDs
        """
        if category is None:
            all_ids: List[str] = []
            for cat in self.categories:
                all_ids.extend(self._cat_ids[cat])
            return all_ids

        if category not in self._cat_ids:
            raise ValueError(f"Unknown category: {category}")

        return list(self._cat_ids[category])


def load_test_suite_into_env(env, level_id: str, test_suite_dir: str) -> None:
    """Convenience function to load a test suite level into an environment.

    Args:
        env: NPP environment instance
        level_id: Level ID to load (e.g., 'simple_000')
        test_suite_dir: Path to test suite directory
    """
    loader = TestSuiteLoader(test_suite_dir)
    level = loader.get_level(level_id)
    env.unwrapped.nplay_headless.load_map_from_map_data(level["map_data"])


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m nclone.evaluation.test_suite_loader <test_suite_dir>")
        sys.exit(1)

    test_suite_dir = sys.argv[1]
    loader = TestSuiteLoader(test_suite_dir)

    print("=" * 70)
    print("NPP-RL Test Suite Information")
    print("=" * 70)
    print()
    print(f"Test suite directory: {test_suite_dir}")
    print(f"Total levels: {loader.get_level_count()}")
    print()
    print("Levels by category:")
    for category in loader.CATEGORIES:
        count = loader.get_level_count(category)
        print(f"  {category:15s}: {count:3d} levels")
    print()

    # Show a sample level
    print("Sample level (simple_000):")
    level = loader.get_level("simple_000")
    print(f"  Level ID: {level['level_id']}")
    print(f"  Category: {level['category']}")
    print(f"  Seed: {level['seed']}")
    print(f"  Description: {level['metadata']['description']}")
    print(f"  Difficulty tier: {level['metadata']['difficulty_tier']}")
    print(f"  Map data length: {len(level['map_data'])} bytes")
    print()
    print("=" * 70)
