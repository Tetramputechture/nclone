"""Unit and integration tests for graph-based reachability augmentations.

Run with:
    python -m pytest nclone/offline_rl/dataset/augmentation/test_reachability_augmentations.py -v

Tests are divided into three layers:
  1. Pure unit tests (no graph building) – coordinate conversions, buffer
     expansion, bounding box helpers.
  2. Behavioural unit tests (graph building required) – verify that protected
     tiles are preserved, unreachable tiles are modified, entity/spawn offsets
     are correct after translation.
  3. End-to-end simulator tests (nclone_native required) – augmented replays
     still produce winning episodes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_minimal_attract_data(
    *,
    tiles: Optional[List[int]] = None,
    spawn: tuple = (20, 20),
    entities: Optional[list] = None,
    inputs: Optional[List[int]] = None,
) -> dict:
    """Return a minimal AttractData dict for testing.

    Default tiles: all 0 (fully open level).
    Default spawn: map-data units (20, 20) ≈ playable tile (col=3, row=3).
    """
    if tiles is None:
        tiles = [0] * 966
    if entities is None:
        entities = [
            {"type": 4, "x": 20, "y": 40, "orientation": 0, "mode": 0},
            {"type": 3, "x": 24, "y": 40, "orientation": 0, "mode": 0},
        ]
    if inputs is None:
        inputs = [0] * 10

    return {
        "tiles": list(tiles),
        "entities": list(entities),
        "ninja_spawn": spawn,
        "header": bytes(184),
        "demo_data": list(inputs),
    }


def _make_walled_attract_data() -> dict:
    """Return an attract data dict with two rooms separated by a vertical wall.

    Layout (playable 42×23 tile grid):
      cols  0-8   : open room (tile 0) – accessible from spawn
      col   9     : solid wall (tile 1) – blocks horizontal movement
      cols 10-41  : open room (tile 0) – unreachable from spawn

    Spawn at map-data unit (20, 48) = world pixel (120, 288)
           = playable tile (col=4, row=11).

    This gives a well-defined reachable area (cols 0-8) and a large unreachable
    region (cols 10-41) for exercising both augmentations.
    """
    tiles = [0] * 966
    for row in range(23):
        tiles[9 + row * 42] = 1  # solid wall at col 9
    return _make_minimal_attract_data(
        tiles=tiles,
        spawn=(20, 48),  # playable tile (col=4, row=11)
        entities=[
            {"type": 4, "x": 20, "y": 48, "orientation": 0, "mode": 0},
            {"type": 3, "x": 24, "y": 48, "orientation": 0, "mode": 0},
        ],
    )


# ---------------------------------------------------------------------------
# Layer 1: Pure unit tests (no graph building)
# ---------------------------------------------------------------------------


class TestAttractDataToLevelData(unittest.TestCase):
    """Validate the AttractData → LevelData coordinate conversion."""

    def setUp(self) -> None:
        from .reachability_augment_utils import attract_data_to_level_data
        self.convert = attract_data_to_level_data

    def test_tile_shape(self) -> None:
        data = _make_minimal_attract_data()
        ld = self.convert(data)
        self.assertEqual(ld.tiles.shape, (23, 42))

    def test_tile_values_preserved(self) -> None:
        """Every tile value should be faithfully transferred."""
        tiles = list(range(34)) * (966 // 34) + list(range(966 % 34))
        data = _make_minimal_attract_data(tiles=tiles)
        ld = self.convert(data)
        flat = ld.tiles.flatten().tolist()
        self.assertEqual(flat, tiles)

    def test_spawn_converted_to_world_pixels(self) -> None:
        """ninja_spawn map-data units → start_position world pixels (* 6)."""
        data = _make_minimal_attract_data(spawn=(20, 48))
        ld = self.convert(data)
        self.assertEqual(ld.start_position, (120, 288))

    def test_entity_coords_converted_to_world_pixels(self) -> None:
        """Entity x/y map-data units → world pixels (* 6)."""
        entities = [{"type": 4, "x": 30, "y": 15, "orientation": 0, "mode": 0}]
        data = _make_minimal_attract_data(entities=entities)
        ld = self.convert(data)
        self.assertEqual(ld.entities[0]["x"], 180)  # 30 * 6
        self.assertEqual(ld.entities[0]["y"], 90)   # 15 * 6

    def test_wrong_tile_count_raises(self) -> None:
        data = _make_minimal_attract_data()
        data["tiles"] = data["tiles"][:100]
        with self.assertRaises(ValueError):
            self.convert(data)


class TestGetProtectedTiles(unittest.TestCase):
    """Buffer expansion around reachable tile set."""

    def setUp(self) -> None:
        from .reachability_augment_utils import get_protected_tiles
        self.expand = get_protected_tiles

    def test_single_tile_buffer_1(self) -> None:
        """A single tile at (5, 5) with buffer=1 produces a 3×3 patch."""
        protected = self.expand({(5, 5)}, buffer=1)
        expected = {(c, r) for c in range(4, 7) for r in range(4, 7)}
        self.assertEqual(protected, expected)

    def test_clamps_to_grid_boundaries(self) -> None:
        """Buffer cannot extend outside [0,41] × [0,22]."""
        protected = self.expand({(0, 0)}, buffer=1)
        for col, row in protected:
            self.assertGreaterEqual(col, 0)
            self.assertGreaterEqual(row, 0)
            self.assertLessEqual(col, 41)
            self.assertLessEqual(row, 22)

    def test_zero_buffer_is_identity(self) -> None:
        reachable = {(3, 3), (4, 4), (10, 10)}
        self.assertEqual(self.expand(reachable, buffer=0), reachable)

    def test_reachable_subset_of_protected(self) -> None:
        reachable = {(10, 10), (11, 11)}
        protected = self.expand(reachable, buffer=2)
        for t in reachable:
            self.assertIn(t, protected)


class TestGetReachableBoundingBox(unittest.TestCase):
    """Bounding box of a tile set."""

    def setUp(self) -> None:
        from .reachability_augment_utils import get_reachable_bounding_box
        self.bbox = get_reachable_bounding_box

    def test_single_tile(self) -> None:
        self.assertEqual(self.bbox({(7, 3)}), (7, 3, 7, 3))

    def test_multiple_tiles(self) -> None:
        tiles = {(2, 1), (10, 5), (4, 3)}
        self.assertEqual(self.bbox(tiles), (2, 1, 10, 5))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.bbox(set())


# ---------------------------------------------------------------------------
# Layer 2: Behavioural unit tests (graph building required)
# ---------------------------------------------------------------------------


class TestUnreachableTileRandomizer(unittest.TestCase):
    """Verify that only unreachable tiles are modified."""

    def setUp(self) -> None:
        from .unreachable_tile_randomizer import UnreachableTileRandomizer
        # Fixed seed for deterministic tests.
        self.aug = UnreachableTileRandomizer(seed=0)

    def test_output_has_correct_tile_count(self) -> None:
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(len(result["tiles"]), 966)

    def test_protected_tiles_unchanged(self) -> None:
        """Tiles in the reachable + buffer zone must not change."""
        from .reachability_augment_utils import get_reachable_tiles, get_protected_tiles

        data = _make_walled_attract_data()
        reachable = get_reachable_tiles(data)
        protected = get_protected_tiles(reachable, buffer=1)

        result = self.aug.augment(data)

        for col, row in protected:
            idx = col + row * 42
            self.assertEqual(
                result["tiles"][idx],
                data["tiles"][idx],
                f"Protected tile at ({col},{row}) was modified",
            )

    def test_unreachable_tiles_can_be_modified(self) -> None:
        """Tiles outside the protected zone should be eligible for randomization.

        With a fixed seed and 34 tile types available, at least some tiles in
        the large unreachable region (cols 10-41) should differ from the
        original all-zero tiles.
        """
        data = _make_walled_attract_data()
        result = self.aug.augment(data)

        changed = sum(
            1
            for row in range(23)
            for col in range(10, 42)
            if result["tiles"][col + row * 42] != data["tiles"][col + row * 42]
        )
        self.assertGreater(changed, 0, "No unreachable tiles were randomized")

    def test_non_tile_fields_unchanged(self) -> None:
        """entities, ninja_spawn, header, and demo_data must not change."""
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(result["ninja_spawn"], data["ninja_spawn"])
        self.assertEqual(result["demo_data"], data["demo_data"])
        self.assertEqual(result["header"], data["header"])
        self.assertEqual(result["entities"], data["entities"])

    def test_wrong_tile_count_raises(self) -> None:
        data = _make_minimal_attract_data()
        data["tiles"] = data["tiles"][:500]
        with self.assertRaises(ValueError):
            self.aug.augment(data)

    def test_input_not_mutated(self) -> None:
        """The augmentation must not modify the input AttractData."""
        data = _make_walled_attract_data()
        original_tiles = list(data["tiles"])
        self.aug.augment(data)
        self.assertEqual(data["tiles"], original_tiles)


class TestReachableAreaTranslator(unittest.TestCase):
    """Verify that translation shifts the geometry and coordinates correctly."""

    def setUp(self) -> None:
        from .reachable_area_translator import ReachableAreaTranslator
        self.aug = ReachableAreaTranslator(seed=42)

    def test_output_has_correct_tile_count(self) -> None:
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(len(result["tiles"]), 966)

    def test_demo_data_unchanged(self) -> None:
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(result["demo_data"], data["demo_data"])

    def test_header_unchanged(self) -> None:
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(result["header"], data["header"])

    def test_spawn_shifted_by_tile_offset(self) -> None:
        """Ninja spawn must shift by exactly (dx * 4, dy * 4) map-data units."""
        from .reachability_augment_utils import (
            get_reachable_tiles,
            get_protected_tiles,
            get_reachable_bounding_box,
            _MAP_UNITS_PER_TILE,
        )

        data = _make_walled_attract_data()
        orig_spawn = data["ninja_spawn"]

        # Run augmentation and infer the offset from the spawn shift.
        result = self.aug.augment(data)
        new_spawn = result["ninja_spawn"]

        dx_units = new_spawn[0] - orig_spawn[0]
        dy_units = new_spawn[1] - orig_spawn[1]

        # dx and dy must be multiples of _MAP_UNITS_PER_TILE (4).
        self.assertEqual(dx_units % _MAP_UNITS_PER_TILE, 0)
        self.assertEqual(dy_units % _MAP_UNITS_PER_TILE, 0)

        dx_tiles = dx_units // _MAP_UNITS_PER_TILE
        dy_tiles = dy_units // _MAP_UNITS_PER_TILE

        # The shifted bounding box must still lie within [0,41] × [0,22].
        reachable = get_reachable_tiles(data)
        protected = get_protected_tiles(reachable, buffer=1)
        min_col, min_row, max_col, max_row = get_reachable_bounding_box(protected)

        self.assertGreaterEqual(min_col + dx_tiles, 0)
        self.assertLessEqual(max_col + dx_tiles, 41)
        self.assertGreaterEqual(min_row + dy_tiles, 0)
        self.assertLessEqual(max_row + dy_tiles, 22)

    def test_entity_coords_shifted_consistently(self) -> None:
        """Entity coordinates shift by the same offset as ninja spawn."""
        data = _make_walled_attract_data()
        orig_spawn = data["ninja_spawn"]
        orig_entities = data["entities"]

        result = self.aug.augment(data)
        new_spawn = result["ninja_spawn"]

        dx_units = new_spawn[0] - orig_spawn[0]
        dy_units = new_spawn[1] - orig_spawn[1]

        for orig_ent, new_ent in zip(orig_entities, result["entities"]):
            self.assertEqual(new_ent["x"], orig_ent["x"] + dx_units)
            self.assertEqual(new_ent["y"], orig_ent["y"] + dy_units)

    def test_tiles_in_translated_region_are_preserved(self) -> None:
        """Tiles that were in the bounding box must appear in the new position."""
        from .reachability_augment_utils import (
            get_reachable_tiles,
            get_protected_tiles,
            get_reachable_bounding_box,
            _MAP_UNITS_PER_TILE,
        )

        data = _make_walled_attract_data()
        orig_tiles = data["tiles"]

        reachable = get_reachable_tiles(data)
        protected = get_protected_tiles(reachable, buffer=1)
        min_col, min_row, max_col, max_row = get_reachable_bounding_box(protected)

        result = self.aug.augment(data)
        new_tiles = result["tiles"]
        new_spawn = result["ninja_spawn"]
        orig_spawn = data["ninja_spawn"]

        dx_tiles = (new_spawn[0] - orig_spawn[0]) // _MAP_UNITS_PER_TILE
        dy_tiles = (new_spawn[1] - orig_spawn[1]) // _MAP_UNITS_PER_TILE

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                new_col = col + dx_tiles
                new_row = row + dy_tiles
                if 0 <= new_col < 42 and 0 <= new_row < 23:
                    src_idx = col + row * 42
                    dst_idx = new_col + new_row * 42
                    self.assertEqual(
                        new_tiles[dst_idx],
                        orig_tiles[src_idx],
                        f"Tile at ({col},{row}) → ({new_col},{new_row}) was not copied correctly",
                    )

    def test_non_identity_translation(self) -> None:
        """With room to translate, the position must actually change."""
        data = _make_walled_attract_data()
        result = self.aug.augment(data)
        self.assertNotEqual(
            result["ninja_spawn"],
            data["ninja_spawn"],
            "ReachableAreaTranslator returned identity translation for a level "
            "that has room to move",
        )

    def test_no_op_when_bbox_fills_grid(self) -> None:
        """When the bounding box spans the full grid, return unchanged data."""
        from .reachable_area_translator import ReachableAreaTranslator

        # All tiles open → reachable area is the entire 42×23 grid.
        data = _make_minimal_attract_data(tiles=[0] * 966)
        aug = ReachableAreaTranslator(seed=0)
        result = aug.augment(data)
        # Tiles may differ (filled with random) but spawn/entities/demo are same.
        self.assertEqual(result["ninja_spawn"], data["ninja_spawn"])
        self.assertEqual(result["demo_data"], data["demo_data"])

    def test_wrong_tile_count_raises(self) -> None:
        data = _make_minimal_attract_data()
        data["tiles"] = data["tiles"][:500]
        with self.assertRaises(ValueError):
            self.aug.augment(data)

    def test_input_not_mutated(self) -> None:
        data = _make_walled_attract_data()
        original_tiles = list(data["tiles"])
        original_spawn = data["ninja_spawn"]
        self.aug.augment(data)
        self.assertEqual(data["tiles"], original_tiles)
        self.assertEqual(data["ninja_spawn"], original_spawn)

    def test_different_seeds_give_different_offsets(self) -> None:
        """Two instances with different seeds should (almost certainly) disagree."""
        from .reachable_area_translator import ReachableAreaTranslator

        data = _make_walled_attract_data()
        results = [ReachableAreaTranslator(seed=s).augment(data) for s in range(5)]
        spawns = [r["ninja_spawn"] for r in results]
        unique_spawns = set(spawns)
        # With 32+ valid dx values, 5 seeds are very likely to give ≥2 distinct positions.
        self.assertGreater(len(unique_spawns), 1)


# ---------------------------------------------------------------------------
# Layer 3: End-to-end simulator tests (requires nclone_native + fixture)
# ---------------------------------------------------------------------------


class TestReachabilityAugsEndToEnd(unittest.TestCase):
    """Run augmented replays through the simulator and confirm they win.

    Automatically skipped if nclone_native is not installed or no fixture
    .replay files are found.
    """

    FIXTURE_CANDIDATES = [
        Path("/home/tetra/projects/nclone/datasets/000basics"),
    ]

    @classmethod
    def _find_fixture(cls) -> Optional[Path]:
        for candidate_dir in cls.FIXTURE_CANDIDATES:
            if candidate_dir.is_dir():
                for f in sorted(candidate_dir.iterdir()):
                    if f.suffix == ".replay":
                        return f
        return None

    def setUp(self) -> None:
        try:
            import nclone_native  # noqa: F401
        except ImportError:
            self.skipTest("nclone_native not installed")

        self.fixture = self._find_fixture()
        if self.fixture is None:
            self.skipTest("No .replay fixture found")

        # Skip if the fixture itself does not succeed (sanity check).
        from .verify_augmentation import verify_replay_file
        orig = verify_replay_file(self.fixture)
        if not orig.success:
            self.skipTest(
                f"Fixture {self.fixture.name} does not succeed "
                f"(frames={orig.frames}); cannot validate augmentations"
            )

    def _augment_and_verify(self, aug_class, **kwargs) -> None:
        """Helper: run augmentation on the fixture and verify the output wins."""
        from .attract_augmentor import AttractAugmentor
        from .verify_augmentation import verify_replay_file

        aug = aug_class(**kwargs)
        with tempfile.TemporaryDirectory() as out_dir:
            augmentor = AttractAugmentor([aug])
            written = augmentor.augment_file(str(self.fixture), out_dir)
            if not written:
                self.skipTest(
                    f"augment_file produced no output for {self.fixture.name}"
                )
            result = verify_replay_file(written[0])
            self.assertTrue(
                result.success,
                f"{aug_class.__name__} augmented replay did not win: "
                f"{result.error} (frames={result.frames})",
            )

    def test_unreachable_tile_randomizer_wins(self) -> None:
        from .unreachable_tile_randomizer import UnreachableTileRandomizer
        self._augment_and_verify(UnreachableTileRandomizer, seed=7)

    def test_reachable_area_translator_wins(self) -> None:
        from .reachable_area_translator import ReachableAreaTranslator
        self._augment_and_verify(ReachableAreaTranslator, seed=7)


if __name__ == "__main__":
    unittest.main()
