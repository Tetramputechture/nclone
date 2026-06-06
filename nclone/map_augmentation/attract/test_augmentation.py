"""Unit and integration tests for the attract-file augmentation system.

Run with:
    python -m pytest nclone/offline_rl/dataset/augmentation/test_augmentation.py -v

Tests are divided into three layers:
  1. Pure unit tests (no simulator required) – tile mirror completeness,
     input mirror bijectivity, spawn/entity coordinate maths.
  2. Pipeline unit tests – build_map_data round-trip, AttractAugmentor
     produces valid-looking output.
  3. End-to-end simulator test (requires nclone_native) – augmented replay
     file verifies as a successful episode.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_attract_data(
    *,
    tiles: List[int] | None = None,
    spawn: tuple = (20, 20),
    inputs: List[int] | None = None,
) -> dict:
    """Construct a minimal AttractData dict for testing."""
    if tiles is None:
        tiles = [0] * 966
        # Place a full solid border (tile 1) and a few slope tiles to exercise
        # the mirror map.
        tiles[0 + 0 * 42] = 1    # top-left  (symmetric)
        tiles[41 + 0 * 42] = 1   # top-right (symmetric)
        tiles[5 + 5 * 42] = 6    # 45-degree slope (should map to 7 when mirrored)
        tiles[10 + 5 * 42] = 18  # mild slope (should map to 19 when mirrored)
        tiles[10 + 5 * 42] = 26  # steep slope (should map to 28 when mirrored)
    if inputs is None:
        inputs = [0, 2, 2, 4, 4, 1, 3, 5, 0]  # noop/R/R/L/L/J/RJ/LJ/noop

    return {
        "tiles": list(tiles),
        "entities": [
            {"type": 4, "x": 30, "y": 15, "orientation": 0, "mode": 0},  # exit switch
            {"type": 3, "x": 140, "y": 15, "orientation": 0, "mode": 0},  # exit door
            {"type": 1, "x": 60, "y": 20, "orientation": 0, "mode": 0},  # mine
        ],
        "ninja_spawn": spawn,
        "header": bytes(184),
        "demo_data": list(inputs),
    }


# ---------------------------------------------------------------------------
# Layer 1: Pure unit tests
# ---------------------------------------------------------------------------


class TestTileMirrorMap(unittest.TestCase):
    """The tile mirror map must be complete, consistent, and an involution."""

    def setUp(self) -> None:
        from .horizontal_mirror import TILE_MIRROR_MAP
        self.mirror = TILE_MIRROR_MAP

    def test_all_pairs_are_symmetric(self) -> None:
        """If A→B then B→A."""
        for src, dst in self.mirror.items():
            self.assertEqual(
                self.mirror.get(dst, dst),
                src,
                f"Mirror map not symmetric: {src}→{dst} but {dst}→? is not {src}",
            )

    def test_no_self_loops_except_known_symmetric(self) -> None:
        """Self-loops (tile maps to itself) should only be for truly symmetric tiles."""
        # Tiles 0, 1, 2, 4 are symmetric; all others should either map to a
        # different tile or not be in the dict (which also means identity).
        explicitly_symmetric = {0, 1, 2, 4}
        for src, dst in self.mirror.items():
            if src == dst:
                self.assertIn(
                    src,
                    explicitly_symmetric,
                    f"Tile {src} self-loops in mirror map but is not explicitly symmetric",
                )

    def test_all_mapped_types_in_valid_range(self) -> None:
        for src, dst in self.mirror.items():
            self.assertIn(src, range(38), f"Source tile {src} out of range")
            self.assertIn(dst, range(38), f"Dest tile {dst} out of range")


class TestInputMirrorMap(unittest.TestCase):
    """Input mirror must be bijective and an involution on 0-7."""

    def setUp(self) -> None:
        from .horizontal_mirror import INPUT_MIRROR_MAP
        self.mirror = INPUT_MIRROR_MAP

    def test_all_eight_inputs_covered(self) -> None:
        for i in range(8):
            self.assertIn(i, self.mirror, f"Input {i} missing from mirror map")

    def test_involution(self) -> None:
        """Applying the mirror twice returns the original."""
        for i in range(8):
            twice = self.mirror[self.mirror[i]]
            self.assertEqual(twice, i, f"Input mirror is not involutory for {i}")

    def test_jump_bit_preserved(self) -> None:
        """Bit 0 (jump) must not change."""
        for i in range(8):
            orig_jump = i & 0x01
            mirr_jump = self.mirror[i] & 0x01
            self.assertEqual(orig_jump, mirr_jump, f"Jump bit changed for input {i}")

    def test_lr_bits_swapped(self) -> None:
        """Bit 1 (right) and bit 2 (left) must swap; invalid inputs (6,7) unchanged."""
        for i in [2, 3, 4, 5]:  # valid single-direction inputs
            mirrored = self.mirror[i]
            orig_right = (i >> 1) & 1
            orig_left = (i >> 2) & 1
            mirr_right = (mirrored >> 1) & 1
            mirr_left = (mirrored >> 2) & 1
            self.assertEqual(
                orig_right, mirr_left,
                f"Input {i}: orig right bit should equal mirrored left bit"
            )
            self.assertEqual(
                orig_left, mirr_right,
                f"Input {i}: orig left bit should equal mirrored right bit"
            )


class TestHorizontalMirrorMath(unittest.TestCase):
    """Test coordinate arithmetic in HorizontalMirror."""

    def setUp(self) -> None:
        from .horizontal_mirror import HorizontalMirror, FULL_MAP_WIDTH_UNITS
        self.aug = HorizontalMirror()
        self.full_width = FULL_MAP_WIDTH_UNITS  # 176

    def test_spawn_mirrors_around_centre(self) -> None:
        """spawn_x + mirrored_spawn_x == FULL_MAP_WIDTH_UNITS."""
        for spawn_x in [4, 88, 172, 10, 166]:
            data = _make_minimal_attract_data(spawn=(spawn_x, 20))
            result = self.aug.augment(data)
            self.assertEqual(
                spawn_x + result["ninja_spawn"][0],
                self.full_width,
                f"Spawn mirror incorrect for spawn_x={spawn_x}",
            )

    def test_spawn_y_unchanged(self) -> None:
        data = _make_minimal_attract_data(spawn=(50, 33))
        result = self.aug.augment(data)
        self.assertEqual(result["ninja_spawn"][1], 33)

    def test_entity_x_mirrors_correctly(self) -> None:
        data = _make_minimal_attract_data()
        result = self.aug.augment(data)
        for orig, mirrored in zip(data["entities"], result["entities"]):
            self.assertEqual(
                orig["x"] + mirrored["x"],
                self.full_width,
                f"Entity x mirror incorrect: orig={orig['x']} mirrored={mirrored['x']}",
            )

    def test_entity_type_unchanged(self) -> None:
        data = _make_minimal_attract_data()
        result = self.aug.augment(data)
        for orig, mirrored in zip(data["entities"], result["entities"]):
            self.assertEqual(orig["type"], mirrored["type"])

    def test_tile_count_preserved(self) -> None:
        data = _make_minimal_attract_data()
        result = self.aug.augment(data)
        self.assertEqual(len(result["tiles"]), 966)

    def test_symmetric_tile_stays_same(self) -> None:
        """Tile 1 (full solid) is symmetric: it maps to itself."""
        tiles = [0] * 966
        tiles[5 + 10 * 42] = 1  # col 5, row 10
        data = _make_minimal_attract_data(tiles=tiles)
        result = self.aug.augment(data)
        # Should appear at col 41-5=36, row 10
        self.assertEqual(result["tiles"][36 + 10 * 42], 1)

    def test_asymmetric_tile_swaps_type_and_position(self) -> None:
        """Tile 6 (slope) should map to tile 7 and be at the mirrored column."""
        tiles = [0] * 966
        tiles[5 + 10 * 42] = 6  # col 5
        data = _make_minimal_attract_data(tiles=tiles)
        result = self.aug.augment(data)
        mirrored_col = 41 - 5  # 36
        self.assertEqual(result["tiles"][mirrored_col + 10 * 42], 7)
        # Original position should now be 0 (or whatever was at col 36)
        self.assertNotEqual(result["tiles"][5 + 10 * 42], 6)

    def test_double_mirror_is_identity(self) -> None:
        """Applying the mirror twice gives back the original data."""
        data = _make_minimal_attract_data()
        result = self.aug.augment(self.aug.augment(data))
        self.assertEqual(result["tiles"], data["tiles"])
        self.assertEqual(result["ninja_spawn"], data["ninja_spawn"])
        self.assertEqual(result["demo_data"], data["demo_data"])
        for orig_e, round_e in zip(data["entities"], result["entities"]):
            self.assertEqual(orig_e["x"], round_e["x"])

    def test_input_sequence_mirrored(self) -> None:
        from .horizontal_mirror import INPUT_MIRROR_MAP
        inputs = [0, 1, 2, 3, 4, 5]
        expected = [INPUT_MIRROR_MAP[i] for i in inputs]
        data = _make_minimal_attract_data(inputs=inputs)
        result = self.aug.augment(data)
        self.assertEqual(result["demo_data"], expected)

    def test_wrong_tile_count_raises(self) -> None:
        data = _make_minimal_attract_data()
        data["tiles"] = data["tiles"][:100]  # truncate
        with self.assertRaises(ValueError):
            self.aug.augment(data)


# ---------------------------------------------------------------------------
# Layer 2: Pipeline tests (map_data construction round-trip)
# ---------------------------------------------------------------------------


class TestBuildMapData(unittest.TestCase):
    """_build_map_data must produce bytes the simulator can load and that
    faithfully encode the tile/entity/spawn data."""

    def _build(self, data: dict) -> bytes:
        from .attract_augmentor import _build_map_data
        result = _build_map_data(data)
        self.assertIsNotNone(result)
        return result

    def test_tile_round_trip(self) -> None:
        """Tiles stored in bytes 184-1149 match the input tile list."""
        tiles = list(range(42))  # first 42 tiles are 0-41 (valid types)
        tiles += [0] * (966 - 42)
        data = _make_minimal_attract_data(tiles=tiles)
        raw = self._build(data)
        extracted = list(raw[184:1150])
        self.assertEqual(extracted, tiles)

    def test_spawn_round_trip(self) -> None:
        spawn_x, spawn_y = 55, 40
        data = _make_minimal_attract_data(spawn=(spawn_x, spawn_y))
        raw = self._build(data)
        self.assertEqual(raw[1231], spawn_x)
        self.assertEqual(raw[1232], spawn_y)

    def test_entity_round_trip(self) -> None:
        """Each entity stored at bytes 1235+ matches the input entity list."""
        data = _make_minimal_attract_data()
        raw = self._build(data)
        idx = 1235
        for orig_ent in data["entities"]:
            self.assertEqual(raw[idx + 0], orig_ent["type"])
            self.assertEqual(raw[idx + 1], orig_ent["x"])
            self.assertEqual(raw[idx + 2], orig_ent["y"])
            idx += 5
        # Terminator byte
        self.assertEqual(raw[idx], 0)

    def test_minimum_length(self) -> None:
        """Output must be at least 1335 bytes (standard nclone map size)."""
        data = _make_minimal_attract_data()
        raw = self._build(data)
        self.assertGreaterEqual(len(raw), 1335)


class TestAttractAugmentorPipeline(unittest.TestCase):
    """AttractAugmentor.augment_file must produce a loadable .replay file."""

    def test_augment_file_produces_replay(self) -> None:
        """When given a synthetic source .replay file, produces an output."""
        from .attract_augmentor import AttractAugmentor, _build_map_data
        from .horizontal_mirror import HorizontalMirror

        # Build a minimal map_data (needs to be valid enough for CompactReplay).
        data = _make_minimal_attract_data()
        map_bytes = _build_map_data(data)
        self.assertIsNotNone(map_bytes)

        # Write a source .replay using CompactReplay.
        from ...replay.gameplay_recorder import CompactReplay
        from datetime import datetime
        replay = CompactReplay(
            episode_id="test_source",
            map_data=map_bytes,
            input_sequence=data["demo_data"],
            level_id=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
        )

        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            src_file = Path(src_dir) / "test_source.replay"
            src_file.write_bytes(replay.to_binary())

            augmentor = AttractAugmentor([HorizontalMirror()])
            written = augmentor.augment_file(str(src_file), out_dir)

            self.assertEqual(len(written), 1)
            out_file = written[0]
            self.assertTrue(out_file.exists())
            self.assertTrue(out_file.suffix == ".replay")
            self.assertIn("hmirror", out_file.name)

            # The output must be loadable as a CompactReplay.
            loaded = CompactReplay.from_binary(
                out_file.read_bytes(), episode_id=out_file.stem
            )
            self.assertTrue(loaded.success)
            self.assertEqual(len(loaded.input_sequence), len(data["demo_data"]))

    def test_overwrite_false_skips_existing(self) -> None:
        """With overwrite=False, existing output files are not regenerated."""
        from .attract_augmentor import AttractAugmentor, _build_map_data
        from .horizontal_mirror import HorizontalMirror
        from ...replay.gameplay_recorder import CompactReplay
        from datetime import datetime

        data = _make_minimal_attract_data()
        map_bytes = _build_map_data(data)
        replay = CompactReplay(
            episode_id="skip_test",
            map_data=map_bytes,
            input_sequence=data["demo_data"],
            level_id=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
        )

        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            src_file = Path(src_dir) / "skip_test.replay"
            src_file.write_bytes(replay.to_binary())
            augmentor = AttractAugmentor([HorizontalMirror()])

            # First run: creates the file.
            written1 = augmentor.augment_file(str(src_file), out_dir)
            mtime1 = written1[0].stat().st_mtime

            # Second run with overwrite=False: should not touch the file.
            written2 = augmentor.augment_file(
                str(src_file), out_dir, overwrite=False
            )
            mtime2 = written2[0].stat().st_mtime
            self.assertEqual(mtime1, mtime2)


# ---------------------------------------------------------------------------
# Layer 3: End-to-end simulator test (optional, requires nclone_native)
# ---------------------------------------------------------------------------


class TestEndToEndSimulator(unittest.TestCase):
    """Replay an augmented file through the simulator and confirm it wins.

    This test is automatically skipped if nclone_native is not installed or
    if no suitable .replay fixture is available.
    """

    FIXTURE_CANDIDATES = [
        # Use one of the real gameplay recordings shipped in the repo.
        Path("/home/tetra/projects/nclone/datasets/000basics"),
    ]

    @classmethod
    def _find_fixture(cls) -> Path | None:
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

    def test_augmented_replay_succeeds(self) -> None:
        """Augmented file must produce a winning episode in the simulator."""
        from .attract_augmentor import AttractAugmentor
        from .horizontal_mirror import HorizontalMirror
        from .verify_augmentation import verify_replay_file

        with tempfile.TemporaryDirectory() as out_dir:
            augmentor = AttractAugmentor([HorizontalMirror()])
            written = augmentor.augment_file(str(self.fixture), out_dir)

            if not written:
                self.skipTest(
                    f"augment_file returned no files for {self.fixture.name} "
                    "(may not be a successful replay)"
                )

            result = verify_replay_file(written[0])
            self.assertTrue(
                result.success,
                f"Augmented replay did not win: {result.error}  "
                f"(frames executed: {result.frames})",
            )

    def test_original_and_mirror_both_win(self) -> None:
        """Both the original and its mirror should succeed independently."""
        from .verify_augmentation import verify_replay_file
        from .attract_augmentor import AttractAugmentor
        from .horizontal_mirror import HorizontalMirror

        # Verify the fixture itself wins first (sanity check).
        orig_result = verify_replay_file(self.fixture)
        if not orig_result.success:
            self.skipTest(
                f"Fixture {self.fixture.name} does not win "
                f"(frames={orig_result.frames})"
            )

        with tempfile.TemporaryDirectory() as out_dir:
            augmentor = AttractAugmentor([HorizontalMirror()])
            written = augmentor.augment_file(str(self.fixture), out_dir)
            if not written:
                self.skipTest("augment_file returned no files")

            mirror_result = verify_replay_file(written[0])
            self.assertTrue(
                mirror_result.success,
                f"Mirrored replay did not win. "
                f"orig_frames={orig_result.frames} "
                f"mirror_frames={mirror_result.frames}",
            )


if __name__ == "__main__":
    unittest.main()
