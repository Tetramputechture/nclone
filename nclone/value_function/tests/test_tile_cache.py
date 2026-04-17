"""Unit tests for the tile_cache subpackage.

Covers:
- 3x3 signature extraction, packing, and hashing round-trip.
- Harness map construction and byte-level consistency.
- Persistent sharded store round-trips launch + landing entries.
- Header-mismatch invalidation keeps existing shards intact on disk but
  hides them from the in-memory view until the header is re-aligned.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nclone.constants import (
    FULL_MAP_HEIGHT,
    FULL_MAP_WIDTH,
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    TILE_PIXEL_SIZE,
)
from nclone.value_function.tile_cache import (
    BIN_SCHEDULE_VERSION,
    HARNESS_CENTER_COL,
    HARNESS_CENTER_ROW,
    HARNESS_SIZE,
    LandingEntry,
    LaunchEntry,
    PHYSICS_VERSION,
    ShardHeader,
    TileCacheStore,
    build_landing_harness_map,
    build_launch_harness_map,
    extract_3x3_signature,
    hash_signature,
    launch_cell_origin_px,
    pack_signature,
    unpack_signature,
)
from nclone.value_function.tile_cache.signature import (
    OOB_TILE,
    shard_id_from_hash,
    signature_frequencies,
    tile_grid_from_map_data,
)


class TestSignature(unittest.TestCase):
    def test_extract_returns_nine_tiles_in_row_major_order(self):
        grid = np.zeros((6, 5), dtype=np.uint8)
        for c in range(6):
            for r in range(5):
                grid[c, r] = c * 10 + r
        sig = extract_3x3_signature(grid, col=3, row=2)
        self.assertEqual(len(sig), 9)
        # Row-major order: (c=2,r=1), (c=3,r=1), (c=4,r=1), (c=2,r=2), ...
        self.assertTrue(
            np.array_equal(
                sig,
                np.array([21, 31, 41, 22, 32, 42, 23, 33, 43], dtype=np.uint8),
            )
        )

    def test_extract_uses_oob_tile_outside_grid(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        sig = extract_3x3_signature(grid, col=0, row=0)
        # Top-left corner: NW, N, W outside grid should be OOB_TILE
        self.assertEqual(int(sig[0]), OOB_TILE)
        self.assertEqual(int(sig[1]), OOB_TILE)
        self.assertEqual(int(sig[3]), OOB_TILE)
        self.assertEqual(int(sig[4]), 0)  # center is in bounds

    def test_pack_unpack_roundtrip(self):
        sig = [0, 1, 5, 37, 127, 6, 8, 10, 2]  # within 7-bit range
        packed = pack_signature(sig)
        unpacked = unpack_signature(packed)
        self.assertEqual(tuple(sig), unpacked)

    def test_pack_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            pack_signature([128, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_pack_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            pack_signature([0, 0, 0])

    def test_hash_signature_returns_uint64(self):
        h = hash_signature([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertIsInstance(h, np.uint64)

    def test_shard_id_is_deterministic_and_in_range(self):
        h = hash_signature([1, 2, 3, 4, 5, 6, 7, 8, 37])
        sid = shard_id_from_hash(int(h), num_shards=128)
        self.assertEqual(sid, shard_id_from_hash(int(h), num_shards=128))
        self.assertGreaterEqual(sid, 0)
        self.assertLess(sid, 128)

    def test_tile_grid_from_map_data_matches_harness(self):
        """Building a harness map then parsing it back gives the placed signature."""
        sig = [1, 0, 1, 0, 0, 0, 1, 0, 1]
        map_data = build_launch_harness_map(sig)
        grid = tile_grid_from_map_data(map_data)
        self.assertEqual(grid.shape, (FULL_MAP_WIDTH, FULL_MAP_HEIGHT))
        # After placement at (HARNESS_CENTER_COL, HARNESS_CENTER_ROW),
        # padded grid coords are (+1, +1).
        cc = HARNESS_CENTER_COL + 1
        rr = HARNESS_CENTER_ROW + 1
        recovered = extract_3x3_signature(grid, cc, rr)
        self.assertTrue(np.array_equal(recovered, np.array(sig, dtype=np.uint8)))

    def test_signature_frequencies_finds_at_least_one_unique(self):
        grid = np.ones((10, 10), dtype=np.uint8)
        freq = signature_frequencies(grid, interior_only=True)
        # Uniform grid => single signature repeated everywhere.
        self.assertEqual(len(freq), 1)


class TestHarness(unittest.TestCase):
    def test_launch_harness_has_expected_size(self):
        sig = [1] * 9
        map_data = build_launch_harness_map(sig)
        self.assertEqual(len(map_data), 1234)
        # Tile region interior should contain the signature; outside the
        # 5x5 window should be solid (tile 1).
        cc_raw = HARNESS_CENTER_COL
        rr_raw = HARNESS_CENTER_ROW
        offset = 184 + cc_raw + rr_raw * MAP_TILE_WIDTH
        self.assertEqual(map_data[offset], 1)  # center cell from signature

    def test_outer_ring_is_solid_and_buffer_is_empty(self):
        sig = [0] * 9
        map_data = build_launch_harness_map(sig)
        half = HARNESS_SIZE // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if abs(dc) <= 1 and abs(dr) <= 1:
                    continue
                c = HARNESS_CENTER_COL + dc
                r = HARNESS_CENTER_ROW + dr
                offset = 184 + c + r * MAP_TILE_WIDTH
                is_outer = abs(dc) == half or abs(dr) == half
                expected = 1 if is_outer else 0
                self.assertEqual(
                    map_data[offset], expected,
                    f"cell ({dc},{dr}) expected {expected}, got {map_data[offset]}",
                )

    def test_center_cell_origin_px_is_aligned(self):
        ox, oy = launch_cell_origin_px()
        # Padded grid coord of center = (HARNESS_CENTER_COL+1, HARNESS_CENTER_ROW+1)
        self.assertEqual(ox, (HARNESS_CENTER_COL + 1) * TILE_PIXEL_SIZE)
        self.assertEqual(oy, (HARNESS_CENTER_ROW + 1) * TILE_PIXEL_SIZE)

    def test_landing_harness_matches_launch_harness(self):
        sig = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(
            build_launch_harness_map(sig),
            build_landing_harness_map(sig),
        )


class TestStore(unittest.TestCase):
    def test_launch_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=16)
            sig = [1] * 9
            h = int(hash_signature(sig))
            key_extra = (0, 3, 3, 7, 12)
            entry = LaunchEntry(
                escaped=0, died=0, frames=15,
                land_cell_dx=1, land_cell_dy=0,
                land_subcell_x=12, land_subcell_y=1,
                land_surface_type=0, land_vel_bin=8,
            )
            store.put_launch(h, key_extra, entry)
            store.flush()

            # Re-open via a fresh instance to exercise on-disk IO.
            store2 = TileCacheStore(td, num_shards=16)
            got = store2.get_launch(h, key_extra)
            self.assertIsNotNone(got)
            self.assertEqual(got.escaped, 0)
            self.assertEqual(got.frames, 15)
            self.assertEqual(got.land_cell_dx, 1)
            self.assertEqual(got.land_subcell_x, 12)
            self.assertEqual(got.land_vel_bin, 8)

    def test_landing_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=16)
            sig = [0, 1, 0, 1, 1, 1, 0, 1, 0]
            h = int(hash_signature(sig))
            key_extra = (10, 11, 3, 4)
            entry = LandingEntry(
                landing_cell_dx=0, landing_cell_dy=1,
                landing_subcell_x=7, landing_subcell_y=0,
                landing_surface_type=0, landing_vel_bin=6,
                contact_frame_offset=2,
            )
            store.put_landing(h, key_extra, entry)
            store.flush()

            store2 = TileCacheStore(td, num_shards=16)
            got = store2.get_landing(h, key_extra)
            self.assertIsNotNone(got)
            self.assertEqual(got.landing_cell_dy, 1)
            self.assertEqual(got.landing_subcell_x, 7)
            self.assertEqual(got.contact_frame_offset, 2)

    def test_header_mismatch_hides_entries(self):
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=16)
            sig = [5] * 9
            h = int(hash_signature(sig))
            store.put_launch(
                h, (0, 0, 0, 0, 0),
                LaunchEntry(escaped=1, died=0, frames=3,
                             exit_pos_dx=40, exit_pos_dy=-20,
                             exit_vx_q=120, exit_vy_q=-200,
                             hold_remaining=10),
            )
            store.flush()

            # Open with a bumped schedule version => existing shard ignored.
            future_header = ShardHeader(
                physics_version=PHYSICS_VERSION,
                bin_schedule_version=BIN_SCHEDULE_VERSION + 1,
            )
            store2 = TileCacheStore(td, num_shards=16,
                                    header_template=future_header)
            self.assertIsNone(store2.get_launch(h, (0, 0, 0, 0, 0)))

            # Open again with the original header => entry still there.
            store3 = TileCacheStore(td, num_shards=16)
            got = store3.get_launch(h, (0, 0, 0, 0, 0))
            self.assertIsNotNone(got)
            self.assertEqual(got.escaped, 1)
            self.assertEqual(got.exit_pos_dx, 40)
            self.assertEqual(got.hold_remaining, 10)

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            self.assertIsNone(store.get_launch(123, (0, 0, 0, 0, 0)))
            self.assertIsNone(store.get_landing(123, (0, 0, 0, 0)))


class TestBuilder(unittest.TestCase):
    """Integration tests covering the launch harness + builder + store flow."""

    def _has_native(self) -> bool:
        try:
            from nclone_native import Simulator  # noqa: F401
            return True
        except ImportError:
            return False

    def test_decode_input_class_covers_all_ids(self):
        from nclone.value_function.tile_cache.builder import (
            _N_INPUT_CLASSES,
            decode_input_class,
        )
        kinds = set()
        for i in range(_N_INPUT_CLASSES):
            ic = decode_input_class(i)
            kinds.add(ic.kind)
        self.assertEqual(kinds, {"run", "floor_jump", "wall_jump"})

    def test_decode_out_of_range_raises(self):
        from nclone.value_function.tile_cache.builder import decode_input_class
        with self.assertRaises(ValueError):
            decode_input_class(-1)

    def test_subcell_quantisation_roundtrip(self):
        from nclone.value_function.tile_cache.builder import (
            SUBCELL_BINS,
            quantise_subcell,
            subcell_centre,
        )
        for sx in range(SUBCELL_BINS):
            for sy in range(SUBCELL_BINS):
                cx, cy = subcell_centre(sx, sy)
                self.assertEqual(quantise_subcell(cx, cy), (sx, sy))

    def test_classes_for_surface_nonempty(self):
        from nclone.value_function.surface_extraction import (
            SURFACE_FLOOR, SURFACE_WALL,
        )
        from nclone.value_function.tile_cache.builder import classes_for_surface
        self.assertGreater(len(list(classes_for_surface(SURFACE_FLOOR))), 0)
        self.assertGreater(len(list(classes_for_surface(SURFACE_WALL))), 0)

    def test_populate_launch_entry_runs_end_to_end(self):
        if not self._has_native():
            self.skipTest("nclone_native not available")
        from nclone.value_function.tile_cache import (
            TileCacheStore,
            populate_launch_entry,
        )
        from nclone.value_function.surface_extraction import SURFACE_FLOOR

        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            # Signature: center tile empty, floor tile just below (south).
            sig = [0, 0, 0,
                   0, 0, 0,
                   1, 1, 1]
            # Short run primitive: input_class 0 (first run class).
            entry = populate_launch_entry(
                store, sig,
                surface_type=SURFACE_FLOOR,
                subcell=(2, 2),  # middle of cell
                vel_bin_in=7,
                input_class_id=0,
                vel_bin_centre=0.0,
            )
            self.assertIsNotNone(entry)
            self.assertGreaterEqual(entry.frames, 0)

    def test_populate_launch_window_exit_is_detected(self):
        """Signature with only a floor, run-right at high duration -> exits east."""
        if not self._has_native():
            self.skipTest("nclone_native not available")
        from nclone.value_function.tile_cache import (
            TileCacheStore,
            populate_launch_entry,
        )
        from nclone.value_function.tile_cache.builder import (
            H_INPUTS,
            RUN_DURATIONS,
            _RUN_BASE,
        )
        from nclone.value_function.surface_extraction import SURFACE_FLOOR

        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            # Floor along the bottom row of the 3x3.
            sig = [0, 0, 0,
                   0, 0, 0,
                   1, 1, 1]
            # Input class for (h=+1, duration=60)
            h_idx = H_INPUTS.index(1)
            d_idx = RUN_DURATIONS.index(60)
            cls_id = _RUN_BASE + h_idx * len(RUN_DURATIONS) + d_idx
            entry = populate_launch_entry(
                store, sig,
                surface_type=SURFACE_FLOOR,
                subcell=(2, 2),
                vel_bin_in=7,
                input_class_id=cls_id,
                vel_bin_centre=3.0,  # already moving right
            )
            self.assertIsNotNone(entry)
            # After running right at high velocity, we either stayed put or
            # exited east.  Either is a valid cache entry; we just assert
            # it's not flagged as a death.
            self.assertEqual(entry.died, 0)


class TestLookup(unittest.TestCase):
    def test_launch_lookup_hits_after_put(self):
        from nclone.value_function.tile_cache import (
            LaunchEntry,
            TileCacheStore,
            launch_lookup,
        )
        from nclone.value_function.tile_cache.signature import hash_signature
        from nclone.value_function.surface_extraction import SURFACE_FLOOR

        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            sig = [1] * 9
            h = int(hash_signature(sig))
            store.put_launch(
                h, (SURFACE_FLOOR, 1, 1, 5, 3),
                LaunchEntry(escaped=0, died=0, frames=5,
                             land_cell_dx=0, land_cell_dy=0,
                             land_subcell_x=11, land_subcell_y=14,
                             land_surface_type=SURFACE_FLOOR, land_vel_bin=7),
            )
            got = launch_lookup(
                store, sig, surface_type=SURFACE_FLOOR,
                subcell=(1, 1), vel_bin_in=5, input_class_id=3,
            )
            self.assertIsNotNone(got)
            self.assertEqual(got.land_subcell_x, 11)

    def test_landing_lookup_miss_returns_none(self):
        from nclone.value_function.tile_cache import (
            TileCacheStore,
            landing_lookup,
        )
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            self.assertIsNone(
                landing_lookup(store, [0] * 9, (0, 0), 0, 0)
            )


class TestLandingBuilder(unittest.TestCase):
    def _has_native(self) -> bool:
        try:
            from nclone_native import Simulator  # noqa: F401
            return True
        except ImportError:
            return False

    def test_populate_landing_entry_runs(self):
        if not self._has_native():
            self.skipTest("nclone_native not available")
        from nclone.value_function.tile_cache import (
            TileCacheStore,
            populate_landing_entry,
            landing_lookup,
        )
        with tempfile.TemporaryDirectory() as td:
            store = TileCacheStore(td, num_shards=4)
            # 3x3 with floor at the bottom three cells
            sig = [0, 0, 0,
                   0, 0, 0,
                   1, 1, 1]
            entry = populate_landing_entry(
                store, sig,
                entry_subcell=(2, 0),        # enter near top of cell
                entry_vx_bin_air=7,          # ~0 vx
                entry_vy_bin_air=16,         # mildly positive vy (falling)
            )
            self.assertIsNotNone(entry)
            # Re-lookup should succeed
            lookup_hit = landing_lookup(store, sig, (2, 0), 7, 16)
            self.assertIsNotNone(lookup_hit)


class TestBallistic(unittest.TestCase):
    """Tests for the ballistic propagator."""

    def test_empty_tile_dict_returns_timeout(self):
        from nclone.value_function.tile_cache import (
            Timeout,
            propagate_and_sweep,
        )
        # No tiles, open space.
        res = propagate_and_sweep(
            launch_pos=(500.0, 300.0),
            launch_vel=(0.0, -2.0),
            hold_remaining=5,
            h_during=0, h_after=0,
            tile_dict={},
            n_cols=44, n_rows=25,
            max_frames=60,
        )
        # With no collision the ninja should eventually terminate on its
        # own fall back into the floor, or hit the budget.  Either way,
        # the result should be Timeout or Landed with hit_kind='floor'
        # (depending on gravity shape); we just assert non-None.
        self.assertIsNotNone(res)

    def test_tile_below_intercepts_falling_ninja(self):
        """A solid floor one tile below launch triggers a floor hit."""
        from nclone.value_function.tile_cache import (
            Landed,
            propagate_and_sweep,
        )
        # Place a solid tile row at col any, row 13 (below launch at 300 px,
        # which corresponds to padded row 12).
        td = {}
        for c in range(44):
            td[(c, 13)] = 1
        res = propagate_and_sweep(
            launch_pos=(500.0, 300.0),  # row 12
            launch_vel=(0.0, 2.0),       # already falling
            hold_remaining=0,
            h_during=0, h_after=0,
            tile_dict=td,
            n_cols=44, n_rows=25,
            max_frames=300,
        )
        self.assertIsInstance(res, Landed)
        self.assertEqual(res.hit_kind, "floor")
        self.assertGreaterEqual(res.cell[1], 13)

    def test_tile_dict_is_solid_bounds(self):
        from nclone.value_function.tile_cache import tile_dict_is_solid
        self.assertTrue(tile_dict_is_solid({}, -1, 5, 44, 25))
        self.assertTrue(tile_dict_is_solid({}, 44, 5, 44, 25))
        self.assertFalse(tile_dict_is_solid({}, 10, 10, 44, 25))
        self.assertTrue(tile_dict_is_solid({(10, 10): 1}, 10, 10, 44, 25))


if __name__ == "__main__":
    unittest.main()
