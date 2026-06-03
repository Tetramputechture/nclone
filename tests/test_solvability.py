"""Tests for the reachability-based solvability gate."""

from __future__ import annotations

import pytest

from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from nclone.map_generation.map import Map
from nclone.map_generation.solvability import is_map_solvable
from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation.generator_configs import get_bucket_entries

NEW_BUCKETS = ["CURVES", "MULTI_JUMP", "COMPOSITE", "PRECISION"]


def test_simple_flat_level_is_solvable():
    m = Map(seed=0)
    m.reset()
    m.set_tiles_bulk([0] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
    floor_row = 15
    for x in range(MAP_TILE_WIDTH):
        for y in range(floor_row, MAP_TILE_HEIGHT - 1):
            m.set_tile(x, y, 1)
    m.set_ninja_spawn(3, floor_row - 1, 1)
    m.add_entity(3, 30, floor_row - 1, 0, 0, 26, floor_row - 1)
    assert is_map_solvable(m.map_data()) is True


def test_sealed_switch_is_unsolvable():
    """Switch/door buried in solid with the ninja in a sealed pocket -> False."""
    m = Map(seed=0)
    m.reset()
    m.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
    m.set_tile(3, 3, 0)
    m.set_tile(3, 4, 0)
    m.set_ninja_spawn(3, 3, 1)
    m.add_entity(3, 30, 10, 0, 0, 28, 10)  # buried goals
    assert is_map_solvable(m.map_data()) is False


@pytest.mark.parametrize("bucket", NEW_BUCKETS)
def test_new_bucket_presets_pass_gate(bucket):
    """Every new-bucket preset is solvable-by-construction for sampled seeds."""
    for tier in (1, 2, 3, 4):
        for gen_type, preset, _w in get_bucket_entries(bucket, tier):
            for seed in (0, 42):
                gen = GeneratorFactory.create_from_preset(gen_type, preset, seed=seed)
                gen.generate(seed=seed)
                assert is_map_solvable(gen.map_data()), (
                    f"{bucket} T{tier} {gen_type}:{preset} seed={seed} not solvable"
                )
