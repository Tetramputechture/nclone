"""Structure-descriptor tests: it must separate sparse open rooms from rich levels."""

from __future__ import annotations

import pytest

from nclone.levelgen.structure import (
    describe_tiles,
    describe_grid,
    grid_from_tiles,
    structure_bin,
    GRID_W,
    GRID_H,
)


def _empty_bordered_room():
    tiles = [0] * (GRID_W * GRID_H)
    for x in range(GRID_W):
        tiles[x] = 1
        tiles[x + (GRID_H - 1) * GRID_W] = 1
    for y in range(GRID_H):
        tiles[y * GRID_W] = 1
        tiles[GRID_W - 1 + y * GRID_W] = 1
    return tiles


def _room_with_platforms(n_platforms=4):
    tiles = _empty_bordered_room()
    # add floating 3-wide platforms in the interior
    for k in range(n_platforms):
        gy = 4 + 3 * k
        gx = 5 + 4 * k
        if gy >= GRID_H - 1:
            break
        for dx in range(3):
            tiles[(gx + dx) + gy * GRID_W] = 1
    return tiles


def test_empty_room_scores_zero_and_flags_open():
    d = describe_tiles(_empty_bordered_room())
    assert d.structure_score == 0.0
    assert d.is_open_room is True
    assert d.n_platform_components == 0
    assert d.standable_surfaces == 0
    assert d.n_air_components == 1


def test_platforms_increase_score_and_clear_open_flag():
    empty = describe_tiles(_empty_bordered_room())
    rich = describe_tiles(_room_with_platforms(4))
    assert rich.structure_score > empty.structure_score
    assert rich.is_open_room is False
    assert rich.n_platform_components >= 3
    assert rich.standable_surfaces >= 3


def test_score_is_monotone_in_platform_count():
    s1 = describe_tiles(_room_with_platforms(1)).structure_score
    s4 = describe_tiles(_room_with_platforms(4)).structure_score
    assert s4 >= s1


def test_score_bounded_unit_interval():
    for tiles in (_empty_bordered_room(), _room_with_platforms(4), [1] * (GRID_W * GRID_H)):
        d = describe_tiles(tiles)
        assert 0.0 <= d.structure_score <= 1.0


def test_structure_bin_range():
    assert structure_bin(0.0) == 0
    assert structure_bin(1.0) == 3
    assert structure_bin(0.5, n_bins=4) == 2
    for s in (0.0, 0.1, 0.4, 0.7, 1.0):
        assert 0 <= structure_bin(s) <= 3


def test_grid_roundtrip_shape():
    g = grid_from_tiles(_empty_bordered_room())
    assert len(g) == GRID_H and len(g[0]) == GRID_W
    assert describe_grid(g).structure_score == 0.0


@pytest.mark.parametrize("preset", [("maze", "large"), ("composite", "long_chain_t4")])
def test_real_procgen_levels_are_not_open_rooms(preset):
    pytest.importorskip("nclone_native")
    from nclone.map_generation.generator_factory import GeneratorFactory

    bucket, p = preset
    g = GeneratorFactory.create_from_preset(bucket, p, 0)
    g.generate(seed=7)
    g.clean_placement()
    from nclone.levelgen.structure import describe_map_data

    d = describe_map_data(g.map_data())
    assert d.is_open_room is False
    assert d.structure_score > 0.1
