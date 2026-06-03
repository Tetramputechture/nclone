"""Reproducibility, entity-purity and mechanic-presence tests for the new
CURVES / MULTI_JUMP / COMPOSITE / PRECISION buckets.

Entity constraint (Layer 1): every level must contain ONLY toggle mines
(types 1/21), exactly one exit door (3) + exit switch (4), and the player.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from nclone.map_generation.generator_configs import get_bucket_entries
from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation import curve_utils as cu

TILE_HEADER_LEN = 184
TILE_GRID_LEN = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
ENTITY_SECTION_START = 1235
MINE_TYPES = (1, 21)
ALLOWED_ENTITY_TYPES = {1, 21, 3, 4}

NEW_BUCKETS = ["CURVES", "MULTI_JUMP", "COMPOSITE", "PRECISION"]


def _all_entries():
    out = []
    for bucket in NEW_BUCKETS:
        for tier in (1, 2, 3, 4):
            for gen_type, preset, _w in get_bucket_entries(bucket, tier):
                out.append((bucket, tier, gen_type, preset))
    return out


def _build(gen_type: str, preset: str, seed: int):
    gen = GeneratorFactory.create_from_preset(gen_type, preset, seed=seed)
    gen.generate(seed=seed)
    return gen


def _tile_grid(map_data: List[int]) -> List[int]:
    return list(map_data[TILE_HEADER_LEN : TILE_HEADER_LEN + TILE_GRID_LEN])


def _parse_entities(map_data: List[int]) -> List[Tuple[int, int, int]]:
    """Strict parser: exit-door pair (10 ints) or mine (5 ints); anything else
    is a mechanic-purity violation."""
    tail = list(map_data[ENTITY_SECTION_START:])
    out: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(tail):
        etype = tail[i]
        if etype == 3:
            assert i + 10 <= len(tail), "truncated exit-door record"
            out.append((3, tail[i + 1], tail[i + 2]))
            out.append((4, tail[i + 6], tail[i + 7]))
            i += 10
        elif etype in MINE_TYPES:
            assert i + 5 <= len(tail), "truncated mine record"
            out.append((etype, tail[i + 1], tail[i + 2]))
            i += 5
        else:
            raise AssertionError(
                f"unsupported entity type {etype} at tail offset {i}"
            )
    return out


@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_new_buckets_reproducible(seed):
    for _b, _t, gen_type, preset in _all_entries():
        a = _build(gen_type, preset, seed).map_data()
        b = _build(gen_type, preset, seed).map_data()
        assert a == b, f"{gen_type}:{preset} (seed={seed}) not reproducible"


def test_new_buckets_entity_purity_and_single_exit():
    for _b, _t, gen_type, preset in _all_entries():
        for seed in (0, 1, 42):
            ents = _parse_entities(_build(gen_type, preset, seed).map_data())
            types = {e[0] for e in ents}
            assert types <= ALLOWED_ENTITY_TYPES, (
                f"{gen_type}:{preset} emitted disallowed entity types {types}"
            )
            doors = [e for e in ents if e[0] == 3]
            switches = [e for e in ents if e[0] == 4]
            assert len(doors) == 1 and len(switches) == 1, (
                f"{gen_type}:{preset} got {len(doors)} doors / {len(switches)} switches"
            )


def test_new_buckets_tile_grid_nontrivial():
    for _b, _t, gen_type, preset in _all_entries():
        grid = _tile_grid(_build(gen_type, preset, 0).map_data())
        assert any(t == 0 for t in grid) and any(t != 0 for t in grid)


def test_curves_emit_curved_tiles():
    """Every CURVES level uses at least one verified walkable curve tile."""
    for tier in (1, 2, 3, 4):
        for gen_type, preset, _w in get_bucket_entries("CURVES", tier):
            grid = set(_tile_grid(_build(gen_type, preset, 0).map_data()))
            assert grid & cu.WALKABLE_CURVE_TILES, (
                f"CURVES {preset} has no curved tile (got curve-ids "
                f"{grid & set(range(10, 18))})"
            )


def test_precision_always_has_a_mine():
    for tier in (1, 2, 3, 4):
        for gen_type, preset, _w in get_bucket_entries("PRECISION", tier):
            for seed in (0, 1, 42):
                ents = _parse_entities(_build(gen_type, preset, seed).map_data())
                assert any(e[0] in MINE_TYPES for e in ents), (
                    f"PRECISION {preset} seed={seed} has no mine"
                )


def _interior_gap_count(grid: List[int]) -> int:
    """Count empty runs in the bottom support row bounded by solid on both
    sides -- i.e. the number of pits the ninja must jump over."""
    row = MAP_TILE_HEIGHT - 2
    cells = [grid[x + row * MAP_TILE_WIDTH] for x in range(MAP_TILE_WIDTH)]
    solids = [x for x, c in enumerate(cells) if c != 0]
    if not solids:
        return 0
    lo, hi = min(solids), max(solids)
    count, in_gap = 0, False
    for x in range(lo, hi + 1):
        if cells[x] == 0 and not in_gap:
            in_gap = True
            count += 1
        elif cells[x] != 0:
            in_gap = False
    return count


def test_multi_jump_gaps_presets_require_multiple_jumps():
    """The gap-based MULTI_JUMP presets carve >=2 pits (>=2 sequenced jumps)."""
    checks = {"two_gaps_t1": 2, "many_gaps_t4": 2}
    seen = 0
    for tier in (1, 2, 3, 4):
        for gen_type, preset, _w in get_bucket_entries("MULTI_JUMP", tier):
            if preset not in checks:
                continue
            grid = _tile_grid(_build(gen_type, preset, 0).map_data())
            assert _interior_gap_count(grid) >= checks[preset], (
                f"{preset}: expected >= {checks[preset]} pits"
            )
            seen += 1
    assert seen >= 1


def test_composite_spans_most_of_canvas():
    """The long COMPOSITE course occupies a wide span of the canvas."""
    grid = _tile_grid(_build("composite", "long_chain_t4", 0).map_data())
    solid_cols = [
        x
        for x in range(MAP_TILE_WIDTH)
        if any(grid[x + y * MAP_TILE_WIDTH] != 0 for y in range(MAP_TILE_HEIGHT))
    ]
    assert solid_cols and (max(solid_cols) - min(solid_cols)) >= 28
