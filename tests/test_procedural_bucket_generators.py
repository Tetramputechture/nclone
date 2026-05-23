"""Reproducibility + mechanic-purity tests for BUCKET_CATALOG generators.

Covers the dedicated Phase-1 generators that back the
mechanic-primary benchmark taxonomy:

* ``MapBasicJump``  (``basic_jump`` in ``GENERATOR_REGISTRY``)
* ``MapWallJump``   (``wall_jump``)
* ``MapMinePath``   (``mine_path``)
* ``MapSlopes``     (``slopes``)

For each generator we assert:

1. Reproducibility -- two generators built from the same ``(preset, seed)``
   pair produce byte-identical ``map_data()`` output, including ninja spawn
   and entity list. This is a hard requirement for the benchmark pipeline
   because the dataset script hashes ``map_data`` for dedup and because the
   ``BSRS`` learned value function expects a stable level id per seed.
2. Mechanic purity -- the generated tile + entity layout contains only the
   features the bucket is supposed to isolate. Specifically:
   * ``basic_jump`` and ``wall_jump`` must contain *zero* toggle-mine
     entities (types 1 and 21). Their only non-ninja entity is the exit
     door (type 3).
   * ``mine_path`` must contain at least one toggle-mine entity and must
     not contain any entity other than toggle mines and the exit door.
3. Tile sanity -- the generated tile grid matches the 42 x 23 size the rest
   of the pipeline assumes and contains at least one solid and one empty
   tile (so we're not accidentally emitting an all-solid or all-empty
   canvas).

Entity layout reference: ``nclone/map_generation/map.py::Map.map_data``.
Exit doors occupy 10 ints (5 for the door + 5 for the switch); all other
supported entities in Phase 1 occupy 5 ints.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    GENERATOR_PRESETS,
    PHASE_1_BUCKETS,
)
from nclone.map_generation.generator_factory import (
    GENERATOR_REGISTRY,
    GeneratorFactory,
)


TILE_HEADER_LEN = 184
TILE_GRID_LEN = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
ENTITY_SECTION_START = 1235

MINE_ENTITY_TYPES = (1, 21)
EXIT_DOOR_ENTITY_TYPE = 3
EXIT_SWITCH_ENTITY_TYPE = 4
LOCKED_DOOR_TYPE = 6
TRAP_DOOR_TYPE = 8

# BASIC_JUMP mixed-secondary presets (tier 3/4) intentionally place mines.
BASIC_JUMP_MIXED_MINE_PRESETS = frozenset(
    {"single_gap_with_mine_t3", "double_gap_mines_t4"}
)

DEDICATED_GENERATORS = {"basic_jump", "wall_jump", "mine_path", "slopes"}
# Tile ids 6–9 (45°) and 18–33 (mild/steep slopes)
SLOPE_TILE_TYPES = set(range(6, 10)) | set(range(18, 34))


def _extract_tile_grid(map_data: List[int]) -> List[int]:
    return list(map_data[TILE_HEADER_LEN : TILE_HEADER_LEN + TILE_GRID_LEN])


def _extract_dedicated_entities(map_data: List[int]) -> List[Tuple[int, int, int]]:
    """Parse entities emitted by the three dedicated Phase-1 generators.

    These generators only ever emit two entity shapes:

    * ``MapBasicJump`` / ``MapWallJump``: exactly one exit-door pair
      (10 ints total: 5 for the door + 5 for the paired switch type 4).
    * ``MapMinePath``: one exit-door pair (10 ints) plus N toggle-mine
      entries (5 ints each).

    Returns a flat ``[(type, x, y), ...]`` list with the inline exit switch
    unpacked into its own ``(EXIT_SWITCH_ENTITY_TYPE, sx, sy)`` entry so
    purity assertions can count mines and doors independently.

    This parser is intentionally strict -- any entity shape outside the
    above set raises ``AssertionError``. That's the mechanic-purity
    signal we want the test to enforce.
    """
    tail = list(map_data[ENTITY_SECTION_START:])
    entities: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(tail):
        etype = tail[i]
        if etype == EXIT_DOOR_ENTITY_TYPE:
            assert i + 10 <= len(tail), "truncated exit-door record"
            door_x, door_y = tail[i + 1], tail[i + 2]
            sx, sy = tail[i + 6], tail[i + 7]
            entities.append((EXIT_DOOR_ENTITY_TYPE, door_x, door_y))
            entities.append((EXIT_SWITCH_ENTITY_TYPE, sx, sy))
            i += 10
            continue
        if etype in MINE_ENTITY_TYPES:
            assert i + 5 <= len(tail), "truncated mine record"
            entities.append((etype, tail[i + 1], tail[i + 2]))
            i += 5
            continue
        raise AssertionError(
            f"Dedicated generator emitted unsupported entity type {etype} "
            f"at tail offset {i}; mechanic-purity broken"
        )
    return entities


def _iter_phase1_entries():
    for bucket in PHASE_1_BUCKETS:
        for tier, entries in BUCKET_CATALOG.get(bucket, {}).items():
            for gen_type, preset, _ in entries:
                yield bucket, tier, gen_type, preset


@pytest.fixture(scope="module")
def phase1_entries():
    entries = list(_iter_phase1_entries())
    assert entries, "BUCKET_CATALOG has no Phase-1 entries to test"
    return entries


def _build(gen_type: str, preset: str, seed: int):
    """Build via the factory so we exercise the same code path as the script."""
    assert gen_type in GENERATOR_REGISTRY, f"Unknown generator {gen_type!r}"
    assert preset in GENERATOR_PRESETS.get(gen_type, {}), (
        f"Missing preset {gen_type}:{preset!r}"
    )
    gen = GeneratorFactory.create_from_preset(gen_type, preset, seed=seed)
    gen.generate(seed=seed)
    return gen


@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_phase1_generators_are_reproducible(phase1_entries, seed):
    """Same ``(gen_type, preset, seed)`` -> identical ``map_data()``."""
    for _bucket, _tier, gen_type, preset in phase1_entries:
        a = _build(gen_type, preset, seed).map_data()
        b = _build(gen_type, preset, seed).map_data()
        assert a == b, (
            f"{gen_type}:{preset} (seed={seed}) not reproducible: "
            f"len(a)={len(a)} len(b)={len(b)}"
        )


def test_phase1_tile_grid_has_both_solid_and_empty(phase1_entries):
    """Every Phase-1 level has a nontrivial tile grid (>=1 solid, >=1 empty)."""
    for _bucket, _tier, gen_type, preset in phase1_entries:
        md = _build(gen_type, preset, seed=0).map_data()
        grid = _extract_tile_grid(md)
        assert len(grid) == TILE_GRID_LEN, (
            f"{gen_type}:{preset} tile grid size {len(grid)} != {TILE_GRID_LEN}"
        )
        assert any(t == 0 for t in grid), f"{gen_type}:{preset} has no empty tiles"
        assert any(t != 0 for t in grid), f"{gen_type}:{preset} has no solid tiles"


def _dedicated_entries(phase1_entries):
    return [
        (bucket, tier, gen_type, preset)
        for bucket, tier, gen_type, preset in phase1_entries
        if gen_type in DEDICATED_GENERATORS
    ]


def test_basic_jump_and_wall_jump_mine_purity(phase1_entries):
    """WALL_JUMP never emits mines; pure BASIC_JUMP presets never emit mines.

    Tier 3/4 mixed presets in BASIC_JUMP (``*with_mine*``) must place at
    least one mine.
    """
    checked_pure = 0
    checked_mixed = 0
    for bucket, _tier, gen_type, preset in _dedicated_entries(phase1_entries):
        if bucket not in ("BASIC_JUMP", "WALL_JUMP"):
            continue
        md = _build(gen_type, preset, seed=0).map_data()
        entities = _extract_dedicated_entities(md)
        mines = [e for e in entities if e[0] in MINE_ENTITY_TYPES]
        if bucket == "WALL_JUMP":
            assert not mines, f"{bucket}/{gen_type}:{preset} emitted mines: {mines}"
        elif gen_type == "basic_jump" and preset in BASIC_JUMP_MIXED_MINE_PRESETS:
            assert mines, f"{bucket}/{gen_type}:{preset} expected at least one mine"
            checked_mixed += 1
        else:
            assert not mines, (
                f"{bucket}/{gen_type}:{preset} emitted mines: {mines}"
            )
            if bucket == "BASIC_JUMP":
                checked_pure += 1
    assert checked_pure > 0, "no pure BASIC_JUMP entries exercised"
    assert checked_mixed > 0, "no mixed-mine BASIC_JUMP entries exercised"


def test_slopes_emits_slopes_tiles_and_mine_budget(phase1_entries):
    """SLOPES bucket: tiles include a slope; optional mines follow tier cap."""
    checked = 0
    for bucket, tier, gen_type, preset in phase1_entries:
        if gen_type != "slopes" or bucket != "SLOPES":
            continue
        md = _build(gen_type, preset, seed=0).map_data()
        grid = _extract_tile_grid(md)
        assert any(t in SLOPE_TILE_TYPES for t in grid), (
            f"{preset}: expected at least one slope tile in grid"
        )
        entities = _extract_dedicated_entities(md)
        mines = [e for e in entities if e[0] in MINE_ENTITY_TYPES]
        if tier <= 2:
            assert not mines, f"{preset} tier {tier} must not have mines, got {mines}"
        if tier == 3:
            assert len(mines) <= 1, f"{preset} T3: at most 1 mine, got {len(mines)}"
        if tier == 4:
            assert len(mines) <= 2, f"{preset} T4: at most 2 mines, got {len(mines)}"
        checked += 1
    assert checked == 4, f"expected 4 SLOPES catalog entries, got {checked}"


def test_mine_path_contains_at_least_one_mine(phase1_entries):
    """MINE_NAVIGATION must produce at least one toggle mine per level."""
    checked = 0
    for bucket, _tier, gen_type, preset in _dedicated_entries(phase1_entries):
        if bucket != "MINE_NAVIGATION":
            continue
        md = _build(gen_type, preset, seed=0).map_data()
        entities = _extract_dedicated_entities(md)
        mines = [e for e in entities if e[0] in MINE_ENTITY_TYPES]
        assert mines, f"{gen_type}:{preset} produced zero mines"
        checked += 1
    assert checked > 0, "no MINE_NAVIGATION entries exercised"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_dedicated_generators_emit_exactly_one_exit_door(phase1_entries, seed):
    """Dedicated Phase-1 generators emit exactly one exit door + switch pair."""
    checked = 0
    for _bucket, _tier, gen_type, preset in _dedicated_entries(phase1_entries):
        md = _build(gen_type, preset, seed=seed).map_data()
        entities = _extract_dedicated_entities(md)
        doors = [e for e in entities if e[0] == EXIT_DOOR_ENTITY_TYPE]
        switches = [e for e in entities if e[0] == EXIT_SWITCH_ENTITY_TYPE]
        assert len(doors) == 1 and len(switches) == 1, (
            f"{gen_type}:{preset} (seed={seed}) got {len(doors)} doors / "
            f"{len(switches)} switches (expected exactly 1 of each)"
        )
        checked += 1
    assert checked > 0, "no dedicated generator entries exercised"
