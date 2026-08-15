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


# Entity types the simulator actually instantiates (each has an entity class in
# entity_factory.get_entity_class_for_type). Types outside ALLOWED_INSTANTIATED
# are out of scope for the RL training subset; the exit *switch* (type 4) is
# created as the exit door's child slot, not a class of its own.
_SIM_INSTANTIATED_TYPES = {1, 2, 3, 5, 6, 8, 10, 11, 14, 17, 20, 21, 24, 25, 26, 28}
_ALLOWED_INSTANTIATED_TYPES = {1, 2, 3, 21}  # mines, gold, exit door (+ child switch)


def _instantiated_entity_types(map_data: List[int]) -> set:
    """Mirror ``MapLoader.load_map_entities``: scan from 1230, stride 5, to the
    end of the buffer, returning the set of types the simulator would actually
    instantiate. This is the ground truth for "what the RL agent sees" and is
    robust to zero-padding gaps (unlike a break-on-first-zero parse)."""
    found = set()
    idx = 1230
    n = len(map_data)
    while idx + 4 < n:
        t = map_data[idx]
        if t in _SIM_INSTANTIATED_TYPES:
            found.add(t)
        idx += 5
    return found


def _is_hazard_entry(gen_type: str) -> bool:
    """gamma-1 hazard buckets are exactly the `hazard_overlay` entries."""
    return gen_type == "hazard_overlay"


#: Buckets that deliberately introduce entities OUTSIDE the phase-1 profile,
#: with the types they may add. The profile gates official-corpus INGEST; procgen
#: is not gated by it, so a procedural bucket CAN reach past it -- and this
#: allowlist is what keeps that a decision rather than an accident.
#:
#: THWUMP: thwump (20) and shove thwump (28) are `phase4` types. Including them
#: at gamma-1 is defensible on three specific grounds, all checked:
#:   * the reachability graph BLOCKS solid entity bodies
#:     (`graph/reachability/entity_mask.py`), so routes and the cost-to-go field
#:     are correct on these levels rather than routing straight through;
#:   * `threat_ttl` covers thwumps, so they are perceivable even with
#:     `--enable-solid-map` off (measured: 1991 threatened cells on a T2 level);
#:   * `map_hazard_overlay` re-routes the DRESSED level for solid classes instead
#:     of replaying the clean base's tape, so solvability is certified after the
#:     bodies are placed.
#: They are also the only source of a non-zero `solid_map` anywhere in gamma-1,
#: since the official corpus at phase 1 cannot produce one.
_OUT_OF_PHASE_BY_DESIGN = {
    "THWUMP": {20, 28},
}


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_the_mechanic_spine_stays_within_the_legacy_entity_subset(
    phase1_entries, seed
):
    """Every NON-hazard Phase-1 level stays within the RL entity subset.

    Contract: the simulator may only instantiate toggle mines (1/21), gold (2),
    and exactly one exit door (3) + exit switch (4) -- no locked/trap doors,
    drones, thwumps, bounce blocks, launch pads, one-ways, boost pads, etc.
    This guards the regression where ``horizontal_corridor`` leaked incidental
    locked doors and ``add_random_entities_outside_playspace`` could emit
    hazards, and mirrors the production ``subset_ok`` gate.

    NARROWED FOR gamma-1, deliberately and no further. `PHASE_1_BUCKETS` used to
    mean "toggle mines only", and the gamma-1 hazard buckets are in it BY
    DESIGN -- a DRONE bucket that contained no drone would be the bug. So the
    contract still applies in full to the mechanic spine, where it has caught
    real leaks, and the hazard buckets get their own, stricter check below:
    they must contain their declared classes and nothing else.
    """
    from nclone.levelgen.labeling import subset_ok

    checked = 0
    for bucket, tier, gen_type, preset in phase1_entries:
        if _is_hazard_entry(gen_type):
            continue
        md = _build(gen_type, preset, seed=seed).map_data()
        disallowed = _instantiated_entity_types(md) - _ALLOWED_INSTANTIATED_TYPES
        assert not disallowed, (
            f"{bucket}/{gen_type}:{preset} (seed={seed}) instantiates "
            f"out-of-subset entity types {sorted(disallowed)}"
        )
        assert subset_ok(md), (
            f"{bucket}/{gen_type}:{preset} (seed={seed}) failed subset_ok "
            f"(entity classes or door/switch count outside contract)"
        )
        checked += 1
    assert checked > 0, "no Phase-1 spine entries exercised"


@pytest.mark.parametrize("tier", [1, 4])
def test_hazard_buckets_emit_their_declared_classes_and_nothing_else(
    phase1_entries, tier
):
    """A hazard bucket exists to make failures ATTRIBUTABLE.

    That only works if the bucket contains what its name says: a DRONE level
    carrying a stray turret makes "the agent fails on drones" unfalsifiable, and
    it fails silently because the level is perfectly valid. Two directions are
    checked, and the second is the one that bites:

    * nothing outside `PHASE1_HAZARD_SCOPE` (would put the level outside the
      corpus the run is defined against), and
    * nothing outside the preset's OWN `hazard_classes` (plus the legacy
      mines/gold/exit set, which every base generator may emit).

    Scoped to T1 and T4, one seed each. Building a hazard level costs ~13 s (the
    d* certificate dominates), so the full 7 buckets x 4 tiers x several seeds is
    a ~15-minute test — too slow to keep in the default suite, and it buys little:
    T1 is the on-ramp and T4 is the ceiling, and the presets in between are the
    same code with different constants.
    """
    from nclone.constants.entity_types import ENTITY_PROFILES
    from nclone.map_generation.map_hazard_overlay import HAZARD_SPECS

    seed = 0
    phase1_scope = {int(t) for t in ENTITY_PROFILES["phase1"]}
    checked = 0
    for bucket, entry_tier, gen_type, preset in phase1_entries:
        if not _is_hazard_entry(gen_type) or entry_tier != tier:
            continue
        cfg = GENERATOR_PRESETS[gen_type][preset]
        declared = {
            etype
            for cname in cfg.hazard_classes
            for etype, *_rest in HAZARD_SPECS[cname].types
        }
        md = _build(gen_type, preset, seed=seed).map_data()
        got = _instantiated_entity_types(md)

        allowed_scope = phase1_scope | _OUT_OF_PHASE_BY_DESIGN.get(bucket, set())
        out_of_phase = got - allowed_scope
        assert not out_of_phase, (
            f"{bucket}/{preset} (seed={seed}) emits {sorted(out_of_phase)}, "
            "which is outside the phase-1 entity scope the run is defined "
            "against. If that is intended, add it to "
            "_OUT_OF_PHASE_BY_DESIGN with the reason -- an out-of-scope entity "
            "that nobody decided on is how a bucket ends up training a mechanic "
            "the planner cannot route."
        )
        undeclared = got - declared - _ALLOWED_INSTANTIATED_TYPES
        assert not undeclared, (
            f"{bucket}/{preset} (seed={seed}) emits {sorted(undeclared)}, which "
            f"this preset does not declare (declared: {sorted(declared)}). A "
            "bucket that contains hazards it is not named for cannot attribute "
            "a failure to anything."
        )
        checked += 1
    assert checked > 0, "no hazard-bucket entries exercised"


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
