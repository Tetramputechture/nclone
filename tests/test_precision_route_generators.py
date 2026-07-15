"""Regression tests for the precision-route curriculum generators (audit P0-3).

Covers the four sub-tasks that added the missing geometry class:

(a) ``long_jump`` wired + extended  -- gap 3->6(+extreme 7-8) x landing 6->1,
    optional mines on the landing edge; MULTI_JUMP / PRECISION T3-T4 entries.
(b) ``drop_landing`` NEW generator + DROP_LANDING bucket -- graded drops 3->17
    tiles onto 1-4 tile ledges with lateral offset and edge mines; T3+ crosses
    the survivable free-fall boundary so braking must be learned.
(c) PRECISION unclamped -- ``gap_width`` sweep 2->3 and ledge clamp 3->2 on new
    harder presets ONLY (legacy presets byte-identical, guarded by hashes).
(d) compose: ``drop_shaft`` / ``wall_chain`` components,
    ``vertical_course_1x3`` + ``serpentine_full_field`` templates, and the
    platform_stairs DESCENT profile presets.

Physics facts asserted here were measured on the real sim (see the calibration
test): an uncontrolled walk-off free-fall survives up to 10 tiles and dies from
11; a wall-slide brake survives 12+.

NB: the heavier checks (20-seed solvability, scripted-jump wins) exercise the
headless sim; the whole module runs in a few minutes.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import pytest

from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    CURRICULUM_SCHEDULE,
    GENERATOR_PRESETS,
    PHASE_1_BUCKETS,
    validate_bucket_catalog,
)
from nclone.map_generation.generator_factory import GeneratorFactory

TILE_HEADER_LEN = 184
TILE_GRID_LEN = MAP_TILE_WIDTH * MAP_TILE_HEIGHT
ENTITY_SECTION_START = 1235
MINE_TYPES = (1, 21)

# Every preset/tier added by the audit-P0-3 work (gen_type, preset).
NEW_PRESETS: List[Tuple[str, str]] = [
    ("long_jump", "gap_t3"),
    ("long_jump", "gap_t4"),
    ("long_jump", "edge_mine_t3"),
    ("long_jump", "edge_mine_t4"),
    ("drop_landing", "drop_t1"),
    ("drop_landing", "drop_t2"),
    ("drop_landing", "drop_t3"),
    ("drop_landing", "drop_t4"),
    ("precision", "wide_gap_ledges_t3"),
    ("precision", "wide_gap_narrow_t4"),
    ("platform_stairs", "stairs_desc_t2"),
    ("platform_stairs", "stairs_desc_t3"),
    ("platform_stairs", "stairs_desc_t4"),
    ("template_vertical_course_1x3", "default"),
    ("template_serpentine_full_field", "default"),
]

# The physics-unverified-by-design extreme kept OUT of every bucket.
UNWIRED_EXTREME = ("long_jump", "max_gap_extreme")


def _build(gen_type: str, preset: str, seed: int):
    gen = GeneratorFactory.create_from_preset(gen_type, preset, seed=seed)
    gen.generate(seed=seed)
    return gen


def _md(gen_type: str, preset: str, seed: int) -> List[int]:
    return _build(gen_type, preset, seed).map_data()


def _tile_grid(map_data: List[int]) -> List[int]:
    return list(map_data[TILE_HEADER_LEN : TILE_HEADER_LEN + TILE_GRID_LEN])


def _tile(grid: List[int], x: int, y: int) -> int:
    return grid[x + y * MAP_TILE_WIDTH]


def _sha(map_data: List[int]) -> str:
    return hashlib.sha256(bytes(str(map_data), "utf8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Catalog wiring
# ---------------------------------------------------------------------------

def test_catalog_validates_and_new_wiring_present():
    assert validate_bucket_catalog() == []
    # long_jump folded into MULTI_JUMP + PRECISION harder tiers.
    mj = {(g, p) for t in (3, 4) for g, p, _ in BUCKET_CATALOG["MULTI_JUMP"][t]}
    assert ("long_jump", "gap_t3") in mj and ("long_jump", "gap_t4") in mj
    pr = {(g, p) for t in (3, 4) for g, p, _ in BUCKET_CATALOG["PRECISION"][t]}
    assert ("long_jump", "edge_mine_t3") in pr
    assert ("long_jump", "edge_mine_t4") in pr
    assert ("precision", "wide_gap_ledges_t3") in pr
    assert ("precision", "wide_gap_narrow_t4") in pr
    # DROP_LANDING bucket exists, is scheduled, and is Phase-1 in-scope.
    assert set(BUCKET_CATALOG["DROP_LANDING"]) == {1, 2, 3, 4}
    assert "DROP_LANDING" in CURRICULUM_SCHEDULE
    assert "DROP_LANDING" in PHASE_1_BUCKETS
    # Descent presets wired into PLATFORM_STAIRS harder tiers.
    ps = {(g, p) for t in (2, 3, 4) for g, p, _ in BUCKET_CATALOG["PLATFORM_STAIRS"][t]}
    assert {("platform_stairs", f"stairs_desc_t{t}") for t in (2, 3, 4)} <= ps
    # The gap 7-8 extreme stays OUT of every bucket (reachability gate is
    # optimistic near the jump envelope; see generator_configs long_jump note).
    all_entries = {
        (g, p)
        for tiers in BUCKET_CATALOG.values()
        for entries in tiers.values()
        for g, p, _ in entries
    }
    assert UNWIRED_EXTREME not in all_entries
    assert UNWIRED_EXTREME[1] in GENERATOR_PRESETS["long_jump"]


# ---------------------------------------------------------------------------
# Legacy output must be byte-identical (golden hashes captured pre-change)
# ---------------------------------------------------------------------------

GOLDEN = {
    ("precision", "two_ledges_t1", 0): "ba2769d2b6f62b37",
    ("precision", "two_ledges_t1", 7): "611037dbe9b37835",
    ("precision", "ledges_t2", 0): "7ed5789f3f2ee6d7",
    ("precision", "ledges_t2", 7): "0d8f05fa088c4ce5",
    ("precision", "ledges_t3", 0): "c65154d54a46a0be",
    ("precision", "narrow_ledges_t4", 0): "43b6dc00c6738f82",
    ("platform_stairs", "stairs_t1", 0): "e9404dd952563ea8",
    ("platform_stairs", "stairs_t2", 0): "fe781e5a55887cd2",
    ("platform_stairs", "stairs_t3", 0): "ca9449eb90200cf7",
    ("platform_stairs", "stairs_t4", 0): "21fdb11d15f8a085",
    ("platform_stairs", "stairs_t3_5", 0): "32cad60b4073a08a",
    ("template_long_jump_course_2x3", "default", 0): "d31779b7da6fd59c",
    ("template_mine_gauntlet_2x3", "default", 0): "1238595af5ec5440",
    ("template_movement_mix_2x3", "default", 0): "246b0898111cac81",
    ("template_flat_intro_2x3", "default", 0): "2a8bb7341e648317",
    ("template_mixed_traverse_2x3", "default", 0): "cdf935d316c574cc",
}


def test_legacy_presets_and_templates_byte_identical():
    """The unclamp/descent/vertical work must not perturb existing output:
    legacy PRECISION presets, ascent PLATFORM_STAIRS presets, and every
    pre-existing compose template (new components must not enter their
    filters) reproduce their pre-change map_data exactly."""
    for (gt, preset, seed), want in GOLDEN.items():
        got = _sha(_md(gt, preset, seed))
        assert got == want, f"{gt}:{preset} seed={seed} changed: {want} -> {got}"


# ---------------------------------------------------------------------------
# Reproducibility of everything new
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 12345])
def test_new_presets_reproducible(seed):
    for gt, preset in NEW_PRESETS + [UNWIRED_EXTREME]:
        a = _md(gt, preset, seed)
        b = _md(gt, preset, seed)
        assert a == b, f"{gt}:{preset} (seed={seed}) not reproducible"


# ---------------------------------------------------------------------------
# Solvability: N=20 seeds per new preset must pass the reachability gate
# ---------------------------------------------------------------------------

def test_new_presets_100pct_solvable():
    from nclone.map_generation.solvability import is_map_solvable

    for gt, preset in NEW_PRESETS:
        fails = [
            seed
            for seed in range(30000, 30020)
            if not is_map_solvable(_md(gt, preset, seed))
        ]
        assert not fails, f"{gt}:{preset} unsolvable at seeds {fails}"


# ---------------------------------------------------------------------------
# PRECISION unclamp geometry
# ---------------------------------------------------------------------------

def _support_row_gap_widths(grid: List[int]) -> List[int]:
    """Widths of the empty runs in the bottom support row between solid spans
    (== the carved gap widths for the precision floor-level layout)."""
    row = MAP_TILE_HEIGHT - 2
    cells = [_tile(grid, x, row) for x in range(MAP_TILE_WIDTH)]
    solids = [x for x, c in enumerate(cells) if c != 0]
    if not solids:
        return []
    widths, run = [], 0
    for x in range(min(solids), max(solids) + 1):
        if cells[x] == 0:
            run += 1
        elif run:
            widths.append(run)
            run = 0
    return widths


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_precision_gap_widths(seed):
    """Legacy presets keep the fixed 2-tile gap; wide presets carve 3."""
    for preset, want in (
        ("ledges_t3", 2),
        ("narrow_ledges_t4", 2),
        ("wide_gap_ledges_t3", 3),
        ("wide_gap_narrow_t4", 3),
    ):
        widths = _support_row_gap_widths(_tile_grid(_md("precision", preset, seed)))
        assert widths, f"{preset} seed={seed}: no gaps found"
        assert all(w == want for w in widths), (
            f"{preset} seed={seed}: gap widths {widths}, expected all == {want}"
        )


# ---------------------------------------------------------------------------
# DROP_LANDING physics: the braking boundary
# ---------------------------------------------------------------------------

SURVIVABLE_FREEFALL_TILES = 10  # measured: 10 survives, 11+ dies (walk-off)


def _headless():
    from nclone.nplay_headless import NPlayHeadless

    return NPlayHeadless(
        render_mode=None,
        enable_animation=False,
        enable_logging=False,
        enable_debug_overlay=False,
        enable_rendering=False,
    )


def _run(nh, md, policy, frames=800):
    nh.load_map_from_map_data(list(md))
    nh.reset()
    n = nh.sim.ninja
    air = 0
    for f in range(frames):
        h, j = policy(f, n)
        nh.tick(h, j)
        if n.airborn:
            air += 1
        if nh.ninja_has_died():
            return ("DIED", air)
        if nh.ninja_has_won():
            return ("WON", air)
    return ("SURVIVED", air)


def test_drop_tier_configs_cross_braking_boundary():
    """T3/T4 DROP_LANDING presets must EXCEED the survivable free-fall depth
    (and skip the ambiguous depth-11 boundary); T1/T2 must stay below it."""
    presets = GENERATOR_PRESETS["drop_landing"]
    for name in ("drop_t3", "drop_t4"):
        cfg = presets[name]
        assert cfg.min_depth >= SURVIVABLE_FREEFALL_TILES + 2, (
            f"{name}: min_depth {cfg.min_depth} does not exceed the "
            f"survivable free-fall boundary ({SURVIVABLE_FREEFALL_TILES})"
        )
    for name in ("drop_t1", "drop_t2"):
        cfg = presets[name]
        assert cfg.max_depth <= SURVIVABLE_FREEFALL_TILES, (
            f"{name}: max_depth {cfg.max_depth} exceeds the survivable band"
        )


def test_freefall_boundary_calibration():
    """The physics fact the tiers are built on: a walk-off free-fall from rest
    survives a 10-tile drop and dies on a 12-tile drop (flat-floor impact)."""
    from nclone.map_generation.map import Map

    nh = _headless()

    def shaft(depth):
        m = Map(seed=0)
        m.set_tiles_bulk([1] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))
        top, w = 2, 8
        for x in range(2, 2 + w):
            for y in range(top - 1, top + depth):
                m.set_tile(x, y, 0)
        m.set_tile(2, top, 1)
        m.set_ninja_spawn(2, top - 1, orientation=1)
        m.add_entity(3, 2 + w - 1, top + depth - 1, 0, 0, 3, top + depth - 1)
        return m.map_data()

    walkoff = lambda f, n: ((0 if n.airborn else 1), 0)
    assert _run(nh, shaft(SURVIVABLE_FREEFALL_TILES), walkoff)[0] != "DIED"
    assert _run(nh, shaft(SURVIVABLE_FREEFALL_TILES + 2), walkoff)[0] == "DIED"


def test_drop_t3_requires_braking_and_wallhug_survives():
    """On real generated T3 drops: an unbraked walk-off DIES on impact, while
    holding into the ledge-flush far wall (wall-slide brake) survives -- the
    exact skill the tier exists to teach. Deterministic seeds."""
    nh = _headless()
    for seed in range(30000, 30005):
        md = _md("drop_landing", "drop_t3", seed)
        results = {}
        for d in (1, -1):
            walkoff = lambda f, n, d=d: ((0 if n.airborn else d), 0)
            hug = lambda f, n, d=d: (d, 0)
            results[d] = (_run(nh, md, walkoff), _run(nh, md, hug))
        # The drop direction is the one where the walk-off spends more frames
        # airborne (the other direction just walks into the chamber wall).
        d = max(results, key=lambda k: results[k][0][1])
        walkoff_res, hug_res = results[d]
        assert walkoff_res[0] == "DIED", (
            f"drop_t3 seed={seed}: unbraked walk-off should die, got {walkoff_res}"
        )
        assert hug_res[0] in ("SURVIVED", "WON"), (
            f"drop_t3 seed={seed}: wall-slide brake should survive, got {hug_res}"
        )


# ---------------------------------------------------------------------------
# LONG_JUMP: geometry + a scripted real-physics win on the widest default gap
# ---------------------------------------------------------------------------

def _parse_entities(map_data: List[int]) -> List[Tuple[int, float, float]]:
    tail = list(map_data[ENTITY_SECTION_START:])
    out: List[Tuple[int, float, float]] = []
    i = 0
    while i < len(tail):
        etype = int(tail[i])
        if etype == 3:
            out.append((3, tail[i + 1], tail[i + 2]))
            out.append((4, tail[i + 6], tail[i + 7]))
            i += 10
        elif etype in MINE_TYPES:
            out.append((etype, tail[i + 1], tail[i + 2]))
            i += 5
        else:
            raise AssertionError(f"unexpected entity type {etype} at {i}")
    return out


@pytest.mark.parametrize("preset,n_mines", [("edge_mine_t3", 1), ("edge_mine_t4", 2)])
def test_long_jump_edge_mines_present_and_clear_of_objectives(preset, n_mines):
    for seed in (0, 1, 42, 30000):
        ents = _parse_entities(_md("long_jump", preset, seed))
        mines = [e for e in ents if e[0] in MINE_TYPES]
        assert len(mines) == n_mines, (
            f"{preset} seed={seed}: expected {n_mines} edge mines, got {len(mines)}"
        )
        # Mines never share a column with the switch/door (units are 4/tile;
        # mines carry the +1.5-tile centering offset -> compare tile indices).
        obj_cols = {int(e[1]) // 4 for e in ents if e[0] in (3, 4)}
        mine_cols = {int(e[1] // 4 - 1) for e in mines}
        assert not (obj_cols & mine_cols), (
            f"{preset} seed={seed}: edge mine overlaps switch/door column"
        )


def _long_jump_scripted_win(nh, md) -> bool:
    """Run right, jump at the gap lip, then sweep the landing for the switch."""
    grid = _tile_grid(md)

    nh.load_map_from_map_data(list(md))
    nh.reset()
    n = nh.sim.ninja
    col = int(n.xpos // 24) - 1
    row = int(n.ypos // 24) - 1
    gr = row + 1
    while _tile(grid, col, gr) == 0:
        gr += 1
    x = col
    while x < MAP_TILE_WIDTH - 1 and _tile(grid, x, gr) != 0:
        x += 1
    gap_start, gw = x, 0
    while x < MAP_TILE_WIDTH - 1 and _tile(grid, x, gr) == 0:
        x += 1
        gw += 1
    lip_px = (gap_start + 1) * 24
    gap_end_px = (gap_start + gw + 1) * 24

    for trigger_px in (12, 24, 36):
        nh.load_map_from_map_data(list(md))
        nh.reset()
        n = nh.sim.ninja
        jumping, landed_f = False, None
        for f in range(900):
            if landed_f is None:
                if not jumping and not n.airborn and n.xpos >= lip_px - trigger_px:
                    jumping = True
                h, j = 1, (1 if jumping else 0)
                if jumping and not n.airborn and n.xpos > gap_end_px:
                    landed_f = f
                    h, j = 0, 0
            else:
                t = (f - landed_f) % 160
                h, j = (-1 if t < 80 else 1), 0
            nh.tick(h, j)
            if nh.ninja_has_died():
                break
            if nh.ninja_has_won():
                return True
    return False


def test_long_jump_widest_default_gap_scripted_win():
    """gap_t4 (5-6 tile gap, the widest DEFAULT tier) must be winnable by a
    scripted full-speed run + lip jump on the real sim -- guarding against the
    optimistic reachability gate admitting physically-impossible gaps."""
    nh = _headless()
    for seed in (30000, 30001, 30002):
        md = _md("long_jump", "gap_t4", seed)
        assert _long_jump_scripted_win(nh, md), (
            f"gap_t4 seed={seed}: scripted run+jump could not win"
        )


# ---------------------------------------------------------------------------
# PLATFORM_STAIRS descent profile
# ---------------------------------------------------------------------------

def test_stairs_descent_spawns_high_and_exits_low():
    """Descent presets spawn ON the top platform (high) with the door at the
    base platform (low) -- the reverse of the ascent profile."""
    for preset in ("stairs_desc_t2", "stairs_desc_t3", "stairs_desc_t4"):
        for seed in (0, 1, 42):
            gen = _build("platform_stairs", preset, seed)
            md = gen.map_data()
            ents = _parse_entities(md)
            door_y = next(e[2] for e in ents if e[0] == 3)
            spawn_y = gen.ninja_spawn_y
            assert spawn_y < door_y, (
                f"{preset} seed={seed}: descent spawn (y={spawn_y}) not above "
                f"door (y={door_y})"
            )


# ---------------------------------------------------------------------------
# Serpentine full-field template structure
# ---------------------------------------------------------------------------

def test_serpentine_route_spans_all_rows_and_floors_are_sealed():
    """The snake must be forced through every grid row: the two separating
    floors are solid across the interior EXCEPT exactly one 2-wide shaft each
    (at opposite ends), the switch is in the top row's far corner, and the
    spawn + exit door are in the bottom row."""
    from nclone.map_generation.compose.templates.vertical import SerpentineFullField

    for seed in (0, 7, 30000):
        gen = _build("template_serpentine_full_field", "default", seed)
        md = gen.map_data()
        grid = _tile_grid(md)

        # Row bands (3 even rows over interior y 1..21): floors at 7, 14, 21.
        fl0, fl1 = 7, 14
        right = set(SerpentineFullField.RIGHT_SHAFT)
        left = set(SerpentineFullField.LEFT_SHAFT)
        holes1 = {x for x in range(1, MAP_TILE_WIDTH - 1) if _tile(grid, x, fl1) == 0}
        holes0 = {x for x in range(1, MAP_TILE_WIDTH - 1) if _tile(grid, x, fl0) == 0}
        assert holes1 == right, f"seed={seed}: mid floor holes {holes1} != {right}"
        assert holes0 == left, f"seed={seed}: top floor holes {holes0} != {left}"

        # Objectives: switch high in the top band far corner; spawn + door low.
        ents = _parse_entities(md)
        switch = next(e for e in ents if e[0] == 4)
        door = next(e for e in ents if e[0] == 3)
        # Entity coords are 4 units/tile (+small type offsets): the switch must
        # sit at/above the top band's floor line, i.e. y_units <= (fl0+1)*4.
        assert switch[2] <= (fl0 + 1) * 4, f"seed={seed}: switch not in the top row"
        assert switch[1] > (MAP_TILE_WIDTH // 2) * 4, (
            f"seed={seed}: switch not in the far (right) half"
        )
        assert door[2] > fl1 * 4, f"seed={seed}: exit door not in the bottom row"
        assert gen.ninja_spawn_y > fl1 * 4, f"seed={seed}: spawn not in the bottom row"


# ---------------------------------------------------------------------------
# New compose components stay out of the legacy templates
# ---------------------------------------------------------------------------

def test_vertical_components_registered_but_fenced_from_legacy_templates():
    from nclone.map_generation.compose.registry import (
        COMPONENT_REGISTRY,
        TEMPLATE_REGISTRY,
        template_possible_tags,
    )

    assert "drop_shaft" in COMPONENT_REGISTRY
    assert "wall_chain" in COMPONENT_REGISTRY
    assert "vertical_course_1x3" in TEMPLATE_REGISTRY
    assert "serpentine_full_field" in TEMPLATE_REGISTRY
    # The new components' tags must not be admitted by any PRE-EXISTING
    # template's cell filters (else their seeded output would change).
    new_tags = {"move:fall.drop", "move:fall.control", "move:walljump.chain",
                "terrain:shaft"}
    for tname in ("flat_intro_2x3", "mixed_traverse_2x3", "long_jump_course_2x3",
                  "mine_gauntlet_2x3", "movement_mix_2x3"):
        cls = TEMPLATE_REGISTRY[tname]
        filters = list(getattr(cls, "cells", {}).values())
        for comp_name in ("drop_shaft", "wall_chain"):
            comp = COMPONENT_REGISTRY[comp_name]
            assert not any(f(comp) for f in filters), (
                f"legacy template {tname} would now admit {comp_name}"
            )
    # And min_h fences them out of half-height (2-row grid) cells regardless.
    for comp_name in ("drop_shaft", "wall_chain"):
        assert COMPONENT_REGISTRY[comp_name].min_h >= 12
