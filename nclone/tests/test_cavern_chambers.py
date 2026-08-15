"""Guards for the cavern/chamber generator.

The generator's whole value is that it is solvable *by construction* -- it is
meant for runtime curriculum sampling, where there is no reseed-and-retry gate
in front of it. An unsolvable level there is not a bad level, it is a silently
poisoned one: the free-space cost-to-go goes ``inf``, which kills the PBRS and
goal-distance signals for that episode (see ``test_dataset_solvable``).

So the load-bearing assertions here are (1) every preset is solvable under the
jump-aware traversability graph, and (2) the two route topologies the generator
claims to produce -- outbound and out-and-back -- actually come out that way,
since that is the reason it exists alongside the single-area generators.
"""

from collections import Counter, deque

import pytest

from nclone.constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH
from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation.map_cavern_chambers import MapCavernChambers
from nclone.map_generation.solvability import is_map_solvable


PRESETS = [
    "cavern_t1",
    "cavern_t2",
    "cavern_t3",
    "cavern_t4",
    "cavern_linear",
    "cavern_nonlinear",
    "cavern_dense_mines",
]
PRESETS += ["cavern_oneway", "cavern_traversal", "cavern_gold", "cavern_full"]
HAZARD_PRESETS = ["cavern_t4", "cavern_full"]
ZAP_DRONE, MINI_DRONE, ROCKET, TURRET = 14, 26, 18, 19
DRONE_TYPES = (ZAP_DRONE, MINI_DRONE)
EMPLACEMENT_TYPES = (ROCKET, TURRET)
MINED_PRESETS = ["cavern_t2", "cavern_t3", "cavern_t4", "cavern_dense_mines"]
GOLD_PRESETS = ["cavern_t3", "cavern_t4", "cavern_dense_mines", "cavern_gold"]
SEEDS = range(8)
MINE_TYPES = (1, 21)
GOLD = 2
ONE_WAY, BOUNCE_BLOCK, LAUNCH_PAD, BOOST_PAD = 11, 17, 10, 24


def _build(preset: str, seed: int) -> MapCavernChambers:
    gen = GeneratorFactory.create_from_preset("cavern_chambers", preset, seed=seed)
    gen.generate(seed)
    return gen


def _tile_distances(gen: MapCavernChambers, src, blocked=frozenset()):
    """4-connected flood fill over empty tiles, ignoring gravity."""
    dist = {src: 0}
    queue = deque([src])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in dist or nxt in blocked:
                continue
            if not (0 <= nxt[0] < MAP_TILE_WIDTH and 0 <= nxt[1] < MAP_TILE_HEIGHT):
                continue
            if gen.get_tile(*nxt) != 0:
                continue
            dist[nxt] = dist[(x, y)] + 1
            queue.append(nxt)
    return dist


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_is_solvable(preset):
    """No seed of any preset may emit an unreachable switch or door."""
    unsolvable = [s for s in SEEDS if not is_map_solvable(_build(preset, s).map_data())]
    assert not unsolvable, f"{preset}: unsolvable seeds {unsolvable}"


@pytest.mark.parametrize("preset", PRESETS)
def test_goal_triple_is_well_formed(preset):
    """Exactly one door + switch, on standable ground, in distinct places."""
    for seed in SEEDS:
        gen = _build(preset, seed)
        assert gen.entity_counts["exit_door"] == 1
        ninja, switch, door = gen._goal_cells
        assert len({ninja, switch, door}) == 3, f"{preset}/{seed}: goals collide"
        for cell in (ninja, switch, door):
            assert gen.get_tile(*cell) == 0, f"{preset}/{seed}: {cell} is inside rock"
            below = gen.get_tile(cell[0], cell[1] + 1)
            assert below not in (0, -1), f"{preset}/{seed}: {cell} has no floor"


@pytest.mark.parametrize("preset", PRESETS)
def test_chambers_are_multiple_and_linked(preset):
    """The point of the generator: more than one area, all of them connected."""
    for seed in SEEDS:
        gen = _build(preset, seed)
        assert len(gen.chambers) >= 2, f"{preset}/{seed}: only one chamber"
        # A spanning tree over N chambers has N-1 edges; extras add cycles.
        assert len(gen.chamber_edges) >= len(gen.chambers) - 1


def test_route_topologies_come_out_as_configured():
    """``nonlinear_prob`` really does flip the switch and the door around.

    Outbound means the switch is no further from spawn than the door (one run
    out); out-and-back means the switch is *past* the door, so the agent crosses
    the level and returns. Both are checked on tile distance, which is the route
    shape the chamber graph imposes regardless of physics.
    """
    shape = Counter()
    for preset, expected in (("cavern_linear", "outbound"), ("cavern_nonlinear", "back")):
        for seed in range(12):
            gen = _build(preset, seed)
            ninja, switch, door = gen._goal_cells
            dist = _tile_distances(gen, ninja)
            assert switch in dist and door in dist, f"{preset}/{seed}: goal cut off"
            got = "outbound" if dist[switch] <= dist[door] else "back"
            shape[(preset, got == expected)] += 1

    # Not 12/12: the layouts are randomized, and a degenerate graph can collapse
    # the two shapes onto each other. A strong majority is the real contract.
    assert shape[("cavern_linear", True)] >= 10
    assert shape[("cavern_nonlinear", True)] >= 10


def _mines(gen):
    """Yield each mine as ``(entity_type, gx, gy)`` in generator-frame tile units.

    ``add_entity`` stores map-data units (tiles x 4) and the sim wraps the map in
    a 1-tile border, so undoing both recovers coordinates comparable to
    ``get_tile``.
    """
    data = gen.entity_data
    for i in range(0, len(data), 5):
        if data[i] in MINE_TYPES:
            yield data[i], data[i + 1] / 4.0 - 1.0, data[i + 2] / 4.0 - 1.0


@pytest.mark.parametrize("preset", MINED_PRESETS)
def test_every_mine_is_stuck_to_a_surface(preset):
    """A mine centre must land ON a tile boundary, empty one side, solid the other.

    This is the assertion that catches the classic failure: passing raw tile
    coordinates to ``add_entity`` floats a mine 48px off its surface (see the
    +1.5/+2.0 convention note in ``map_cavern_chambers``). A floating mine is not
    a visual nit -- it is a hazard in the middle of the run lane instead of on
    the floor the agent is running along.
    """
    seen = 0
    for seed in SEEDS:
        gen = _build(preset, seed)
        for _mtype, gx, gy in _mines(gen):
            on_h_edge = abs(gy - round(gy)) < 1e-6
            on_v_edge = abs(gx - round(gx)) < 1e-6
            assert on_h_edge or on_v_edge, f"{preset}/{seed}: mine floating at {gx},{gy}"
            if on_h_edge:
                above = gen.get_tile(int(gx), int(gy - 0.25))
                below = gen.get_tile(int(gx), int(gy + 0.25))
                assert {above, below} == {0, 1}, (
                    f"{preset}/{seed}: mine at {gx},{gy} has above={above} below={below}"
                )
            else:
                left = gen.get_tile(int(gx - 0.25), int(gy))
                right = gen.get_tile(int(gx + 0.25), int(gy))
                assert {left, right} == {0, 1}, (
                    f"{preset}/{seed}: mine at {gx},{gy} has left={left} right={right}"
                )
            seen += 1
    assert seen > 0, f"{preset} placed no mines at all"


@pytest.mark.parametrize("preset", MINED_PRESETS)
def test_mines_never_seal_the_route(preset):
    """Goals stay connected even with every mine treated as a solid wall.

    The generator enforces this pessimistically at build time, which is what
    makes it a guarantee rather than a hope: if the route survives mines-as-walls
    then no combination of toggle states can close it.
    """
    for seed in SEEDS:
        gen = _build(preset, seed)
        blocked = set(gen.mine_cells)
        ninja, switch, door = gen._goal_cells
        dist = _tile_distances(gen, ninja, blocked)
        assert switch in dist and door in dist, (
            f"{preset}/{seed}: mines sever the route ({len(blocked)} mines)"
        )


@pytest.mark.parametrize("preset", MINED_PRESETS)
def test_mines_keep_clear_of_spawn(preset):
    """No mine within the spawn-clearance radius.

    A mine that close is a frame-1 death (armed) or a spawn trap (untoggled that
    arms the moment the ninja steps off it) -- and `is_map_solvable` reports the
    goals as perfectly reachable either way, so nothing else catches it.
    """
    from nclone.map_generation.solvability import (
        _build_level_data,
        spawn_mine_clearance_ok,
    )

    for seed in SEEDS:
        gen = _build(preset, seed)
        level = _build_level_data(gen.map_data())
        assert spawn_mine_clearance_ok(level), f"{preset}/{seed}: mine on the spawn"


def test_mines_never_land_in_a_wall_jump_shaft():
    """Vertical corridor runs stay mine-free.

    A 2-wide shaft is climbed by chaining wall-jumps off both faces; a mine
    partway up is undodgeable, and no reachability check models that. Corridors
    are excluded entirely by default, so this also covers the opt-in path.
    """
    for preset in MINED_PRESETS:
        for seed in SEEDS:
            gen = _build(preset, seed)
            offenders = [c for c in gen.mine_cells if c in gen._shaft_cells]
            assert not offenders, f"{preset}/{seed}: mines in shaft cells {offenders}"


def test_mine_type_mix_tracks_the_official_corpus():
    """Roughly a quarter of mines ship armed (type 21), as in official levels."""
    kinds = Counter()
    for seed in range(20):
        gen = _build("cavern_dense_mines", seed)
        kinds.update(mtype for mtype, _gx, _gy in _mines(gen))
    total = sum(kinds.values())
    assert total > 100, f"only {total} mines sampled"
    armed = kinds[21] / total
    assert 0.15 < armed < 0.40, f"armed share {armed:.2f} off the 0.26 target"


def test_mine_free_presets_stay_mine_free():
    for preset in ("cavern_t1", "cavern_linear", "cavern_nonlinear"):
        for seed in SEEDS:
            gen = _build(preset, seed)
            assert not list(_mines(gen)), f"{preset}/{seed} placed mines"


def _entities(gen, types):
    """Yield ``(type, orientation, gx, gy)`` in generator-frame tile units."""
    data = gen.entity_data
    for i in range(0, len(data), 5):
        if data[i] in types:
            yield data[i], data[i + 3], data[i + 1] / 4.0 - 1.0, data[i + 2] / 4.0 - 1.0


def test_traversal_furniture_is_mounted_correctly():
    """Each traversal type lands on the geometry its mechanic requires.

    A one-way in solid rock is invisible; a launch pad firing into a ceiling is a
    trap rather than a shortcut; a bounce block plugged into a wall is a doorstop.
    The alignments asserted here are the ones measured off the official corpus.
    """
    counts = Counter()
    for seed in SEEDS:
        gen = _build("cavern_traversal", seed)
        for etype, orient, gx, gy in _entities(
            gen, (ONE_WAY, BOUNCE_BLOCK, LAUNCH_PAD, BOOST_PAD)
        ):
            cx = int(gx - 0.5)
            if etype == ONE_WAY:
                # Flush on a tile boundary, normal up, air both sides: a platform
                # you pass up through and land on top of.
                cy = int(gy) - 1
                assert orient == 6, f"one-way orientation {orient} (want 6, normal up)"
                assert abs(gy - round(gy)) < 1e-6, f"one-way off-boundary at {gy}"
                assert gen.get_tile(cx, cy) == 0 and gen.get_tile(cx, cy + 1) == 0
            elif etype == BOUNCE_BLOCK:
                cy = int(gy - 0.5)
                assert gen.get_tile(cx, cy) == 0, "bounce block inside rock"
                assert gen.get_tile(cx, cy + 1) == 0, "bounce block is a doorstop"
            elif etype == LAUNCH_PAD:
                cy = int(gy) - 1
                assert orient == 6, f"launch pad orientation {orient} (want 6, up)"
                assert gen.get_tile(cx, cy + 1) == 1, "launch pad has no floor"
                assert gen.get_tile(cx, cy - 3) == 0, "launch pad fires into a ceiling"
            else:
                cy = int(gy - 0.5)
                assert gen.get_tile(cx, cy + 1) == 1, "boost pad has no floor"
            counts[etype] += 1
    for etype in (ONE_WAY, BOUNCE_BLOCK, LAUNCH_PAD, BOOST_PAD):
        assert counts[etype] > 0, f"cavern_traversal never placed type {etype}"


def test_traversal_furniture_never_seals_the_route():
    """Obstructing furniture is proved harmless on the mines-as-walls model.

    A one-way is strictly more permeable than a wall and a bounce block is
    springy, so a route that survives with their cells fully blocked survives
    with the real entities in place. This is what makes one-ways safe to place
    despite being, by construction, a directional valve.
    """
    for preset in ("cavern_oneway", "cavern_traversal"):
        for seed in SEEDS:
            gen = _build(preset, seed)
            blocked = set(gen.mine_cells) | {c for c, _t in gen.entity_cells}
            ninja, switch, door = gen._goal_cells
            dist = _tile_distances(gen, ninja, blocked)
            assert switch in dist and door in dist, (
                f"{preset}/{seed}: furniture severs the route"
            )


def test_traversal_furniture_stays_out_of_corridors():
    """No furniture in a connecting tunnel, and none in a wall-jump shaft.

    A corridor is the one place with no way around, so a one-way there is a valve
    on the only route and a bounce block is a plug.
    """
    for preset in ("cavern_oneway", "cavern_traversal"):
        for seed in SEEDS:
            gen = _build(preset, seed)
            cells = {c for c, _t in gen.entity_cells}
            assert not (cells & gen._corridor_cells), f"{preset}/{seed}: in a corridor"
            assert not (cells & gen._shaft_cells), f"{preset}/{seed}: in a shaft"


def test_traversal_is_off_unless_a_preset_asks():
    """Phase-3 entities must never appear by accident.

    Bounce blocks and launch pads change the cost-to-go field, so a tier that
    silently grew them would invalidate PBRS and the demo corpus without anyone
    choosing that.
    """
    phase3 = (BOUNCE_BLOCK, LAUNCH_PAD)
    for preset in ("cavern_t1", "cavern_t2", "cavern_t3", "cavern_t4",
                   "cavern_dense_mines", "cavern_linear", "cavern_nonlinear"):
        for seed in SEEDS:
            gen = _build(preset, seed)
            found = list(_entities(gen, phase3 + (ONE_WAY, BOOST_PAD)))
            assert not found, f"{preset}/{seed} emitted traversal entities {found}"


def _gold(gen):
    """Yield gold as ``(gx, gy)`` in generator-frame tile units.

    Gold is the only entity here that takes an ``add_entity`` offset
    (GOLD_OFFSET_UNITS), which is what puts an integer coordinate at its cell's
    centre -- so undoing it, not the border, recovers the placement grid.
    """
    data = gen.entity_data
    for i in range(0, len(data), 5):
        if data[i] == GOLD:
            yield (data[i + 1] - 6) / 4.0, (data[i + 2] - 6) / 4.0


@pytest.mark.parametrize("preset", GOLD_PRESETS)
def test_gold_is_open_and_reachable(preset):
    """No gold buried in rock, and none walled into a pocket.

    Unreachable gold is worse than no gold: it cannot be collected, but it still
    consumes observation bandwidth and drags on the orienteering potential, which
    scores every remaining gold as if a detour existed.
    """
    seen = 0
    for seed in SEEDS:
        gen = _build(preset, seed)
        reachable = gen._reachable_cells(gen._goal_cells[0])
        for gx, gy in _gold(gen):
            for cell in gen._gold_footprint(gx, gy):
                assert gen.get_tile(*cell) == 0, f"{preset}/{seed}: gold in rock {cell}"
                assert cell in reachable, f"{preset}/{seed}: gold walled off {cell}"
            seen += 1
    assert seen > 0, f"{preset} placed no gold"


@pytest.mark.parametrize("preset", GOLD_PRESETS)
def test_gold_comes_in_clusters_not_scatter(preset):
    """Gold must arrive in blobs, which is what the collection planner assumes.

    The planner orders CLUSTERS exactly (Held-Karp over super-nodes) and sweeps
    within them; that only works because 93% of official gold is clustered at a
    48px radius. Scattered gold would silently push every level into the
    heuristic fallback.
    """
    for seed in SEEDS:
        gen = _build(preset, seed)
        pts = [(gx, gy) for gx, gy in _gold(gen)]
        if len(pts) < 4:
            continue
        # Each gold should have a neighbour within the corpus's 48px (2-tile)
        # clustering radius -- i.e. it belongs to a blob, not to nothing.
        lonely = [
            p
            for p in pts
            if not any(
                q is not p and abs(p[0] - q[0]) <= 2 and abs(p[1] - q[1]) <= 2
                for q in pts
            )
        ]
        assert not lonely, f"{preset}/{seed}: {len(lonely)} isolated gold"


def test_gold_cluster_shapes_and_sizes():
    """Every declared shape gets used, and cluster sizes stay in the configured band."""
    shapes = Counter()
    for seed in range(24):
        gen = _build("cavern_gold", seed)
        for shape, count, _anchor in gen.gold_clusters:
            shapes[shape] += 1
            assert 3 <= count <= 12, f"cluster of {count} outside the 3-12 band"
    for shape in ("grid", "triangle", "ring", "arc", "blob"):
        assert shapes[shape] > 0, f"shape {shape!r} never produced (got {dict(shapes)})"


def test_gold_density_tracks_the_official_corpus():
    """Counts land in the official band rather than an order off it.

    Official: 24 gold in 5 clusters at the median, p90 51 gold / 11 clusters.
    """
    gold, clusters = [], []
    for seed in range(30):
        gen = _build("cavern_gold", seed)
        gold.append(sum(1 for _ in _gold(gen)))
        clusters.append(len(gen.gold_clusters))
    gold.sort()
    clusters.sort()
    median_gold = gold[len(gold) // 2]
    median_clusters = clusters[len(clusters) // 2]
    assert 10 <= median_gold <= 70, f"median gold {median_gold}"
    assert 3 <= median_clusters <= 12, f"median clusters {median_clusters}"


def test_gold_does_not_sit_on_other_entities():
    """Gold never lands on a mine or a piece of traversal furniture."""
    for seed in SEEDS:
        gen = _build("cavern_traversal", seed)
        taken = set(gen.mine_cells) | {c for c, _t in gen.entity_cells}
        for gx, gy in _gold(gen):
            for cell in gen._gold_footprint(gx, gy):
                assert cell not in taken, f"gold on an entity at {cell}"


@pytest.mark.parametrize("preset", HAZARD_PRESETS)
def test_hazard_counts_respect_their_caps(preset):
    """Drones and emplacements stay inside the configured budgets.

    Emplacement sparsity is not an arbitrary cap: official runs rockets at
    1/1/2/3 per level (p25/50/75/90) with a crowd of 1, and turrets at 1/1/2/4.
    A cavern crowded with turrets would be off-distribution in a way no amount of
    geometry variety compensates for.
    """
    gen0 = _build(preset, 0)
    for seed in SEEDS:
        gen = _build(preset, seed)
        counts = Counter(t for t, _o, _x, _y in _entities(gen, DRONE_TYPES + EMPLACEMENT_TYPES))
        drones = sum(counts[t] for t in DRONE_TYPES)
        empl = sum(counts[t] for t in EMPLACEMENT_TYPES)
        assert drones <= gen0.MAX_DRONES, f"{preset}/{seed}: {drones} drones"
        assert empl <= gen0.MAX_EMPLACEMENTS, f"{preset}/{seed}: {empl} emplacements"


def test_hazards_sit_on_their_own_placement_lattice():
    """Each class lands on the grid its own entity uses.

    The lattice is per TYPE, not per class, and a miss is silent: a zap drone
    steps between 24px cell centres while a mini drone steps on a 12px grid, so a
    mini placed on the zap lattice begins mid-step and its whole patrol is
    phase-shifted against whatever the forecast observation channels predict.
    Values are asserted against the corpus measurements in ``HAZARD_SPECS``.
    """
    seen = Counter()
    for seed in range(16):
        gen = _build("cavern_full", seed)
        for etype, _orient, gx, gy in _entities(
            gen, DRONE_TYPES + EMPLACEMENT_TYPES
        ):
            # Back to bordered pixels, which is the frame the lattices are in.
            px, py = (gx + 1.0) * 24.0, (gy + 1.0) * 24.0
            if etype == ZAP_DRONE:
                assert (px % 24, py % 24) == (12, 12), f"zap drone off-lattice {px},{py}"
            elif etype == MINI_DRONE:
                assert px % 24 in (6, 18) and py % 24 in (6, 18), (
                    f"mini drone off its 12px lattice at {px},{py}"
                )
            else:
                assert px % 12 == 0 and py % 12 == 0, f"emplacement off-lattice {px},{py}"
            seen[etype] += 1
    for etype in DRONE_TYPES + EMPLACEMENT_TYPES:
        assert seen[etype] > 0, f"type {etype} never placed"


@pytest.mark.parametrize("preset", HAZARD_PRESETS)
def test_no_hazard_can_kill_the_spawn(preset):
    """Nothing lethal near the spawn, and no emplacement with a free shot at it.

    An unavoidable death before the agent has taken an action is not difficulty,
    it is a poisoned episode -- and unlike the hazard overlay, which replays a
    routed tape and classifies the outcome, a runtime generator has no gate that
    would catch one after the fact.
    """
    for seed in SEEDS:
        gen = _build(preset, seed)
        ninja = gen._goal_cells[0]
        for cell, kind in gen.hazard_cells:
            chebyshev = max(abs(cell[0] - ninja[0]), abs(cell[1] - ninja[1]))
            assert chebyshev >= gen.HAZARD_SPAWN_CLEARANCE, (
                f"{preset}/{seed}: {kind} {chebyshev} tiles from spawn"
            )
            if kind == "emplacement":
                for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    assert ninja not in gen._ray(cell, step), (
                        f"{preset}/{seed}: emplacement has a clear shot at spawn"
                    )


def test_emplacements_are_mounted_with_a_sightline():
    """A turret needs a wall to sit on and something to see.

    Both are hitscan or homing classes: with no line of sight they are pure
    observation bandwidth attached to nothing, which the hazard census calls out
    as worse than not placing them.
    """
    for seed in SEEDS:
        gen = _build("cavern_full", seed)
        for cell, kind in gen.hazard_cells:
            if kind != "emplacement":
                continue
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
            mounted = any(
                gen.get_tile(cell[0] + s[0], cell[1] + s[1]) == 1 for s in steps
            )
            sight = max(len(gen._ray(cell, s)) for s in steps)
            assert mounted, f"seed {seed}: emplacement floating at {cell}"
            assert sight >= gen.HAZARD_MIN_SIGHTLINE, f"seed {seed}: no sightline"


def test_hazards_are_off_unless_a_preset_asks():
    for preset in ("cavern_t1", "cavern_t2", "cavern_linear", "cavern_nonlinear"):
        for seed in SEEDS:
            gen = _build(preset, seed)
            found = list(_entities(gen, DRONE_TYPES + EMPLACEMENT_TYPES))
            assert not found, f"{preset}/{seed} emitted hazards {found}"


def test_generation_is_deterministic_per_seed():
    for seed in (0, 5):
        first = _build("cavern_t3", seed).map_data()
        second = _build("cavern_t3", seed).map_data()
        assert first == second


def test_tile_palette_is_shaped_not_noise():
    """Slopes and curves must appear, and must stay a minority.

    Both directions matter. A pure 0/1 palette means the smoothing pass silently
    stopped working and the levels are boxes; a *large* shaped fraction means it
    ran away into the uniform-tile-noise regime that put earlier procgen off the
    human manifold. The official corpus sits near 4% shaped tiles.
    """
    hist = Counter()
    for seed in range(12):
        hist.update(_build("cavern_t3", seed).tile_data)
    total = sum(hist.values())
    shaped = sum(count for tile, count in hist.items() if tile > 1)
    assert len(hist) >= 15, f"only {len(hist)} tile types used"
    assert 0.005 < shaped / total < 0.12, f"shaped fraction {shaped / total:.3f}"
