"""`map_hazard_overlay` placement preconditions and certification semantics.

The expensive half of this generator (route, replay, count ladder) is exercised
by `npp-rl/scripts/audit_hazard_overlay.py`, which measures what it EMITS
against the official census -- that is the check that matters and it costs
~15 s per level. What is tested here is everything that can be checked without
the planner, and specifically the class of bug that made the first version
silently wrong: preconditions that pass while placing the entity somewhere it
cannot function.

The phantom-mine incident is the precedent. All PLATFORMING / MINE_NAVIGATION
mines were buried in solid rock while poisoning 60-72% of standable ground, and
nothing failed -- the levels generated, the dataset built, the training ran. A
hazard placed without its precondition is WORSE than no hazard: it costs
observation bandwidth and teaches nothing.
"""

from __future__ import annotations

import pytest

from nclone.map_generation.map_hazard_overlay import (
    EMPTY_TILE,
    GRID_W,
    HAZARD_SPECS,
    MapHazardOverlay,
    SOLID_TILES,
    TILE_START,
    _ORIENT_VECTOR,
)

_PX = 24.0


def _flat_world(floor_row: int = 10) -> list[int]:
    """A map_data with one solid floor row and empty space above and below."""
    md = [0] * 2000
    for x in range(GRID_W):
        md[TILE_START + floor_row * GRID_W + x] = 1
    return md


def _y_of_row(row: int) -> float:
    """Top edge of tile row `row`, in the entity pixel frame.

    `_tile_at` maps a pixel to `int((p - 24) // 24)`, so row r spans
    [24(r+1), 24(r+2)).
    """
    return _PX * (row + 1)


# ---------------------------------------------------------------------------
# the spec table
# ---------------------------------------------------------------------------


def test_every_spec_declares_a_lattice_per_TYPE():
    """A zap drone steps between 24 px cell centres and a mini drone on a 12 px
    grid, so "patrol drone" is two lattices, not one. A drone off its own
    lattice starts mid-step between cells and its whole patrol is phase-shifted
    relative to anything the forecast channels predict."""
    for name, spec in HAZARD_SPECS.items():
        assert spec.types, name
        for etype, weight, grid, ox, oy in spec.types:
            assert weight > 0
            assert grid in (12.0, 24.0), f"{name}/{etype}: grid {grid}"
            assert 0.0 <= ox < grid and 0.0 <= oy < grid, f"{name}/{etype}"


def test_the_measured_drone_lattices_match_the_corpus():
    """5292/5292 zap drones sit at (12, 12) mod 24; 5230/5230 mini drones at
    (6|18, 6|18) mod 24, which is the 12 px grid offset by 6. Both follow
    directly from DRONE_GRID_SIZE = 24 and MINI_DRONE_GRID_SIZE = 12."""
    by_type = {t: (g, ox, oy)
               for t, _w, g, ox, oy in HAZARD_SPECS["patrol_drone"].types}
    assert by_type[14] == (24.0, 12.0, 12.0)
    assert by_type[26] == (12.0, 6.0, 6.0)


def test_orientation_is_declared_only_where_the_sim_reads_it():
    """`utils.entity_factory.create_entity_instance` passes orientation only for
    LAUNCH_PAD / ONE_WAY / THWUMP / TURRET and the drone family. Declaring
    orientations for anything else writes a byte the sim ignores, which then
    reads as calibration that is not calibration."""
    takes_orientation = {14, 15, 26, 12, 13, 23, 19, 20}
    for name, spec in HAZARD_SPECS.items():
        if not spec.orientations:
            continue
        for etype, *_rest in spec.types:
            assert etype in takes_orientation, (
                f"{name} declares orientations but type {etype} ignores them"
            )


def test_only_thwumps_are_solid():
    solid = {n for n, s in HAZARD_SPECS.items() if s.is_solid}
    assert solid == {"thwump", "shove_thwump"}


def test_uncalibrated_classes_are_flagged():
    """thwump / shove_thwump are out of phase-1 ingest scope and eye_bat has zero
    instances in the corpus, so none of their numbers is a measurement. The flag
    is what stops them being read as one."""
    assert {n for n, s in HAZARD_SPECS.items() if s.uncalibrated} == {
        "thwump", "shove_thwump", "eye_bat"
    }
    for n, s in HAZARD_SPECS.items():
        if not s.uncalibrated:
            assert 0.0 < s.near_route_fraction < 1.0, n
            assert s.count_p25 <= s.count_p50 <= s.count_p75 <= s.count_p90, n


def test_count_bands_are_ordered_and_crowd_is_consistent():
    for n, s in HAZARD_SPECS.items():
        assert s.crowd_p50 <= s.crowd_p90, n
        assert s.crowd_p90 >= 1, n


# ---------------------------------------------------------------------------
# preconditions
# ---------------------------------------------------------------------------


def test_has_support_probes_exactly_one_tile_and_no_further():
    """The bound is inclusive of 24 px and nothing beyond. An earlier version
    incremented before testing its loop condition and probed 30 px, so the
    generator's idea of "supported" disagreed with the census's -- and a support
    statistic that is not the one official was measured with is not a
    calibration."""
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    floor_top = _y_of_row(10)

    assert g._has_support(md, 100.0, floor_top - 6.0)
    assert g._has_support(md, 100.0, floor_top - 24.0)
    assert not g._has_support(md, 100.0, floor_top - 30.0)


def test_has_support_ignores_non_solid_tiles():
    """Slopes and curves are not in SOLID_TILES, and the census measured official
    support with the fully-solid set."""
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    md[TILE_START + 10 * GRID_W + 4] = 6  # a slope, under x tile 4
    x = _PX * 4 + 36.0
    assert not g._has_support(md, x, _y_of_row(10) - 12.0)


def test_seat_lands_in_free_space_on_the_types_own_lattice():
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    # Start within SEAT_MAX_DROP_PX (120) of the floor: the drop limit is real
    # and a hazard seated from five tiles up would land somewhere unrelated to
    # the reachable cell it was drawn from.
    seat = g._seat_on_surface(md, 300.0, _y_of_row(10) - 96.0,
                              lattice=24.0, offset=12.0)
    assert seat is not None
    _sx, sy = seat
    assert sy % 24.0 == 12.0, "off the type's lattice phase"
    assert g._is_free(md, *seat), "seated INSIDE the surface"
    assert g._has_support(md, *seat), "seated but not supported"


def test_seat_returns_None_rather_than_reaching_past_the_drop_limit():
    g = MapHazardOverlay(1)
    md = [0] * 2000  # no floor anywhere
    assert g._seat_on_surface(md, 300.0, 60.0, lattice=12.0, offset=0.0) is None


def test_clear_lane_rejects_a_mount_facing_a_wall():
    """A thwump mounted against a wall it immediately charges into never moves,
    which is the solid-body equivalent of a mine buried in rock."""
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    for y in range(GRID_W):
        pass
    # Wall at tile column 6, spanning the rows above the floor.
    for row in range(5, 10):
        md[TILE_START + row * GRID_W + 6] = 1

    x = _PX * 5 + 36.0  # inside column 5, just left of the wall
    y = _y_of_row(8) + 12.0
    assert not g._clear_lane(md, x, y, orientation=0, tiles=3), "0 is +x, INTO the wall"
    assert g._clear_lane(md, x, y, orientation=4, tiles=3), "4 is -x, away from it"


def test_orientation_vectors_match_the_simulators_own_table():
    """`physics.map_orientation_to_vector`. 0 is (+1, 0) -- NOT "up", which is
    the misconception that made `map_augmentation.transforms._ORIENT_FLIP` wrong
    for every axis-aligned case."""
    from nclone.physics import map_orientation_to_vector

    for o in range(8):
        want = map_orientation_to_vector(o)
        got = _ORIENT_VECTOR[o]
        assert got[0] == pytest.approx(want[0]), o
        assert got[1] == pytest.approx(want[1]), o


def test_los_is_blocked_by_geometry_and_clear_without_it():
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=20)
    y = 200.0
    assert g._has_los(md, 100.0, y, [(400.0, y)])
    for row in range(6, 10):
        md[TILE_START + row * GRID_W + 10] = 1
    assert not g._has_los(md, 100.0, _y_of_row(7) + 12.0,
                          [(400.0, _y_of_row(7) + 12.0)])


def test_anchors_are_free_and_split_by_real_support():
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    # Nodes hugging the floor: snapping these to a lattice puts several INSIDE
    # the floor, which is the failure `_anchors_for` filters out.
    nodes = [(x * 1.0, _y_of_row(10) - 4.0) for x in range(100, 500, 13)]
    grounded, airborne = g._anchors_for(md, nodes, grid=24.0, ox=12.0, oy=12.0)
    assert grounded, "no grounded anchors at all -- the seat extension is dead"
    for px, py in grounded:
        assert g._is_free(md, px, py)
        assert g._has_support(md, px, py)
        assert py % 24.0 == 12.0
    for px, py in airborne:
        assert g._is_free(md, px, py)
        assert not g._has_support(md, px, py)


def test_unknown_hazard_class_is_a_loud_error():
    g = MapHazardOverlay(1)
    g.HAZARD_CLASSES = ("no_such_hazard",)
    with pytest.raises(ValueError, match="no known class"):
        g.generate(seed=1)


# ---------------------------------------------------------------------------
# certification semantics
# ---------------------------------------------------------------------------


def test_a_lethal_level_ships_no_demo():
    """A lethal level's route is not a winning route. Shipping it as BC data
    would teach the policy to walk into the hazard -- and these levels are kept
    on purpose (they are the ones that teach timing), so the guard has to live
    at commit time rather than at selection time."""
    g = MapHazardOverlay(1)
    md = _flat_world()
    g._commit(md, [(100.0, 100.0, 14, 0, 0)], "lethal", [1, 2, 3], 99,
              "spec", 1.0, {})
    assert g.hazard_outcome == "lethal"
    assert g.route_actions is None
    assert g.route_frames is None


def test_a_safe_level_ships_its_route():
    g = MapHazardOverlay(1)
    md = _flat_world()
    g._commit(md, [(100.0, 100.0, 14, 0, 0)], "safe", [1, 2, 3], 99,
              "spec", 1.0, {})
    assert g.hazard_outcome == "safe"
    assert g.route_actions == [1, 2, 3]
    assert g.route_frames == 99


def test_achieved_counts_describe_what_was_ADOPTED_not_what_was_drawn():
    """Reporting the draw was wrong by up to the whole count ladder: a thinned
    level reported 5 hazards and emitted 1, so an audit reading `_achieved`
    would certify a density the bucket does not have -- exactly the mistake this
    generator exists to stop making."""
    g = MapHazardOverlay(1)
    md = _flat_world()
    placements = [(100.0, 100.0, 14, 0, 0), (200.0, 100.0, 26, 0, 0),
                  (300.0, 100.0, 16, 0, 0)]
    g._commit(md, placements, "safe", [], None, "spec", 0.25,
              {"patrol_drone": 99, "floor_guard": 99})
    assert g._achieved == {"patrol_drone": 2, "floor_guard": 1}


# --- spawn clearance ---------------------------------------------------------
# `_sample_class` rejects candidates inside SPAWN_CLEARANCE_PX, but that is a
# filter inside the sampler and nothing downstream re-checked it. A drone,
# rocket or turret spawned on top of the ninja decides the episode before the
# policy observes anything: the level teaches nothing and still counts as a
# death, and the only symptom is a bucket that reads 0.000 forever -- which is
# indistinguishable from an encoding bug. Hence a post-condition at the single
# commit point, and these tests.

def _spawned_at(md: list[int], px: float, py: float) -> list[int]:
    from nclone.map_generation.map_hazard_overlay import SPAWN_X_IDX, SPAWN_Y_IDX
    md = list(md)
    md[SPAWN_X_IDX] = px / 6.0
    md[SPAWN_Y_IDX] = py / 6.0
    return md


@pytest.mark.parametrize("etype,name", [(14, "patrol_drone"), (26, "mini_drone"),
                                        (18, "rocket"), (19, "turret")])
def test_a_hazard_inside_the_spawn_clearance_is_a_loud_error(etype, name):
    """The three classes named in the requirement plus the mini drone: a
    point-blank spawn must RAISE, not be silently dropped -- a violation means
    the sampler's guard is broken and filtering would hide that."""
    g = MapHazardOverlay(1)
    md = _spawned_at(_flat_world(), 240.0, 216.0)
    # 24 px away: half the clearance.
    with pytest.raises(AssertionError, match="spawn clearance"):
        g._commit(md, [(264.0, 216.0, etype, 0, 0)], "safe", [], None,
                  "spec", 1.0, {})


def test_the_spawn_clearance_boundary_is_inclusive():
    """`never LESS than 48 px` -- exactly 48 px is legal, 47.9 is not."""
    g = MapHazardOverlay(1)
    md = _spawned_at(_flat_world(), 240.0, 216.0)
    clear = float(MapHazardOverlay.SPAWN_CLEARANCE_PX)
    g._commit(md, [(240.0 + clear, 216.0, 14, 0, 0)], "safe", [], None,
              "spec", 1.0, {})
    assert g._achieved == {"patrol_drone": 1}
    with pytest.raises(AssertionError):
        MapHazardOverlay(1)._commit(
            md, [(240.0 + clear - 0.1, 216.0, 14, 0, 0)], "safe", [], None,
            "spec", 1.0, {})


def test_the_spawn_clearance_cannot_be_lowered_by_a_preset():
    """`GeneratorFactory` upper-cases every config field onto the generator, so
    a `spawn_clearance_px` field would silently override the guard. There is
    none, and this test is what keeps it that way."""
    from dataclasses import fields
    from nclone.map_generation.generator_configs import HazardOverlayConfig

    assert MapHazardOverlay.SPAWN_CLEARANCE_PX == 48.0
    assert not any(f.name.upper() == "SPAWN_CLEARANCE_PX"
                   for f in fields(HazardOverlayConfig))


# --- placement preconditions and density ------------------------------------
# Both of these were reported from watching the levels and then confirmed by
# measuring the packed corpus:
#   * floor guards airborne -- only 32% grounded at FLOOR_GUARD/T1, against a
#     spec target of 0.83 that was itself wrong (a floor guard charges ALONG a
#     floor; the 0.83 is an artefact of probing official geometry with
#     SOLID_TILES only, and 691/767 official instances sit on the floor line).
#   * DRONE far too dense -- 4.07 hazards/1k reachable cells at T2 against
#     official's 0.91 (4.5x), 6.98 at T4 with a p90 of 117.9. The cause is
#     count bands measured CONDITIONAL on the class being present, applied to
#     every level with no area term.

def test_floor_guards_require_ground_and_target_full_support():
    spec = HAZARD_SPECS["floor_guard"]
    assert spec.requires_support is True
    assert spec.supported_fraction == 1.0


def test_only_ground_dwelling_classes_require_support():
    """A hard precondition on a flying class would empty its pool."""
    for name in ("patrol_drone", "chaser_drone", "death_ball", "eye_bat",
                 "rocket", "turret", "dual_laser"):
        assert HAZARD_SPECS[name].requires_support is False, name


def test_an_unsupported_mount_is_rejected_for_a_support_required_class():
    g = MapHazardOverlay(1)
    md = _flat_world(floor_row=10)
    # `_tile_at` offsets pixel coords ONE TILE from the tile-grid origin, so the
    # solid row at tile y=10 occupies py in [11*24, 12*24). Deriving the probe
    # points instead of assuming them: the first version of this test asserted
    # against py = 240 and failed, and the geometry was wrong, not the code.
    floor_top = (10 + 1) * _PX
    assert g._tile_at(md, 120.0, floor_top + 1.0) in SOLID_TILES
    assert g._has_support(md, 120.0, floor_top - 6.0) is True
    assert g._has_support(md, 120.0, floor_top - 144.0) is False


def test_the_density_cap_bounds_count_by_reachable_area():
    """The count band still sets the shape; area sets the ceiling."""
    import random
    g = MapHazardOverlay(1)
    g.SINGLE_INSTANCE = False
    g.COUNT_SCALE = 1.0
    g.DENSITY_CAP_PER_1K = 2.0
    spec = HAZARD_SPECS["patrol_drone"]          # p25..p75 = 8..28
    rng = random.Random(0)
    # A small chamber: 500 reachable cells -> at most 1 hazard.
    assert g._target_count(rng, spec, 500) <= 1
    # A large one: 10_000 cells -> the band applies, capped at 20.
    big = g._target_count(rng, spec, 10_000)
    assert 1 <= big <= 20


def test_the_density_cap_never_zeroes_a_bucket():
    """A cap that returns 0 would silently produce hazard-free hazard levels --
    the failure this generator exists to prevent."""
    import random
    g = MapHazardOverlay(1)
    g.SINGLE_INSTANCE = False
    g.DENSITY_CAP_PER_1K = 0.1
    assert g._target_count(random.Random(0), HAZARD_SPECS["patrol_drone"], 10) >= 1


def test_the_tiers_ramp_density_monotonically():
    from nclone.map_generation.generator_configs import GENERATOR_PRESETS
    caps = [GENERATOR_PRESETS["hazard_overlay"][f"drone_hazard_t{t}"]
            .density_cap_per_1k for t in (1, 2, 3, 4)]
    assert caps == sorted(caps) and caps[0] < caps[-1], caps


def test_the_density_cap_is_expressed_in_CELLS_not_graph_nodes():
    """The unit mismatch that made the cap inert on its first attempt.

    The reachability graph stores traversal STATES, not positions -- measured
    8,540 nodes over 2,196 cells, ~3.9x. `DENSITY_CAP_PER_1K` is calibrated
    against official in 12 px cells, so feeding it raw node counts inflated the
    ceiling ~4x and it stopped binding: a cap of 2.0 permitted 17-31 hazards
    against a draw band of 8-28. The regenerated DRONE T2 came out at 5.21/1k,
    WORSE than the 4.07 it was meant to fix, and an 8-seed spot check missed it.
    """
    g = MapHazardOverlay(1)
    # Four nodes per 12 px cell, the ratio the graph actually produces.
    nodes = [(x * 12.0 + o, y * 12.0 + o2)
             for x in range(20) for y in range(10)
             for o, o2 in ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0))]
    assert len(nodes) == 800
    assert g._standable_cells(nodes) == 200, g._standable_cells(nodes)


def test_the_cap_actually_binds_on_a_realistic_area():
    """A cap that never fires is the failure mode, so assert it FIRES."""
    import random
    g = MapHazardOverlay(1)
    g.SINGLE_INSTANCE = False
    g.COUNT_SCALE = 1.0
    g.DENSITY_CAP_PER_1K = 2.0
    spec = HAZARD_SPECS["patrol_drone"]      # draws 8..28
    # 2,200 cells is a typical procgen chamber -> ceiling of 4, well under the
    # draw band, so every draw must be truncated.
    for seed in range(20):
        assert g._target_count(random.Random(seed), spec, 2200) <= 4
