"""The gamma-1 hazard buckets, and the catalog invariant they had to respect.

`generate_benchmark_dataset.py` derives every bucket's seed offset from its
POSITION in `BUCKET_CATALOG` (`bucket_index * BUCKET_STRIDE`). Inserting a bucket
mid-catalog therefore renumbers every bucket after it and silently regenerates
them from different seeds -- the levels still build, they are just not the same
levels, and nothing reports it. That is why `MINE_FIELD` sits at the end rather
than next to `MINE_NAVIGATION` where it reads better, and it is why gamma-1
fills the pre-existing `DRONE` / `THWUMP` / `DEATH_BALL` placeholders IN PLACE
and appends everything genuinely new after `MINE_FIELD`.
"""

from __future__ import annotations

import pytest

from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    CURRICULUM_SCHEDULE,
    GENERATOR_PRESETS,
    PHASE_1_BUCKETS,
    validate_bucket_catalog,
)
from nclone.map_generation.generator_factory import GENERATOR_REGISTRY
from nclone.map_generation.map_hazard_overlay import HAZARD_SPECS

#: The catalog order as of the gamma-1 additions. A diff here is either a
#: deliberate append (extend this list) or the silent renumbering above.
_EXPECTED_ORDER = [
    "FLAT_RUN", "BASIC_JUMP", "WALL_JUMP", "VERTICAL_SHAFT", "MINE_NAVIGATION",
    "SLOPES", "CHAMBER_MAZE", "CURVES", "MULTI_JUMP", "COMPOSITE",
    "PLATFORMING", "PLATFORM_STAIRS", "LOOP_CORRIDOR", "PRECISION",
    "DROP_LANDING", "FAR_OBJECTIVE", "DENSE_COMPOSITE", "SWITCH_PUZZLE",
    "LAUNCH_PAD", "THWUMP", "DRONE", "DEATH_BALL", "MINE_FIELD",
    # gamma-1 appends, in CURRICULUM_SCHEDULE order.
    "BEAM", "ROCKET", "FLOOR_GUARD", "HAZARD_FIELD",
]

_NEW_BUCKETS = ("DRONE", "FLOOR_GUARD", "BEAM", "ROCKET", "THWUMP",
                "DEATH_BALL", "HAZARD_FIELD")


def test_catalog_order_is_append_only():
    assert list(BUCKET_CATALOG) == _EXPECTED_ORDER, (
        "BUCKET_CATALOG order changed. Every bucket's generation seed is its "
        "INDEX here, so a reorder regenerates buckets from different seeds "
        "without any error. Append only."
    )


def test_the_filled_placeholders_did_not_move():
    """DRONE / THWUMP / DEATH_BALL were empty placeholders that already held
    catalog positions, so filling them costs no renumbering. This pins that they
    were filled rather than re-added at the end."""
    order = list(BUCKET_CATALOG)
    for name in ("THWUMP", "DRONE", "DEATH_BALL"):
        assert order.index(name) < order.index("MINE_FIELD"), (
            f"{name} moved past MINE_FIELD; it must be filled in its original "
            "placeholder position"
        )


def test_the_genuinely_new_buckets_are_after_MINE_FIELD():
    order = list(BUCKET_CATALOG)
    for name in ("BEAM", "ROCKET", "FLOOR_GUARD", "HAZARD_FIELD"):
        assert order.index(name) > order.index("MINE_FIELD")


def test_phase2_placeholders_stay_empty():
    """SWITCH_PUZZLE and LAUNCH_PAD need traversal entities that change the
    reachability graph; neither can be served by an overlay."""
    assert BUCKET_CATALOG["SWITCH_PUZZLE"] == {}
    assert BUCKET_CATALOG["LAUNCH_PAD"] == {}


@pytest.mark.parametrize("bucket", _NEW_BUCKETS)
def test_every_new_bucket_has_all_four_tiers(bucket):
    tiers = BUCKET_CATALOG[bucket]
    assert sorted(tiers) == [1, 2, 3, 4], (
        f"{bucket} is missing a tier; the curriculum's tier_min/tier_max "
        "selection would silently resolve to a smaller level set"
    )
    for entries in tiers.values():
        assert entries


@pytest.mark.parametrize("bucket", _NEW_BUCKETS)
def test_t1_is_a_single_isolated_instance_with_no_lethal_levels(bucket):
    """For a mechanic the agent has never seen, tier 1 must be a single instance
    in generous geometry -- the `shaft_t2_one_mine` precedent. And none of its
    levels may require timing: a T1 the agent cannot solve by walking the route
    is not an on-ramp, it is a bucket that reads 0.000 forever."""
    for gen, preset, _w in BUCKET_CATALOG[bucket][1]:
        cfg = GENERATOR_PRESETS[gen][preset]
        assert cfg.single_instance is True, f"{bucket}/T1 {preset}"
        assert cfg.target_lethal_fraction == 0.0, f"{bucket}/T1 {preset}"


@pytest.mark.parametrize("bucket", _NEW_BUCKETS)
def test_the_lethal_share_rises_with_tier(bucket):
    """The real difficulty ramp. Hazard COUNT is not one: the census measures
    official density as flat across the game's own difficulty ordering for every
    class, so a count ladder builds a T4 easier than official's T1."""
    shares = []
    for tier in (1, 2, 3, 4):
        entry = BUCKET_CATALOG[bucket][tier][0]
        shares.append(GENERATOR_PRESETS[entry[0]][entry[1]]
                      .target_lethal_fraction)
    assert shares == sorted(shares) and shares[-1] > shares[0], shares


@pytest.mark.parametrize("bucket", _NEW_BUCKETS)
def test_every_entry_names_a_real_generator_class_and_hazard_class(bucket):
    for tier, entries in BUCKET_CATALOG[bucket].items():
        for gen, preset, weight in entries:
            assert gen in GENERATOR_REGISTRY, f"{bucket}/T{tier}: {gen}"
            assert preset in GENERATOR_PRESETS[gen], f"{bucket}/T{tier}: {preset}"
            assert weight > 0
            for cname in GENERATOR_PRESETS[gen][preset].hazard_classes:
                assert cname in HAZARD_SPECS, (
                    f"{bucket}/T{tier}: unknown hazard class {cname!r}"
                )


def test_catalog_validation_reports_nothing_new():
    """The catalog carries pre-existing errors for `template_*` generators that
    were never registered. Assert only that the hazard work added none."""
    bad = [e for e in validate_bucket_catalog() if "hazard_overlay" in e]
    assert not bad, bad


def test_curriculum_schedule_introduces_hazards_after_the_mechanic_spine():
    """Every hazard bucket is a threat laid over morphology the agent has
    already learned, so the whole spine must precede the block -- and
    HAZARD_FIELD, the composite, must come last of the hazards."""
    idx = {b: i for i, b in enumerate(CURRICULUM_SCHEDULE)}
    for b in _NEW_BUCKETS:
        assert idx[b] > idx["PRECISION"], b
        assert idx[b] < idx["COMPOSITE"], b
    assert idx["DRONE"] == min(idx[b] for b in _NEW_BUCKETS), (
        "patrol drones lead: tier C, exactly forecastable, the cleanest "
        "introduction to a threat that moves"
    )
    assert idx["HAZARD_FIELD"] == max(idx[b] for b in _NEW_BUCKETS)


def test_phase_1_buckets_picks_up_the_hazard_block():
    """`PHASE_1_BUCKETS` filters CURRICULUM_SCHEDULE on non-emptiness, which is
    what let the hazard buckets join by acquiring catalog entries."""
    for b in _NEW_BUCKETS:
        assert b in PHASE_1_BUCKETS
    assert "SWITCH_PUZZLE" not in PHASE_1_BUCKETS
    assert "LAUNCH_PAD" not in PHASE_1_BUCKETS


def test_a_multi_class_preset_emits_ALL_its_classes_not_one_of_them():
    """`classes_per_level` defaults to 1, which picks ONE declared class per
    level. That is right where the classes are alternatives (BEAM's turret vs
    dual laser are separate in official, and rarely co-occur) and WRONG where the
    point is the contrast.

    Measured on the first DRONE build: `drone_mixed` at the default emitted 6 of
    48 T3/T4 levels with no patrol drone at all -- chaser-only levels inside the
    bucket whose gate is supposed to certify patrol drones. The preset exists to
    put a forecastable threat and an unforecastable one in the SAME level.
    """
    for tier in (3, 4):
        presets = [p for _g, p, _w in BUCKET_CATALOG["DRONE"][tier]
                   if "mixed" in p]
        assert presets, f"DRONE/T{tier} lost its mixed entry"
        for preset in presets:
            cfg = GENERATOR_PRESETS["hazard_overlay"][preset]
            assert cfg.classes_per_level == len(cfg.hazard_classes), (
                f"{preset} declares {len(cfg.hazard_classes)} classes but draws "
                f"{cfg.classes_per_level} per level, so levels carrying only "
                "one of them are emitted into a bucket gated on the other"
            )


def test_single_class_presets_cannot_ask_for_more_classes_than_they_declare():
    for gen, presets in GENERATOR_PRESETS.items():
        if gen != "hazard_overlay":
            continue
        for name, cfg in presets.items():
            assert 1 <= cfg.classes_per_level <= len(cfg.hazard_classes), name


def test_only_the_thwump_bucket_carries_solid_bodies():
    """THWUMP is the sole source of a non-zero `solid_map` in gamma-1: every
    solid-body type is out of phase-1 ingest scope, so the official corpus
    cannot produce one. If this ever fails, the `solid_map` gate in
    GAMMA_1_PLAN.md §3.3 became readable somewhere else too."""
    solid = set()
    for bucket in _NEW_BUCKETS:
        for entries in BUCKET_CATALOG[bucket].values():
            for gen, preset, _w in entries:
                for cname in GENERATOR_PRESETS[gen][preset].hazard_classes:
                    if HAZARD_SPECS[cname].is_solid:
                        solid.add(bucket)
    assert solid == {"THWUMP"}, solid
