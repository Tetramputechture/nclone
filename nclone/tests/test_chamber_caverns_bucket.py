"""Guards for the CHAMBER_CAVERNS benchmark bucket (gamma-4).

The bucket exists to close the LAYOUT half of the procgen-vs-official gap: every
other procedural bucket emits a single-area level, while official N++ levels are
several distinct areas the ninja traverses between. `map_cavern_chambers` builds
that macro-structure directly.

What must not regress:

1. **Entity scope.** `CHAMBER_CAVERNS` must stay inside phase-2 scope, because a
   phase-2 run's reachability graph and cost-to-go field are only correct there.
   `cavern_traversal` is split into its own `CHAMBER_CAVERNS_TRAVERSAL` bucket
   precisely because it leaves that scope -- it emits BOUNCE_BLOCK and LAUNCH_PAD
   (phase 3, "the only phase that changes the cost-to-go field") and BOOST_PAD,
   which is outside phase 3 as well. Folding it into a tier of the main bucket
   would silently train a phase-2 run on levels whose PBRS potential is wrong.

2. **The catalog append is append-ONLY.** `generate_benchmark_dataset` derives
   each bucket's seed offset from its POSITION in `BUCKET_CATALOG`, so inserting
   mid-catalog renumbers every later bucket and silently regenerates it from
   different seeds.

3. **All eleven cavern presets are reachable** from one bucket or the other, so
   "all types" does not quietly become "the ones someone remembered".

4. **The traversal bucket stays out of the phase-1/2 default set**, i.e. out of
   `CURRICULUM_SCHEDULE` and therefore out of `PHASE_1_BUCKETS`.
"""
from __future__ import annotations

import collections

import pytest

from nclone.constants.entity_types import ENTITY_PROFILES, EntityType
from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    BUCKET_TIERS,
    CURRICULUM_SCHEDULE,
    GENERATOR_PRESETS,
    PHASE_1_BUCKETS,
)
from nclone.map_generation.generator_factory import GeneratorFactory

MAIN = "CHAMBER_CAVERNS"
TRAVERSAL = "CHAMBER_CAVERNS_TRAVERSAL"
_ENTITY_NAMES = {v: k for k, v in vars(EntityType).items() if isinstance(v, int)}


def _entity_types(map_data) -> collections.Counter:
    ents = map_data[1235:]
    return collections.Counter(
        ents[i] for i in range(0, len(ents) - 4, 5) if ents[i]
    )


def _generate(bucket: str, seeds: int = 4) -> collections.Counter:
    seen = collections.Counter()
    for tier, entries in BUCKET_CATALOG[bucket].items():
        for gen, preset, _w in entries:
            for s in range(seeds):
                seed = 5000 + tier * 131 + s
                g = GeneratorFactory.create_from_preset(gen, preset, seed)
                g.generate(seed=seed)
                seen.update(_entity_types(g.map_data()))
    return seen


def test_both_buckets_are_registered():
    assert MAIN in BUCKET_CATALOG
    assert TRAVERSAL in BUCKET_CATALOG
    assert sorted(BUCKET_CATALOG[MAIN]) == list(BUCKET_TIERS)
    assert sorted(BUCKET_CATALOG[TRAVERSAL]) == list(BUCKET_TIERS)


def test_catalog_append_is_append_only():
    """The two new buckets must be the LAST entries, so every pre-existing
    bucket keeps its seed offset."""
    names = list(BUCKET_CATALOG)
    assert names[-2:] == [MAIN, TRAVERSAL], (
        f"catalog tail is {names[-3:]}; inserting mid-catalog renumbers every "
        "later bucket's seed offset"
    )


def test_every_cavern_preset_is_reachable_from_a_bucket():
    """'All types' must mean all of them."""
    declared = set(GENERATOR_PRESETS["cavern_chambers"])
    used = {
        preset
        for bucket in (MAIN, TRAVERSAL)
        for entries in BUCKET_CATALOG[bucket].values()
        for _gen, preset, _w in entries
    }
    assert used == declared, f"unreferenced cavern presets: {sorted(declared - used)}"


def test_traversal_bucket_is_excluded_from_the_phase_1_set():
    assert MAIN in CURRICULUM_SCHEDULE
    assert MAIN in PHASE_1_BUCKETS
    assert TRAVERSAL not in CURRICULUM_SCHEDULE, (
        "the traversal bucket is phase-3 entity scope; listing it here puts it "
        "into PHASE_1_BUCKETS and every phase-1/2 run picks it up by default"
    )
    assert TRAVERSAL not in PHASE_1_BUCKETS


def test_main_bucket_stays_inside_phase_2_entity_scope():
    """The load-bearing safety property: a phase-2 run's cost-to-go field is only
    correct for phase-2 entities, and PBRS is computed from that field."""
    phase2 = set(ENTITY_PROFILES["phase2"])
    seen = _generate(MAIN)
    assert seen, "generation produced no entities at all"
    outside = sorted(
        _ENTITY_NAMES.get(t, t) for t in seen if t not in phase2
    )
    assert not outside, f"{MAIN} emitted out-of-scope entity types: {outside}"


def test_traversal_bucket_really_does_leave_phase_2_scope():
    """Pins WHY the split exists. If this ever passes trivially (no out-of-scope
    types), the separation has become pointless and should be revisited rather
    than kept as cargo."""
    phase2 = set(ENTITY_PROFILES["phase2"])
    seen = _generate(TRAVERSAL)
    outside = {_ENTITY_NAMES.get(t, t) for t in seen if t not in phase2}
    assert outside, (
        "cavern_traversal no longer leaves phase-2 scope; the dedicated bucket "
        "may no longer be needed"
    )
    assert {"BOUNCE_BLOCK", "LAUNCH_PAD"} & outside


@pytest.mark.parametrize("bucket", [MAIN, TRAVERSAL])
def test_generated_levels_pass_the_dataset_builders_validity_gates(bucket):
    """`generate_benchmark_dataset` rejects a level with no exit door or no
    spawn and retries with another seed; a bucket that fails these routinely
    would burn its retry budget and silently under-fill."""
    for tier, entries in BUCKET_CATALOG[bucket].items():
        for gen, preset, _w in entries:
            for s in range(3):
                seed = 1000 + tier * 97 + s
                g = GeneratorFactory.create_from_preset(gen, preset, seed)
                g.generate(seed=seed)
                md = g.map_data()
                assert md[1156] != 0, f"{preset} seed={seed}: no exit door"
                assert not (md[1231] == 0 and md[1232] == 0), (
                    f"{preset} seed={seed}: no ninja spawn"
                )


def test_tiers_ramp_chamber_count():
    """The tier ladder must actually be a ladder: T4 levels should carry more
    entities than T1 on average, reflecting more chambers and more contents."""
    def mean_entities(tier: int) -> float:
        total = n = 0
        for gen, preset, _w in BUCKET_CATALOG[MAIN][tier]:
            for s in range(4):
                seed = 7000 + tier * 53 + s
                g = GeneratorFactory.create_from_preset(gen, preset, seed)
                g.generate(seed=seed)
                total += sum(_entity_types(g.map_data()).values())
                n += 1
        return total / max(n, 1)

    assert mean_entities(4) > mean_entities(1), "the tier ladder does not ramp"
