"""The phased training scopes must keep meaning what the plan says they mean.

`docs/ENTITY_INTEGRATION_PLAN.md` §9 phases the full-entity rollout by corpus
coverage, and those percentages are the argument for the phase order -- phase 1
is worth doing first *because* hazards alone triple the reachable corpus without
touching the planner. If a profile silently drifts (an entity moved between
`PURE_HAZARD_TYPES` and `TRAVERSAL_TYPES`, say), the phase boundaries stop
matching the plan and a run's declared scope stops meaning what its name says.

These are measured against the real official corpus, so they double as a
regression test on the map parser and on the hazard/traversal partition itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nclone.constants.entity_types import (
    ENTITY_PROFILES,
    MINIMAL_TRAINING_SUBSET,
    NATIVE_IMPLEMENTED_TYPES,
    PHASE1_HAZARD_SCOPE,
    PHASE2_SCOPE,
    PHASE3_SCOPE,
    PHASE4_SCOPE,
    PURE_HAZARD_TYPES,
    TRAVERSAL_TYPES,
    EntityType,
    map_in_scope,
)

_OFFICIAL = Path(__file__).resolve().parents[1] / "nclone" / "maps" / "official"

# (profile, expected S*-family coverage %). Tolerance is one map in 2382
# (~0.05%), so these pin the exact set, not a ballpark.
_EXPECTED_S_COVERAGE = {
    "minimal": 14.2,
    "phase1": 43.3,
    "phase2": 63.5,
    "phase3": 90.5,
    "phase4": 100.0,
}


@pytest.fixture(scope="module")
def corpus():
    """(is_s_family, entity_type_set) for every official map."""
    from nclone.levelgen.labeling import entity_counts

    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    rows = []
    for path in sorted(_OFFICIAL.rglob("*")):
        if not path.is_file():
            continue
        try:
            counts = entity_counts(path.read_bytes())
        except Exception:
            continue
        rows.append((path.parent.name.startswith("S"), set(counts)))
    assert len(rows) > 3000, f"corpus looks truncated: {len(rows)} maps"
    return rows


def test_phases_are_strictly_nested():
    """Each phase must ADD to the previous one. A phase that drops a type would
    make a resumed run's scope non-monotonic and silently shrink its corpus."""
    chain = [
        MINIMAL_TRAINING_SUBSET,
        PHASE1_HAZARD_SCOPE,
        PHASE2_SCOPE,
        PHASE3_SCOPE,
        PHASE4_SCOPE,
    ]
    for earlier, later in zip(chain, chain[1:]):
        assert earlier < later, f"{sorted(earlier - later)} lost between phases"


def test_phase1_adds_only_hazards():
    """The whole argument for phase 1 landing first is that it needs NO planner
    work, which holds only if every type it adds is a pure hazard."""
    added = PHASE1_HAZARD_SCOPE - MINIMAL_TRAINING_SUBSET
    assert added <= PURE_HAZARD_TYPES, (
        f"phase 1 admits non-hazard types {sorted(added - PURE_HAZARD_TYPES)}; "
        "those change routing and would invalidate the untouched planner."
    )


def test_phase2_adds_only_types_the_graph_already_models():
    """Phase 2 is observation-only for the same reason: locked doors and
    one-ways are already in `graph/reachability/entity_mask.py`."""
    added = PHASE2_SCOPE - PHASE1_HAZARD_SCOPE
    assert added == {
        EntityType.LOCKED_DOOR,
        EntityType.LOCKED_DOOR_SWITCH,
        EntityType.ONE_WAY,
    }


def test_phase3_adds_traversal_types_needing_new_graph_work():
    added = PHASE3_SCOPE - PHASE2_SCOPE
    non_phantom = added - {EntityType.TRAP_DOOR_SWITCH}
    assert non_phantom <= TRAVERSAL_TYPES, (
        f"phase 3 admits {sorted(non_phantom - TRAVERSAL_TYPES)}, which are not "
        "traversal types -- phase 3 is defined as the planner-work phase."
    )


def test_phase4_covers_everything_the_simulators_run():
    assert NATIVE_IMPLEMENTED_TYPES <= PHASE4_SCOPE


@pytest.mark.parametrize("profile,expected_pct", sorted(_EXPECTED_S_COVERAGE.items()))
def test_profile_reproduces_documented_coverage(corpus, profile, expected_pct):
    s_maps = [entities for is_s, entities in corpus if is_s]
    in_scope = sum(1 for e in s_maps if map_in_scope(e, profile))
    pct = 100.0 * in_scope / len(s_maps)
    assert pct == pytest.approx(expected_pct, abs=0.06), (
        f"profile {profile!r} covers {pct:.1f}% of the S* family, but "
        f"ENTITY_INTEGRATION_PLAN.md §9 documents {expected_pct}%. Either the "
        f"profile drifted or the plan's phasing table needs updating -- they "
        f"must not disagree."
    )


def test_ninja_spawn_never_puts_a_map_out_of_scope():
    """Type 0 is on every map; treating it as an entity would empty every scope."""
    assert map_in_scope({EntityType.NINJA}, "minimal")
    assert map_in_scope({EntityType.NINJA, EntityType.GOLD}, "minimal")


def test_profile_names_resolve():
    for name in ("minimal", "native", "phase1", "phase2", "phase3", "phase4", "full"):
        assert name in ENTITY_PROFILES
    assert ENTITY_PROFILES["full"] == PHASE4_SCOPE
