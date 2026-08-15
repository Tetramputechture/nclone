"""Guards on the entity capability/contract profiles.

The bug these exist to prevent: the ingest allowed-set was a bare literal
``{1,2,3,4,21}`` that nobody updated while `nclone-native` gained parity on every
entity type, so the human corpus stayed at ~8% reach for the whole effort. The
fix was to name the profiles; these tests keep them honest as entities change.
"""

from __future__ import annotations

from nclone.constants.entity_types import (
    ENTITY_PROFILES,
    MINIMAL_TRAINING_SUBSET,
    NATIVE_IMPLEMENTED_TYPES,
    PHANTOM_ENTITY_TYPES,
    PURE_HAZARD_TYPES,
    TRAVERSAL_TYPES,
    EntityType,
)
from nclone.levelgen.labeling import subset_ok
from nclone.map_augmentation.attract.filter_attracts import check_file  # noqa: F401


def _map_data(entity_types):
    """Minimal well-formed map_data carrying the given entity types."""
    md = [0] * 1235
    for t in entity_types:
        md.extend([t, 20, 20, 0, 0])
    md.extend([0, 0, 0, 0, 0])
    return md


class TestProfilePartition:
    def test_hazard_and_traversal_partition_the_widening(self):
        """Every type `native` adds over `minimal` is classified as exactly one of
        hazard (strippable) or traversal (not strippable).

        A new entity type that lands without being classified fails here rather
        than silently defaulting to "safe to delete", which would corrupt demos.
        """
        widening = NATIVE_IMPLEMENTED_TYPES - MINIMAL_TRAINING_SUBSET
        assert PURE_HAZARD_TYPES | TRAVERSAL_TYPES == widening
        assert not (PURE_HAZARD_TYPES & TRAVERSAL_TYPES)

    def test_phantoms_are_not_implementable_types(self):
        """7 and 9 are switch halves owned by their door -- never constructed."""
        assert not (NATIVE_IMPLEMENTED_TYPES & PHANTOM_ENTITY_TYPES)

    def test_native_is_strictly_wider_than_minimal(self):
        assert MINIMAL_TRAINING_SUBSET < NATIVE_IMPLEMENTED_TYPES

    def test_profiles_registry_matches_the_sets(self):
        assert ENTITY_PROFILES["minimal"] == MINIMAL_TRAINING_SUBSET
        assert ENTITY_PROFILES["native"] == NATIVE_IMPLEMENTED_TYPES

    def test_gate_report_reads_the_shared_profile(self):
        """The gate report must not keep a private copy of "what's implemented"."""
        from scripts.attract_gate_report import IMPLEMENTED

        assert set(IMPLEMENTED) == set(NATIVE_IMPLEMENTED_TYPES)


class TestSubsetOk:
    def test_default_is_the_narrow_generator_contract(self):
        """Widening the *capability* profile must not widen the *contract* one.

        `subset_ok` gates procedurally generated levels: a locked door leaking out
        of a corridor preset is a generator bug, and stays one however much the
        simulator can run.
        """
        md = _map_data([EntityType.LOCKED_DOOR, EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH])
        assert not subset_ok(md)

    def test_explicit_wide_profile_accepts_human_levels(self):
        md = _map_data([EntityType.LOCKED_DOOR, EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH])
        assert subset_ok(md, allowed=NATIVE_IMPLEMENTED_TYPES)

    def test_still_requires_exactly_one_exit_pair(self):
        two_doors = _map_data(
            [EntityType.EXIT_DOOR, EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH]
        )
        assert not subset_ok(two_doors, allowed=NATIVE_IMPLEMENTED_TYPES)

    def test_phantom_door_switch_does_not_reject(self):
        """Regression: the door-switch half is not an entity.

        `subset_ok` counted types 7/9 as ordinary entities, so every locked- or
        trap-door level failed even under the wide profile -- which held the
        packable corpus at 7% *after* the allowed-set had been widened, and made
        the widening look like it had barely helped.
        """
        md = _map_data(
            [
                EntityType.LOCKED_DOOR,
                EntityType.LOCKED_DOOR_SWITCH,
                EntityType.TRAP_DOOR,
                EntityType.TRAP_DOOR_SWITCH,
                EntityType.EXIT_DOOR,
                EntityType.EXIT_SWITCH,
            ]
        )
        assert subset_ok(md, allowed=NATIVE_IMPLEMENTED_TYPES)

    def test_phantoms_alone_do_not_widen_the_narrow_contract(self):
        """Phantom tolerance must not smuggle doors past the generator contract."""
        md = _map_data(
            [
                EntityType.LOCKED_DOOR,
                EntityType.LOCKED_DOOR_SWITCH,
                EntityType.EXIT_DOOR,
                EntityType.EXIT_SWITCH,
            ]
        )
        assert not subset_ok(md)

    def test_minimal_subset_level_passes_both_profiles(self):
        md = _map_data(
            [EntityType.TOGGLE_MINE, EntityType.GOLD, EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH]
        )
        assert subset_ok(md)
        assert subset_ok(md, allowed=NATIVE_IMPLEMENTED_TYPES)


class TestPhantomHandling:
    def test_door_switch_halves_do_not_reject_a_level(self, tmp_path):
        """A locked door writes a second record tagged type 7 holding the switch
        coordinates. Treating that as a disallowed entity rejected every door
        level -- the exact failure the phantom subtraction prevents.
        """
        md = _map_data(
            [
                EntityType.LOCKED_DOOR,
                EntityType.LOCKED_DOOR_SWITCH,
                EntityType.EXIT_DOOR,
                EntityType.EXIT_SWITCH,
            ]
        )
        p = tmp_path / "lvl.bin"
        p.write_bytes(bytes(v & 0xFF for v in md))

        name, passed, reason = check_file(str(p), allowed=NATIVE_IMPLEMENTED_TYPES)
        assert passed, reason

    def test_genuinely_unsupported_type_is_still_rejected(self, tmp_path):
        md = _map_data([99, EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH])
        p = tmp_path / "lvl.bin"
        p.write_bytes(bytes(v & 0xFF for v in md))

        name, passed, reason = check_file(str(p), allowed=NATIVE_IMPLEMENTED_TYPES)
        assert not passed
        assert "99" in reason
