"""The LOCKED_DOOR bucket's doors must gate the route, on essentially every level.

That is the bucket's entire reason to exist. The official corpus cannot supply
it: of its 767 solvable locked-door levels, 473 (62%) reach the exit with every
door shut, so their doors are decoration and the milestone reward correctly pays
nothing on them. A door bucket that inherited that distribution would look
identical in every log to one that worked.

Every check below re-derives necessity from the EMITTED MAP, never from the
generator's own bookkeeping -- a generator that certifies itself certifies
nothing. Two of these fixed real defects found while building the bucket, both of
which produced levels that passed every existing check:

* Placing doors before the smoothing pass, which only removes material and so
  re-opened a route around the gate (2 of 12 T1 levels).
* Requiring BOTH exit switch and exit door to be unreachable doors-shut, which
  rejected every legitimate T1 layout and silently emitted the trivial fallback
  level instead (16 of 16).
"""

import pytest

from nclone.map_generation.generator_configs import BUCKET_CATALOG
from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.value_function.door_necessity import count_locked_doors, necessary_doors


TIER_PRESETS = ("door_t1", "door_t2", "door_t3", "door_t4")


def _gen(preset, seed):
    g = GeneratorFactory.create_from_preset("cavern_chambers", preset, seed=seed)
    g.generate(seed=seed)
    return g


@pytest.mark.parametrize("preset", ("door_probe",) + TIER_PRESETS)
def test_every_level_carries_a_necessary_door(preset):
    """Verified with the same solver the reward reads."""
    n, with_nec = 8, 0
    for s in range(n):
        g = _gen(preset, 500 + s)
        md = g.map_data()
        assert count_locked_doors(md) >= 1, f"{preset} seed {s}: no locked door at all"
        nec = necessary_doors(md, name=f"{preset}_{s}", use_cache=False)
        if nec.solvable and nec.n_necessary > 0:
            with_nec += 1
    # 1.0 is the design target; the slack is the physics flood finding a wall-jump
    # route the generator's tile-level flood cannot express. Official scores 0.38.
    assert with_nec >= n - 1, (
        f"{preset}: only {with_nec}/{n} levels have a NECESSARY door -- the "
        "bucket has regressed toward the official corpus's decorative doors"
    )


def test_a_door_preset_never_emits_a_door_free_level():
    """Including via the fallback.

    A door bucket with no doors in it is invisible downstream: the bucket emits,
    the pool fills, the curriculum advances, and only a necessity audit would
    ever notice. Measured: T3 and T4 hit exactly this, producing 0 doors on 12 of
    12 levels while reporting success.
    """
    for preset in TIER_PRESETS:
        for s in range(6):
            assert count_locked_doors(_gen(preset, 900 + s).map_data()) >= 1


def test_fallback_level_still_carries_a_door():
    """The last-resort layout is where the door-free hole opened up."""
    from nclone.map_generation.map_cavern_chambers import MapCavernChambers

    g = MapCavernChambers(seed=1)
    g.MIN_LOCKED_DOORS, g.MAX_LOCKED_DOORS = 1, 1
    g._fallback()
    assert count_locked_doors(g.map_data()) == 1
    assert g.necessary_door_count == 1


def test_tiers_ramp_dependency_depth():
    """Chain length is the axis the tiers move.

    Sequential subtasks are learned fastest in their natural order, so the ramp
    has to be depth -- T1 one door, T4 a chain of several -- rather than geometry
    alone. If T4 stops out-ranking T1 the tiers have collapsed into one tier.
    """
    means = {}
    for preset in TIER_PRESETS:
        counts = []
        for s in range(12):
            nec = necessary_doors(_gen(preset, 700 + s).map_data(), use_cache=False)
            counts.append(nec.n_necessary if nec.solvable else 0)
        means[preset] = sum(counts) / len(counts)
    # T1 and T2 both sit near 1 by design -- T2's contribution is that SOME of
    # its levels chain, not that all of them do -- so they are not separated
    # here; over 160 audited levels they measure 1.00 and 1.23. The ramp that has
    # to hold is T2 -> T3 -> T4, plus a real gap end to end.
    assert means["door_t2"] >= means["door_t1"] - 0.35, means
    assert means["door_t3"] > means["door_t2"], means
    assert means["door_t4"] > means["door_t3"], means
    assert means["door_t4"] >= means["door_t1"] + 1.5, means


def test_t1_is_a_diagnostic_not_a_warmup():
    """One door, no hazards, no decoys -- so a T1 failure indicts the reward.

    T1 is the tier that tells you whether the mechanic is learnable at all. If it
    carries mines or distractor doors, a failure there is unattributable.
    """
    for s in range(6):
        g = _gen("door_t1", 800 + s)
        assert count_locked_doors(g.map_data()) == 1
        assert not g.mine_cells
        assert not g.hazard_cells
        assert all(nec for _g, _s, nec in g.locked_door_cells)


def test_bucket_is_registered_with_four_tiers():
    tiers = BUCKET_CATALOG.get("LOCKED_DOOR")
    assert tiers and sorted(tiers) == [1, 2, 3, 4]
    for entries in tiers.values():
        assert entries and all(gen == "cavern_chambers" for gen, _p, _w in entries)


def test_decoy_doors_are_not_necessary():
    """T3+ mixes load-bearing doors with decoration on purpose.

    The reward pays only for necessary doors, so the data has to contain the
    distinction -- otherwise the agent learns "open every door" rather than "open
    the doors on the route", which is the wrong lesson for a corpus that is
    mostly decoration.
    """
    saw_decoy = False
    for s in range(12):
        g = _gen("door_t3", 600 + s)
        if any(not nec for _g, _sw, nec in g.locked_door_cells):
            saw_decoy = True
            md = g.map_data()
            nec = necessary_doors(md, use_cache=False)
            if nec.solvable:
                assert nec.n_necessary < count_locked_doors(md)
    if not saw_decoy:
        pytest.skip("no decoy sampled in this seed range")
