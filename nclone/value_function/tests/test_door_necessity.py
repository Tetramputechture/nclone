"""The standalone, cached necessity answer -- the thing the milestone reward reads.

These pin the two properties the reward depends on and one that gates deployment:

1. **Decoupling.** ``necessary_doors`` must answer without building a door-legged
   potential.  That coupling is why the reward was never separable from a field
   the gamma-5 A/B had already refuted.
2. **Agreement.** It must give the same answer as the full solver, which is the
   one the door-aware potential descends.  If the two disagree, the reward's
   phase and the potential's phase disagree, and this area has already been
   burned by exactly that.
3. **Caching.** A cold solve was measured at up to 164 s over the official
   corpus (p99 67 s).  Every subsequent read has to come from a cache or no run
   can afford the flag.
"""

import glob
import os

import pytest

from nclone.value_function.door_necessity import (
    clear_caches,
    count_locked_doors,
    live_solve_count,
    necessary_doors,
)


_OFFICIAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "maps", "official",
)


def _map_bytes(path):
    with open(path, "rb") as f:
        return list(f.read())


def _door_levels(limit=6):
    """A few official levels that carry a locked door, cheapest-looking first.

    Sorted by door count so the suite spends its time on levels whose solve is
    seconds, not minutes -- the tail runs to 380 doors.
    """
    paths = [p for p in glob.glob(os.path.join(_OFFICIAL, "**", "*"), recursive=True)
             if os.path.isfile(p)]
    rows = []
    for p in paths:
        try:
            d = _map_bytes(p)
        except Exception:
            continue
        n = count_locked_doors(d)
        if n:
            rows.append((n, p, d))
    rows.sort(key=lambda r: r[0])
    return rows[:limit]


@pytest.fixture(scope="module")
def door_levels():
    rows = _door_levels()
    if not rows:
        pytest.skip("no official locked-door maps available")
    return rows


def test_no_doors_costs_no_solve():
    """The 77% of levels with no locked door must never reach the solver.

    This is what keeps the flag off the reset path for every non-door bucket.
    """
    clear_caches()
    before = live_solve_count()
    # A map body with no entity records at all: no doors, nothing to solve.
    empty = [0] * 1200
    got = necessary_doors(empty)
    assert got.n_doors == 0
    assert got.order == ()
    assert not got.has_doors
    assert live_solve_count() == before, "a door-free level started a solve"


def test_matches_the_full_solver(door_levels):
    """Necessity computed WITHOUT the legs equals necessity computed with them.

    ``compute_legs=False`` is a pure cost optimization; if it changed the answer
    the reward would credit different doors than the potential's phase does.
    """
    from nclone.value_function.free_space_door_plan import solve

    n_doors, path, data = door_levels[0]
    name = os.path.basename(path)
    with_legs, _ = solve(data, one_way=True, mine_split_topology=True,
                         name=name, compute_legs=True)
    if with_legs is None or not with_legs.solvable:
        pytest.skip(f"{name}: full solver returned no usable plan")
    got = necessary_doors(data, name=name, use_cache=False)
    assert got.solvable
    assert list(got.order) == list(with_legs.necessary_order)
    # And the legs really were skipped -- that is the whole point.
    assert with_legs.door_leg_dstar or not with_legs.necessary_order


def test_second_read_is_a_cache_hit(door_levels):
    """One solve per level, ever. A cold solve reached 164 s on the corpus."""
    _n, path, data = door_levels[0]
    name = os.path.basename(path)
    clear_caches()
    first = necessary_doors(data, name=name)
    before = live_solve_count()
    second = necessary_doors(data, name=name)
    assert live_solve_count() == before, "warm read re-solved"
    assert second.order == first.order
    assert second.cache_tier in ("mem", "disk", "no_doors")

    # Dropping the in-process tier must fall through to disk, not to a re-solve:
    # workers are separate processes and each one paying a cold solve is exactly
    # the reset-deadline failure this cache exists to prevent.
    clear_caches()
    before = live_solve_count()
    third = necessary_doors(data, name=name)
    assert third.order == first.order
    assert live_solve_count() == before, "cold process re-solved instead of reading disk"
    assert third.cache_tier == "disk"


def test_solver_params_key_the_cache(door_levels):
    """Necessity is a statement about a TOPOLOGY, so a different mask is a miss.

    A door that is necessary on the one-way-directional field need not be on the
    undirected one; serving one for the other would be a stale answer, not a
    cheap one.
    """
    _n, path, data = door_levels[0]
    name = os.path.basename(path)
    necessary_doors(data, name=name, one_way=True)
    before = live_solve_count()
    necessary_doors(data, name=name, one_way=False)
    assert live_solve_count() > before, "a different topology was served from cache"


def test_failure_is_empty_and_flagged_not_raised():
    """Never raise, and never claim necessity we could not compute.

    Callers treat an empty order as "no milestone available". They must never
    receive an exception on the reset path, and an unsolvable verdict must be
    visible rather than silently indistinguishable from "no doors needed".
    """
    # Entity records that name a locked door but no goals: the solver cannot
    # build a field, so this exercises the degrade path rather than the algorithm.
    data = [0] * 1235 + [6, 10, 10, 0, 0] + [0] * 40
    got = necessary_doors(data, use_cache=False)
    assert got.n_doors == 1
    assert got.order == ()
    assert isinstance(got.reason, str)
