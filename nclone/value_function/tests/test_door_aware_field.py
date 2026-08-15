"""Invariants of the phase-aware, door-aware free-space potential.

These pin the three properties that make it safe to descend, all of which were
violated by an earlier draft and found by measurement rather than by reasoning:

* **never no-gradient** — every level keeps a finite cost at its spawn, whatever
  the door solver concludes.  The solver calls 9 of 105 demo-verified levels
  "unsolvable", so its verdict must never be able to delete the potential.
* **opening a door is progress** — the potential must not rise across a phase
  boundary.  It did, on 52 of 219 levels, because the tail after the last leg
  was short by a whole switch leg.
* **inert without doors** — a level with no locked door must be bit-identical to
  the plain field.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from nclone.value_function.door_aware_field import DoorAwareFreeSpaceField
from nclone.value_function.free_space_distance import FreeSpaceDistanceSource

_OFFICIAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "maps", "official")


def _map(path):
    with open(path, "rb") as f:
        return [int(b) for b in f.read()]


def _spawn(md):
    import nclone_native
    return nclone_native.Simulator.load(list(md)).get_ninja_pos()


def _levels(pred, limit=1200, want=4, min_doors=1):
    """Official levels whose built field satisfies ``pred``.

    Pre-filters on the locked-door COUNT with the cheap plain source before
    paying for a full plan.  A fixed-prefix scan is not enough: multi-leg levels
    are rare and clustered, so scanning the first few hundred paths silently
    skipped the very tests that matter.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:limit]:
        try:
            md = _map(p)
            base = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True)
            if not base.available or len(base.locked_doors) < min_doors:
                continue
            f = DoorAwareFreeSpaceField(md)
        except Exception:
            continue
        if f.available and pred(f):
            out.append((p, md, f))
            if len(out) >= want:
                break
    return out


pytestmark = pytest.mark.skipif(not os.path.isdir(_OFFICIAL),
                                reason="official maps absent")


def test_spawn_is_always_finite():
    """Whatever the solver concludes, the agent keeps a gradient."""
    checked = 0
    for p in sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:120]:
        md = _map(p)
        f = DoorAwareFreeSpaceField(md)
        if not f.available:
            continue
        sx, sy = _spawn(md)
        v = f.cost_to_completion(float(sx), float(sy), False, [])
        assert np.isfinite(v), f"{p}: spawn potential is inf ({f.fallback_reason})"
        checked += 1
    assert checked > 50


def test_unsolvable_plan_falls_back_rather_than_severing():
    """An 'unsolvable' verdict is measured to be wrong ~9% of the time."""
    got = _levels(lambda f: f.fallback_reason
                  and f.fallback_reason.startswith("plan unsolvable"), want=1)
    if not got:
        pytest.skip("no unsolvable-plan level in range")
    p, md, f = got[0]
    assert f.available and not f.door_aware
    sx, sy = _spawn(md)
    assert np.isfinite(f.cost_to_completion(float(sx), float(sy), False, []))


def test_opening_a_door_never_raises_the_potential():
    """Monotone across every phase boundary, including the last."""
    got = _levels(lambda f: f.door_aware and len(f.necessary_order) >= 2,
                  want=3, min_doors=2)
    if not got:
        pytest.skip("no multi-leg door level in range")
    for p, md, f in got:
        sx, sy = _spawn(md)
        prev = None
        for k in range(len(f.necessary_order) + 1):
            v = f.cost_to_completion(float(sx), float(sy), False,
                                     f.necessary_order[:k])
            assert np.isfinite(v), f"{p}: phase {k} is inf"
            if prev is not None:
                assert v <= prev + 1e-6, (
                    f"{p}: potential ROSE {prev:.1f} -> {v:.1f} opening door {k}")
            prev = v


def test_phase_only_advances_when_all_necessary_doors_open():
    """A pre-collected exit switch must not skip a door the level requires."""
    got = _levels(lambda f: f.door_aware and len(f.necessary_order) >= 2,
                  want=1, min_doors=2)
    if not got:
        pytest.skip("no multi-leg door level in range")
    _p, _md, f = got[0]
    k = len(f.necessary_order)
    assert f.door_phase([]) == 0
    assert f.door_phase(f.necessary_order[:1]) == 1
    assert f.door_phase(f.necessary_order[:k - 1]) == k - 1
    assert f.door_phase(f.necessary_order) is None
    # An unrelated door being open must not advance the phase.
    assert f.door_phase([999]) == 0


def test_inert_on_levels_without_locked_doors():
    """No doors -> exactly the plain field, and said so."""
    checked = 0
    for p in sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:40]:
        md = _map(p)
        base = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True)
        if not base.available or base.locked_doors:
            continue
        f = DoorAwareFreeSpaceField(md)
        assert f.fallback_reason == "no locked doors"
        assert not f.door_aware
        sx, sy = _spawn(md)
        assert f.cost_to_completion(float(sx), float(sy), False, []) == pytest.approx(
            base.cost_to_completion(float(sx), float(sy), False))
        checked += 1
    assert checked > 5
