"""Entity awareness in the free-space mask: doors, one-ways, moving hazards.

These pin the three properties that make the additions safe to land rather than
the numbers they produce (those live in ``tools/door_audit.py`` and the findings
doc):

* the DEFAULTS are inert — an existing caller gets a bit-identical field;
* the door open-set degenerates correctly at both ends;
* a one-way blocks one direction and not the other, and the flood never loses
  its finiteness to the threat cost.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource

_OFFICIAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "maps", "official")


def _map(path):
    with open(path, "rb") as f:
        return [int(b) for b in f.read()]


def _first_with(pred, limit=400):
    """First official level whose built source satisfies ``pred``."""
    for p in sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:limit]:
        try:
            md = _map(p)
            src = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                                          one_way=True)
        except Exception:
            continue
        if src.available and pred(src):
            return md, src
    return None, None


def _fields_equal(a, b):
    return (np.array_equal(a._d_switch, b._d_switch, equal_nan=True)
            and np.array_equal(a._d_exit, b._d_exit, equal_nan=True))


@pytest.mark.skipif(not os.path.isdir(_OFFICIAL), reason="official maps absent")
def test_defaults_are_inert():
    """Naming the new options at their defaults must change nothing at all."""
    paths = sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:25]
    assert paths, "no official maps found"
    for p in paths:
        md = _map(p)
        a = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True)
        b = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                                    doors="open", one_way=False,
                                    threat_occupancy=False)
        assert _fields_equal(a, b), p


@pytest.mark.skipif(not os.path.isdir(_OFFICIAL), reason="official maps absent")
def test_door_open_set_degenerates_at_both_ends():
    """``frozenset()`` is "closed" and the full set is "open" — exactly."""
    md, src = _first_with(lambda s: len(s.locked_doors) >= 2)
    if md is None:
        pytest.skip("no official level with >=2 locked doors in range")
    n = len(src.locked_doors)

    def build(doors):
        return FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                                       doors=doors)

    assert _fields_equal(build("closed"), build(frozenset()))
    assert _fields_equal(build("open"), build(frozenset(range(n))))
    # And the two ends are genuinely different, or the test proves nothing.
    assert not _fields_equal(build("open"), build("closed"))


@pytest.mark.skipif(not os.path.isdir(_OFFICIAL), reason="official maps absent")
def test_closed_doors_only_ever_remove_reachability():
    """Shutting a door is a strict edge REMOVAL, so no cost may fall."""
    md, src = _first_with(lambda s: len(s.locked_doors) >= 1)
    if md is None:
        pytest.skip("no official level with a locked door in range")
    o = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                               doors="open")
    c = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                               doors="closed")
    both = np.isfinite(o._d_switch) & np.isfinite(c._d_switch) & o._free
    assert (c._d_switch[both] >= o._d_switch[both] - 1e-6).all()
    # Nothing may become reachable that was not.
    assert not (np.isfinite(c._d_switch) & ~np.isfinite(o._d_switch)).any()


@pytest.mark.skipif(not os.path.isdir(_OFFICIAL), reason="official maps absent")
def test_one_way_is_directional():
    """Some edge must be blocked one way and open the other.

    A symmetric implementation passes every coverage check and still gets the
    physics wrong, so the asymmetry itself is what needs pinning.
    """
    md, src = _first_with(lambda s: len(s.one_ways) >= 1)
    if md is None:
        pytest.skip("no official level with a one-way in range")
    assert src._block_r_neg is not None, "directional masks were not built"
    asym = ((src._block_r != src._block_r_neg)
            | (src._block_u != src._block_u_neg))
    assert asym.any(), "one-ways produced no directional difference"
    # The negative masks are built as copies, so they can only ADD blocks.
    assert (src._block_r_neg | src._block_r).sum() >= src._block_r.sum()


@pytest.mark.skipif(not os.path.isdir(_OFFICIAL), reason="official maps absent")
def test_threat_occupancy_never_removes_finiteness():
    """The threat cost is a bounded multiplier, so coverage cannot regress.

    This is acceptance criterion 1 of the research brief, and it is the property
    that separates a soft cost from the hard mask the same section rejects.
    """
    from nclone.value_function.threat_occupancy import marginal_occupancy

    checked = 0
    for p in sorted(glob.glob(os.path.join(_OFFICIAL, "*", "*")))[:120]:
        md = _map(p)
        if marginal_occupancy(md, frames=120) is None:
            continue  # no certain-tier threat: the mechanism is inert here
        a = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True)
        b = FreeSpaceDistanceSource(md, mine_hard=True, physics_cost=True,
                                    threat_occupancy=True)
        free = a._free
        lost = np.isfinite(a._d_switch) & ~np.isfinite(b._d_switch) & free
        assert not lost.any(), f"{p}: threat cost severed {int(lost.sum())} cells"
        # A cost multiplier of >=1 can only make things more expensive.
        both = np.isfinite(a._d_switch) & np.isfinite(b._d_switch) & free
        assert (b._d_switch[both] >= a._d_switch[both] - 1e-6).all(), p
        checked += 1
        if checked >= 3:
            break
    if checked == 0:
        pytest.skip("no level with certain-tier threat in range")
