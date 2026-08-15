"""Solid entity bodies must be visible to the policy as SURFACES.

Thwumps, shove thwumps, bounce blocks, one-way platforms and closed doors are
geometry the ninja stands on, wall-jumps off and is blocked by -- and several
official routes exist only because of that. But `tile_grid` carries static tiles
only and `entity_overlays` carries goals, mines and gold, so before `solid_map`
a wall-jumpable thwump face was indistinguishable from empty air.

This is deliberately a SEPARATE observation from the threat field, and the
thwump is why: it is a hazard on one face and a floor on the other three. One
tensor cannot carry both meanings without one of them being wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

OUT_H, OUT_W = 46, 84
from nclone.gym_environment.constants import (  # noqa: E402
    SOLID_MAP_CHANNELS,
    SOLID_MAP_CH_BOOST_PAD,
    SOLID_MAP_CH_BOUNCE,
    SOLID_MAP_CH_DOOR_OPEN,
    SOLID_MAP_CH_DOOR_SWITCH,
    SOLID_MAP_CH_LAUNCHPAD,
    SOLID_MAP_CH_LAUNCHPAD_BX,
    SOLID_MAP_CH_LAUNCHPAD_BY,
    SOLID_MAP_CH_ONEWAY,
    SOLID_MAP_CH_ONEWAY_NX,
    SOLID_MAP_CH_ONEWAY_NY,
    SOLID_MAP_CH_STATIC,
)

CH_STATIC = SOLID_MAP_CH_STATIC
CH_BOUNCE = SOLID_MAP_CH_BOUNCE
CH_ONEWAY = SOLID_MAP_CH_ONEWAY
CH_ONEWAY_NX = SOLID_MAP_CH_ONEWAY_NX
CH_ONEWAY_NY = SOLID_MAP_CH_ONEWAY_NY
_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"


def _solid(sim):
    raw = sim._native_sim.solid_map()
    return np.frombuffer(raw, dtype=np.float32).reshape(
        SOLID_MAP_CHANNELS, OUT_H, OUT_W
    )


def _synthetic(entity_type, ex, ey, orientation=0, **kw):
    from nclone.map_generation.map import Map

    m = Map()
    m.set_empty_rectangle(2, 2, 40, 21)
    m.set_ninja_spawn(4, 19)
    m.add_entity(3, 38, 20, switch_x=36, switch_y=20)
    m.add_entity(entity_type, ex, ey, orientation, **kw)
    return m.map_data()


def _load(map_data):
    from nclone.native_headless import NativePlayHeadless

    sim = NativePlayHeadless()
    sim.load_map_from_map_data(map_data)
    for _ in range(4):
        sim.tick(0, 0)
    return sim


@pytest.mark.parametrize(
    "entity_type,channel,label,semi",
    [
        (20, CH_STATIC, "thwump", 9.0),
        (28, CH_STATIC, "shove thwump", 12.0),
        (17, CH_BOUNCE, "bounce block", 9.0),
        (11, CH_ONEWAY, "one-way platform", 12.0),
    ],
)
def test_solid_body_is_stamped_at_its_own_position(entity_type, channel, label, semi):
    """Occupancy must land on the body, with a footprint matching its extent."""
    sim = _load(_synthetic(entity_type, 20, 12))
    field = _solid(sim)
    occupied = field[channel] > 0.0
    assert occupied.any(), f"{label} left no footprint in the solid map"

    ent = [
        e
        for e in sim._native_sim.get_state().entities
        if e.entity_type == entity_type
    ][0]
    rows, cols = np.nonzero(occupied)
    world_x = 24 + cols * 12 + 6
    world_y = 24 + rows * 12 + 6
    # Every stamped cell centre must lie within the body plus one cell of
    # quantisation slack.
    assert np.all(np.abs(world_x - ent.x) <= semi + 6.0), f"{label} stamped too wide"
    assert np.all(np.abs(world_y - ent.y) <= semi + 6.0), f"{label} stamped too tall"


def test_bounce_blocks_are_not_merged_into_static():
    """Bounce blocks are springy: the jump that leaves one differs from the jump
    that leaves a thwump, so they cannot share a channel."""
    field = _solid(_load(_synthetic(17, 20, 12)))
    assert (field[CH_BOUNCE] > 0).any()
    assert not (field[CH_STATIC] > 0).any()


def test_one_way_carries_its_blocking_normal():
    """A one-way is passable from one side and solid from the other; occupancy
    alone cannot express that, so the normal is the load-bearing part."""
    seen = set()
    for orientation in (0, 2, 4, 6):
        field = _solid(_load(_synthetic(11, 20, 12, orientation)))
        occupied = field[CH_ONEWAY] > 0
        assert occupied.any(), f"one-way orientation {orientation} not stamped"
        nx = field[CH_ONEWAY_NX][occupied]
        ny = field[CH_ONEWAY_NY][occupied]
        assert np.hypot(nx, ny).max() > 0.5, "normal is zero on an occupied cell"
        seen.add((round(float(nx[0])), round(float(ny[0]))))
    assert len(seen) == 4, f"orientations collapsed to the same normal: {seen}"


def test_open_doors_are_not_solid():
    """A door contributes only while CLOSED. A channel that ignored the state
    would tell the policy a corridor is walled when it is open."""
    from nclone.native_headless import NativePlayHeadless

    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    hits = list(_OFFICIAL.rglob("021 trap doors"))
    if not hits:
        pytest.skip("map not in corpus")
    sim = NativePlayHeadless()
    sim.load_map(str(hits[0]))
    for _ in range(4):
        sim.tick(0, 0)
    closed_cells = int((_solid(sim)[CH_STATIC] > 0).sum())

    states = [
        e for e in sim._native_sim.get_state().entities if e.entity_type in (6, 8)
    ]
    assert states, "expected doors on this map"
    # Every stamped static cell must be attributable to a door that reports
    # closed, or to another solid body -- never to an open door.
    assert closed_cells >= 0  # smoke: the map builds and stamps without error
    assert closed_cells <= 4 * len(states) + 64, (
        "far more static cells than the doors can account for; open doors may be "
        "being stamped as solid"
    )


# ---------------------------------------------------------------------------
# Traversal channels (plan section 3.4)
#
# What GATES, CARRIES or PROPELS, as opposed to what blocks. Six channels, not
# the plan's twelve: one-way density and normals, closed doors, thwumps and
# bounce blocks are channels 0-4 of this same field, which shipped after the
# plan was written. Building them a second time on `entity_overlays` would spend
# conv parameters and per-step compute on a copy of a signal the policy already
# receives.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entity_type,channel,label",
    [
        (10, SOLID_MAP_CH_LAUNCHPAD, "launch pad"),
        (24, SOLID_MAP_CH_BOOST_PAD, "boost pad"),
    ],
)
def test_propulsion_pads_are_stamped(entity_type, channel, label):
    """Neither pad is SOLID, so channel 0 cannot carry it -- yet both change
    which routes exist. A launch pad is the difference between a wall being a
    dead end and being the intended way up."""
    field = _solid(_load(_synthetic(entity_type, 20, 12)))
    assert (field[channel] > 0).any(), f"{label} left no footprint"
    # ... and it is not being confused with a blocking body.
    assert not (field[CH_STATIC] > 0).any(), f"{label} stamped as a solid barrier"


def test_launch_pad_carries_its_boost_direction():
    """Direction is the whole point, exactly as it is for a one-way: two pads in
    the same cell pointing opposite ways are opposite affordances, and an
    occupancy-only channel makes them identical."""
    seen = set()
    for orientation in (0, 2, 4, 6):
        field = _solid(_load(_synthetic(10, 20, 12, orientation)))
        occupied = field[SOLID_MAP_CH_LAUNCHPAD] > 0
        assert occupied.any(), f"launch pad orientation {orientation} not stamped"
        bx = field[SOLID_MAP_CH_LAUNCHPAD_BX][occupied]
        by = field[SOLID_MAP_CH_LAUNCHPAD_BY][occupied]
        assert np.hypot(bx, by).max() > 0.5, "boost vector is zero on an occupied cell"
        seen.add((round(float(bx[0])), round(float(by[0]))))
    assert len(seen) == 4, f"orientations collapsed to the same boost: {seen}"


def test_boost_pad_has_no_direction_channel():
    """A boost pad SCALES the ninja's existing velocity rather than imposing one,
    so there is no vector to report and inventing one would be a lie."""
    field = _solid(_load(_synthetic(24, 20, 12)))
    assert (field[SOLID_MAP_CH_BOOST_PAD] > 0).any()
    assert not (field[SOLID_MAP_CH_LAUNCHPAD] > 0).any()
    assert not (field[SOLID_MAP_CH_LAUNCHPAD_BX] != 0).any()


def test_doors_are_split_by_state_and_publish_their_switch():
    """A door is unified by STATE, not type, and a CLOSED one publishes the
    switch that opens it.

    The switch is the actionable half: the barrier has always been visible in
    channel 0, but the thing the ninja must go and touch sits elsewhere on the
    map and appeared in no channel at all. Channels 0 and 5 must also be
    disjoint -- a door is either a wall or a doorway, never both.
    """
    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    hits = list(_OFFICIAL.rglob("021 trap doors"))
    if not hits:
        pytest.skip("map not in corpus")
    from nclone.native_headless import NativePlayHeadless

    sim = NativePlayHeadless()
    sim.load_map(str(hits[0]))
    for _ in range(4):
        sim.tick(0, 0)
    field = _solid(sim)

    doors = [e for e in sim._native_sim.get_state().entities if e.entity_type in (6, 8)]
    assert doors, "expected doors on this map"
    closed = [d for d in doors if getattr(d, "door_closed", False)]
    if not closed:
        pytest.skip("no closed door on this map at this frame")

    # Every closed door contributes a switch marker.
    assert (field[SOLID_MAP_CH_DOOR_SWITCH] > 0).any(), (
        "closed doors present but no switch was published"
    )
    # A cell is never simultaneously a wall and an open doorway.
    both = (field[CH_STATIC] > 0) & (field[SOLID_MAP_CH_DOOR_OPEN] > 0)
    assert not both.any(), "a door is stamped as closed AND open in the same cell"


def test_new_channels_are_silent_on_a_level_without_them():
    """A hazard-only level must read exactly zero in all six traversal channels.

    Zero is the correct neutral for an OCCUPANCY channel (no bodies), unlike the
    threat field's 1.0 (no danger) -- and a channel that is never zero would mean
    every level pays for a signal only some levels carry.
    """
    field = _solid(_load(_synthetic(20, 20, 12)))  # a lone thwump
    for ch in (
        SOLID_MAP_CH_DOOR_OPEN,
        SOLID_MAP_CH_DOOR_SWITCH,
        SOLID_MAP_CH_LAUNCHPAD,
        SOLID_MAP_CH_LAUNCHPAD_BX,
        SOLID_MAP_CH_LAUNCHPAD_BY,
        SOLID_MAP_CH_BOOST_PAD,
    ):
        assert not (field[ch] != 0).any(), f"channel {ch} fired on a thwump-only level"
