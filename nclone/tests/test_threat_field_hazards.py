"""Coverage of the non-drone hazards in the threat field.

The field started as drones only. These tests pin the entities added afterwards
-- rockets, thwumps, shove thwumps, floor guards -- against the corpus, and pin
the tier they land in, because the tier is a claim about predictability and
getting it wrong is the one error a hazard field must not make:

* tier C says "this WILL be lethal at frame t". Under-stating reach here means
  the policy plans a route through something that kills it.
* tier R says "this COULD be lethal by frame t". Over-stating is merely
  conservative; under-stating is the same failure as above.

Each hazard is exercised on a map that ISOLATES it -- no drones, death balls or
eye bats -- so a non-zero field is attributable to the entity under test rather
than to something else on the level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nclone.gym_environment.threat_ttl_reference import (
    CH_CERTAIN_TTL,
    CH_REACTIVE_TTL,
    N_CHANNELS,
    OUT_HEIGHT,
    OUT_WIDTH,
)

_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"
MAX_FRAMES, STRIDE = 64, 4

# Confounders: anything already covered that would make a non-zero field
# ambiguous about which entity produced it.
_CONFOUNDING_TYPES = {12, 13, 14, 15, 25, 26, 27}


def _field(sim):
    raw = sim._native_sim.threat_ttl(MAX_FRAMES, STRIDE)
    return np.frombuffer(raw, dtype=np.float32).reshape(N_CHANNELS, OUT_HEIGHT, OUT_WIDTH)


#: Hazards that can supply a KINEMATIC (committed-velocity) source, and so can
#: legitimately put cells in tier C. Excluded when a test asserts tier C is empty.
_KINEMATIC_TYPES = {16, 20}


def _isolated_map(entity_type: int, also_exclude: frozenset = frozenset()):
    """An official map carrying `entity_type` and no already-covered hazard."""
    from nclone.levelgen.labeling import entity_counts

    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    excluded = (_CONFOUNDING_TYPES | set(also_exclude)) - {entity_type}
    for path in sorted(_OFFICIAL.rglob("*")):
        if not path.is_file():
            continue
        try:
            counts = entity_counts(path.read_bytes())
        except Exception:
            continue
        if set(counts) & excluded:
            continue
        if counts.get(entity_type):
            return path
    pytest.skip(f"no isolated map for entity type {entity_type}")


@pytest.mark.parametrize(
    "entity_type,label",
    [(18, "rocket"), (20, "thwump"), (28, "shove thwump"), (16, "floor guard")],
)
def test_hazard_registers_in_the_reactive_tier(entity_type, label):
    """Each of these is ninja-dependent while idle, so it must produce a
    worst-case envelope rather than nothing at all.

    Before this, all four were invisible to the policy: the field was drone-only,
    so a level whose entire threat was a rocket battery read as completely safe.
    """
    from nclone.native_headless import NativePlayHeadless

    path = _isolated_map(entity_type)
    sim = NativePlayHeadless()
    sim.load_map(str(path))

    peak = 0
    for _ in range(90):
        sim.tick(1, 0)
        peak = max(peak, int((_field(sim)[CH_REACTIVE_TTL] < 1.0).sum()))
    assert peak > 0, (
        f"{label} (type {entity_type}) on {path.name!r} produced an entirely "
        f"empty reactive field -- the entity is not being collected"
    )


def _synthetic(entity_type, ex, ey, orientation=0, nx=10, ny=12):
    """A bare room with the ninja, an exit pair, and one hazard."""
    from nclone.map_generation.map import Map

    m = Map()
    m.set_empty_rectangle(2, 2, 40, 21)
    m.set_ninja_spawn(nx, ny)
    m.add_entity(3, 38, 20, switch_x=36, switch_y=20)
    m.add_entity(entity_type, ex, ey, orientation)
    return m.map_data()


@pytest.mark.parametrize("orientation", [2, 4])
def test_charging_thwump_moves_to_the_certain_tier(orientation):
    """A thwump that has committed to a charge is no longer reactive.

    Its trigger is ninja-dependent, but once `state == 1` the direction is fixed
    and it cannot re-aim, so its short-term future is a straight-line
    extrapolation -- exactly the thing tier C is for. Orientations 2 and 4 face
    the approaching ninja on this layout; 0 and 6 face away and never trigger,
    which is why they are not parametrised here.
    """
    from nclone.native_headless import NativePlayHeadless

    sim = NativePlayHeadless()
    sim.load_map_from_map_data(_synthetic(20, 16, 12, orientation))

    saw_charging = False
    peak_certain = 0
    for i in range(300):
        sim.tick(1, 1 if i % 23 == 0 else 0)
        states = {
            e.state for e in sim._native_sim.get_state().entities if e.entity_type == 20
        }
        saw_charging |= bool(states - {0})
        peak_certain = max(peak_certain, int((_field(sim)[CH_CERTAIN_TTL] < 1.0).sum()))

    assert saw_charging, "the thwump never left its idle state; the test setup is stale"
    assert peak_certain > 0, (
        "a charging thwump produced no CERTAIN-tier cells -- the kinematic "
        "(committed-velocity) source path is not firing"
    )


def test_idle_thwump_is_reactive_not_certain():
    """The converse: an idle thwump must NOT claim a certain future.

    Facing away from the ninja it never charges, so every cell it contributes
    belongs to the envelope tier. If it leaked into tier C the policy would be
    told a stationary hazard is committed to moving.
    """
    from nclone.native_headless import NativePlayHeadless

    sim = NativePlayHeadless()
    sim.load_map_from_map_data(_synthetic(20, 16, 12, orientation=0))
    for i in range(200):
        sim.tick(1, 1 if i % 23 == 0 else 0)
        field = _field(sim)
        assert int((field[CH_CERTAIN_TTL] < 1.0).sum()) == 0
    assert int((field[CH_REACTIVE_TTL] < 1.0).sum()) > 0


def test_flying_rocket_is_reactive_because_it_homes():
    """Rockets must never be extrapolated as ballistic.

    `Rocket::think` calls `projectile.steer_toward(ctx.ninja)` on every state-2
    frame, so a flying rocket re-aims continuously. Treating it as tier C would
    project a straight line and under-state where it can reach -- the exact
    failure mode a hazard field cannot have. This asserts the classification
    rather than the geometry: whatever a rocket contributes lands in tier R.
    """
    from nclone.native_headless import NativePlayHeadless

    # Also exclude thwumps and floor guards: those legitimately produce tier-C
    # cells, so leaving them in would make this assertion meaningless.
    path = _isolated_map(18, also_exclude=frozenset(_KINEMATIC_TYPES))
    sim = NativePlayHeadless()
    sim.load_map(str(path))
    # No assertion that the rocket contributes SOMETHING here: an IDLE launcher
    # is deliberately not a source. It only becomes one on acquisition, and then
    # the 15-frame CHARGING telegraph is what the policy needs to see. A launcher
    # that has not acquired anything is not yet a threat, and dilating it by
    # ROCKET_MAX_SPEED "just in case" would paint most of the level lethal.
    # Reactive registration is covered by
    # `test_hazard_registers_in_the_reactive_tier[18-rocket]`.
    for _ in range(120):
        sim.tick(1, 0)
        assert int((_field(sim)[CH_CERTAIN_TTL] < 1.0).sum()) == 0, (
            "a rocket map produced CERTAIN-tier cells; a homing projectile must "
            "not be extrapolated as committed"
        )


def test_latched_floor_guard_reaches_the_certain_tier():
    """A floor guard that has acquired a target is committed.

    `FloorGuard::think` early-outs forever once `dir != 0` -- it charges until a
    wall or ledge stops it and never re-acquires mid-charge -- so its direction
    is fixed and tier C is correct for it. This runs on a real official map
    because the latch needs the ninja inside a 24px foot-level band with a clear
    tile row between them, which is fiddly to stage synthetically and easy to
    get subtly wrong.
    """
    from nclone.native_headless import NativePlayHeadless

    # Pinned map rather than a search: the latch needs a specific geometry, and
    # the first matching map in corpus order ('006 dead end') happens to place
    # its guards where a straight walk never crosses the band. This one does
    # latch, which is how the kinematic path was first observed firing at all.
    path = _OFFICIAL / "R" / "027 rear guard"
    if not path.is_file():
        pytest.skip("pinned floor-guard map not present")
    sim = NativePlayHeadless()
    sim.load_map(str(path))
    peak = 0
    for _ in range(160):
        sim.tick(1, 0)
        peak = max(peak, int((_field(sim)[CH_CERTAIN_TTL] < 1.0).sum()))
    assert peak > 0, (
        f"no floor guard on {path.name!r} ever latched into the CERTAIN tier; "
        "the kinematic source path may not be firing for type 16"
    )


@pytest.mark.parametrize(
    "orientation,axis,sign",
    [(0, "x", +1), (2, "y", +1), (4, "x", -1), (6, "y", -1)],
)
def test_idle_thwump_threatens_only_its_facing_side(orientation, axis, sign):
    """A thwump's three SAFE sides must stay unshaded.

    A thwump is a solid block with one dangerous face; the other three are
    standable and routes depend on that. Two bugs put danger on them, and this
    catches both:

    * a DISC big enough to contain the face segment (radius 12 + 7 = 19), centred
      on a face only 11px from the body centre, reaches 8px out the far side;
    * a RADIAL envelope -- correct for threats that can turn -- sprays the charge
      reach backwards through the block, which is what a thwump can never do.

    An idle thwump is the strict case: it has not moved, so every lethal cell it
    contributes must lie ahead of its own centre along the facing axis.
    """
    from nclone.native_headless import NativePlayHeadless

    sim = NativePlayHeadless()
    # Ninja parked far away so the thwump stays idle for the whole run.
    sim.load_map_from_map_data(_synthetic(20, 20, 12, orientation, nx=4, ny=19))
    for _ in range(5):
        sim.tick(0, 0)

    body = [e for e in sim._native_sim.get_state().entities if e.entity_type == 20][0]
    field = _field(sim)
    danger = field[CH_REACTIVE_TTL] < 1.0
    assert danger.any(), "an idle thwump should still be lethal on its face"

    rows, cols = np.nonzero(danger)
    world_x = 24 + cols * 12 + 6
    world_y = 24 + rows * 12 + 6
    along = (world_x - body.x) if axis == "x" else (world_y - body.y)
    across = (world_y - body.y) if axis == "x" else (world_x - body.x)

    # Nothing behind the body centre along the facing axis.
    behind = along * sign
    assert behind.min() >= 0.0, (
        f"orientation {orientation} puts danger {abs(behind.min()):.0f}px behind "
        f"the thwump centre -- that is one of its SAFE, standable sides"
    )
    # And the perpendicular spread must match the real capsule (face half-length
    # 7 + kill radius 12 = 19), not a disc's radius plus the charge dilation.
    assert abs(across).max() <= 19.0 + 6.0, (
        f"perpendicular spread {abs(across).max():.0f}px exceeds the face capsule; "
        f"the envelope is being dilated sideways"
    )
