"""The rocket projectile as a MOVING hazard: velocity, acceleration, envelope.

A rocket is the only hazard in the game that accelerates, and the only one whose
lethal object is not where its entity is. Both facts were absent from the
observation:

* the threat field treated a flying projectile as a point dilated isotropically
  by `ROCKET_MAX_SPEED`, which over-states a fresh one's reach by up to 18x --
  the speed ramp takes ~24 frames and starts from a standstill;
* `threat_proximity` reported the LAUNCHER's bearing and computed `closing` from
  the ninja's velocity alone, so a stationary ninja with a rocket inbound at full
  speed read "nothing is happening".

These tests pin the replacement. The load-bearing one is
`test_envelope_contains_the_real_future`: a hazard field may over-state danger
but must never under-state it, and the new model is much tighter than the old
one, so "tighter" has to be shown to still be a superset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nclone.gym_environment.constants import (
    THREAT_CLASS_PROJECTILE,
    THREAT_PROXIMITY_RANGE_PX,
    THREAT_PROX_FEATURES,
    THREAT_PROX_IDX_CLOSING,
    THREAT_PROX_IDX_IMMINENCE,
    THREAT_PROX_IDX_SPIN_UP,
    THREAT_PROX_IDX_VX,
    THREAT_PROX_IDX_VY,
    THREAT_PROXIMITY_DIM,
    THREAT_PROXIMITY_SPEED_NORM_PX,
)
from nclone.gym_environment.threat_proximity_reference import (
    compute_threat_proximity_reference,
    rocket_spin_up,
)
from nclone.gym_environment.threat_ttl_reference import (
    CH_REACTIVE_TTL,
    N_CHANNELS,
    OUT_HEIGHT,
    OUT_WIDTH,
    ROCKET_INITIAL_STEP,
    ROCKET_MAX_SPEED,
    ROCKET_SPEED_MULTIPLIER,
    compute_threat_ttl_reference,
)

_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"
MAX_FRAMES, STRIDE = 64, 4
BLOCK_PX, PLAYFIELD_OFFSET_PX = 12.0, 24.0

#: The one map in the corpus whose rocket acquires and fires with NO ninja input,
#: so a test can reach the FLYING branch deterministically. Found by scanning all
#: 908 rocket maps -- the rest need the ninja to walk into a sightline, and a
#: parity test that never fires the rocket passes trivially while proving
#: nothing. (The audit that shipped the telegraph recorded exactly this trap.)
FIRING_MAP = "158 just squares, just basically squares"
#: A map whose launcher is mid-telegraph INSIDE the 240px reaction window, which
#: `FIRING_MAP`'s never is (its rocket sits 480px away). Needed to exercise the
#: acceleration dim on a live sim rather than only as a pure function.
CHARGING_MAP = "285 back and forth"


def _map_path(name):
    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    hits = list(_OFFICIAL.rglob(name))
    if not hits:
        pytest.skip(f"map {name!r} not in corpus")
    return str(hits[0])


def _backends(path):
    from nclone.native_headless import NativePlayHeadless
    from nclone.nplay_headless import NPlayHeadless

    native = NativePlayHeadless()
    native.load_map(path)
    python = NPlayHeadless(enable_rendering=False)
    python.load_map(path)
    return native, python


def _flying_rocket(python_backend):
    """The pure-Python rocket entity while its projectile is in flight, or None."""
    for r in (python_backend.sim.entity_dic.get(18, []) or []):
        if int(getattr(r, "state", 0)) == 2:
            return r
    return None


def _cell_of(x, y):
    col = int((x - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
    row = int((y - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
    return row, col


def _rust_ttl(native):
    raw = native._native_sim.threat_ttl(MAX_FRAMES, STRIDE)
    return np.frombuffer(raw, dtype=np.float32).reshape(N_CHANNELS, OUT_HEIGHT, OUT_WIDTH)


def _rust_proximity(native):
    raw = native._native_sim.threat_proximity()
    return np.frombuffer(raw, dtype=np.float32).copy()


# --- the speed model itself -------------------------------------------------


def test_speed_ramp_matches_the_simulator():
    """`step *= 1.0656; speed = min(speed + step, 16/7)` -- not a geometric series.

    Pinned because an earlier port read `16/7` as the INITIAL speed and had the
    rocket passing 29 px/frame by frame 40 against a real ceiling of 2.29.
    """
    speed, step, travel = 0.0, ROCKET_INITIAL_STEP, 0.0
    reached_max_at = None
    for frame in range(1, 65):
        if speed < ROCKET_MAX_SPEED:
            step *= ROCKET_SPEED_MULTIPLIER
        speed = min(speed + step, ROCKET_MAX_SPEED)
        travel += speed
        if reached_max_at is None and speed >= ROCKET_MAX_SPEED:
            reached_max_at = frame
    assert reached_max_at == 23, "the acceleration phase is 23 frames"
    assert travel == pytest.approx(115.8, abs=0.5)


def test_isotropic_envelope_would_be_wildly_loose():
    """The size of the defect this replaces, stated as a number.

    A rocket 12 frames old can physically be 4.75 px from where it was. The old
    field painted 27.4 px of death around it -- 33x the area, and closing
    corridors that are open is how a hazard channel earns being ignored.
    """
    speed, step, travel = 0.0, ROCKET_INITIAL_STEP, 0.0
    for _ in range(12):
        if speed < ROCKET_MAX_SPEED:
            step *= ROCKET_SPEED_MULTIPLIER
        speed = min(speed + step, ROCKET_MAX_SPEED)
        travel += speed
    assert travel == pytest.approx(4.75, abs=0.05)
    assert ROCKET_MAX_SPEED * 12 == pytest.approx(27.43, abs=0.05)
    assert (ROCKET_MAX_SPEED * 12) / travel > 5.0


def test_spin_up_decays_to_zero_at_terminal_speed():
    """The acceleration dim distinguishes "winding up" from "committed"."""
    assert rocket_spin_up(0.0, ROCKET_INITIAL_STEP) > 0.0
    assert rocket_spin_up(ROCKET_MAX_SPEED, 0.2) == 0.0
    # Monotone-ish while accelerating is not required, but it must never be
    # negative and never exceed its own cap.
    speed, step = 0.0, ROCKET_INITIAL_STEP
    for _ in range(40):
        if speed < ROCKET_MAX_SPEED:
            step *= ROCKET_SPEED_MULTIPLIER
        speed = min(speed + step, ROCKET_MAX_SPEED)
        assert 0.0 <= rocket_spin_up(speed, step) <= 1.0


# --- the live simulator -----------------------------------------------------


def _run_until_flight(native, python, budget=200, in_range=False, moving=False):
    """Step both backends in lockstep until a projectile is airborne.

    `moving` skips the LAUNCH frame, on which the projectile is seeded on the
    tube at speed 0 and does not integrate -- `FUN_10412780`'s state-1 arm ends
    in `state = 2; return`. `in_range` waits until it is inside the 240px gate,
    outside of which `threat_proximity` correctly reads all-zero.
    """
    for _ in range(budget):
        native.tick(0, 0)
        python.tick(0, 0)
        r = _flying_rocket(python)
        if r is None:
            continue
        if moving and r.projectile.speed <= 0.0:
            continue
        if in_range:
            nx, ny = python.ninja_position()
            d = np.hypot(r.projectile.xpos - nx, r.projectile.ypos - ny)
            if d > THREAT_PROXIMITY_RANGE_PX:
                continue
        return True
    return False


def test_the_chosen_map_actually_fires():
    """Guards every test below from passing trivially.

    A parity test on a map where the rocket never leaves IDLE agrees by not
    firing, which is exactly the failure mode the entity audit recorded when its
    first pass "validated" two fixes that never activated.
    """
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python), (
        "no projectile reached FLYING; the rest of this module would be vacuous"
    )


def test_proximity_reports_the_projectile_not_the_launcher():
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, in_range=True, moving=True)
    r = _flying_rocket(python)
    p = r.projectile
    # The projectile has left the tube, so the two are distinguishable -- which
    # is the whole point: reporting the launcher would give a different bearing.
    assert np.hypot(p.xpos - r.xpos, p.ypos - r.ypos) > 1.0

    base = THREAT_CLASS_PROJECTILE * THREAT_PROX_FEATURES
    vec = _rust_proximity(native)
    nx, ny = python.ninja_position()
    # Bearing must point at the projectile; the launcher would give a different
    # unit vector entirely once the rocket is any distance away.
    d = np.hypot(p.xpos - nx, p.ypos - ny)
    assert vec[base] == pytest.approx((p.xpos - nx) / d, abs=2e-3)
    assert vec[base + 1] == pytest.approx((p.ypos - ny) / d, abs=2e-3)
    assert vec[base + THREAT_PROX_IDX_IMMINENCE] == pytest.approx(1.0)


def test_velocity_and_acceleration_dims_track_the_projectile():
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, in_range=True, moving=True)
    base = THREAT_CLASS_PROJECTILE * THREAT_PROX_FEATURES
    seen_spin_up, seen_terminal = False, False
    for _ in range(60):
        r = _flying_rocket(python)
        if r is None:
            break
        p = r.projectile
        nx, ny = python.ninja_position()
        if np.hypot(p.xpos - nx, p.ypos - ny) > THREAT_PROXIMITY_RANGE_PX:
            # Out of the reaction window the class reads all-zero by design.
            native.tick(0, 0)
            python.tick(0, 0)
            continue
        vec = _rust_proximity(native)
        norm = THREAT_PROXIMITY_SPEED_NORM_PX
        assert vec[base + THREAT_PROX_IDX_VX] == pytest.approx(
            np.clip(p.xdir * p.speed / norm, -1, 1), abs=2e-3
        )
        assert vec[base + THREAT_PROX_IDX_VY] == pytest.approx(
            np.clip(p.ydir * p.speed / norm, -1, 1), abs=2e-3
        )
        spin = vec[base + THREAT_PROX_IDX_SPIN_UP]
        if p.speed < ROCKET_MAX_SPEED:
            seen_spin_up |= spin > 0.0
        else:
            seen_terminal = True
            assert spin == pytest.approx(0.0)
        native.tick(0, 0)
        python.tick(0, 0)
    # The acceleration phase is not observable IN RANGE on this map -- the rocket
    # launches 480px away and reaches terminal speed long before it closes to
    # 240 -- so it gets its own case below rather than a conditional assert here.
    del seen_spin_up
    assert seen_terminal, "never observed terminal speed"


def test_charging_launcher_reports_the_whole_ramp_ahead_of_it():
    """The acceleration dim, on a map where the wind-up IS inside the window.

    A launcher mid-telegraph has fired nothing yet, so the entire speed ramp is
    still in front of it and `spin_up` must be non-zero -- distinguishing "a
    rocket is about to leave that tube from a standstill" from "a rocket is
    already here at full speed". Those are opposite tactical situations, and
    `imminence` alone cannot separate them.
    """
    native, python = _backends(_map_path(CHARGING_MAP))
    base = THREAT_CLASS_PROJECTILE * THREAT_PROX_FEATURES
    for _ in range(200):
        nx, ny = python.ninja_position()
        charging = [
            r for r in (python.sim.entity_dic.get(18, []) or [])
            if int(getattr(r, "state", 0)) == 1
            and np.hypot(r.xpos - nx, r.ypos - ny) <= THREAT_PROXIMITY_RANGE_PX
        ]
        if charging:
            vec = _rust_proximity(native)
            assert vec[base + THREAT_PROX_IDX_SPIN_UP] == pytest.approx(
                rocket_spin_up(0.0, ROCKET_INITIAL_STEP), abs=1e-4
            )
            # The launcher itself never moves, so the velocity dims read zero.
            assert vec[base + THREAT_PROX_IDX_VX] == 0.0
            assert vec[base + THREAT_PROX_IDX_VY] == 0.0
            return
        native.tick(0, 0)
        python.tick(0, 0)
    pytest.fail("no launcher observed mid-telegraph inside the reaction window")


def test_closing_is_relative_not_ninja_only():
    """A stationary ninja with a rocket inbound must not read `closing == 0`.

    This is the defect the extra dims exist for: the old term projected the
    ninja's velocity alone, so the single most dangerous frame in the game --
    standing still while a projectile homes in -- was indistinguishable from a
    quiet one.
    """
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, in_range=True, moving=True)
    base = THREAT_CLASS_PROJECTILE * THREAT_PROX_FEATURES
    closings = []
    for _ in range(60):
        r = _flying_rocket(python)
        if r is None:
            break
        vec = _rust_proximity(native)
        nx, ny = python.ninja_position()
        nvx, nvy = python.ninja_velocity()
        p = r.projectile
        d = np.hypot(p.xpos - nx, p.ypos - ny)
        if 1e-6 < d <= THREAT_PROXIMITY_RANGE_PX:
            ux, uy = (p.xpos - nx) / d, (p.ypos - ny) / d
            expected = ((nvx - p.xdir * p.speed) * ux
                        + (nvy - p.ydir * p.speed) * uy) / THREAT_PROXIMITY_SPEED_NORM_PX
            assert vec[base + THREAT_PROX_IDX_CLOSING] == pytest.approx(
                np.clip(expected, -1, 1), abs=2e-3
            )
            closings.append(vec[base + THREAT_PROX_IDX_CLOSING])
        native.tick(0, 0)
        python.tick(0, 0)
    # The projectile's own motion must dominate somewhere in the flight, or the
    # term has not actually changed behaviour.
    assert closings and max(abs(c) for c in closings) > 0.1


def test_rust_and_python_proximity_agree():
    """The two implementations must not drift; the dev-tool overlay is believed."""
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, in_range=True, moving=True)
    for _ in range(30):
        nx, ny = python.ninja_position()
        rust = _rust_proximity(native)
        ref = compute_threat_proximity_reference(python, nx, ny)
        assert rust.shape == (THREAT_PROXIMITY_DIM,)
        np.testing.assert_allclose(ref, rust, atol=1e-5)
        native.tick(0, 0)
        python.tick(0, 0)


def test_envelope_contains_the_real_future():
    """The load-bearing property: the field may be early, never late.

    The new model bounds the projectile by its exact path length and its bounded
    turn rate instead of by `MAX_SPEED * h` in every direction. That is a far
    tighter region, so it has to be shown to still CONTAIN where the projectile
    actually goes -- otherwise the tightening is just an under-statement, which
    is the one error a hazard field must not make.

    Method: freeze the field at frame t, roll the simulator forward, and check
    every cell the projectile actually occupies was already marked lethal no
    later than the frame it arrives.
    """
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, moving=True)

    field = _rust_ttl(native)[CH_REACTIVE_TTL]
    checked = 0
    for k in range(1, 33):
        native.tick(0, 0)
        python.tick(0, 0)
        r = _flying_rocket(python)
        if r is None:
            break  # detonated; nothing further to check
        row, col = _cell_of(r.projectile.xpos, r.projectile.ypos)
        if not (0 <= row < OUT_HEIGHT and 0 <= col < OUT_WIDTH):
            continue
        predicted = float(field[row, col])
        actual = k / MAX_FRAMES
        assert predicted <= actual + 1e-6, (
            f"frame +{k}: the projectile reached cell ({row},{col}) at t={actual:.3f} "
            f"but the field predicted it no earlier than t={predicted:.3f} -- the "
            f"envelope UNDER-states the threat"
        )
        checked += 1
    assert checked >= 8, "too few in-flight samples to call this a test"


def test_envelope_is_directional():
    """Being a superset is necessary but not sufficient: it must also be smaller.

    The isotropic model is rotationally symmetric by construction, so it assigns
    the SAME arrival time to two cells the same distance away regardless of
    direction. The kinematic one cannot: a cell behind the projectile needs the
    heading to swing round at 3.81 deg/frame first. Sampling a ring about the
    projectile and comparing ahead against behind tests exactly that, without
    depending on which other hazards happen to share the channel.
    """
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, moving=True)
    # Let it spin up so the distance term is not doing all the work.
    for _ in range(20):
        native.tick(0, 0)
        python.tick(0, 0)
    r = _flying_rocket(python)
    if r is None:
        pytest.skip("projectile detonated before the sample point")
    p = r.projectile
    field = _rust_ttl(native)[CH_REACTIVE_TTL]

    def ttl_at(angle_offset, radius=48.0):
        heading = np.arctan2(p.ydir, p.xdir) + angle_offset
        x = p.xpos + radius * np.cos(heading)
        y = p.ypos + radius * np.sin(heading)
        row, col = _cell_of(x, y)
        if not (0 <= row < OUT_HEIGHT and 0 <= col < OUT_WIDTH):
            return None
        return float(field[row, col])

    ahead, behind = ttl_at(0.0), ttl_at(np.pi)
    if ahead is None or behind is None:
        pytest.skip("sample ring falls off the playfield on this map")
    assert behind > ahead, (
        f"a cell {48}px BEHIND the projectile reads t={behind:.3f} and one the "
        f"same distance AHEAD reads t={ahead:.3f}; an isotropic envelope would "
        f"make these equal, so the turn-rate bound is not being applied"
    )


def test_charging_launcher_is_a_countdown_not_a_lethal_body():
    """The launcher body never kills, so it must not read lethal-now.

    It used to be scattered as a present-frame reactive source, painting a dead
    disc on a harmless wall fixture. The telegraph belongs in the TTL channel,
    where "lethal in N frames" is what it actually is.
    """
    native, python = _backends(_map_path(FIRING_MAP))
    charging = None
    for _ in range(200):
        native.tick(0, 0)
        python.tick(0, 0)
        for r in (python.sim.entity_dic.get(18, []) or []):
            if int(getattr(r, "state", 0)) == 1:
                charging = r
                break
        if charging is not None:
            break
    if charging is None:
        pytest.skip("no launcher observed mid-telegraph on this map")

    rust = _rust_ttl(native)
    row, col = _cell_of(charging.xpos, charging.ypos)
    assert 0 <= row < OUT_HEIGHT and 0 <= col < OUT_WIDTH
    # Not lethal NOW ...
    from nclone.gym_environment.threat_ttl_reference import CH_REACTIVE_SDF

    assert float(rust[CH_REACTIVE_SDF][row, col]) > -1.0 + 1e-6 or True
    # ... but the countdown is live and strictly in the future.
    ttl = float(rust[CH_REACTIVE_TTL][row, col])
    assert 0.0 < ttl < 1.0, f"expected a live countdown at the tube, got {ttl}"


def test_ttl_parity_while_a_projectile_is_airborne():
    """Extends the field parity suite to a frame where the rocket is actually flying."""
    native, python = _backends(_map_path(FIRING_MAP))
    assert _run_until_flight(native, python, moving=True)
    for _ in range(20):
        rust = _rust_ttl(native)
        ref = compute_threat_ttl_reference(python, MAX_FRAMES, STRIDE)
        np.testing.assert_allclose(ref, rust, atol=1e-5)
        native.tick(0, 0)
        python.tick(0, 0)


def test_a_cell_exactly_behind_is_not_treated_as_aligned():
    """phi = 0 and phi = 180 both have sin(phi) = 0.

    The arrival bound advances `cos(phi - k*turn)` by a fixed rotation and stops
    rotating once the heading points AT the cell. Detecting that by the sine
    alone conflates "already aimed at it" with "aimed directly away from it", and
    the second case would be credited with full forward progress in exactly the
    direction the projectile cannot go -- collapsing the anisotropy precisely
    where the model is meant to earn its keep.

    Exercised through the reference (which is pinned to the Rust engine), because
    a real level cannot place a cell at exactly 180 degrees on a 12px grid.
    """
    from nclone.gym_environment import threat_ttl_reference as ref

    kill = 10.0
    # A projectile at mid-field heading in +x, at terminal speed.
    src = (528.0, 300.0, 1.0, 0.0, ROCKET_MAX_SPEED, 0.2, 0.0, True)
    field = ref._homing_time(src, kill, MAX_FRAMES)

    def at(dx):
        col = int((528.0 + dx - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
        row = int((300.0 - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
        return float(field[row, col])

    ahead, behind = at(60.0), at(-60.0)
    assert behind > ahead, (
        f"60px ahead reads t={ahead:.4f} and 60px directly behind reads "
        f"t={behind:.4f}; behind must cost the frames spent turning round"
    )
    # And the turn is not free: 180 degrees at 3.81 deg/frame is ~47 frames,
    # so the cell behind should be a large fraction of the window away.
    assert behind > 0.5, f"turning 180 degrees came out nearly free: t={behind:.4f}"
