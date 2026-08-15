"""Pure-Python twin of the `threat_proximity` telegraph block.

The production implementation lives in `nclone-native/src/threat_proximity.rs`
and is what the policy consumes. This module exists for the same reason
`threat_ttl_reference.py` does: the dev tool renders through the pure-Python
backend, which has no Rust simulator behind it.

**A second implementation is a liability, so it is pinned.**
`nclone/tests/test_threat_proximity_parity.py` steps both backends on the same
map and asserts the two vectors agree dim for dim.

# Why the block moved to Rust

It used to be computed here, off `sim.entity_dic`. On the NATIVE backend
`_build_entity_dic` whitelists types `{1, 2, 3, 4, 6, 7, 11, 21}` — mines, gold,
doors, switches, one-ways — and **not one threat type is in that set**. Measured
on `official/R/027 rear guard`, which carries a rocket (18) and a floor guard
(16): the Python backend's `entity_dic` reports `{18: 1}`, the native backend's
reports nothing at all. Training runs `--sim-backend native`, so the whole block
was structurally zero in production while every unit test passed — the tests
exercise the pure `imminence` mapping, and the overlay runs on Python.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    THREAT_CLASS_BEAM,
    THREAT_CLASS_CONTACT,
    THREAT_CLASS_HOMING,
    THREAT_CLASS_PROJECTILE,
    THREAT_PROX_CLASSES,
    THREAT_PROX_FEATURES,
    THREAT_PROXIMITY_DIM,
    THREAT_PROXIMITY_RANGE_PX,
    THREAT_PROXIMITY_SPEED_NORM_PX,
)
from .threat_ttl_reference import (
    ROCKET_INITIAL_STEP,
    ROCKET_LAUNCH_DELAY,
    ROCKET_MAX_SPEED,
    ROCKET_SPEED_MULTIPLIER,
)

#: Turret `+0x20` charge threshold. Mirrors TURRET_CHARGE_THRESHOLD in Rust.
TURRET_CHARGE_THRESHOLD = 90.0
#: Direction index -> unit vector, matching `DIR_TO_VEC` in drone.rs.
_DIR_TO_VEC = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
#: `EntityEvilNinja.PLAYBACK` — the only state in which it is lethal.
EVIL_NINJA_PLAYBACK = 2


def rocket_spin_up(speed: float, step: float) -> float:
    """Speed the projectile gains NEXT frame, as a fraction of its cap.

    The acceleration term. A projectile at `speed = 0.05` and one at
    `speed = 2.28` are 45x apart in how far they travel over the next second and
    the still-frame bearing cannot tell them apart. Reaches exactly 0.0 at
    terminal speed, so "still winding up" and "committed" are distinguishable.
    """
    if speed >= ROCKET_MAX_SPEED:
        return 0.0
    next_step = step * ROCKET_SPEED_MULTIPLIER
    gain = min(speed + next_step, ROCKET_MAX_SPEED) - speed
    return float(np.clip(gain / ROCKET_MAX_SPEED, 0.0, 1.0))


def threat_imminence(e, type_id: int) -> float:
    """Normalised telegraph countdown in [0, 1] for one shooter.

    0.0 means "nothing winding up". 1.0 means "the shot is leaving now". Every
    value below is read from the entity's own state machine rather than assumed,
    because a made-up countdown is worse than none: it would look learnable and
    teach the wrong timing.

    RECOVERING/COOLDOWN deliberately report 0.0. That is not merely "no threat" —
    it is a *safe window*, and it is the most actionable frame the signal has:
    the turret has just fired and cannot fire again for 15 frames, so this is
    exactly when crossing its line is free.

    The single Python source of truth for this mapping: the env method delegates
    here, `_candidates` calls it, and the Rust twin is pinned against the result
    by the parity test.
    """
    state = int(getattr(e, "state", 0) or 0)
    if type_id == 19:  # turret: SEARCHING/TRACKING/FIRING/RECOVERING
        if state == 1:  # TRACKING — charge accumulates toward the commit
            thr = float(getattr(e, "CHARGE_THRESHOLD", TURRET_CHARGE_THRESHOLD)) or TURRET_CHARGE_THRESHOLD
            return float(np.clip(float(getattr(e, "charge", 0.0)) / thr, 0.0, 1.0))
        if state == 2:  # FIRING — committed, the beam is live
            return 1.0
        return 0.0  # SEARCHING (no lock) or RECOVERING (just fired: safe)
    if type_id == 18:  # rocket: IDLE/CHARGING/FLYING
        if state == 1:
            delay = float(getattr(e, "LAUNCH_DELAY", ROCKET_LAUNCH_DELAY)) or ROCKET_LAUNCH_DELAY
            return float(np.clip(float(getattr(e, "charge", 0.0)) / delay, 0.0, 1.0))
        if state == 2:  # FLYING — the projectile is already homing
            return 1.0
        return 0.0
    if type_id in (12, 13):  # shooter drones: IDLE/AIMING/FIRING/COOLDOWN
        if state == 1:
            dur = float(getattr(e, "CHARGE_DURATION", 53.0)) or 53.0
            return float(np.clip(float(getattr(e, "timer", 0.0)) / dur, 0.0, 1.0))
        if state == 2:
            return 1.0
        return 0.0
    # Dual laser and the contact/homing classes have no wind-up: they are either
    # lethal or not, and the FIELD already says which.
    return 0.0


def _entities(nplay_headless, type_id):
    sim = getattr(nplay_headless, "sim", None)
    if sim is None:
        return []
    return [
        e
        for e in (getattr(sim, "entity_dic", {}) or {}).get(type_id, [])
        if getattr(e, "active", True)
    ]


def _candidates(nplay_headless):
    """`(class, x, y, vx, vy, imminence, spin_up)` for every live threat.

    Mirrors `collect_candidates` in threat_proximity.rs. Every `imminence` is
    read from the entity's own state machine rather than assumed: a made-up
    countdown is worse than none, because it looks learnable and teaches the
    wrong timing.
    """
    out = []

    def drone_velocity(d):
        vx, vy = _DIR_TO_VEC[int(getattr(d, "dir", 0)) & 3]
        speed = float(getattr(d, "speed", 0.0))
        return speed * vx, speed * vy

    for type_id in (14, 26):  # zap, mini — contact-lethal patrol
        for d in _entities(nplay_headless, type_id):
            vx, vy = drone_velocity(d)
            out.append((THREAT_CLASS_CONTACT, d.xpos, d.ypos, vx, vy, 0.0, 0.0))

    for d in _entities(nplay_headless, 15):  # chaser
        vx, vy = drone_velocity(d)
        out.append((THREAT_CLASS_HOMING, d.xpos, d.ypos, vx, vy, 0.0, 0.0))

    for type_id in (25, 27):  # death ball, eye bat
        for b in _entities(nplay_headless, type_id):
            out.append((
                THREAT_CLASS_HOMING, b.xpos, b.ypos,
                float(getattr(b, "xspeed", 0.0)), float(getattr(b, "yspeed", 0.0)),
                0.0, 0.0,
            ))

    # Lethal ONLY in PLAYBACK: a dormant one cannot kill and carries the larger
    # dormant radius. Its velocity is not exposed, so the motion dims stay zero
    # rather than being invented.
    for n in _entities(nplay_headless, 22):
        if int(getattr(n, "state", 0)) == EVIL_NINJA_PLAYBACK:
            out.append((THREAT_CLASS_HOMING, n.xpos, n.ypos, 0.0, 0.0, 0.0, 0.0))

    # Shooter drones: the BODY is harmless, so this reports the origin of a beam
    # that is about to exist. COOLDOWN reads 0.0, which is a positive signal —
    # it has just fired and cannot fire again yet.
    for type_id in (12, 13):
        for d in _entities(nplay_headless, type_id):
            imm = threat_imminence(d, type_id)
            out.append((THREAT_CLASS_BEAM, d.xpos, d.ypos, 0.0, 0.0, imm, 0.0))

    for t in _entities(nplay_headless, 19):  # turret: a fixture
        out.append((THREAT_CLASS_BEAM, t.xpos, t.ypos, 0.0, 0.0,
                    threat_imminence(t, 19), 0.0))

    # Dual laser: no wind-up at all — it is either sweeping a lethal line or it
    # is not, and the FIELD already carries which.
    for laser in _entities(nplay_headless, 23):
        out.append((THREAT_CLASS_BEAM, laser.xpos, laser.ypos, 0.0, 0.0, 0.0, 0.0))

    for r in _entities(nplay_headless, 18):
        state = int(getattr(r, "state", 0))
        if state == 2:
            # Report the PROJECTILE, not the launcher. That is the whole point of
            # the class: the launcher is a harmless tube bolted to a wall, and
            # once it has fired the lethal object can be hundreds of px away
            # closing at 2.29 px/frame while the launcher's bearing says nothing.
            p = r.projectile
            out.append((
                THREAT_CLASS_PROJECTILE, p.xpos, p.ypos,
                float(p.xdir) * float(p.speed), float(p.ydir) * float(p.speed),
                1.0, rocket_spin_up(float(p.speed), float(p.step)),
            ))
        elif state == 1:
            # CHARGING: the launcher IS the right object — the projectile does
            # not exist yet and will appear exactly here.
            out.append((
                THREAT_CLASS_PROJECTILE, r.xpos, r.ypos, 0.0, 0.0,
                threat_imminence(r, 18), rocket_spin_up(0.0, ROCKET_INITIAL_STEP),
            ))
        # IDLE acquires by sight in one frame and has committed to nothing.
        # Reporting it would make a harmless fixture read as a permanent threat.

    return out


def compute_threat_proximity_reference(nplay_headless, nx: float, ny: float) -> np.ndarray:
    """`(THREAT_PROXIMITY_DIM,)` float32, mirroring threat_proximity.rs.

    `closing` is the TRUE relative closing rate `(v_ninja - v_threat)·u`,
    positive when the gap is shrinking. Projecting the ninja's velocity alone —
    what this did before — is correct for a mine and wrong for everything in this
    table: a stationary ninja with a rocket inbound at full speed read
    `closing = 0.0`, i.e. "nothing is happening".
    """
    out = np.zeros(THREAT_PROXIMITY_DIM, dtype=np.float32)
    nvx, nvy = nplay_headless.ninja_velocity()

    best = {}
    for cand in _candidates(nplay_headless):
        cls, x, y = cand[0], float(cand[1]), float(cand[2])
        d = float(np.hypot(x - nx, y - ny))
        if d > THREAT_PROXIMITY_RANGE_PX:
            continue
        prev = best.get(cls)
        if prev is None or d < prev[0]:
            best[cls] = (d, cand)

    for cls in range(THREAT_PROX_CLASSES):
        if cls not in best:
            continue
        d, (_, x, y, vx, vy, imminence, spin_up) = best[cls]
        base = cls * THREAT_PROX_FEATURES
        if d > 1e-6:
            ux, uy = (float(x) - nx) / d, (float(y) - ny) / d
        else:
            ux = uy = 0.0
        closing = ((float(nvx) - vx) * ux + (float(nvy) - vy) * uy)
        out[base + 0] = ux
        out[base + 1] = uy
        out[base + 2] = float(np.clip(1.0 - d / THREAT_PROXIMITY_RANGE_PX, 0.0, 1.0))
        out[base + 3] = float(np.clip(closing / THREAT_PROXIMITY_SPEED_NORM_PX, -1.0, 1.0))
        out[base + 4] = float(np.clip(imminence, 0.0, 1.0))
        out[base + 5] = float(np.clip(vx / THREAT_PROXIMITY_SPEED_NORM_PX, -1.0, 1.0))
        out[base + 6] = float(np.clip(vy / THREAT_PROXIMITY_SPEED_NORM_PX, -1.0, 1.0))
        out[base + 7] = float(np.clip(spin_up, 0.0, 1.0))
    return out
