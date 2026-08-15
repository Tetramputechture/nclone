"""Pure-Python reference implementation of the threat time-to-lethal field.

The production field lives in `nclone-native/src/threat_field.rs` and is what the
policy consumes. This module exists for ONE reason: the dev tool renders through
the pure-Python backend (`NPlayHeadless`), because the native simulator has no
renderer at all, so an overlay that could only read the Rust field would never be
visible on the tool where it is actually useful.

**A second implementation is a liability, so it is pinned.**
`nclone/tests/test_threat_ttl_parity.py` loads the same map into both backends,
steps them identically and asserts the two fields agree cell for cell. Without
that, this file would drift from the Rust one and the overlay would quietly stop
showing what the policy sees -- which is worse than having no overlay, because it
would be believed.

Not for the hot path: this is vectorised numpy, fine at dev-tool frame rates, but
it marshals drone positions into Python exactly the way the Rust engine exists to
avoid.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants.physics_constants import NINJA_RADIUS

# Must match threat_field.rs.
OUT_WIDTH = 84
OUT_HEIGHT = 46
BLOCK_PX = 12.0
FINE_HALF_PX = 3.0
PLAYFIELD_OFFSET_PX = 24.0
SDF_SCALE_PX = 24.0

DRONE_RADIUS = 7.5
MINI_DRONE_RADIUS = 4.0
DEATH_BALL_RADIUS = 5.0
EYE_BAT_RADIUS = 5.0

FLOOR_GUARD_RADIUS = 6.0
SHOVE_THWUMP_RADIUS = 8.0
ROCKET_RADIUS = 0.0
THWUMP_SEMI_SIDE = 9.0

#: `EntityEvilNinja.PLAYBACK` -- the only state in which it is lethal.
EVIL_NINJA_PLAYBACK = 2
#: Kill radius about the face SEGMENT (not a disc containing it): the collision
#: test inflates the ninja circle by 2 and tests it against that segment.
THWUMP_FACE_KILL_RADIUS = NINJA_RADIUS + 2.0

DRONE_ZAP_SPEED = 8.0 / 7.0
DEATH_BALL_MAX_SPEED = 0.85
EYE_BAT_MAX_SPEED = 3.0
FLOOR_GUARD_SPEED = 24.0 / 7.0
THWUMP_FORWARD_SPEED = 20.0 / 7.0
ROCKET_MAX_SPEED = 16.0 / 7.0
#: The projectile's speed schedule, which is ninja-INDEPENDENT and therefore
#: exactly known: `step *= MULT; speed = min(speed + step, MAX)` every frame.
ROCKET_INITIAL_STEP = 2.0 / 45.0
ROCKET_SPEED_MULTIPLIER = 1.1 ** (2.0 / 3.0)
#: Homing gain. The heading rotates by `atan(cross * TURN_RATE)` with the sine
#: `cross` in [-1, 1], so 3.81 degrees a frame is the hard ceiling.
ROCKET_TURN_RATE = 1.0 / 15.0
ROCKET_LAUNCH_DELAY = 15.0
#: Shove thwumps shove off at 4 px/frame and return at 1; the envelope takes the
#: larger.
SHOVE_THWUMP_SPEED = 4.0
#: Beams kill on contact with the ninja's own radius; there is no projectile body.
BEAM_KILL_RADIUS = NINJA_RADIUS

# Channel layout, mirroring threat_field.rs.
CH_CERTAIN_SDF = 0
CH_CERTAIN_TTL = 1
CH_REACTIVE_SDF = 2
CH_REACTIVE_TTL = 3
N_CHANNELS = 4


def _block_centres():
    """(rows, cols) grids of output-block centre coordinates, in world px."""
    rows = np.arange(OUT_HEIGHT, dtype=np.float64)
    cols = np.arange(OUT_WIDTH, dtype=np.float64)
    by = PLAYFIELD_OFFSET_PX + (rows + 0.5) * BLOCK_PX
    bx = PLAYFIELD_OFFSET_PX + (cols + 0.5) * BLOCK_PX
    return np.meshgrid(bx, by)  # both (OUT_HEIGHT, OUT_WIDTH)


_BX, _BY = _block_centres()


def _nearest_subcentre_distance(px: float, py: float) -> np.ndarray:
    """Distance from (px, py) to the nearest 6px sub-centre of each output block.

    This is the same analytic pick the Rust scatter makes: the four sub-centres
    sit at +-3px about the block centre, so minimising the squared distance over
    them separates per axis, and the nearest is just the nearer one in x and the
    nearer one in y.
    """
    cx = np.where(px > _BX, _BX + FINE_HALF_PX, _BX - FINE_HALF_PX)
    cy = np.where(py > _BY, _BY + FINE_HALF_PX, _BY - FINE_HALF_PX)
    return np.hypot(cx - px, cy - py)


def _sdf_from(points, kill_radius: float, out: np.ndarray) -> None:
    """Min-merge `clip((d - kill_radius) / 24, -1, 1)` for each point into `out`."""
    for px, py in points:
        d = _nearest_subcentre_distance(float(px), float(py))
        np.minimum(out, np.clip((d - kill_radius) / SDF_SCALE_PX, -1.0, 1.0), out=out)


def _lethal_time_from(points, kill_radius: float, t_norm: float, out: np.ndarray) -> None:
    """Mark cells inside the kill disc as lethal no later than `t_norm`."""
    if t_norm >= 1.0:
        return
    for px, py in points:
        d = _nearest_subcentre_distance(float(px), float(py))
        np.minimum(out, np.where(d <= kill_radius, np.float32(t_norm), np.float32(1.0)), out=out)


def compute_threat_ttl_reference(
    nplay_headless,
    max_frames: int = 64,
    stride: int = 4,
) -> np.ndarray:
    """`(4, OUT_HEIGHT, OUT_WIDTH)` float32 field: [C_sdf, C_ttl, R_sdf, R_ttl].

    Args:
        nplay_headless: backend exposing ``get_drone_positions`` /
            ``get_predicted_drone_positions`` and a ``sim.entity_dic``.
        max_frames: window length in sim frames. 1.0 in a TTL channel means "not
            lethal within this window".
        stride: tier-C sampling interval, in sim frames. The reactive tier is
            closed-form and ignores it.
    """
    field = np.ones((N_CHANNELS, OUT_HEIGHT, OUT_WIDTH), dtype=np.float32)
    max_frames = max(1, int(max_frames))
    stride = max(1, int(stride))

    zap, mini = nplay_headless.get_drone_positions()

    # --- Tier C: deterministic patrol drones -----------------------------
    _sdf_from(zap, NINJA_RADIUS + DRONE_RADIUS, field[CH_CERTAIN_SDF])
    _sdf_from(mini, NINJA_RADIUS + MINI_DRONE_RADIUS, field[CH_CERTAIN_SDF])

    _lethal_time_from(zap, NINJA_RADIUS + DRONE_RADIUS, 0.0, field[CH_CERTAIN_TTL])
    _lethal_time_from(mini, NINJA_RADIUS + MINI_DRONE_RADIUS, 0.0, field[CH_CERTAIN_TTL])

    # The evil ninja: contact-lethal (`logical_collision` -> kill "evil_ninja")
    # yet present in NO collector before this. 58 official maps, 941 instances.
    #
    # Tier C rather than R, and the distinction is real: it replays the ninja's
    # own recorded input tape at a fixed lag, so going forward its motion is a
    # recording, not a reaction to the live ninja.
    #
    # PLAYBACK only. A DORMANT/WAITING one cannot kill and carries the LARGER
    # 10.0 dormant radius, so admitting it would paint a bigger dead zone than
    # the game actually has.
    evil = [
        (float(e.xpos), float(e.ypos), float(getattr(e, "radius", 6.0)))
        for e in _entities(nplay_headless, 22)
        if int(getattr(e, "state", 0)) == EVIL_NINJA_PLAYBACK
    ]
    for ex, ey, er in evil:
        _sdf_from([(ex, ey)], NINJA_RADIUS + er, field[CH_CERTAIN_SDF])
        _lethal_time_from([(ex, ey)], NINJA_RADIUS + er, 0.0, field[CH_CERTAIN_TTL])

    samples: Sequence[int] = list(range(stride, max_frames + 1, stride))
    if samples:
        for frame, (fzap, fmini) in zip(
            samples, nplay_headless.get_predicted_drone_positions(samples)
        ):
            t = frame / max_frames
            _lethal_time_from(fzap, NINJA_RADIUS + DRONE_RADIUS, t, field[CH_CERTAIN_TTL])
            _lethal_time_from(fmini, NINJA_RADIUS + MINI_DRONE_RADIUS, t, field[CH_CERTAIN_TTL])

    # --- Committed-velocity hazards also ride the CERTAIN tier ------------
    kinematic = _kinematic_sources(nplay_headless)
    for kx, ky, vx, vy, kill in kinematic:
        _sdf_from([(kx, ky)], kill, field[CH_CERTAIN_SDF])
        _lethal_time_from([(kx, ky)], kill, 0.0, field[CH_CERTAIN_TTL])
        for frame in samples:
            _lethal_time_from(
                [(kx + vx * frame, ky + vy * frame)],
                kill,
                frame / max_frames,
                field[CH_CERTAIN_TTL],
            )

    # --- Thwumps: segment geometry, forward-only sweep --------------------
    for (x1, y1, x2, y2, vx, vy), committed in _thwump_faces(nplay_headless):
        sdf_ch = CH_CERTAIN_SDF if committed else CH_REACTIVE_SDF
        ttl_ch = CH_CERTAIN_TTL if committed else CH_REACTIVE_TTL
        d = _segment_distance(x1, y1, x2, y2)
        np.minimum(
            field[sdf_ch],
            np.clip((d - THWUMP_FACE_KILL_RADIUS) / SDF_SCALE_PX, -1.0, 1.0),
            out=field[sdf_ch],
        )
        for frame in [0] + list(samples):
            ox, oy = vx * frame, vy * frame
            d = _segment_distance(x1 + ox, y1 + oy, x2 + ox, y2 + oy)
            t = np.float32(frame / max_frames)
            np.minimum(
                field[ttl_ch],
                np.where(d <= THWUMP_FACE_KILL_RADIUS, t, np.float32(1.0)),
                out=field[ttl_ch],
            )

    # --- Homing projectiles: exact speed ramp, bounded turn ---------------
    for src in _projectile_sources(nplay_headless):
        kill = NINJA_RADIUS + ROCKET_RADIUS
        # Only a projectile that EXISTS is lethal now; a charging launcher's
        # body never kills, so it contributes to the countdown and nothing else.
        if src[7]:
            _sdf_from([(src[0], src[1])], kill, field[CH_REACTIVE_SDF])
        np.minimum(
            field[CH_REACTIVE_TTL],
            _homing_time(src, kill, max_frames),
            out=field[CH_REACTIVE_TTL],
        )

    # --- Beams: live lethal segments (turret hitscan, dual laser) ---------
    # A degenerate segment means "no beam this frame" and must be skipped, or
    # every turret paints a permanent dot on itself.
    for x1, y1, x2, y2 in _beam_segments(nplay_headless):
        d = _segment_distance(x1, y1, x2, y2)
        np.minimum(
            field[CH_REACTIVE_SDF],
            np.clip((d - BEAM_KILL_RADIUS) / SDF_SCALE_PX, -1.0, 1.0),
            out=field[CH_REACTIVE_SDF],
        )
        np.minimum(
            field[CH_REACTIVE_TTL],
            np.where(d <= BEAM_KILL_RADIUS, np.float32(0.0), np.float32(1.0)),
            out=field[CH_REACTIVE_TTL],
        )

    # --- Tier R: ninja-reactive threats, worst-case envelope --------------
    for points, radius, speed in _reactive_sources(nplay_headless):
        _sdf_from(points, NINJA_RADIUS + radius, field[CH_REACTIVE_SDF])
        kill = NINJA_RADIUS + radius
        for px, py in points:
            d = _nearest_subcentre_distance(float(px), float(py))
            frames = np.maximum((d - kill) / speed, 0.0)
            t = np.where(frames <= max_frames, frames / max_frames, 1.0).astype(np.float32)
            np.minimum(field[CH_REACTIVE_TTL], t, out=field[CH_REACTIVE_TTL])

    return field


def _thwump_faces(nplay_headless):
    """Each thwump's lethal face SEGMENT plus its forward charge velocity.

    Mirrors `collect_thwumps` in threat_field.rs. Only the facing side of a
    thwump is lethal -- the other three are solid and standable -- so this must
    stay a segment swept FORWARD, never a disc dilated radially.
    """
    half = THWUMP_SEMI_SIDE - 2.0
    offset = THWUMP_SEMI_SIDE + 2.0
    out = []
    for t in _entities(nplay_headless, 20):
        speed = THWUMP_FORWARD_SPEED * t.direction
        if t.is_horizontal:
            fx = t.xpos + offset * t.direction
            seg = (fx, t.ypos - half, fx, t.ypos + half, speed, 0.0)
        else:
            fy = t.ypos + offset * t.direction
            seg = (t.xpos - half, fy, t.xpos + half, fy, 0.0, speed)
        out.append((seg, getattr(t, "state", 0) == 1))
    return out


def _segment_distance(x1, y1, x2, y2):
    """Distance from each output block's nearest sub-centre to a segment."""
    mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cx = np.where(mx > _BX, _BX + FINE_HALF_PX, _BX - FINE_HALF_PX)
    cy = np.where(my > _BY, _BY + FINE_HALF_PX, _BY - FINE_HALF_PX)
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    if len2 > 0.0:
        t = np.clip(((cx - x1) * dx + (cy - y1) * dy) / len2, 0.0, 1.0)
    else:
        t = np.zeros_like(cx)
    return np.hypot(cx - (x1 + t * dx), cy - (y1 + t * dy))


def _entities(nplay_headless, type_id):
    sim = getattr(nplay_headless, "sim", None)
    if sim is None:
        return []
    return [
        e
        for e in (getattr(sim, "entity_dic", {}) or {}).get(type_id, [])
        if getattr(e, "active", True)
    ]


def _kinematic_sources(nplay_headless):
    """`(x, y, vx, vy, kill_radius)` for hazards with a committed velocity.

    Mirrors `collect_kinematic` in threat_field.rs -- see that function for why
    the rocket projectile is NOT here (it homes every frame).
    """
    out = []
    for g in _entities(nplay_headless, 16):  # floor guard, latched
        if getattr(g, "dir", 0):
            out.append(
                (g.xpos, g.ypos, g.dir * FLOOR_GUARD_SPEED, 0.0,
                 NINJA_RADIUS + FLOOR_GUARD_RADIUS)
            )
    # Thwumps are not here -- they are segments, handled by `_thwump_faces`.
    return out


def _reactive_sources(nplay_headless):
    """`(points, radius, per_frame_speed)` for each reactive threat class."""
    if getattr(nplay_headless, "sim", None) is None:
        return []

    def xy(type_id):
        return [(float(e.xpos), float(e.ypos)) for e in _entities(nplay_headless, type_id)]

    sources = [
        (xy(15), DRONE_RADIUS, DRONE_ZAP_SPEED),  # chaser drone
        (xy(25), DEATH_BALL_RADIUS, DEATH_BALL_MAX_SPEED),
        (xy(27), EYE_BAT_RADIUS, EYE_BAT_MAX_SPEED),
        (xy(28), SHOVE_THWUMP_RADIUS, SHOVE_THWUMP_SPEED),
    ]
    flying, charging = [], []
    # Rockets are NOT here: their speed schedule is deterministic and their
    # heading is rate-limited, so `_projectile_sources` / `_homing_time` model
    # them instead. See `HomingProjectile` in threat_field.rs for the numbers.
    del flying, charging
    # Floor guard while dormant, thwump while idle or retracting.
    sources.append(
        (
            [(g.xpos, g.ypos) for g in _entities(nplay_headless, 16)
             if not getattr(g, "dir", 0)],
            FLOOR_GUARD_RADIUS,
            FLOOR_GUARD_SPEED,
        )
    )
    return sources


def _projectile_sources(nplay_headless):
    """Every live rocket projectile plus every launcher mid-telegraph.

    Mirrors `collect_projectiles` in threat_field.rs. Each entry is
    `(x, y, dirx, diry, speed, step, delay, heading_known)`. An IDLE launcher is
    deliberately absent -- it has committed to nothing and its body is harmless.
    """
    out = []
    for r in _entities(nplay_headless, 18):
        state = int(getattr(r, "state", 0))
        if state == 2:
            p = r.projectile
            out.append(
                (float(p.xpos), float(p.ypos), float(p.xdir), float(p.ydir),
                 float(p.speed), float(p.step), 0.0, True)
            )
        elif state == 1:
            delay = max(0.0, ROCKET_LAUNCH_DELAY - float(getattr(r, "charge", 0.0)))
            out.append(
                (float(r.xpos), float(r.ypos), 0.0, 0.0,
                 0.0, ROCKET_INITIAL_STEP, delay, False)
            )
    return out


def _projectile_speed_table(speed: float, step: float, n: int) -> np.ndarray:
    """Per-frame speed for the next n frames, from a live `(speed, step)`.

    Runs the same recurrence as `ProjectileState.advance` rather than a closed
    form: the `min(speed + step, MAX)` cap makes it piecewise, and a second
    derivation is a second thing to keep in step with the simulator.
    """
    table = np.empty(n, dtype=np.float64)
    for k in range(n):
        if speed < ROCKET_MAX_SPEED:
            step *= ROCKET_SPEED_MULTIPLIER
        speed = min(speed + step, ROCKET_MAX_SPEED)
        table[k] = speed
    return table


def _homing_time(src, kill_radius: float, max_frames: int) -> np.ndarray:
    """Normalised earliest-arrival time for one homing projectile.

    To be lethal at a cell the projectile must come within `kill_radius` of it,
    which requires its DISPLACEMENT COMPONENT along the direction to that cell to
    reach `gap = d - kill_radius`. The largest achievable such component after k
    frames is

        sum over i in 1..=k of  v_i * cos(max(phi - i*turn, 0))

    where `phi` is the angle from the current heading to the cell: turn toward it
    at the maximum rate, then hold. That heading trajectory is feasible, so this
    is exact rather than assumed, and it is a NECESSARY condition -- the field can
    only ever say a cell becomes lethal earlier than it really does, which is the
    direction a hazard field may err in.

    See `scatter_homing_time` in threat_field.rs, which this is pinned against.
    """
    x, y, dirx, diry, speed, step, delay, heading_known = src
    n = int(max(0.0, max_frames - delay))
    speeds = _projectile_speed_table(speed, step, n)
    d = _nearest_subcentre_distance(x, y)
    gap = d - kill_radius

    if heading_known:
        cx = np.where(x > _BX, _BX + FINE_HALF_PX, _BX - FINE_HALF_PX)
        cy = np.where(y > _BY, _BY + FINE_HALF_PX, _BY - FINE_HALF_PX)
        safe_d = np.maximum(d, 1e-9)
        c = np.clip(((cx - x) * dirx + (cy - y) * diry) / safe_d, -1.0, 1.0)
        s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    else:
        # A charging launcher has not aimed yet -- its heading is chosen at
        # launch from the live ninja -- so every direction is equally open.
        c = np.ones_like(d)
        s = np.zeros_like(d)

    turn = np.arctan(ROCKET_TURN_RATE)
    cos_t, sin_t = np.cos(turn), np.sin(turn)
    acc = np.zeros_like(d)
    frames = np.full(d.shape, np.inf)
    # Both phi = 0 and phi = 180 give sin(phi) = 0, so the sine alone cannot say
    # whether the heading already points AT the cell or directly AWAY from it.
    # The cosine breaks the tie; without it a cell directly behind the projectile
    # would be credited with full forward progress. See scatter_homing_time.
    aligned = (s <= 0.0) & (c > 0.0)
    for k in range(n):
        c_new = np.where(aligned, c, c * cos_t + s * sin_t)
        s_new = np.where(aligned, s, s * cos_t - c * sin_t)
        c, s = c_new, s_new
        aligned = aligned | (s <= 0.0)
        acc += speeds[k] * np.where(aligned, 1.0, c)
        newly = np.isinf(frames) & (acc >= gap)
        frames[newly] = k + 1
    # Cells already inside the kill disc are lethal the moment it exists.
    frames = np.where(gap <= 0.0, 0.0, frames)

    t = frames + delay
    return np.where(np.isinf(t) | (t > max_frames), 1.0, t / max_frames).astype(np.float32)


def _beam_segments(nplay_headless):
    """Live beam segments for turrets (19), dual lasers (23) and laser drones (13).

    Mirrors `collect_beams` in threat_field.rs. All three expose the raycast
    endpoint they killed along; it collapses onto their own position when there
    is no beam, which is the skip condition.

    The laser drone was missing here and in the Rust collector. Its BODY is
    deliberately not a threat source (``body_is_lethal`` is false for shooters),
    so omitting the beam omitted the entity from every channel -- on the 11
    official maps that carry one, the only lethal thing on the level was
    invisible to the policy.
    """
    out = []
    for type_id in (19, 23, 13):
        for e in _entities(nplay_headless, type_id):
            x1, y1 = float(e.xpos), float(e.ypos)
            x2 = float(getattr(e, "xend", x1))
            y2 = float(getattr(e, "yend", y1))
            if abs(x2 - x1) > 1e-9 or abs(y2 - y1) > 1e-9:
                out.append((x1, y1, x2, y2))
    return out
