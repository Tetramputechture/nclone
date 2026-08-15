"""Dual Laser Entity (Type 23).

Reconstructed from `docs/ENTITY_INTEGRATION_SPEC.md` §8.6
(`think` = 0x10401220, ctor = 0x10400960). Second-biggest gate in the corpus:
type 23 is on 16% of attract maps.

`think` is a 16-line dispatcher on a mode field at `+0x20`, which is the two-mode
split the abandoned `entity_laser.py` draft never reconciled:

    mode 0  spinner   0x10401270, 190 lines   pi/600, pi/300, no sqrt
    mode 1  surface   0x10401670, 412 lines   0.2, 0.1, sqrt, miss sentinel

Mode identification is verified by constant fingerprint, and
`EntityDualLaser::projectSurfaceLaserOntoInitialSurface()` exists as an assertion
string, so surface mode is real and not a misreading. The two modes touch
**disjoint field regions above `+0x1C`**, so they are modelled here as two code
paths over one object rather than as shared state.

Type 23 is thinkable but NOT movable (spec §4) -- surface mode writes `+0x10`
and `+0x18` itself, so it must not be added to the move pass to compensate.

### Spinner mode: two ordering facts that cause divergence if missed

1. **The hit test uses the pre-step angle.** `+0x48` is committed only after the
   ninja loop finishes, so the angle must not be advanced first.
2. **Angle wrapping uses `(double)(float)2π`**, i.e. `6.283185005187988`, not
   `2π` in full `f64` (spec §6.3). The wrap point is measurably not
   `6.283185307179586`, and this is exactly the kind of low-bit difference the
   differential parity harness asserts on.

Also: π/600 and π/300 are **not** two spin speeds. π/300 is the per-frame angular
step and π/600 is the hit tolerance -- exactly half a step, so the accepted arc
equals the arc swept, with no gaps and no double coverage. That relationship is
why the two constants differ by 2x.

### What is verified and what is not

Verified: the mode dispatch, the whole spinner control flow and its constants,
the float-rounded TAU, the disjoint field regions, the 10-iteration surface cap,
and that surface mode moves the entity itself.

**Not** verified: the two surface steppers (`FUN_1050E9E0` flat, `FUN_1050ECA0`
sloped), `DAT_105723A0` (a threshold in the walk's termination test), and the
range gate `FUN_1050E8D0` are all unread ([U]). Surface mode below is therefore a
reconstruction from the spec's structure plus the abandoned draft's partial
reading; only its *shape* (bounded crawl along the nearest surface, beam along the
outward normal) is spec-backed.
"""

import math
import os

from ..entities import Entity
from ..physics import (
    beam_hits_ninja,
    gather_segments_from_region,
    get_raycast_distance,
    get_single_closest_point,
    map_orientation_to_vector,
    raycast_vs_player,
)


_SEAT_RULE = os.environ.get("NCLONE_AB_LASER_SEAT_RULE") == "1"


class EntityDualLaser(Entity):
    """Type 23. A spinning beam, or a beam that crawls along level geometry."""

    SPINNER = 0
    SURFACE = 1

    MAX_COUNT_PER_LEVEL = 128

    RADIUS = 5.9
    #: `0x10572508`. Per-frame angular step in spinner mode.
    SPIN_STEP = 0.010471975
    #: `0x10572500`. Hit tolerance -- exactly half a step, by design.
    HIT_TOLERANCE = 0.0052359875
    #: `(double)(float)pi` and `(double)(float)2pi` as stored in `.rdata`. The
    #: game computed these in `float` and widened, so they are NOT math.pi/tau.
    PI_F32 = 3.1415927410125732
    TAU_F32 = 6.283185005187988
    #: `0x10571D98`. Surface-mode sub-step travel, and the ten sub-steps per frame.
    SURFACE_SPEED = 0.1
    SURFACE_ITERATIONS = 10
    #: `+0xA8`, an `atan2` the ctor computes from `+0xA0` and the 24px cell:
    #: `+0xA0 = sqrt((0.1k*24)^2 + (18.1k*24 - 24)^2) * 1.000001` with
    #: `k = 1/sqrt(327.62)`, i.e. `k` normalises `(0.1, 18.1)` -- note
    #: `0.1^2 + 18.1^2 = 327.62` exactly, and `18.1 = 24 - RADIUS`. The step is
    #: therefore `atan2(SURFACE_SPEED, 24 - RADIUS)`.
    #:
    #: This used to be the literal `0.005524747952943415`, a value the port had
    #: invented, and the comment here concluded it "matches to seven digits, so
    #: the corner rate was never the bug". It was: seven digits is 5.8e-8 of
    #: error per sub-step, and a laser pivoting a corner takes ten sub-steps a
    #: frame, so `10138435`'s laser leaves the game's on frame 1. Measured
    #: against the game's own `+0x68/+0x70` normal, the atan2 is exact to a ULP
    #: while the literal is off by 5.8e-8.
    SURFACE_CORNER_SPEED = math.atan2(SURFACE_SPEED, 24.0 - RADIUS)
    #: `0x10572510`. The beam angle is LERPED toward the surface normal at 20% a
    #: frame, it does not snap to it. This is the behavioural difference that
    #: matters: a snapping beam points straight at whatever the crawl just walked
    #: past, so it lands on the ninja on the frame it rounds a corner.
    BEAM_LERP = 0.2
    #: `0x105723A0` (a true f64 pi) bounds the wrap; `0x105723A8` is a HAND-TYPED
    #: 7-digit 2pi and is the step. Both verbatim -- a full-precision tau drifts.
    #: This is a third pi in one entity, distinct from the PI_F32 pair above.
    WRAP_BOUND = 3.141592653589793
    WRAP_STEP = 6.283185
    #: `0x1056C358` / `0x10571CB8` / `_PTR_105724F8`.
    SURFACE_GATHER = 12.0
    DEAD_END_TOLERANCE = 0.01
    ANCHOR_EPSILON = 1.1920928955078125e-07

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        # No logical collision of its own (spec §6.6) -- the emitter body is
        # harmless, only the beam kills.
        self.is_logical_collidable = False

        # Mode is decided at construction by proximity to geometry: a laser
        # mounted on a wall crawls, one placed in open space spins. The ctor's
        # `24.0` is the likely query radius; 12/7 are the abandoned draft's
        # numbers and are the only concrete reading available.
        result, closest_point = get_single_closest_point(
            self.sim, self.xpos, self.ypos, 12
        )
        # `get_single_closest_point` returns 0 for "nothing found", 1 for an
        # outside edge and **-1 for an inside edge** -- -1 is a hit, not an error.
        # This port read it as an error and sent it to SURFACE, and sent the
        # genuine no-hit case there too, against its own rule directly above.
        #
        # Both halves are wrong and the captures settle both. On `1011932` four
        # lasers sit at (744, 168), (672, 384), (456, 312) and (264, 432) with no
        # segment within 24px; the game leaves `+0x58`..`+0x70` zero on all four,
        # which only the spinner path does, while this port seated them on a
        # surface that is not there and swept a beam from it. Those four plus two
        # each on `1010625` and `1010640` were the whole gap.
        #
        # Scored over all 136 lasers in the capture batches: no-hit -> SPINNER
        # with `dist < 7` gives 136/136, at every query radius from 8 to 32. The
        # old branch gives 128/136 and no (radius, threshold) pair in that family
        # closes it, because the miss was the branch, not the numbers.
        if closest_point is None or (
            result == -1 and os.environ.get("NCLONE_AB_LASER_EMPTY_SURFACE") == "1"
        ):
            self.mode = (
                self.SURFACE
                if os.environ.get("NCLONE_AB_LASER_EMPTY_SURFACE") == "1"
                else self.SPINNER
            )
        else:
            dist = math.sqrt(
                (closest_point[0] - self.xpos) ** 2
                + (closest_point[1] - self.ypos) ** 2
            )
            self.mode = self.SURFACE if dist < 7 else self.SPINNER

        #: `+0x50`. Spin direction/multiplier, from the map's `mode` byte.
        self.spin = -1 if mode == 1 else 1
        #: Surface mode only crawls when `mode == 1`; see `_think_surface`.
        self._crawls = mode == 1
        self._laser_seat_ab = os.environ.get("NCLONE_AB_NO_LASER_SEAT") != "1"
        #: Restore the retired `mode == 1` crawl gate -- see `_think_surface`.
        self._mode_gate_ab = os.environ.get("NCLONE_AB_LASER_MODE_GATE") == "1"
        #: Walk flush candidates horizontal-first again -- see `_seat_on_surface`.
        self._hfirst_ab = os.environ.get("NCLONE_AB_LASER_HFIRST") == "1"
        #: Surface mode flips `spin` at a dead end, so the reset needs the
        #: original; spinner mode never changes it.
        self._initial_spin = self.spin
        self.xend, self.yend = self.xpos, self.ypos

        if self.mode == self.SPINNER:
            dx, dy = map_orientation_to_vector(orientation)
            #: `+0x48`. Current beam angle.
            self.angle = math.atan2(dy, dx)
        else:
            self.angle = 0.0
            # Outward surface normal; the crawl runs perpendicular to it and the
            # beam eases toward it.
            self.xnormal, self.ynormal = 0.0, -1.0
            self._seat_on_surface()
            #: `+0x58`/`+0x60`. The contact point the crawl last sat on. Whether
            #: it moves between sub-steps is what distinguishes sliding along a
            #: surface from being pinned on a convex corner.
            #
            # `_seat_on_surface` sets both of these from the winning segment. It
            # only leaves them unset if the laser has no surface within 24px at
            # all, which the fallback below covers.
            if not hasattr(self, "xanchor"):
                self.xanchor = self.xpos - self.xnormal * self.RADIUS
                self.yanchor = self.ypos - self.ynormal * self.RADIUS
            #: `+0xB0`. The beam's own angle, which LAGS the surface normal.
            self.beam_angle = math.atan2(self.ynormal, self.xnormal)

        self._initial_state = (
            self.xpos, self.ypos, self.angle,
            getattr(self, "xnormal", 0.0), getattr(self, "ynormal", -1.0),
        )
        self._initial_cell = self.cell

    def think(self):
        if self.mode == self.SPINNER:
            self._think_spinner()
        else:
            self._think_surface()

    # -- mode 0: spinner ---------------------------------------------------

    def _think_spinner(self):
        angle = self.angle  # PRE-step angle: everything below uses this one.
        angle_new = self.spin * self.SPIN_STEP + angle
        # Explicit wrap against the float-rounded TAU, not a modulo against 2pi.
        while angle_new >= self.TAU_F32:
            angle_new -= self.TAU_F32
        while angle_new < 0.0:
            angle_new += self.TAU_F32

        dx = math.cos(angle)
        dy = math.sin(angle)
        length = get_raycast_distance(self.sim, self.xpos, self.ypos, dx, dy)
        if length:
            self.xend = self.xpos + dx * length
            self.yend = self.ypos + dy * length
        else:
            self.xend, self.yend = self.xpos, self.ypos

        ninja = self.sim.ninja
        if ninja is not None and ninja.is_valid_target():
            if raycast_vs_player(
                self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos
            ):
                ninja_angle = math.atan2(
                    ninja.ypos - self.ypos, ninja.xpos - self.xpos
                )
                diff = ninja_angle - angle
                # Wrap into [-pi, pi) using the same float-rounded constants.
                while diff >= self.PI_F32:
                    diff -= self.TAU_F32
                while diff < -self.PI_F32:
                    diff += self.TAU_F32
                # Two kill paths, matching spec §8.6's [V] control flow: the
                # centre inside the swept arc, or the drawn beam clipping the
                # hitbox. The second stands in for `FUN_1050B070`, which is [U].
                #
                # Measured: across 524 type-23 attract demos the spinner scores
                # ZERO kills by either path -- every laser death is surface mode.
                # So this branch is not the source of type 23's false-kill rate,
                # and narrowing it to a centre-only test (which the pi/600 =
                # half-of-pi/300 tiling argues for) changes nothing on the corpus.
                # Left as the spec reads it until `FUN_1050B070` is decompiled.
                if abs(diff) <= self.HIT_TOLERANCE:
                    # Ninja centre is inside the swept arc.
                    ninja.kill(0, 0, 0, 0, 0, cause="laser_spinner")
                elif beam_hits_ninja(
                    self.xpos, self.ypos, self.xend, self.yend, ninja
                ):
                    # Centre is outside the arc, but the drawn beam still
                    # clips the ninja's hitbox.
                    ninja.kill(0, 0, 0, 0, 0, cause="laser_spinner")

        self.angle = angle_new  # committed only now

    # -- mode 1: surface ---------------------------------------------------

    @staticmethod
    def _segment_normal(segment, a=0.0, b=0.0):
        """A tile segment's outward normal at the contact point `(a, b)`.

        Tile segments are wound consistently, so the outward normal is
        `(-dy, dx)` (round-7 §3, `docs/decomp/TILE_SEGMENT_TABLES.md`). It does
        NOT depend on which side the query point is on: `is_back_facing` only
        says the laser spawned inside the solid, and the game seats it on the
        open side anyway. See `_seat_on_surface`.

        Circular segments have no `px`/`py`; their outward normal is radial,
        inverted on a concave arc.
        """
        # A DOOR states its facing outright. Its segment is wound low-to-high
        # like every other, because that winding is the numerically exact one for
        # `get_closest_point` and the ninja stands on these -- see
        # `entity_door_base.py`. So the facing cannot be recovered from `(-py,
        # px)` and is carried alongside instead.
        door_normal = getattr(segment, "door_normal", None)
        if door_normal is not None:
            return door_normal

        px = getattr(segment, "px", None)
        py = getattr(segment, "py", None)
        if px is not None and py is not None:
            plen = math.sqrt(px * px + py * py)
            if plen == 0:
                return None, None
            return -py / plen, px / plen
        radius = getattr(segment, "radius", 0.0)
        if not radius:
            return None, None
        sign = 1.0 if getattr(segment, "convex", True) else -1.0
        return (a - segment.xpos) / radius * sign, (b - segment.ypos) / radius * sign

    def _seat_on_surface(self):
        """Snap onto the nearest surface and derive the outward normal.

        Stands in for `EntityDualLaser::projectSurfaceLaserOntoInitialSurface()`,
        which the binary calls once at spawn and which is a separate function from
        the per-frame walk.

        Read off the round-9 captures (`docs/ENTITY_RE_ROUND9_FINDINGS.md` §1.5),
        which carry the game's own `xnormal`/`ynormal` at `+0x68`/`+0x70` and
        `xanchor`/`yanchor` at `+0x58`/`+0x60`:

        * the ANCHOR is the closest point on the surface, not the map coordinate --
          `10134679`'s lasers at map `y = 546` anchor at `y = 552`, six pixels down
          onto the floor;
        * the BODY sits one `RADIUS` out along the outward normal;
        * the normal is the SURFACE's, and orientation has nothing to do with it --
          `10138730` has six lasers on the same wall with orientations 3, 1 and 6,
          and the game gives all six `(1, 0)`.

        This needs the winning SEGMENT, not just the closest point, because a
        flush-mounted laser sits exactly on its surface and `pos - closest` is the
        zero vector. `get_single_closest_point` throws the segment away, so the
        search is inlined here; the scoring (the `- 0.1` nudge favouring
        front-facing hits) matches it exactly.
        """
        if os.environ.get("NCLONE_AB_NO_LASER_SEAT") == "1":
            # Legacy seating: the normal comes from the closest POINT, which is the
            # zero vector for a flush-mounted laser, so it stays (0, -1) and the
            # body sits one radius inside its own wall. Known wrong; kept only as
            # an escape hatch. Round 9 §1.5.
            result, closest_point = get_single_closest_point(
                self.sim, self.xpos, self.ypos, 24
            )
            if not closest_point:
                return
            dx = self.xpos - closest_point[0]
            dy = self.ypos - closest_point[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist == 0:
                return
            self.xnormal, self.ynormal = dx / dist, dy / dist
            return

        rad = 24
        segments = gather_segments_from_region(
            self.sim, self.xpos - rad, self.ypos - rad, self.xpos + rad, self.ypos + rad
        )
        best = None
        flush = []
        for segment in segments:
            is_back_facing, a, b = segment.get_closest_point(self.xpos, self.ypos)
            raw_sq = (self.xpos - a) ** 2 + (self.ypos - b) ** 2
            if raw_sq == 0.0:
                flush.append((segment, a, b, is_back_facing))
            if _SEAT_RULE:
                # The game's rule: strict `<` on the RAW distance, with the
                # front-facing preference applied only on an EXACT tie.
                if best is None or raw_sq < best[0]:
                    best = (raw_sq, segment, a, b, is_back_facing)
                elif (raw_sq == best[0] and best[4] and not is_back_facing):
                    best = (raw_sq, segment, a, b, is_back_facing)
            else:
                distance_sq = raw_sq - (0.0 if is_back_facing else 0.1)
                if best is None or distance_sq < best[0]:
                    best = (distance_sq, segment, a, b, is_back_facing)
        if best is None:
            return
        _, segment, a, b, is_back_facing = best

        # A laser mounted flush in a corner touches SEVERAL segments at distance
        # zero -- on `10134679` the two half-edges at (294, 552) tie exactly, one
        # wound each way. `get_single_closest_point`'s strict `<` then just keeps
        # whichever the gather happened to yield first, and nclone's gather order
        # is not the game's. Break the tie on meaning instead: the outward normal
        # is the one that points into open space.
        if len(flush) > 1:
            # Walk the flush candidates in the GAME's per-cell emission order.
            #
            # `FUN_1050D6E0` emits a cell's sides as left, bottom, right, top,
            # then the interior -- so a VERTICAL face is offered first. This
            # port's accumulator emits every horizontal face and then every
            # vertical one, so on a convex corner, where both candidates point
            # into open space and the loop below takes the first that does, the
            # horizontal always won.
            #
            # `10139043` is that corner: the laser spawns at (480, 312) with a
            # horizontal face and a vertical face both at distance zero. The port
            # seats on the horizontal and gets (0, -1); the game's captured
            # normal on frame 1 is (-0.9985, -0.0552), which is (-1, 0) -- the
            # vertical -- pivoted by 0.0552 rad, exactly ten sub-steps of
            # `SURFACE_CORNER_SPEED`. So the game seated on the vertical face and
            # the corner pivot did the rest inside frame 1.
            #
            # `NCLONE_AB_LASER_HFIRST=1` restores the horizontal-first walk.
            if not self._hfirst_ab:
                flush = sorted(flush, key=lambda t: getattr(t[0], "px", 1) != 0)
            for seg, fa, fb, fback in flush:
                nx, ny = self._segment_normal(seg, fa, fb)
                if nx is None:
                    continue
                cx = int((fa + nx * 12.0) // 24)
                cy = int((fb + ny * 12.0) // 24)
                if self.sim.tile_dic.get((cx, cy), 0) == 0:
                    segment, a, b, is_back_facing = seg, fa, fb, fback
                    break

        # The sign comes from the SEGMENT's winding, never from `pos - closest`.
        #
        # This port used `pos - closest` and fell back to the winding only for a
        # flush mount. That is wrong whenever the laser's map coordinate is
        # inside the solid, which is common: `10138522`'s laser 9 spawns at
        # x = 138, six pixels inside the wall at x = 144, and the game still puts
        # the body at x = 149.9 on the open side. `pos - closest` gives the
        # inward normal there and the beam fires into the wall.
        #
        # Scored against the game's own `+0x68`/`+0x70` over all 128 surface
        # lasers in the two capture batches: `pos - closest` 63, winding 120,
        # winding plus the flush tie-break above 126. The two it still misses are
        # one map whose captured normal is already off-axis on frame 1, i.e. a
        # crawl difference rather than a seating one.
        if os.environ.get("NCLONE_AB_NO_LASER_WINDING") == "1":
            dx = self.xpos - a
            dy = self.ypos - b
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                self.xnormal, self.ynormal = dx / dist, dy / dist
            else:
                nx, ny = self._segment_normal(segment, a, b)
                if nx is None:
                    return
                self.xnormal, self.ynormal = nx, ny
        else:
            nx, ny = self._segment_normal(segment, a, b)
            if nx is None:
                return
            self.xnormal, self.ynormal = nx, ny

        self.xanchor, self.yanchor = a, b
        self.xpos = a + self.xnormal * self.RADIUS
        self.ypos = b + self.ynormal * self.RADIUS

    def _think_surface(self):
        """Crawl along the surface, then fire along the outward normal.

        The crawl is capped at 10 iterations per frame -- a hard loop bound, not a
        convergence test, and observable when the laser reaches a corner
        (spec §8.6 point 1).
        """
        g = self.SURFACE_GATHER
        segments = gather_segments_from_region(
            self.sim, self.xpos - g, self.ypos - g, self.xpos + g, self.ypos + g
        )
        # A door in the gather region used to disable the crawl outright. That was
        # a proxy fitted to the captures -- every one of the 15 lasers the game
        # froze and this crawl moved had a door segment in region -- and it is
        # too strong. `FUN_10401670` gates on the door being the segment the
        # crawl would step ONTO, not on its being nearby: the `else` arm of the
        # dead-end test falls into the same handler when the winning segment's
        # kind is 3, and kind 3 is the door (it takes the linear closest-point
        # routine `FUN_1050ECA0` like an ordinary tile edge, but with `back`
        # forced false -- exactly `oriented=False`). See `_is_dead_end` below.
        #
        # On `10138435` the difference is the whole demo: a door edge at
        # y=336 enters the 12 px region on frame 385 while the laser is crawling
        # a tile floor at y=324, so the region test freezes it 57 px short of
        # where the game has it.
        # EVERY surface laser crawls. The mode byte picks the spin SIGN and
        # nothing else.
        #
        # A previous round gated the crawl on `mode == 1`, because `10134679`'s
        # eight mode-0 lasers and `10138730`'s six "do not move a pixel in 200
        # frames". They do not move because the crawl PINS them -- the dead-end
        # reversal below flips `spin` every sub-step once the contact point stops
        # advancing, so a laser mounted on a short face oscillates inside a
        # fraction of a pixel and reads as stationary at whole-pixel resolution.
        # Fifty of the 136 captured surface lasers move by 0.2 or less per frame,
        # which is exactly that, and `10138727`'s single laser crawls a full 1.0
        # (ten sub-steps of `SURFACE_SPEED`) on frame 1 with a mode byte of 0.
        #
        # Scored as the fraction of (laser, frame) pairs whose body is bit-equal
        # to the game's over the first 120 frames of every captured demo:
        # mode-gated 10440/15360 = 67.97%, crawl-always 13560/15360 = 88.28%.
        # `NCLONE_AB_LASER_MODE_GATE=1` restores the gate.
        if self._laser_seat_ab and not self._crawls and self._mode_gate_ab:
            segments = []
        if segments:
            for _ in range(self.SURFACE_ITERATIONS):
                # Probe one sub-step along the surface, perpendicular to the normal.
                tx = -self.ynormal * self.spin * self.SURFACE_SPEED
                ty = self.xnormal * self.spin * self.SURFACE_SPEED
                xnew = self.xpos + tx
                ynew = self.ypos + ty

                best_d2 = None
                best = None
                best_back = False
                best_seg = None
                for segment in segments:
                    back, a, b = segment.get_closest_point(xnew, ynew)
                    d2 = (xnew - a) ** 2 + (ynew - b) ** 2
                    # Strict `<`, then one tie-break: on an EXACT tie a
                    # front-facing segment displaces a back-facing one.
                    if best_d2 is None or d2 < best_d2:
                        best_d2, best, best_back, best_seg = d2, (a, b), bool(back), segment
                    elif d2 == best_d2 and best_back and not back:
                        best, best_back, best_seg = (a, b), False, segment
                if best is None:
                    break

                vx = xnew - best[0]
                vy = ynew - best[1]

                # Two ways into the tangent check, and the binary spells both:
                #
                #   * a dead end -- the probe has walked past the anchor and is
                #     now heading INTO the surface. Without this the crawl drives
                #     through the end of a surface and the beam ends up pointing
                #     somewhere the real one never aims.
                #   * the winning segment is a DOOR (kind 3). `FUN_10401670`'s
                #     `else` arm re-enters the same handler for it.
                #
                # Either way the laser only advances if the step still has a
                # positive component along the tangent; otherwise it reverses and
                # stays put. Note a door is crawlable -- it is refused only when
                # the step would turn back on itself.
                if (
                    (self.xpos - self.xanchor) * vx
                    + (self.ypos - self.yanchor) * vy
                    <= self.DEAD_END_TOLERANCE
                    or not getattr(best_seg, "oriented", True)
                ):
                    if tx * vx + ty * vy <= 0.0:
                        self.spin = -self.spin
                        continue

                moved = math.hypot(best[0] - self.xanchor, best[1] - self.yanchor)
                if moved >= self.ANCHOR_EPSILON:
                    # The contact point moved: snap the normal onto it. The sign
                    # comes from which side of the directed segment we are on.
                    dist = math.hypot(vx, vy)
                    if dist == 0.0:
                        break
                    sign = -1.0 if best_back else 1.0
                    self.xnormal = vx / dist * sign
                    self.ynormal = vy / dist * sign
                else:
                    # The contact point did NOT move -- we are pinned on a convex
                    # corner, so rotate the normal in place instead of sliding.
                    angle = math.atan2(self.ynormal, self.xnormal)
                    angle += self.SURFACE_CORNER_SPEED * self.spin
                    self.xnormal = math.cos(angle)
                    self.ynormal = math.sin(angle)

                self.xanchor, self.yanchor = best
                self.xpos = best[0] + self.xnormal * self.RADIUS
                self.ypos = best[1] + self.ynormal * self.RADIUS

            self.grid_move()

        # The beam does NOT point along the normal -- it eases toward it, 20% of
        # the remaining angle per frame, so it lags the crawl.
        diff = math.atan2(self.ynormal, self.xnormal) - self.beam_angle
        while abs(diff) >= self.WRAP_BOUND:
            diff -= math.copysign(self.WRAP_STEP, diff)
        self.beam_angle += diff * self.BEAM_LERP
        bx = math.cos(self.beam_angle)
        by = math.sin(self.beam_angle)

        length = get_raycast_distance(self.sim, self.xpos, self.ypos, bx, by)
        if length:
            self.xend = self.xpos + bx * length
            self.yend = self.ypos + by * length
        else:
            self.xend, self.yend = self.xpos, self.ypos
            length = None

        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        if not raycast_vs_player(self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos):
            return
        # `length`, not `hypot(xend - xpos, yend - ypos)`: `FUN_1050B070` takes the
        # raycast's own distance as `param_5`. See `check_lineseg_vs_ninja`.
        if beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ninja, length):
            ninja.kill(0, 0, 0, 0, 0, cause="laser_surface")

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        (
            self.xpos,
            self.ypos,
            self.angle,
            self.xnormal,
            self.ynormal,
        ) = self._initial_state
        self.cell = self._initial_cell
        self.xend, self.yend = self.xpos, self.ypos
        self.active = True
        if self.mode == self.SURFACE:
            self.spin = self._initial_spin
            self.xanchor = self.xpos - self.xnormal * self.RADIUS
            self.yanchor = self.ypos - self.ynormal * self.RADIUS
            self.beam_angle = math.atan2(self.ynormal, self.xnormal)
