"""Pull per-frame dynamic geometry (ninja and entities) from the live sim.

This is the seam that reads simulation state, once per rendered frame. It never
mutates the sim.

Entity and ninja geometry is built by instancing precomputed unit templates
(sphere, box, spiky mine) with vectorized numpy: one array operation per
primitive class per frame rather than a Python trig loop per entity. On software
GL the Python mesh build, not the GPU, was the frame-time bottleneck before
this, at roughly 23 ms per frame against 1.3 ms after.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .. import render_utils
from .geometry import DEPTH, MeshBuilder

# The ninja renders solid black (matching the 2D game). A single flat color also
# means overlapping bone segments / joints can't show as darker seams.
NINJA_COLOR = (0.0, 0.0, 0.0)
NINJA_RADIUS = 10.0

# Ninja speed (sim px/frame) treated as "full speed" by every presentation
# effect keyed off velocity: trail colour, camera closeness, speed FOV, the
# animated background. Roughly the sim's top horizontal run speed.
MAX_SPEED = 3.4
# Vertical speed treated as full for the background's jump-impulse channel;
# a jump peaks noticeably higher than the horizontal run speed.
MAX_VSPEED = 4.6

# Exit door entity type; it vanishes once the ninja wins (state 8).
_EXIT_DOOR_TYPE = 3


# --- precomputed unit templates (triangle soup: positions + normals) --------

def _unit_sphere(lat: int = 10, lon: int = 14):
    """Unit sphere at the origin; for a unit sphere position == normal."""
    def vert(i, j):
        theta = math.pi * i / lat
        phi = 2.0 * math.pi * j / lon
        return (math.sin(theta) * math.cos(phi),
                math.cos(theta),
                math.sin(theta) * math.sin(phi))
    pts = []
    for i in range(lat):
        for j in range(lon):
            a = vert(i, j); b = vert(i + 1, j)
            c = vert(i + 1, j + 1); d = vert(i, j + 1)
            pts += [a, b, c, a, c, d]
    p = np.array(pts, dtype=np.float32)
    return p, p.copy()  # normal == position on the unit sphere


def _unit_box():
    """Unit box [-1,1]^3 with correct per-face normals (reuses MeshBuilder)."""
    mb = MeshBuilder()
    mb.add_box(-1, -1, -1, 1, 1, 1, (0.0, 0.0, 0.0))
    v = mb.build()
    return v[:, 0:3].copy(), v[:, 3:6].copy()


def _fib_dirs(n: int):
    """``n`` roughly-even unit directions on the sphere (Fibonacci lattice)."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        t = ga * i
        out.append((math.cos(t) * r, y, math.sin(t) * r))
    return out


def _perp_basis(d):
    """Two orthonormal vectors spanning the plane perpendicular to ``d``."""
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref); u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(d, u); v /= (np.linalg.norm(v) or 1.0)
    return u, v


def _unit_mine(n_spikes: int = 12, inner: float = 0.66, base_r: float = 0.14):
    """Spiky mine: a reduced inner sphere with outward spikes reaching radius 1.

    Built once as a unit template (tip at radius 1 = the mine's true radius, core
    shrunk so the 3-sided spikes have room to protrude), then instanced like any
    other sphere, so it costs the same per frame as a plain sphere entity.
    """
    mb = MeshBuilder()
    mb.add_sphere((0.0, 0.0, 0.0), inner, (0.0, 0.0, 0.0), lat=6, lon=8)
    for dd in _fib_dirs(n_spikes):
        d = np.asarray(dd, dtype=np.float64)
        u, v = _perp_basis(d)
        base_c = d * (inner * 0.9)
        apex = d * 1.0  # spike tip at the true radius
        ring = [base_c + (math.cos(a) * u + math.sin(a) * v) * base_r
                for a in (0.0, 2.094395, 4.18879)]  # 3-sided spike
        for i in range(3):
            mb.add_tri(ring[i], ring[(i + 1) % 3], apex, (0.0, 0.0, 0.0))
    m = mb.build()
    return m[:, 0:3].copy(), m[:, 3:6].copy()


_MINE_TYPES = frozenset({1, 21})  # toggle mine: untoggled (1) + toggled (21)

_SPHERE_POS, _SPHERE_NRM = _unit_sphere()
_BOX_POS, _BOX_NRM = _unit_box()
_MINE_POS, _MINE_NRM = _unit_mine()
_EMPTY = np.zeros((0, 9), dtype=np.float32)


def _instances(tpl_pos, tpl_nrm, centers, scales, colors) -> np.ndarray:
    """Instance a unit template: out[m] = center[m] + scale[m] * template.

    ``scales`` is per-axis (M, 3) so it covers uniform spheres and boxes. Fully
    vectorized; returns (M*V, 9).
    """
    if not centers:
        return _EMPTY
    C = np.asarray(centers, dtype=np.float32)          # (M, 3)
    S = np.asarray(scales, dtype=np.float32)           # (M, 3)
    COL = np.asarray(colors, dtype=np.float32)         # (M, 3)
    M, V = C.shape[0], tpl_pos.shape[0]
    pos = C[:, None, :] + S[:, None, :] * tpl_pos[None, :, :]     # (M, V, 3)
    nrm = np.broadcast_to(tpl_nrm[None, :, :], (M, V, 3))
    col = np.broadcast_to(COL[:, None, :], (M, V, 3))
    return np.concatenate([pos, nrm, col], axis=2).reshape(M * V, 9).astype(np.float32)


def _saturate(rgb, factor: float = 1.55):
    """Push a colour away from its own luma to make entities read as vibrant."""
    r, g, b = rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return tuple(min(max(luma + (c - luma) * factor, 0.0), 1.0) for c in rgb)


# Explicit bright colours for the key entities (flat-lit, so they read as pure
# hues rather than the muted source palette). Other types get a saturation lift.
_MINE_RED = (0.96, 0.13, 0.13)
_VIBRANT = {2: (1.00, 0.88, 0.08)}  # gold -> bright yellow


def _entity_color(entity_type: int):
    if entity_type in _MINE_TYPES:
        return _MINE_RED
    if entity_type in _VIBRANT:
        return _VIBRANT[entity_type]
    rgb = render_utils.ENTITYCOLORS_RGB.get(entity_type)
    if rgb is None or rgb == (0.0, 0.0, 0.0):
        return (0.6, 0.6, 0.6)
    return _saturate(rgb)


def build_dynamic_mesh(sim) -> np.ndarray:
    """Build the mesh for all active entities + the ninja for the current frame.

    Circular entities (those with a RADIUS: gold, mines, drones, launch pads,
    death balls, ...) render as spheres at their physical radius; square entities
    (SEMI_SIDE: thwumps, bounce blocks, one-way platforms) render as boxes.
    """
    zc = DEPTH / 2.0
    won = sim.ninja is not None and getattr(sim.ninja, "state", -1) == 8
    sph_c, sph_s, sph_col = [], [], []
    box_c, box_s, box_col = [], [], []
    mine_c, mine_s, mine_col = [], [], []

    for entity_list in sim.entity_dic.values():
        for e in entity_list:
            if not getattr(e, "active", False):
                continue
            if won and e.type == _EXIT_DOOR_TYPE:
                continue  # exit door disappears once the level is won
            xpos = getattr(e, "xpos", None)
            ypos = getattr(e, "ypos", None)
            if xpos is None or ypos is None:
                continue
            color = _entity_color(e.type)
            radius = getattr(e, "RADIUS", None)
            if radius:
                r = float(radius)
                if e.type in _MINE_TYPES:  # spiky mine template
                    mine_c.append((xpos, -ypos, zc)); mine_s.append((r, r, r))
                    mine_col.append(color)
                else:
                    sph_c.append((xpos, -ypos, zc)); sph_s.append((r, r, r))
                    sph_col.append(color)
            else:
                semi = getattr(e, "SEMI_SIDE", None)
                half = float(semi) if semi else 6.0
                box_c.append((xpos, -ypos, zc)); box_s.append((half, half, half))
                box_col.append(color)

    chunks = [
        _instances(_SPHERE_POS, _SPHERE_NRM, sph_c, sph_s, sph_col),
        _instances(_MINE_POS, _MINE_NRM, mine_c, mine_s, mine_col),
        _instances(_BOX_POS, _BOX_NRM, box_c, box_s, box_col),
    ]

    ninja = sim.ninja
    if ninja is not None:
        chunks.append(_ninja_mesh(ninja, NINJA_COLOR))

    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return _EMPTY
    return np.concatenate(chunks, axis=0)


def _lerp3(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u)


def _trail_color(s: float):
    """Speed 0..1 -> trail colour: cool blue (slow) through cyan to hot magenta."""
    s = max(0.0, min(1.0, s))
    if s < 0.5:
        return _lerp3((0.15, 0.35, 1.0), (0.20, 1.0, 0.95), s / 0.5)
    return _lerp3((0.20, 1.0, 0.95), (1.0, 0.35, 0.85), (s - 0.5) / 0.5)


class NinjaTrail:
    """A thin, fluttering speed-coloured bandana streaming from the ninja's head.

    Keeps a short history of head positions + speed; ``build`` emits a thin ribbon
    anchored at the ninja's top that flutters (an animated lateral sine wave, free
    at the tail, still at the head) and reads blue->magenta from tail to head as
    the ninja speeds up. It thins/shortens toward rest. Rendered translucent.
    """

    def __init__(self, maxlen: int = 60, width: float = 0.5,
                 hmax: float = MAX_SPEED, anchor_up: float | None = None,
                 samples: int = 28,
                 min_len: float = 18.0, max_len: float = 130.0,
                 flutter_amp: float = 9.0, wave_count: float = 2.3,
                 osc_gain: float = 0.95, descend: float | None = None):
        self._pts: deque = deque(maxlen=maxlen)
        self.width = width      # half-width in px (ribbon ~2*width tall)
        self.hmax = hmax        # ninja speed that saturates the colour/length
        self.anchor_up = NINJA_RADIUS * 0.8 if anchor_up is None else anchor_up
        # How far the tail sinks below the head anchor (down toward the ground).
        self.descend = (self.anchor_up + NINJA_RADIUS) if descend is None else descend
        self.samples = max(2, int(samples))   # >= 2, the ribbon needs a segment
        self.min_len = min_len
        self.max_len = max_len
        self.flutter_amp = flutter_amp
        self.wave_count = wave_count
        self.osc_gain = osc_gain            # phase advance per frame at top speed
        self._phase = 0.0
        self._dir = (-1.0, 0.0)             # trailing direction fallback (at rest)

    def reset(self) -> None:
        self._pts.clear()
        self._phase = 0.0

    def push(self, ninja) -> None:
        if ninja is None:
            return
        vx = float(getattr(ninja, "xspeed", 0.0) or 0.0)
        vy = float(getattr(ninja, "yspeed", 0.0) or 0.0)
        spd = math.hypot(vx, vy)
        s = min(spd / self.hmax, 1.0)
        # Anchor at the top of the ninja (world +y is up).
        self._pts.append((float(ninja.xpos), -float(ninja.ypos) + self.anchor_up, s))
        # Trailing (world) direction; kept for rest/extrapolation. World y = -sim y.
        if spd > 0.3:
            m = math.hypot(vx, -vy)
            self._dir = (-vx / m, vy / m)
        # Oscillation advances only with speed -> no flutter at rest, faster fast.
        self._phase += self.osc_gain * s

    @staticmethod
    def _point_at(poly, d, fallback_dir):
        """Point at arc-distance ``d`` from the head along ``poly`` (head..tail),
        extrapolating straight past the tail (fallback_dir if degenerate)."""
        if len(poly) == 1:
            return (poly[0][0] + fallback_dir[0] * d, poly[0][1] + fallback_dir[1] * d)
        acc = 0.0
        for i in range(len(poly) - 1):
            ax, ay = poly[i]
            bx, by = poly[i + 1]
            seg = math.hypot(bx - ax, by - ay)
            if seg < 1e-6:
                continue
            if acc + seg >= d:
                t = (d - acc) / seg
                return (ax + (bx - ax) * t, ay + (by - ay) * t)
            acc += seg
        dx, dy = fallback_dir
        for i in range(len(poly) - 1, 0, -1):
            ax, ay = poly[i - 1]
            bx, by = poly[i]
            seg = math.hypot(bx - ax, by - ay)
            if seg > 1e-6:
                dx, dy = (bx - ax) / seg, (by - ay) / seg
                break
        lx, ly = poly[-1]
        return (lx + dx * (d - acc), ly + dy * (d - acc))

    def build(self) -> np.ndarray:
        pts = list(self._pts)
        n = len(pts)
        if n < 1:
            return _EMPTY
        s_head = pts[-1][2]
        zc = DEPTH / 2.0 - 1.0
        poly = [(pts[n - 1 - i][0], pts[n - 1 - i][1]) for i in range(n)]  # head..tail

        # Bounded length: min at rest -> max at top speed.
        length = self.min_len + (self.max_len - self.min_len) * s_head
        amp = self.flutter_amp * s_head                 # no flutter at rest
        head_col = _trail_color(min(1.0, s_head))
        tail_col = _trail_color(max(0.0, s_head * 0.35))
        two_pi = 2.0 * math.pi

        K = self.samples
        left, right, cols = [], [], []
        for k in range(K):
            u = k / (K - 1)                             # 0 head .. 1 tail
            d = u * length
            cx, cy = self._point_at(poly, d, self._dir)
            nx, ny = self._point_at(poly, d + 2.0, self._dir)
            tx, ty = nx - cx, ny - cy
            tl = math.hypot(tx, ty) or 1.0
            px, py = -ty / tl, tx / tl                  # unit perpendicular
            wob = amp * (u ** 0.7) * math.sin(u * self.wave_count * two_pi - self._phase)
            cwx, cwy = cx + px * wob, cy + py * wob
            cwy -= self.descend * (u ** 1.3)             # tail sinks toward the ground
            hw = self.width * (0.55 + 0.45 * (1.0 - u))  # thin, ~uniform, slight taper
            left.append((cwx + px * hw, cwy + py * hw))
            right.append((cwx - px * hw, cwy - py * hw))
            cols.append(_lerp3(head_col, tail_col, u))
        verts = []
        for i in range(K - 1):
            (lx0, ly0), (rx0, ry0), c0 = left[i], right[i], cols[i]
            (lx1, ly1), (rx1, ry1), c1 = left[i + 1], right[i + 1], cols[i + 1]

            def V(px, py, c):
                return (px, py, zc, 0.0, 0.0, 1.0, c[0], c[1], c[2])
            verts += [V(lx0, ly0, c0), V(rx0, ry0, c0), V(rx1, ry1, c1),
                      V(lx0, ly0, c0), V(rx1, ry1, c1), V(lx1, ly1, c1)]
        return np.array(verts, dtype=np.float32)


# --- speed-driven ninja motion blur (echo smear) ---------------------------
# A handful of translated, alpha-faded copies of the ninja mesh trailing along
# its velocity vector. Cheap per-object motion blur: one small mesh copy per
# ghost (numpy translate) and one blended draw each, gated to only appear while
# the ninja is actually moving. At rest there are no ghosts (returns None).
#
# The echoes trail BEHIND the ninja: the nearest copy hugs the body (a small
# GHOST_CLEARANCE) and they fan out from there, so they read as a continuous
# directional smear off the back. Any inner overlap is hidden because the solid
# ninja is always drawn last, over them.
NINJA_GHOSTS = 6          # echo copies at top speed
GHOST_CLEARANCE = 3.0     # px the nearest echo sits behind the ninja centre
GHOST_MAX_OFFSET = 30.0   # px the smear fans out beyond the clearance at top speed
GHOST_MAX_ALPHA = 0.26    # opacity of the nearest echo at top speed


def build_ninja_ghosts(ninja, speed01: float, strength: float = 1.0,
                       n: int = NINJA_GHOSTS, color=NINJA_COLOR):
    """Trailing echo copies of the ninja mesh for speed-scaled motion blur.

    Returns ``(verts, verts_per_ghost, alphas)`` where ``verts`` packs the ghost
    copies farthest-first (so the nearer, more opaque echoes draw last), or
    ``None`` when the ninja is (near) stationary. ``alphas[i]`` is the blend
    weight for the ghost occupying rows ``[i*V : (i+1)*V]``.
    """
    if ninja is None or strength <= 0.0 or speed01 <= 0.02:
        return None
    vx = float(getattr(ninja, "xspeed", 0.0) or 0.0)
    vy = float(getattr(ninja, "yspeed", 0.0) or 0.0)
    spd = math.hypot(vx, vy)
    if spd < 1e-4:
        return None
    # Trail OPPOSITE the motion. World +y is up (= -sim y), so world velocity is
    # (vx, -vy); the smear direction is its negation, normalized.
    dirx, diry = -vx / spd, vy / spd
    base = _ninja_mesh(ninja, color)
    if base is None or len(base) == 0:
        return None
    V = base.shape[0]
    # Nearest echo clears the body; the rest fan out behind it, farther the faster.
    clearance = GHOST_CLEARANCE * strength
    span = GHOST_MAX_OFFSET * strength * speed01
    peak = GHOST_MAX_ALPHA * min(strength, 1.0) * speed01
    # One tiled copy of the base mesh, then a per-copy offset added in place:
    # avoids n separate array copies and concatenation.
    out = np.tile(base, (n, 1))
    alphas = []
    for k, i in enumerate(range(n, 0, -1)):       # farthest (faintest) first
        f = i / n                                 # 0..1 back along the smear
        dist = clearance + span * f
        rows = slice(k * V, (k + 1) * V)
        out[rows, 0] += dirx * dist
        out[rows, 1] += diry * dist
        alphas.append(peak * (1.0 - 0.8 * f))     # nearer echoes are more opaque
    return out, V, alphas


def _ninja_mesh(ninja, color) -> np.ndarray:
    """Ninja geometry: a bone skeleton if animation data is present, else a sphere.

    Bones are stored as normalized [x, y] offsets; the 2D renderer scales them by
    2*NINJA_RADIUS around the ninja position (see entity_renderer._draw_ninja).
    Limbs are sharp-edged rectangular prisms (no rounded joints); no contact
    shadow is drawn for the ninja.
    """
    zc = DEPTH / 2.0
    bones = getattr(ninja, "bones", None)
    if bones:
        mb = MeshBuilder()
        s = 2.0 * NINJA_RADIUS
        cx, cy = ninja.xpos, ninja.ypos
        pts = [(cx + bx * s, -(cy + by * s), zc) for (bx, by) in bones]
        for i, j in render_utils.LIMBS:
            if i < len(pts) and j < len(pts):
                mb.add_segment(pts[i], pts[j], 0.5, color)
        return mb.build()
    return _instances(_SPHERE_POS, _SPHERE_NRM,
                      [(ninja.xpos, -ninja.ypos, zc)],
                      [(NINJA_RADIUS, NINJA_RADIUS, NINJA_RADIUS)], [color])
