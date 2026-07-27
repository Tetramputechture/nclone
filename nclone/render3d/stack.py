"""Stacked multi-level mode: up to three levels along the depth (z) axis.

The live level sits at the play plane (z ~ 0). Its neighbours in a folder of
maps are rendered as faint, depth-offset panes:

    next level  -> pushed INTO the screen (-z), a little brighter (anticipation)
    prev level  -> pulled toward the camera (+z), fainter (already cleared)

When the ninja wins, the camera dollies forward through the stack while a bright
"warp" streak carries the agent from the exit into the next slab; on arrival the
window advances (next -> live, live -> prev, a fresh next is loaded) and the
camera resets. Because the geometry is laid out symmetrically around the live
level, that reset is seamless. The folder cycles forever (wrapping), so a folder
of N maps streams past three-at-a-time.

Only the live level owns a simulation; the neighbours are static snapshots, so
the cost over single-level mode is two extra (cheap, blended) mesh draws.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import numpy as np

from .geometry import TILE, DEPTH, MeshBuilder, build_level_mesh
from .scene import build_dynamic_mesh, NINJA_RADIUS

# Depth centre of the play slab (entities live here in single-level mode too).
_ZC = DEPTH / 2.0

# Bright cyan for the teleport warp streak.
_WARP_COLOR = (0.35, 0.95, 1.0)


# --------------------------------------------------------------------------- #
# map discovery
# --------------------------------------------------------------------------- #

_SKIP_EXT = (".py", ".png", ".jpg", ".md", ".txt", ".replay", ".attract",
             ".json", ".pkl", ".npz")


def discover_maps(path: str) -> list[str]:
    """Sorted list of map files under ``path`` (a folder, or a single file)."""
    if os.path.isfile(path):
        return [path]
    hits = []
    for p in sorted(glob.glob(os.path.join(path, "*"))):
        if os.path.isdir(p) or p.lower().endswith(_SKIP_EXT):
            continue
        hits.append(p)
    if not hits:
        raise SystemExit(f"--stack: no map files found in {path!r}")
    return hits


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #

def _load_sim(map_path: str, anim: bool = False):
    from ..nplay_headless import NPlayHeadless
    nh = NPlayHeadless(enable_rendering=False, enable_animation=anim)
    nh.load_map(map_path)
    return nh


def _try_load_sim(map_path: str, anim: bool = False):
    """Load a level, returning None (with a warning) on failure, so the cycle
    tolerate a stray unreadable/invalid map in the folder instead of crashing."""
    try:
        return _load_sim(map_path, anim)
    except Exception as exc:                      # noqa: BLE001 - report and skip
        print(f"[stack] skipping {os.path.basename(map_path)}: {exc}")
        return None


def _offset_z(mesh: np.ndarray | None, dz: float) -> np.ndarray | None:
    """Return a copy of an Nx9 mesh translated by ``dz`` along z."""
    if mesh is None or len(mesh) == 0:
        return mesh
    out = mesh.copy()
    out[:, 2] += dz
    return out


def _recolor(mesh: np.ndarray | None, desat: float, tint, tint_amt: float):
    """Desaturate a neighbour slab toward its luma, then tint it, so the two
    inactive levels read as background depth rather than competing with the live
    one. Operates on the colour channels (cols 6:9); geometry is untouched."""
    if mesh is None or len(mesh) == 0 or (desat <= 0.0 and tint_amt <= 0.0):
        return mesh
    out = mesh.copy()
    col = out[:, 6:9]
    luma = (col * np.array([0.299, 0.587, 0.114], "f4")).sum(1, keepdims=True)
    col = col + (luma - col) * desat
    col = col + (np.asarray(tint, "f4")[None, :] - col) * tint_amt
    out[:, 6:9] = np.clip(col, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# director
# --------------------------------------------------------------------------- #

class StackDirector:
    """Owns the live sim + the two neighbour snapshots and the teleport state
    machine. The driver loop polls :meth:`update` each frame, then reads the live
    sim, teleport progress, and camera dolly to render."""

    PLAY = "play"
    TELEPORT = "teleport"

    def __init__(self, r, maps, args, source_factory):
        self.r = r
        self.maps = maps
        self.n = len(maps)
        self.args = args
        self._source_factory = source_factory
        self.spacing = float(args.stack_spacing)
        self.tp_len = max(1, int(args.stack_teleport))
        # Scatter blocks triple up across the three slabs into visual noise, so
        # they are opt-in (--stack-scatter) rather than on by the single-level default.
        self.scatter = (not args.no_scatter) and bool(getattr(args, "stack_scatter", False))
        self.guardrails = not args.no_guardrails

        self.idx = 0
        self.state = self.PLAY
        self.tp = 0                       # teleport frame counter
        self.active = None                # live NPlayHeadless
        self.source = None
        self.active_static = None         # live level tile mesh (z=0)
        self.static_version = 0           # bumped whenever a new live level loads
        self._prev_slab = None            # dict(static, dynamic) at +spacing
        self._next_slab = None            # dict(static, dynamic) at -spacing
        self._pending_prev = None         # won snapshot -> becomes prev on commit

        self._start_active()
        self._build_next()
        # A prev pane only renders with three or more maps in the folder
        # (_register drops it otherwise), so skip loading that level entirely.
        if self.n > 2:
            self._build_prev(self._start_snapshot((self.idx - 1) % self.n))
        self._register()

    # -- slab construction -------------------------------------------------- #

    def _snapshot(self, sim, dz: float) -> dict:
        static = _offset_z(build_level_mesh(sim.tile_dic, self.scatter,
                                            self.guardrails), dz)
        dynamic = _offset_z(build_dynamic_mesh(sim), dz)
        return {"static": static, "dynamic": dynamic}

    def _tinted(self, slab: dict, next_side: bool) -> dict:
        """Apply the neighbour desaturation/tint for rendering."""
        # next = cool blue (upcoming), prev = warm grey (behind you).
        tint = (0.45, 0.62, 0.95) if next_side else (0.72, 0.68, 0.62)
        return {
            "static": _recolor(slab["static"], self.args.stack_desat, tint, 0.22),
            "dynamic": _recolor(slab["dynamic"], self.args.stack_desat * 0.4,
                                tint, 0.10),
        }

    def _start_snapshot(self, idx: int) -> dict | None:
        """Fresh start-state snapshot of ``maps[idx]`` (for a not-yet-played
        neighbour), at z=0 (the caller shifts it onto its pane)."""
        nh = _try_load_sim(self.maps[idx])
        return self._snapshot(nh.sim, 0.0) if nh is not None else None

    def _build_next(self) -> None:
        default_spawn = (44 * TILE / 2.0, -(25 * TILE / 2.0))
        nh = _try_load_sim(self.maps[(self.idx + 1) % self.n])
        if nh is None:
            self._next_slab = None
            self._next_spawn = default_spawn
            self._next_stub = None
            return
        # The pane itself only renders with two or more maps, but the spawn and
        # facing are still needed to aim the warp streak and the arrival camera.
        self._next_slab = self._snapshot(nh.sim, -self.spacing) if self.n > 1 else None
        nj = nh.sim.ninja
        # World-space spawn of the upcoming level, so the warp streak connects to
        # it and the camera pans/orients to frame it (seamless hand-off in x/y).
        if nj is not None:
            self._next_spawn = (float(nj.xpos), -float(nj.ypos))
            # A stand-in ninja at the next spawn (sim coords + facing, at rest) so
            # the first-person camera can compute the correct ride-along pose for
            # the upcoming level's start node, arriving oriented behind it.
            self._next_stub = SimpleNamespace(
                xpos=float(nj.xpos), ypos=float(nj.ypos),
                facing=float(getattr(nj, "facing", 1) or 1),
                xspeed=0.0, yspeed=0.0)
        else:
            self._next_spawn = default_spawn
            self._next_stub = None

    def _build_prev(self, snap_at0: dict | None) -> None:
        """``snap_at0`` is a z=0 snapshot; shift it out to the prev pane (+z)."""
        if snap_at0 is None:
            self._prev_slab = None
            return
        self._prev_slab = {
            "static": _offset_z(snap_at0["static"], self.spacing),
            "dynamic": _offset_z(snap_at0["dynamic"], self.spacing),
        }

    def _register(self) -> None:
        slabs = []
        if self.n > 1 and self._next_slab is not None:
            t = self._tinted(self._next_slab, next_side=True)
            slabs.append({"static": t["static"], "dynamic": t["dynamic"],
                          "alpha": self.args.stack_next_alpha, "near": False})
        if self.n > 2 and self._prev_slab is not None:
            t = self._tinted(self._prev_slab, next_side=False)
            slabs.append({"static": t["static"], "dynamic": t["dynamic"],
                          "alpha": self.args.stack_prev_alpha, "near": True})
        self.r.set_ghost_slabs(slabs)

    # -- live level lifecycle ---------------------------------------------- #

    def _start_active(self) -> None:
        """Load the live level at ``self.idx``, skipping forward over any maps
        that fail to load (so one bad file in the folder can't stall the cycle)."""
        for _ in range(self.n):
            nh = _try_load_sim(self.maps[self.idx], anim=self.args.anim)
            if nh is not None:
                self.active = nh
                self.source = self._source_factory(self.maps[self.idx])
                self.active_static = build_level_mesh(
                    nh.sim.tile_dic, self.scatter, self.guardrails)
                self.static_version += 1
                return
            self.idx = (self.idx + 1) % self.n
        raise SystemExit("[stack] no loadable maps found")

    def retry(self) -> None:
        """Ninja died (or the player asked): reset the live level and its input
        source, keeping the same map."""
        self.active.reset()
        if self.source is not None:
            self.source.reset()

    def begin_teleport(self) -> None:
        """Start the warp to the next level (on a win, or on a manual skip)."""
        # Capture the won state (ninja at exit, door gone) to become the prev
        # pane; only worth building when a prev pane will actually render.
        self._pending_prev = (self._snapshot(self.active.sim, 0.0)
                              if self.n > 2 else None)
        self.state = self.TELEPORT
        self.tp = 0

    def _commit(self) -> None:
        """Advance the window: next -> live, live(won) -> prev, load a new next."""
        self.idx = (self.idx + 1) % self.n
        self._start_active()
        self._build_prev(self._pending_prev)
        self._pending_prev = None
        self._build_next()
        self._register()
        self.state = self.PLAY
        self.tp = 0

    # -- per-frame update --------------------------------------------------- #

    def update(self) -> None:
        if self.state == self.PLAY:
            hor, jump = self.source.poll() if self.source is not None else (0, 0)
            self.active.tick(hor, jump)
            if self.active.ninja_has_won():
                self.begin_teleport()
            elif self.active.ninja_has_died():
                self.retry()
        elif self.state == self.TELEPORT:
            self.tp += 1
            if self.tp >= self.tp_len:
                self._commit()

    # -- outputs for the driver -------------------------------------------- #

    def teleport_ease(self) -> float:
        """Smoothstep 0..1 across the teleport (0 while playing)."""
        if self.state != self.TELEPORT:
            return 0.0
        p = min(self.tp / float(self.tp_len), 1.0)
        return p * p * (3.0 - 2.0 * p)

    def dolly_z(self) -> float:
        """Camera z-offset: eases from 0 (viewing live) to -spacing (viewing next)."""
        return -self.spacing * self.teleport_ease()

    def teleport_target_xy(self):
        """World x/y the warp camera pans to: the upcoming level's spawn."""
        return getattr(self, "_next_spawn", (44 * TILE / 2.0, -(25 * TILE / 2.0)))

    def next_spawn_stub(self):
        """Stand-in ninja at the upcoming level's start node (or None), for the
        first-person camera to orient its arrival pose. Its geometry lives one
        slab deep (z = _ZC - spacing)."""
        return getattr(self, "_next_stub", None)

    def main_alpha(self) -> float:
        """Opacity of the live level: dissolves fully as the camera dives through
        it, so that by the moment of commit only the (now opaque) next slab shows
        making the hand-off to the new live level seamless."""
        return max(0.0, 1.0 - 1.25 * self.teleport_ease())

    def build_active_dynamic(self) -> np.ndarray:
        """Live entity/ninja mesh for this frame."""
        return build_dynamic_mesh(self.active.sim)

    def overlay_mesh(self):
        """Always-on-top warp streak during a teleport (None otherwise)."""
        if self.state != self.TELEPORT:
            return None
        return self._warp_streak(self.teleport_ease())

    def _warp_streak(self, ease: float) -> np.ndarray | None:
        """A comet of bright spheres connecting the exit of the live level to the
        start node of the next one, across the depth gap.

        The path runs from the current exit ``(ex, ey, _ZC)`` to the next level's
        spawn one slab deep ``(nx, ny, _ZC - spacing)``, so the streak literally
        threads the agent from where it finished to where it will begin. The lead
        reaches farther along that path as the teleport progresses. Drawn on top
        (see Renderer.set_overlay_mesh) so it stays visible as it crosses behind
        the dissolving live level."""
        ninja = self.active.sim.ninja
        if ninja is None:
            return None
        ex, ey = float(ninja.xpos), -float(ninja.ypos)      # current exit (world)
        nx, ny = self._next_spawn                           # next start (world)
        sx, sy, sz = ex, ey, _ZC
        tx, ty, tz = nx, ny, _ZC - self.spacing
        K = 18
        lead = int(round(ease * K)) + 1
        mb = MeshBuilder()
        for i in range(min(lead, K)):
            f = i / float(K - 1)                  # 0 at exit .. 1 at next start
            p = (sx + (tx - sx) * f, sy + (ty - sy) * f, sz + (tz - sz) * f)
            r = NINJA_RADIUS * (1.15 - 0.7 * f)   # fat head tapering to the tip
            # brighten toward white at the tip for a hot core
            c = tuple(min(1.0, ch + 0.55 * f) for ch in _WARP_COLOR)
            mb.add_sphere(p, max(r, 1.5), c, lat=6, lon=8)
        return mb.build()

    @property
    def label(self) -> str:
        return os.path.basename(self.maps[self.idx])


# --------------------------------------------------------------------------- #
# camera + shared setup
# --------------------------------------------------------------------------- #

def _stack_camera(args):
    """Orbit camera framed to show all three depth-stacked slabs at an angle."""
    from .camera import OrbitCamera
    cx = 44 * TILE / 2.0
    cy = -(25 * TILE / 2.0)
    target = (cx, cy, _ZC)
    radius = 44 * TILE * 1.05 / max(args.zoom, 0.01)
    cam = OrbitCamera(target, radius=radius, elevation_deg=args.stack_elevation)
    cam.azimuth = np.radians(args.stack_azimuth)
    return cam


def _make_source_factory(args, keyboard: bool):
    """Build the per-level input-source factory for the live level."""
    from .input_source import KeyboardSource, HoldSource, PatternSource
    from . import app as _app

    def factory(map_path: str):
        if keyboard and not args.stack_hold:
            return KeyboardSource()
        hor, jump = _app.HOLD_INPUTS.get(args.stack_hold or "right", (0, 0))
        if args.jump_period:
            return PatternSource(hor, args.jump_period, args.jump_len)
        return HoldSource(hor, jump)

    return factory


def _setup_renderer(ctx, args):
    from .renderer import Renderer
    from .geometry import build_backplane_mesh
    r = Renderer(ctx)
    # The animated cube field is beautiful but visually swamps the faint stacked
    # panes, so stacked mode keeps a clean backdrop unless --stack-bg-anim is set.
    r.bg_anim = bool(getattr(args, "stack_bg_anim", False)) and not args.plain_bg
    r.set_backplane(build_backplane_mesh())
    return r


class _Travel:
    """Per-level travel tracker feeding the animated background uniforms."""

    def __init__(self):
        self.reset(None)

    def reset(self, ninja):
        self.x0 = float(ninja.xpos) if ninja else 0.0
        self.y0 = float(ninja.ypos) if ninja else 0.0
        self.hw = 0.0

    def sample(self, ninja):
        if ninja is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        from .app import _vel_frac
        h, v = _vel_frac(ninja)
        ntx = float(ninja.xpos) - self.x0
        nty = float(ninja.ypos) - self.y0
        self.hw = max(self.hw, abs(ntx) + abs(nty))
        return h, v, ntx, nty, self.hw


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #

def _advance_camera(cam, director):
    """Update the orbit camera: gentle follow in-plane while playing; during a
    teleport, pan x/y toward the upcoming spawn while dollying z through the stack
    so the arrival lands already framing the new level (seamless hand-off)."""
    from pyrr import Vector3
    from .app import _follow_step
    ninja = director.active.sim.ninja
    if director.state == director.TELEPORT:
        if getattr(cam, "_tp_start", None) is None:
            cam._tp_start = (float(cam.target.x), float(cam.target.y))
        ease = director.teleport_ease()
        sx, sy = cam._tp_start
        tx, ty = director.teleport_target_xy()
        fx, fy = sx + (tx - sx) * ease, sy + (ty - sy) * ease
    else:
        cam._tp_start = None
        if director.state == director.PLAY and ninja is not None:
            t = _follow_step(cam, ninja)         # eases target xy, z -> _ZC
            fx, fy = float(t.x), float(t.y)
        else:
            fx, fy = float(cam.target.x), float(cam.target.y)
    z = _ZC + director.dolly_z()
    cam.target = Vector3([fx, fy, z], dtype="f4")
    if director.args.spin:
        cam.azimuth += director.args.spin_rate


class _StackCam:
    """Camera for stacked mode: orbit (default) or agent first-person (--fpv).

    Both produce ``(mvp, cam_pos)`` from :meth:`update`. During a teleport the
    orbit path dollies its target through the stack (see ``_advance_camera``);
    the first-person path dives the eye forward along -z toward the upcoming
    spawn, so the agent literally flies through into the next level.
    """

    def __init__(self, args):
        self.args = args
        self.fpv = bool(getattr(args, "fpv", False))
        self.orbit = None if self.fpv else _stack_camera(args)
        self._eye = None            # first-person eye/target (smoothed)
        self._tgt = None
        self._fov = args.fov
        self._tp = None             # (start_eye, start_tgt) captured at warp start

    def toggle(self) -> None:
        """Swap orbit <-> first-person live (window 'C' key)."""
        self.fpv = not self.fpv
        if self.fpv:
            self._eye = self._tgt = self._tp = None
            self._fov = self.args.fov
        else:
            self.orbit = _stack_camera(self.args)

    def update(self, director, aspect):
        if not self.fpv:
            _advance_camera(self.orbit, director)
            return self.orbit.mvp(aspect), self.orbit.eye()

        from .app import (_agent_fpv_pose, _speed_fov, _dolly_scale, _lookat_mvp)
        args = self.args
        ninja = director.active.sim.ninja
        if self._eye is None:
            self._eye, self._tgt = _agent_fpv_pose(ninja, args)
        if director.state == director.TELEPORT:
            # Freeze the ride-along pose at warp start, then ease toward the proper
            # ride-along pose for the NEXT level's start node, computed from a
            # stand-in ninja at its spawn (correct facing), shifted one slab deep.
            # So the camera flies through and arrives oriented behind the agent.
            if self._tp is None:
                self._tp = (self._eye.copy(), self._tgt.copy())
            ease = director.teleport_ease()
            s_eye, s_tgt = self._tp
            stub = director.next_spawn_stub()
            if stub is not None:
                end_eye, end_tgt = _agent_fpv_pose(stub, args)
                deep = np.array([0.0, 0.0, -director.spacing], "f4")
                end_eye, end_tgt = end_eye + deep, end_tgt + deep
            else:                                  # no next spawn: dive straight in
                tx, ty = director.teleport_target_xy()
                end_tgt = np.array([tx, ty, _ZC - director.spacing], "f4")
                end_eye = end_tgt + (s_eye - s_tgt)
            self._eye = s_eye + (end_eye - s_eye) * ease
            self._tgt = s_tgt + (end_tgt - s_tgt) * ease
        else:
            if self._tp is not None:
                # Just committed: the world swapped the next level to z=0, so undo
                # the one-slab dive in one step: the view is unchanged, keeping
                # the hand-off seamless, and normal ride-along easing resumes.
                deep = np.array([0.0, 0.0, director.spacing], "f4")
                self._eye = self._eye + deep
                self._tgt = self._tgt + deep
                self._tp = None
            self._fov = (self._fov * 0.85
                         + _speed_fov(ninja, args.fov, args.fov_speed) * 0.15)
            des_eye, des_tgt = _agent_fpv_pose(
                ninja, args, dist_scale=_dolly_scale(args.fov, self._fov))
            self._eye = self._eye * 0.82 + des_eye * 0.18
            self._tgt = self._tgt * 0.82 + des_tgt * 0.18
        return _lookat_mvp(self._eye, self._tgt, aspect, self._fov), self._eye


def run_stack_record(args, maps) -> None:
    import moderngl
    import imageio

    ctx = moderngl.create_standalone_context(backend="egl")
    r = _setup_renderer(ctx, args)
    from .app import _parse_bg, _dash_footfall
    from .postfx import SceneTarget
    from .renderer import TRAIL_MAX
    from collections import deque

    W, H = args.width, args.height
    bg = _parse_bg(args.bg)
    director = StackDirector(r, maps, args, _make_source_factory(args, keyboard=False))
    cam = _StackCam(args)
    target = SceneTarget(ctx, W, H, samples=args.msaa)
    aspect = W / H

    writer = imageio.get_writer(args.record, fps=args.fps, macro_block_size=None,
                                quality=8, codec="libx264")
    travel = _Travel()
    travel.reset(director.active.sim.ninja)
    footfalls = deque(maxlen=TRAIL_MAX)
    sp_h = 0.0
    uploaded_static = -1
    hard_cap = args.max_frames or 3600

    for frame in range(hard_cap):
        director.update()
        if director.static_version != uploaded_static:   # new live level loaded
            uploaded_static = director.static_version
            r.set_static_mesh(director.active_static)
            travel.reset(director.active.sim.ninja)
            footfalls.clear()

        ninja = director.active.sim.ninja
        r.set_dynamic_mesh(director.build_active_dynamic())
        if ninja is not None and director.state == director.PLAY:
            footfalls.append(_dash_footfall(ninja))
        r.set_dash_trail(footfalls)
        r.teleport_k = director.teleport_ease()
        r.set_overlay_mesh(director.overlay_mesh())

        h, v, ntx, nty, hw = travel.sample(ninja)
        sp_h = sp_h * 0.88 + h * 0.12
        mvp, cam_pos = cam.update(director, aspect)

        target.use()
        r.render(mvp, clear=bg, cam_pos=cam_pos, time=frame / float(args.fps),
                 hspeed=sp_h, vspeed=v, netx=ntx, nety=nty, travelmax=hw,
                 main_alpha=director.main_alpha())
        target.resolve()
        buf = np.frombuffer(target.read_rgb(), np.uint8)
        writer.append_data(buf.reshape(H, W, 3)[::-1])

    writer.close()
    print(f"wrote {args.record}  frames={hard_cap}  maps={director.n}  "
          f"spacing={director.spacing}")


def run_stack_window(args, maps) -> None:
    from . import app as _app
    _app._ensure_window_gl_env()  # may re-exec
    import pygame
    import moderngl
    from collections import deque
    from .app import _parse_bg, _dash_footfall
    from .renderer import TRAIL_MAX

    pygame.init()
    pygame.display.set_caption("nclone 3D - stacked")
    if args.msaa and args.msaa > 1:
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, int(args.msaa))
    try:
        pygame.display.set_mode((args.width, args.height),
                                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    except pygame.error:
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        pygame.display.set_mode((args.width, args.height),
                                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    ctx = moderngl.create_context()

    r = _setup_renderer(ctx, args)
    bg = _parse_bg(args.bg)
    director = StackDirector(r, maps, args,
                             _make_source_factory(args, keyboard=True))
    cam = _StackCam(args)
    w, h = args.width, args.height

    travel = _Travel()
    travel.reset(director.active.sim.ninja)
    footfalls = deque(maxlen=TRAIL_MAX)
    sp_h = 0.0
    uploaded_static = -1
    clock = pygame.time.Clock()

    print(f"stacked window: {w}x{h}  GL={ctx.info['GL_RENDERER']}  "
          f"maps={director.n}  spacing={director.spacing}  "
          f"cam={'first-person' if cam.fpv else 'orbit'}")
    print("move: <-/-> or A/D   jump: Z/Space/Up/W")
    print("beat a level to warp to the next; C toggles first-person, "
          "N skips, R retries, ESC quits")

    dragging = False
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                w, h = e.w, e.h
                ctx.viewport = (0, 0, w, h)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging and not cam.fpv:
                cam.orbit.azimuth += e.rel[0] * 0.01
                cam.orbit.elevation = float(np.clip(
                    cam.orbit.elevation - e.rel[1] * 0.01, -1.4, 1.4))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r and director.state == director.PLAY:
                    director.retry()
                elif e.key == pygame.K_n and director.state == director.PLAY:
                    director.begin_teleport()   # skip: warp without winning
                elif e.key == pygame.K_c:
                    cam.toggle()                 # orbit <-> first-person
                elif e.key == pygame.K_o:
                    args.spin = not args.spin

        keys = pygame.key.get_pressed()
        if not cam.fpv:                          # orbit-only rotate/zoom
            if keys[pygame.K_q]:
                cam.orbit.azimuth -= 0.03
            if keys[pygame.K_e]:
                cam.orbit.azimuth += 0.03
            if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                cam.orbit.radius *= 0.98
            if keys[pygame.K_MINUS]:
                cam.orbit.radius *= 1.02

        director.update()
        if director.static_version != uploaded_static:
            uploaded_static = director.static_version
            r.set_static_mesh(director.active_static)
            travel.reset(director.active.sim.ninja)
            footfalls.clear()

        ninja = director.active.sim.ninja
        r.set_dynamic_mesh(director.build_active_dynamic())
        if ninja is not None and director.state == director.PLAY:
            footfalls.append(_dash_footfall(ninja))
        r.set_dash_trail(footfalls)
        r.teleport_k = director.teleport_ease()
        r.set_overlay_mesh(director.overlay_mesh())

        hval, vval, ntx, nty, hw = travel.sample(ninja)
        sp_h = sp_h * 0.88 + hval * 0.12
        mvp, cam_pos = cam.update(director, w / max(h, 1))

        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        r.render(mvp, clear=bg, cam_pos=cam_pos,
                 time=pygame.time.get_ticks() * 0.001, hspeed=sp_h, vspeed=vval,
                 netx=ntx, nety=nty, travelmax=hw, main_alpha=director.main_alpha())
        pygame.display.flip()

        tag = "WARP" if director.state == director.TELEPORT else "PLAY"
        pygame.display.set_caption(
            f"nclone 3D - stacked  [{director.idx + 1}/{director.n}] "
            f"{director.label}  {'first-person' if cam.fpv else 'orbit'}  {tag}")
        clock.tick(60)

    pygame.quit()


def run_stack(args) -> None:
    maps = discover_maps(args.stack)
    if args.record:
        run_stack_record(args, maps)
    else:
        run_stack_window(args, maps)
