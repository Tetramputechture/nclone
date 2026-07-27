"""Entry point for the 3D frontend: CLI, camera wiring, and the three run loops.

Three rendering paths, all stepping the same simulation and reading its state
each frame:
  * window  : pygame (SDL) + moderngl, live and interactive
  * record  : moderngl standalone EGL -> offscreen FBO -> MP4
  * headless: the same offscreen path, one frame -> PNG

Usage:
    python -m nclone.render3d.app --map nclone/test_maps/basics
    python -m nclone.render3d.app --map nclone/test_maps/basics --headless --out /tmp/level.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from .scene import MAX_SPEED, MAX_VSPEED

# ---------------------------------------------------------------------------
# GL environment bootstrap.
#
# Under WSLg, on-screen GLX context creation fails, so a visible window needs
# the SDL Wayland backend plus an unversioned libGL.so on the loader path.
# Neither can be changed after the process starts, so we build a symlink shim
# in the user cache and re-exec once with the right environment.
#
# This only applies to Linux sessions running a Wayland compositor, and only to
# the window path. Headless (EGL) rendering needs none of it. Set
# NCLONE_RENDER3D_NO_BOOTSTRAP=1 to opt out entirely.
# ---------------------------------------------------------------------------
_BOOTSTRAP_SENTINEL = "NCLONE_RENDER3D_BOOTSTRAPPED"

_SHIM_SONAMES = (("libGL.so", "libGL.so.1"),
                 ("libEGL.so", "libEGL.so.1"),
                 ("libgbm.so", "libgbm.so.1"))


def _package_root() -> str:
    """Directory containing the ``nclone`` package."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _shim_dir() -> str:
    """User-cache directory for the unversioned GL symlinks.

    Deliberately not inside the package: an installed package tree is often
    read-only and is the wrong place for machine-local state.
    """
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "nclone", "gl_shims")


def _bootstrap_applies() -> bool:
    """True when the WSLg/Wayland window workaround is relevant to this host."""
    if os.environ.get("NCLONE_RENDER3D_NO_BOOTSTRAP"):
        return False
    if os.environ.get(_BOOTSTRAP_SENTINEL):
        return False
    return sys.platform.startswith("linux") and bool(
        os.environ.get("WAYLAND_DISPLAY"))


def _ensure_window_gl_env() -> None:
    """Re-exec once with the SDL/loader environment a Wayland window needs.

    Returns without doing anything when the workaround does not apply, or when
    the current environment already provides it.
    """
    if not _bootstrap_applies():
        return
    shim = _shim_dir()
    try:
        os.makedirs(shim, exist_ok=True)
    except OSError as exc:
        print(f"render3d: cannot create GL shim dir {shim}: {exc}")
        return
    # Symlink unversioned names to the versioned SONAMEs the loader can find.
    for name, soname in _SHIM_SONAMES:
        link = os.path.join(shim, name)
        if os.path.islink(link):
            continue
        target = _find_soname(soname)
        if not target:
            continue
        try:
            os.symlink(target, link)
        except OSError:
            pass
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("SDL_VIDEODRIVER") and shim in ld.split(os.pathsep):
        return  # already set up by the caller; no re-exec needed
    env = dict(os.environ)
    env[_BOOTSTRAP_SENTINEL] = "1"
    env.setdefault("SDL_VIDEODRIVER", "wayland")
    env["LD_LIBRARY_PATH"] = shim + (os.pathsep + ld if ld else "")
    os.execve(sys.executable,
              [sys.executable, "-m", "nclone.render3d.app", *sys.argv[1:]], env)


def _find_soname(soname: str) -> str | None:
    """Resolve a versioned SONAME to a path via ``ldconfig -p`` (glibc only)."""
    import subprocess
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if soname in line and "=>" in line:
            return line.split("=>")[-1].strip()
    return None


# ---------------------------------------------------------------------------

def _default_map() -> str:
    root = _package_root()
    for cand in ("nclone/test_maps/basics", "nclone/test_maps/flat-run-open"):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(root, "nclone/test_maps/*"))
    hits = [h for h in hits if not h.endswith((".py", ".png", ".md", ".attract"))]
    if not hits:
        raise SystemExit("no map found; pass --map")
    return hits[0]


def _load_sim(map_path: str, anim: bool = False):
    from nclone.nplay_headless import NPlayHeadless
    nh = NPlayHeadless(enable_rendering=False, enable_animation=anim)
    nh.load_map(map_path)
    return nh


def _setup(args):
    """Build (nh, input_source, label) from either --replay or --map."""
    from nclone.nplay_headless import NPlayHeadless

    if getattr(args, "replay", None):
        from .replay_loader import load_attract_replay
        from .input_source import ReplaySource
        map_data, inputs, name = load_attract_replay(args.replay)
        nh = NPlayHeadless(enable_rendering=False, enable_animation=args.anim)
        nh.load_map_from_map_data(map_data)
        return nh, ReplaySource(inputs), name or "replay"

    nh = _load_sim(args.map, args.anim)
    label = os.path.basename(args.map)
    # Offscreen recording has no keyboard: drive with a scripted input. A
    # --jump-period pulses jump on top of the --hold direction (a bhop pattern
    # that solves geometry-rich levels a constant hold can't).
    if getattr(args, "record", None):
        hor, jump = HOLD_INPUTS.get(args.hold, (0, 0))
        if getattr(args, "jump_period", 0):
            from .input_source import PatternSource
            return nh, PatternSource(hor, args.jump_period, args.jump_len), label
        from .input_source import HoldSource
        return nh, HoldSource(hor, jump), label

    from .input_source import KeyboardSource
    return nh, KeyboardSource(), label


def _make_camera(width, height, elevation_deg=28.0):
    from .camera import OrbitCamera
    from .geometry import TILE, DEPTH
    # level extent in sim pixels: 44 x 25 tiles
    cx = 44 * TILE / 2.0
    cy_world = -(25 * TILE / 2.0)
    target = (cx, cy_world, DEPTH / 2.0)
    radius = 44 * TILE * 0.85
    return OrbitCamera(target, radius=radius, elevation_deg=elevation_deg)


# Named scripted drives for --hold / --stack-hold: (horizontal, jump) held every
# frame. Also the choice list for those CLI flags.
HOLD_INPUTS = {
    "none": (0, 0), "left": (-1, 0), "right": (1, 0), "jump": (0, 1),
    "right-jump": (1, 1), "left-jump": (-1, 1),
}

# Named background presets; --bg also accepts "r,g,b" floats in [0,1].
_BG_PRESETS = {
    "white": (1.0, 1.0, 1.0),
    "light": (0.93, 0.93, 0.94),
    "slate": (0.09, 0.10, 0.13),
    "dark": (0.05, 0.05, 0.06),
}


def _parse_bg(s: str):
    """Turn a --bg string (preset name or 'r,g,b') into an RGB float tuple."""
    if s in _BG_PRESETS:
        return _BG_PRESETS[s]
    try:
        parts = tuple(float(x) for x in s.replace(" ", "").split(","))
    except ValueError:
        parts = ()
    if len(parts) == 3:
        return tuple(min(max(c, 0.0), 1.0) for c in parts)
    raise SystemExit(f"--bg: unknown preset '{s}' (try white/light/slate/dark or 'r,g,b')")


def _blur_config(args):
    """Edge motion blur settings, as ``(strength, first_person_only)``.

    ``--motion-blur`` < 0 (the default) means auto: blur only while the
    first-person camera is active. 0 disables it. A positive value scales the
    strength and enables blur in every camera mode.
    """
    mb = float(args.motion_blur)
    if mb == 0.0:
        return 0.0, False
    if mb < 0.0:
        return float(args.motion_blur_strength), True
    return float(args.motion_blur_strength) * mb, False


def _blur_for_mode(strength: float, fp_only: bool, first_person: bool) -> bool:
    """Whether the blur composite should run for the current camera mode."""
    return strength > 0.0 and (first_person or not fp_only)


# Eye movement (world px/frame) that saturates the edge motion blur.
_BLUR_FULL_SPEED = 8.0


def _blur_step(cam_pos, prev_cam_pos, strength: float):
    """Per-frame blur strength from camera movement, plus the new previous pose.

    Returns ``(strength_this_frame, cam_pos_for_next_frame)``. The first frame
    has no previous pose and so renders sharp.
    """
    cur = np.asarray(cam_pos, dtype="f4").copy()
    if strength <= 0.0 or prev_cam_pos is None:
        return 0.0, cur
    speed = float(np.linalg.norm(cur - prev_cam_pos))
    return strength * min(speed / _BLUR_FULL_SPEED, 1.0), cur


def _agent_fpv_pose(ninja, args, dist_scale: float = 1.0):
    """Desired (eye, target) for the agent-perspective camera.

    The camera rides just behind + above the ninja along its facing direction
    and looks ahead down the corridor. ``dist_scale`` uniformly scales the eye
    offset (a dolly): the loops set it from the FOV ratio so that, as the
    speed-FOV widens, the camera dollies in to keep the agent roughly the same
    on-screen size while the periphery rushes by. Returns world-space vectors.
    """
    from .geometry import TILE, DEPTH
    if ninja is None:
        c = np.array([44 * TILE / 2.0, -(25 * TILE / 2.0), DEPTH / 2.0 + 200.0], "f4")
        return c, c + np.array([0.0, 0.0, -1.0], "f4")
    facing = float(getattr(ninja, "facing", 1) or 1)
    k = float(dist_scale)
    nx, ny = ninja.xpos, -ninja.ypos
    eye = np.array([nx - facing * args.fp_back * k, ny + args.fp_up * k,
                    DEPTH / 2.0 + args.fp_out * k], dtype="f4")
    tgt = np.array([nx + facing * args.fp_ahead, ny + args.fp_up * 0.25,
                    DEPTH / 2.0], dtype="f4")
    return eye, tgt


def _dolly_scale(base_fov, fov):
    """Eye-offset scale that keeps a subject the same on-screen size as the FOV
    widens: distance ∝ 1/tan(fov/2)."""
    return float(np.tan(np.radians(base_fov) * 0.5) / np.tan(np.radians(fov) * 0.5))


def _speed_fov(ninja, base_fov, add_max, max_speed=MAX_SPEED):
    """Widen the FOV with the ninja's speed to exaggerate the sense of velocity.

    The rise is eased (frac**1.7) so the FOV ramps up more gradually at low/mid
    speed while still reaching the same min (rest) and max (top speed) extremes.
    """
    if ninja is None or add_max <= 0:
        return base_fov
    vx = float(getattr(ninja, "xspeed", 0.0) or 0.0)
    vy = float(getattr(ninja, "yspeed", 0.0) or 0.0)
    frac = min(float(np.hypot(vx, vy)) / max_speed, 1.0) ** 1.7
    return base_fov + add_max * frac


def _speed01(ninja, hmax=MAX_SPEED):
    """Overall ninja speed as a 0..1 fraction (for the run-over dash colouring)."""
    if ninja is None:
        return 0.0
    vx = abs(float(getattr(ninja, "xspeed", 0.0) or 0.0))
    vy = abs(float(getattr(ninja, "yspeed", 0.0) or 0.0))
    return min(float(np.hypot(vx, vy)) / hmax, 1.0)


def _dash_footfall(ninja):
    """(world_x, surface_y, speed) of the ninja's footing this frame, for dashes."""
    from .scene import NINJA_RADIUS
    return (float(ninja.xpos), -float(ninja.ypos) - NINJA_RADIUS, _speed01(ninja))


def _vel_frac(ninja, hmax=MAX_SPEED, vmax=MAX_VSPEED):
    """(horizontal, vertical) speed as 0..1 fractions, for the background shader.

    Kept as separate axes: horizontal is a continuous run build-up, vertical is
    usually a jump impulse. Both are unsmoothed here; callers apply temporal
    filtering (EMA on horizontal, peak-hold decay on vertical).
    """
    if ninja is None:
        return 0.0, 0.0
    vx = abs(float(getattr(ninja, "xspeed", 0.0) or 0.0))
    vy = abs(float(getattr(ninja, "yspeed", 0.0) or 0.0))
    return min(vx / hmax, 1.0), min(vy / vmax, 1.0)


def _lookat_mvp(eye, target, aspect, fov_deg, near=1.0, far=8000.0):
    """Build a look-at MVP from world eye/target (for the agent FPV camera)."""
    from pyrr import Matrix44, Vector3
    view = Matrix44.look_at(Vector3(np.asarray(eye, "f4")),
                            Vector3(np.asarray(target, "f4")),
                            Vector3([0.0, 1.0, 0.0]), dtype="f4")
    proj = Matrix44.perspective_projection(fov_deg, aspect, near, far, dtype="f4")
    return (proj * view).astype("f4")


def run_headless(args) -> None:
    import moderngl
    from PIL import Image
    from .geometry import build_backplane_mesh, build_level_mesh
    from .scene import build_dynamic_mesh, build_ninja_ghosts

    nh = _load_sim(args.map)
    tiles = build_level_mesh(nh.sim.tile_dic, not args.no_scatter, not args.no_guardrails)

    # Optionally drive the sim forward with a scripted hold before rendering, so
    # the still can show the ninja mid-motion rather than at spawn.
    _nj = nh.sim.ninja
    x0, y0 = (float(_nj.xpos), float(_nj.ypos)) if _nj else (0.0, 0.0)
    hor, jump = HOLD_INPUTS.get(args.hold, (0, 0))
    for _ in range(args.steps):
        nh.tick(hor, jump)
    dynamic = build_dynamic_mesh(nh.sim)

    ctx = moderngl.create_standalone_context(backend="egl")
    from .renderer import Renderer
    from .postfx import SceneTarget
    r = Renderer(ctx)
    r.bg_anim = not args.plain_bg
    r.set_static_mesh(tiles)
    r.set_backplane(build_backplane_mesh())
    r.set_dynamic_mesh(dynamic)
    if not args.no_ninja_blur:
        r.set_ninja_ghosts(build_ninja_ghosts(
            nh.sim.ninja, _speed01(nh.sim.ninja), args.ninja_blur_strength))

    bg = _parse_bg(args.bg)
    target = SceneTarget(ctx, args.width, args.height, samples=args.msaa)

    aspect = args.width / args.height
    if args.fpv:
        fov = _speed_fov(nh.sim.ninja, args.fov, args.fov_speed)
        eye, tgt = _agent_fpv_pose(nh.sim.ninja, args,
                                   dist_scale=_dolly_scale(args.fov, fov))
        mvp = _lookat_mvp(eye, tgt, aspect, fov)
        cam_pos = eye
    else:
        cam = _make_camera(args.width, args.height, args.elevation)
        cam.radius /= max(args.zoom, 0.01)
        cam.azimuth = np.radians(args.azimuth)
        mvp = cam.mvp(aspect)
        cam_pos = cam.eye()

    target.use()
    h_raw, v_raw = _vel_frac(nh.sim.ninja)
    _nj = nh.sim.ninja
    ntx, nty = (_nj.xpos - x0, _nj.ypos - y0) if _nj else (0.0, 0.0)
    r.render(mvp, clear=bg, cam_pos=cam_pos, time=args.steps * 0.05 + 4.0,
             hspeed=h_raw, vspeed=v_raw, netx=ntx, nety=nty,
             travelmax=abs(ntx) + abs(nty))
    target.resolve()

    data = target.read_rgb()
    img = Image.frombytes("RGB", (args.width, args.height), data).transpose(
        Image.FLIP_TOP_BOTTOM
    )
    img.save(args.out)
    ninja = nh.sim.ninja
    where = (f"ninja={tuple(round(v) for v in nh.ninja_position())} "
             f"state={ninja.state}" if ninja is not None else "ninja=none")
    print(f"wrote {args.out} ({args.width}x{args.height})  frame={nh.sim.frame}  "
          f"msaa={target.samples}  {where}  "
          f"tiles={len(tiles)//3} tris, dynamic={len(dynamic)//3} tris")
    r.release()
    target.release()


_STATE_NAMES = {
    0: "immobile", 1: "running", 2: "sliding", 3: "jumping", 4: "falling",
    5: "wall-slide", 6: "dead", 7: "dying", 8: "WON", 9: "disabled",
}


def _follow_step(cam, ninja, dead_zone: float = 30.0, ease: float = 0.14,
                 lookahead: float = 9.0):
    """Dead-zone follow with velocity look-ahead and gentle easing.

    The target holds still while the ninja stays within ``dead_zone`` px of it
    (no jitter on small moves); past that it eases toward a point biased ahead
    in the ninja's travel direction. Returns the new target as a Vector3.
    """
    from pyrr import Vector3
    from .geometry import DEPTH

    vx = float(getattr(ninja, "xspeed", 0.0) or 0.0)
    vy = float(getattr(ninja, "yspeed", 0.0) or 0.0)
    desired = np.array(
        [ninja.xpos + vx * lookahead, -(ninja.ypos + vy * lookahead), DEPTH / 2.0],
        dtype="f4")
    cur = np.array(list(cam.target), dtype="f4")
    delta = desired - cur
    dist = float(np.hypot(delta[0], delta[1]))
    if dist > dead_zone:
        cur = cur + delta * ((dist - dead_zone) / dist) * ease
        cur[2] = DEPTH / 2.0
    return Vector3(cur, dtype="f4")


def run_record(args) -> None:
    """Offscreen: play the input source, encode every frame to an MP4."""
    import moderngl
    import imageio
    from pyrr import Vector3
    from .renderer import Renderer, TRAIL_MAX
    from .geometry import build_backplane_mesh, build_level_mesh
    from .scene import build_dynamic_mesh, build_ninja_ghosts, NinjaTrail
    from .postfx import SceneTarget

    nh, source, label = _setup(args)
    W, H = args.width, args.height
    bg = _parse_bg(args.bg)
    blur_strength, blur_fp_only = _blur_config(args)
    blur_on = _blur_for_mode(blur_strength, blur_fp_only, args.fpv)

    ctx = moderngl.create_standalone_context(backend="egl")
    r = Renderer(ctx)
    r.bg_anim = not args.plain_bg
    r.set_static_mesh(build_level_mesh(nh.sim.tile_dic, not args.no_scatter, not args.no_guardrails))
    r.set_backplane(build_backplane_mesh())
    sp_h, sp_v = 0.0, 0.0   # smoothed horizontal (EMA) + vertical (peak-hold)

    target = SceneTarget(ctx, W, H, samples=args.msaa)
    # When compositing motion blur we present into a plain single-sample FBO,
    # then read that; otherwise we read the resolved scene directly.
    out_fbo = None
    if blur_on:
        out_fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 3)])

    fpv = args.fpv
    aspect = W / H
    if fpv:
        # Agent-perspective camera chases the ninja; the input source drives the
        # ninja (replay / --hold), so the flythrough follows the agent itself.
        eye, tgt = _agent_fpv_pose(nh.sim.ninja, args)
        cur_fov = args.fov
    else:
        cam = _make_camera(W, H, args.elevation)
        cam.radius /= max(args.zoom, 0.01)
        cam.azimuth = np.radians(args.azimuth)
        level_target = np.array(list(cam.target), dtype="f4")
    follow = args.follow or bool(args.replay)

    writer = imageio.get_writer(args.record, fps=args.fps, macro_block_size=None,
                                quality=8, codec="libx264")
    hard_cap = args.max_frames or ((len(source) or 3600) + args.tail)

    from collections import deque
    trail = None if args.no_trail else NinjaTrail()
    footfalls = deque(maxlen=TRAIL_MAX)
    _nj = nh.sim.ninja
    x0, y0 = (float(_nj.xpos), float(_nj.ypos)) if _nj else (0.0, 0.0)

    frame = 0
    settle = 0
    prev_cam_pos = None
    travel_hw = 0.0                    # travel high-water mark (never decreases)
    while frame < hard_cap:
        hor, jump = source.poll()
        nh.tick(hor, jump)
        r.set_dynamic_mesh(build_dynamic_mesh(nh.sim))
        if not args.no_ninja_blur:
            r.set_ninja_ghosts(build_ninja_ghosts(
                nh.sim.ninja, _speed01(nh.sim.ninja), args.ninja_blur_strength))
        if nh.sim.ninja is not None:
            footfalls.append(_dash_footfall(nh.sim.ninja))
            r.set_dash_trail(footfalls)
        if trail is not None:
            trail.push(nh.sim.ninja)
            r.set_trail(trail.build())

        h_raw, v_raw = _vel_frac(nh.sim.ninja)
        sp_h = sp_h * 0.88 + h_raw * 0.12          # EMA -> stable at top run speed
        sp_v = max(sp_v * 0.90, v_raw)             # peak-hold -> jump burst decays

        if fpv:
            # Ease the agent-cam pose toward its desired follow position so facing
            # flips don't snap; the speed-FOV widens and the camera dollies in to
            # match, keeping the agent the same on-screen size.
            cur_fov = cur_fov * 0.85 + _speed_fov(nh.sim.ninja, args.fov, args.fov_speed) * 0.15
            des_eye, des_tgt = _agent_fpv_pose(nh.sim.ninja, args,
                                               dist_scale=_dolly_scale(args.fov, cur_fov))
            eye = eye * 0.82 + des_eye * 0.18
            tgt = tgt * 0.82 + des_tgt * 0.18
            mvp = _lookat_mvp(eye, tgt, aspect, cur_fov)
            cam_pos = eye
        else:
            if follow and nh.sim.ninja:
                cam.target = _follow_step(cam, nh.sim.ninja)
            else:
                cam.target = Vector3(
                    np.array(list(cam.target), "f4") * 0.9 + level_target * 0.1,
                    dtype="f4")
            if args.spin:
                cam.azimuth += args.spin_rate
            # Follow cam moves radially closer behind the ninja as speed rises.
            r0 = cam.radius
            cam.radius = r0 * (1.0 - 0.32 * sp_h) if follow else r0
            mvp = cam.mvp(aspect)
            cam_pos = cam.eye()
            cam.radius = r0

        # Edge blur scales with how far the eye moved this frame; works the same
        # for the agent camera and the orbit/follow camera.
        strength, prev_cam_pos = _blur_step(cam_pos, prev_cam_pos,
                                            blur_strength if blur_on else 0.0)

        _nj = nh.sim.ninja
        ntx, nty = (_nj.xpos - x0, _nj.ypos - y0) if _nj else (0.0, 0.0)
        travel_hw = max(travel_hw, abs(ntx) + abs(nty))
        target.use()
        r.render(mvp, clear=bg, cam_pos=cam_pos, time=frame / float(args.fps),
                 hspeed=sp_h, vspeed=sp_v, netx=ntx, nety=nty, travelmax=travel_hw)
        target.resolve()
        if blur_on:
            target.present(out_fbo, strength)
            buf = np.frombuffer(out_fbo.read(components=3, alignment=1), np.uint8)
        else:
            buf = np.frombuffer(target.read_rgb(), np.uint8)
        writer.append_data(buf.reshape(H, W, 3)[::-1])
        frame += 1

        # With an explicit --max-frames we render exactly that many (e.g. a
        # fixed-length orbit showcase). Otherwise hold a short tail once the run
        # is over (replay spent, or ninja won / died) so the outcome shows.
        if not args.max_frames:
            terminal = (source.exhausted() or nh.ninja_has_won()
                        or nh.ninja_has_died())
            if terminal:
                settle += 1
                if settle >= args.tail:
                    break

    writer.close()
    ninja = nh.sim.ninja
    state = "none" if ninja is None else ninja.state
    print(f"wrote {args.record}  frames={frame}  fps={args.fps}  "
          f"level='{label}'  won={nh.ninja_has_won()}  final_state={state}")
    r.release()
    target.release()
    if out_fbo is not None:
        for tex in out_fbo.color_attachments:
            tex.release()
        out_fbo.release()


def run_window(args) -> None:
    _ensure_window_gl_env()  # may re-exec
    import pygame
    import moderngl

    pygame.init()
    pygame.display.set_caption("nclone 3D")
    # Request a multisampled default framebuffer. If the driver rejects the
    # requested sample count we step down rather than fail.
    if args.msaa and args.msaa > 1:
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, int(args.msaa))
    try:
        pygame.display.set_mode((args.width, args.height),
                                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    except pygame.error:
        # Retry at 2x, then without MSAA, if the requested config is unavailable.
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 2)
        try:
            pygame.display.set_mode((args.width, args.height),
                                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        except pygame.error:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            pygame.display.set_mode((args.width, args.height),
                                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    ctx = moderngl.create_context()
    # A multisampled default framebuffer (requested via the GL attributes above)
    # is resolved automatically by GL; moderngl has no MULTISAMPLE enable flag.

    nh, source, label = _setup(args)
    from .renderer import Renderer, TRAIL_MAX
    from .scene import build_dynamic_mesh, build_ninja_ghosts, NinjaTrail
    from .geometry import build_backplane_mesh, build_level_mesh
    from pyrr import Vector3

    tiles = build_level_mesh(nh.sim.tile_dic, not args.no_scatter, not args.no_guardrails)
    r = Renderer(ctx)
    r.bg_anim = not args.plain_bg
    r.set_static_mesh(tiles)
    r.set_backplane(build_backplane_mesh())
    sp_h, sp_v = 0.0, 0.0   # smoothed horizontal (EMA) + vertical (peak-hold)

    bg = _parse_bg(args.bg)
    blur_strength, blur_fp_only = _blur_config(args)

    cam = _make_camera(args.width, args.height, args.elevation)
    cam.radius /= max(args.zoom, 0.01)
    level_target = Vector3(list(cam.target), dtype="f4")
    clock = pygame.time.Clock()
    w, h = args.width, args.height

    # Camera modes: 0 = orbit, 1 = follow, 2 = first-person (agent perspective).
    # In agent perspective the camera rides the ninja and the keyboard drives the
    # ninja itself (you play the agent), not a detached fly camera.
    _MODE_NAMES = ("orbit", "follow", "first-person")
    if args.fpv:
        cam_mode = 2
    elif args.follow or bool(args.replay):
        cam_mode = 1
    else:
        cam_mode = 0
    spin = args.spin
    paused = False
    fp_eye, fp_tgt = _agent_fpv_pose(nh.sim.ninja, args)
    fp_fov = args.fov
    prev_cam_pos = None
    from collections import deque
    trail = None if args.no_trail else NinjaTrail()
    footfalls = deque(maxlen=TRAIL_MAX)
    _nj = nh.sim.ninja
    x0, y0 = (float(_nj.xpos), float(_nj.ypos)) if _nj else (0.0, 0.0)
    travel_hw = 0.0                    # travel high-water mark (never decreases)

    # Offscreen target only needed for the motion-blur composite; without it the
    # scene renders straight to the (multisampled) default framebuffer. Built
    # whenever blur is configured at all, since the camera mode can change with C.
    from .postfx import SceneTarget
    fx = SceneTarget(ctx, w, h, samples=args.msaa) if blur_strength > 0.0 else None

    blur_desc = "off" if blur_strength <= 0.0 else (
        "first-person only" if blur_fp_only else "on")
    print(f"window: {w}x{h}  GL={ctx.info['GL_RENDERER']}  "
          f"level='{label}'  source={source.name}  tiles={len(tiles)//3} tris  "
          f"msaa={args.msaa}  motion_blur={blur_desc}")
    print("move: <-/-> or A/D   jump: Z/Space/Up/W")
    print("cam: C cycle mode (orbit/follow/first-person), drag or Q/E rotate, +/- zoom, "
          "F follow, O spin   R reset, P pause, ESC quit")
    print("first-person: rides the agent; A/D + jump drive the agent you're watching")

    dragging = False
    running = True
    frame_no = 0
    while running:
        if args.frames and frame_no >= args.frames:
            break
        frame_no += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                w, h = max(1, e.w), max(1, e.h)
                ctx.viewport = (0, 0, w, h)
                if fx is not None:
                    fx.resize(w, h)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging and cam_mode != 2:
                cam.azimuth += e.rel[0] * 0.01
                cam.elevation = float(np.clip(cam.elevation - e.rel[1] * 0.01,
                                              -1.4, 1.4))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    nh.reset()
                    source.reset()
                    footfalls.clear()
                    if nh.sim.ninja:
                        x0, y0 = float(nh.sim.ninja.xpos), float(nh.sim.ninja.ypos)
                    if trail is not None:
                        trail.reset()
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_f:
                    cam_mode = 1 if cam_mode != 1 else 0
                elif e.key == pygame.K_o:
                    spin = not spin
                elif e.key == pygame.K_c:
                    cam_mode = (cam_mode + 1) % 3

        # --- advance the simulation one physics frame from the input source ---
        # The keyboard always drives the ninja, including in first-person, where
        # you *are* the agent (A/D + jump), and the camera rides along.
        hor, jump = source.poll()
        if not paused:
            nh.tick(hor, jump)

        # --- rebuild dynamic geometry from the live sim ---
        r.set_dynamic_mesh(build_dynamic_mesh(nh.sim))
        if not args.no_ninja_blur:
            r.set_ninja_ghosts(None if paused else build_ninja_ghosts(
                nh.sim.ninja, _speed01(nh.sim.ninja), args.ninja_blur_strength))
        if nh.sim.ninja is not None and not paused:
            footfalls.append(_dash_footfall(nh.sim.ninja))
        r.set_dash_trail(footfalls)
        if trail is not None:
            if not paused:
                trail.push(nh.sim.ninja)
            r.set_trail(trail.build())

        # --- speed (smoothed) drives camera closeness, FOV, blur, background ---
        h_raw, v_raw = _vel_frac(nh.sim.ninja)
        sp_h = sp_h * 0.88 + h_raw * 0.12          # EMA -> stable at top run speed
        sp_v = max(sp_v * 0.90, v_raw)             # peak-hold -> jump burst decays

        # --- camera update ---
        keys = pygame.key.get_pressed()
        aspect = w / max(h, 1)
        if cam_mode == 2:
            # Agent-perspective: ease the camera toward its ride-along pose so
            # facing flips glide instead of snapping. The speed-FOV widens and the
            # camera dollies in to match, keeping the agent the same size.
            fp_fov = fp_fov * 0.85 + _speed_fov(nh.sim.ninja, args.fov, args.fov_speed) * 0.15
            des_eye, des_tgt = _agent_fpv_pose(nh.sim.ninja, args,
                                               dist_scale=_dolly_scale(args.fov, fp_fov))
            fp_eye = fp_eye * 0.82 + des_eye * 0.18
            fp_tgt = fp_tgt * 0.82 + des_tgt * 0.18
            mvp = _lookat_mvp(fp_eye, fp_tgt, aspect, fp_fov)
            cam_pos = fp_eye
        else:
            if keys[pygame.K_q]:
                cam.azimuth -= 0.03
            if keys[pygame.K_e]:
                cam.azimuth += 0.03
            if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                cam.radius *= 0.98
            if keys[pygame.K_MINUS]:
                cam.radius *= 1.02
            if spin:
                cam.azimuth += 0.01
            if cam_mode == 1 and nh.sim.ninja:
                cam.target = _follow_step(cam, nh.sim.ninja)
            else:
                cam.target = Vector3(
                    (np.array(list(cam.target), dtype="f4") * 0.85
                     + np.array(list(level_target), dtype="f4") * 0.15),
                    dtype="f4")
            # Follow cam moves radially closer behind the ninja as speed rises.
            r0 = cam.radius
            cam.radius = r0 * (1.0 - 0.32 * sp_h) if cam_mode == 1 else r0
            mvp = cam.mvp(aspect)
            cam_pos = cam.eye()
            cam.radius = r0

        blur_now = fx is not None and _blur_for_mode(
            blur_strength, blur_fp_only, cam_mode == 2)
        strength, prev_cam_pos = _blur_step(cam_pos, prev_cam_pos,
                                            blur_strength if blur_now else 0.0)

        # --- render (optionally through the motion-blur composite) ---
        t_now = pygame.time.get_ticks() * 0.001
        _nj = nh.sim.ninja
        ntx, nty = (_nj.xpos - x0, _nj.ypos - y0) if _nj else (0.0, 0.0)
        travel_hw = max(travel_hw, abs(ntx) + abs(nty))
        if blur_now:
            fx.use()
            r.render(mvp, clear=bg, cam_pos=cam_pos, time=t_now, hspeed=sp_h,
                     vspeed=sp_v, netx=ntx, nety=nty, travelmax=travel_hw)
            fx.resolve()
            fx.present(ctx.screen, strength)
        else:
            ctx.screen.use()
            ctx.viewport = (0, 0, w, h)
            r.render(mvp, clear=bg, cam_pos=cam_pos, time=t_now, hspeed=sp_h,
                     vspeed=sp_v, netx=ntx, nety=nty, travelmax=travel_hw)
        pygame.display.flip()

        state = getattr(nh.sim.ninja, "state", -1)
        tag = "PAUSED " if paused else ""
        pygame.display.set_caption(
            f"nclone 3D  {label}  frame {nh.sim.frame}  {tag}"
            f"cam={_MODE_NAMES[cam_mode]}  state={_STATE_NAMES.get(state, state)}"
        )
        clock.tick(60)

    r.release()
    if fx is not None:
        fx.release()
    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="nclone 3D frontend: play, record, or render the simulation")
    ap.add_argument("--map", default=None, help="path to a map file")
    ap.add_argument("--headless", action="store_true",
                    help="render one frame offscreen to --out")
    ap.add_argument("--out", default="/tmp/nclone_level.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=15.0,
                    help="camera azimuth in degrees (start angle for --spin)")
    ap.add_argument("--elevation", type=float, default=30.0,
                    help="camera elevation angle in degrees")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="camera zoom factor (>1 = closer; good with --follow)")
    ap.add_argument("--steps", type=int, default=0,
                    help="headless: advance the sim N frames before rendering")
    ap.add_argument("--hold", default="none", choices=list(HOLD_INPUTS),
                    help="headless: input to hold during --steps")
    ap.add_argument("--jump-period", type=int, default=0,
                    help="record: pulse jump every N frames on top of --hold (bhop)")
    ap.add_argument("--jump-len", type=int, default=3,
                    help="record: frames to hold jump within each --jump-period cycle")
    ap.add_argument("--spin", action="store_true", help="auto-orbit the camera")
    ap.add_argument("--spin-rate", type=float, default=0.004,
                    help="record: radians/frame of auto-orbit when --spin")
    ap.add_argument("--follow", action="store_true",
                    help="camera follows the ninja (implied by --replay)")
    ap.add_argument("--frames", type=int, default=0,
                    help="window: auto-quit after N frames (0 = run until closed)")
    # stacked multi-level mode (up to 3 levels along the depth axis)
    ap.add_argument("--stack", default=None,
                    help="path to a FOLDER of maps: play them stacked in depth, "
                         "warping to the next level on completion (cycles forever)")
    ap.add_argument("--stack-spacing", type=float, default=320.0,
                    help="stack: depth (px) between adjacent level slabs")
    ap.add_argument("--stack-teleport", type=int, default=34,
                    help="stack: frames of warp animation between levels")
    ap.add_argument("--stack-hold", default=None, choices=list(HOLD_INPUTS),
                    help="stack: auto-drive the live level with this held input "
                         "(default: keyboard in window mode, 'right' when recording)")
    ap.add_argument("--stack-next-alpha", type=float, default=0.34,
                    help="stack: opacity of the upcoming (next) level pane")
    ap.add_argument("--stack-prev-alpha", type=float, default=0.16,
                    help="stack: opacity of the completed (previous) level pane")
    ap.add_argument("--stack-desat", type=float, default=0.55,
                    help="stack: how far neighbour panes desaturate toward grey")
    ap.add_argument("--stack-azimuth", type=float, default=24.0,
                    help="stack: camera azimuth (deg); angles the depth stack")
    ap.add_argument("--stack-elevation", type=float, default=16.0,
                    help="stack: camera elevation (deg)")
    ap.add_argument("--stack-bg-anim", action="store_true",
                    help="stack: keep the busy animated cube background (off by "
                         "default in stacked mode so the depth stack reads clearly)")
    ap.add_argument("--stack-scatter", action="store_true",
                    help="stack: keep the scattered decoration blocks around each "
                         "slab (off by default; they triple up and clutter the stack)")
    # replay + video
    ap.add_argument("--replay", default=None,
                    help="path to an npp_attract demo file to play back")
    ap.add_argument("--record", default=None,
                    help="encode an MP4 to this path (offscreen) instead of a window")
    ap.add_argument("--fps", type=int, default=60, help="record: video frame rate")
    ap.add_argument("--tail", type=int, default=90,
                    help="record: extra frames after the replay ends")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="record: hard cap on frames (0 = replay length + tail)")
    ap.add_argument("--anim", action="store_true",
                    help="enable ninja animation (bones)")
    # appearance
    ap.add_argument("--bg", default="white",
                    help="background: preset (white/light/slate/dark) or 'r,g,b'")
    ap.add_argument("--msaa", type=int, default=4, choices=[0, 2, 4, 8],
                    help="MSAA sample count (0 = off); llvmpipe MSAA is costly")
    ap.add_argument("--plain-bg", action="store_true",
                    help="disable the animated geometric background (solid --bg)")
    ap.add_argument("--no-scatter", action="store_true",
                    help="disable the scattered 3D blocks radiating around the level")
    ap.add_argument("--no-guardrails", action="store_true",
                    help="disable the segmented rails floating above ground/slope tiles")
    ap.add_argument("--no-trail", action="store_true",
                    help="disable the speed-coloured trail behind the ninja")
    ap.add_argument("--no-ninja-blur", action="store_true",
                    help="disable the speed-driven motion-blur echoes on the ninja")
    ap.add_argument("--ninja-blur-strength", type=float, default=1.0,
                    help="scalar multiplier on the ninja motion-blur echo smear")
    # first-person (agent-perspective) camera + motion blur
    ap.add_argument("--fpv", action="store_true",
                    help="agent-perspective camera: ride the ninja, you drive it")
    ap.add_argument("--fov", type=float, default=65.0,
                    help="first-person camera field of view (degrees)")
    ap.add_argument("--fp-back", type=float, default=70.0,
                    help="fpv: camera distance behind the ninja along facing (px)")
    ap.add_argument("--fp-up", type=float, default=34.0,
                    help="fpv: camera height above the ninja (px)")
    ap.add_argument("--fp-out", type=float, default=95.0,
                    help="fpv: camera distance out of the play slab, +z (px)")
    ap.add_argument("--fp-ahead", type=float, default=90.0,
                    help="fpv: look-ahead distance in front of the ninja (px)")
    ap.add_argument("--fov-speed", type=float, default=30.0,
                    help="fpv: extra degrees of FOV added at top ninja speed (0 = off)")
    ap.add_argument("--motion-blur", type=float, default=-1.0,
                    help="edge motion blur: <0 auto (first-person only), 0 off, "
                         ">0 strength scale and enabled in every camera mode")
    ap.add_argument("--motion-blur-strength", type=float, default=1.0,
                    help="scalar multiplier on the edge motion-blur intensity")
    args = ap.parse_args()

    if args.width < 1 or args.height < 1:
        ap.error("--width and --height must be positive")

    if args.stack:
        from .stack import run_stack
        run_stack(args)
        return

    if args.map is None and args.replay is None:
        args.map = _default_map()

    if args.record:
        run_record(args)
    elif args.headless:
        run_headless(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
