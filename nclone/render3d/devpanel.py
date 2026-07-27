"""A 3D view panel for the dev tool, sharing the dev tool's live simulation.

Renders the same ``sim`` the 2D dev interface is driving, offscreen via a
moderngl EGL context, and hands back a pygame surface the dev tool blits into a
reserved strip of its window. So the regular debug interface and the 3D diorama
run side by side off one simulation.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np


class Render3DPanel:
    """Offscreen 3D renderer that produces a pygame surface from a live sim."""

    def __init__(self, width: int, height: int, follow: bool = True,
                 fpv: bool = False, msaa: int = 4):
        import moderngl
        from .renderer import Renderer, TRAIL_MAX
        from .postfx import SceneTarget
        from .scene import NinjaTrail

        self.w, self.h = max(1, int(width)), max(1, int(height))
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.r = Renderer(self.ctx)
        self.target = SceneTarget(self.ctx, self.w, self.h, samples=msaa)
        self.follow = follow
        self.fpv = fpv
        self.cam = None
        self._map_key = None
        self.trail = NinjaTrail()
        self.footfalls: deque = deque(maxlen=TRAIL_MAX)
        self._t = 0.0
        self._sp_h = self._sp_v = 0.0
        self._x0 = self._y0 = 0.0
        # Agent-perspective camera state (defaults mirror the app's FPV CLI).
        self._fa = SimpleNamespace(fp_back=70.0, fp_up=34.0, fp_out=95.0,
                                   fp_ahead=90.0, fov=65.0, fov_speed=30.0)
        self._fp_eye = self._fp_tgt = None
        self._fp_fov = self._fa.fov

    def resize(self, width: int, height: int) -> None:
        """Match a new panel size (the dev-tool window is resizable)."""
        w, h = max(1, int(width)), max(1, int(height))
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h
        self.target.resize(w, h)

    def release(self) -> None:
        """Free the panel's GL objects and its private EGL context."""
        self.r.release()
        self.target.release()
        self.ctx.release()

    # -- map (static geometry) --------------------------------------------

    def _set_map(self, sim) -> None:
        from .geometry import build_backplane_mesh, build_level_mesh
        from .app import _make_camera, _agent_fpv_pose

        self.r.set_static_mesh(build_level_mesh(sim.tile_dic))
        self.r.set_backplane(build_backplane_mesh())
        self.cam = _make_camera(self.w, self.h, 30.0)
        self.cam.radius /= 2.2                      # closer follow than a full orbit
        self.trail.reset()
        self.footfalls.clear()
        nj = sim.ninja
        self._x0, self._y0 = (float(nj.xpos), float(nj.ypos)) if nj else (0.0, 0.0)
        eye, tgt = _agent_fpv_pose(nj, self._fa)
        self._fp_eye, self._fp_tgt = eye.copy(), tgt.copy()
        self._fp_fov = self._fa.fov

    def _maybe_set_map(self, sim) -> None:
        # Rebuild static geometry when a fresh map is loaded (frame resets low).
        if self.cam is None or sim.frame <= 1:
            try:
                sig = hash(frozenset(sim.tile_dic.items()))
            except TypeError:
                sig = id(sim.tile_dic)
            if sig != self._map_key:
                self._map_key = sig
                self._set_map(sim)

    # -- per-frame render --------------------------------------------------

    def render(self, sim):
        """Render the current sim state; return an (w, h) RGB pygame surface."""
        import pygame
        from .scene import build_dynamic_mesh
        from .app import (_follow_step, _vel_frac, _dash_footfall,
                          _agent_fpv_pose, _lookat_mvp, _speed_fov, _dolly_scale)

        self._maybe_set_map(sim)
        nj = sim.ninja

        self.r.set_dynamic_mesh(build_dynamic_mesh(sim))
        if nj is not None:
            self.footfalls.append(_dash_footfall(nj))
            self.r.set_dash_trail(self.footfalls)
            self.trail.push(nj)
            self.r.set_trail(self.trail.build())
            h_raw, v_raw = _vel_frac(nj)
            self._sp_h = self._sp_h * 0.88 + h_raw * 0.12
            self._sp_v = max(self._sp_v * 0.90, v_raw)
            ntx, nty = nj.xpos - self._x0, nj.ypos - self._y0
        else:
            ntx = nty = 0.0
        self._t += 1.0 / 60.0

        # Camera: agent-perspective (FPV) or orbit-follow.
        aspect = self.w / max(self.h, 1)
        if self.fpv and nj is not None:
            fa = self._fa
            self._fp_fov = self._fp_fov * 0.85 + _speed_fov(nj, fa.fov, fa.fov_speed) * 0.15
            des_eye, des_tgt = _agent_fpv_pose(nj, fa, dist_scale=_dolly_scale(fa.fov, self._fp_fov))
            self._fp_eye = self._fp_eye * 0.82 + des_eye * 0.18
            self._fp_tgt = self._fp_tgt * 0.82 + des_tgt * 0.18
            mvp = _lookat_mvp(self._fp_eye, self._fp_tgt, aspect, self._fp_fov)
            cam_pos = self._fp_eye
        else:
            if self.follow and nj is not None:
                self.cam.target = _follow_step(self.cam, nj)
            mvp = self.cam.mvp(aspect)
            cam_pos = self.cam.eye()

        self.target.use()
        self.r.render(mvp, clear=(1.0, 1.0, 1.0), cam_pos=cam_pos, time=self._t,
                      hspeed=self._sp_h, vspeed=self._sp_v, netx=ntx, nety=nty)
        self.target.resolve()

        data = self.target.read_rgb()               # bottom-up RGB, tightly packed
        arr = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)[::-1]
        return pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(),
                                       (self.w, self.h), "RGB")
