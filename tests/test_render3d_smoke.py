"""render3d end-to-end smoke tests against a real GL context.

These compile the shaders and render offscreen, which is the only way to catch
GLSL regressions and buffer-layout mismatches. They skip when moderngl or an
EGL-capable driver is unavailable, so they are safe in a headless CI container
without a GPU only if software GL (llvmpipe) is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")
pytest.importorskip("pyrr")

from nclone.render3d import geometry as G          # noqa: E402
from nclone.render3d import scene as S             # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(backend="egl")
    except Exception as exc:                        # noqa: BLE001 - no GL here
        pytest.skip(f"no offscreen GL context: {exc}")
    yield c
    c.release()


@pytest.fixture
def renderer(ctx):
    from nclone.render3d.renderer import Renderer
    r = Renderer(ctx)
    yield r
    r.release()


def _sim():
    from nclone.nplay_headless import NPlayHeadless
    nh = NPlayHeadless(enable_rendering=False)
    nh.load_map("nclone/test_maps/basics")
    return nh


def test_shaders_compile(renderer):
    """Constructing a Renderer compiles both the scene and background programs."""
    assert renderer.prog is not None
    assert "u_trail" in renderer.prog


def test_offscreen_render_produces_a_non_blank_frame(ctx, renderer):
    from nclone.render3d.app import _make_camera
    from nclone.render3d.postfx import SceneTarget

    nh = _sim()
    renderer.set_static_mesh(G.build_level_mesh(nh.sim.tile_dic))
    renderer.set_backplane(G.build_backplane_mesh())
    renderer.set_dynamic_mesh(S.build_dynamic_mesh(nh.sim))

    target = SceneTarget(ctx, 160, 120, samples=0)
    cam = _make_camera(160, 120)
    target.use()
    renderer.render(cam.mvp(160 / 120), cam_pos=cam.eye())
    target.resolve()
    frame = np.frombuffer(target.read_rgb(), np.uint8).reshape(120, 160, 3)
    target.release()

    assert frame.std() > 1.0, "frame is a flat fill, nothing was drawn"


def test_blur_composite_runs_and_is_a_plain_copy_at_zero_strength(ctx, renderer):
    from nclone.render3d.app import _make_camera
    from nclone.render3d.postfx import SceneTarget

    nh = _sim()
    renderer.set_static_mesh(G.build_level_mesh(nh.sim.tile_dic, scatter=False))
    target = SceneTarget(ctx, 96, 64, samples=0)
    out = ctx.framebuffer(color_attachments=[ctx.texture((96, 64), 3)])
    cam = _make_camera(96, 64)

    target.use()
    renderer.render(cam.mvp(96 / 64), cam_pos=cam.eye())
    target.resolve()
    direct = np.frombuffer(target.read_rgb(), np.uint8).copy()
    target.present(out, 0.0)
    copied = np.frombuffer(out.read(components=3, alignment=1), np.uint8)
    assert np.array_equal(direct, copied)

    target.present(out, 1.0)                       # exercises the tap loop
    for tex in out.color_attachments:
        tex.release()
    out.release()
    target.release()


def test_mesh_slots_release_replaced_buffers(renderer):
    """Re-uploading the level mesh must not leak the previous GL buffer.

    moderngl's default gc_mode is manual, so a dropped Python reference alone
    would leave the GL object allocated for the life of the process.
    """
    seen = {}
    for i in range(1, 25):
        renderer.set_static_mesh(np.zeros((3 * i, 9), dtype=np.float32))
        seen[id(renderer._static.vbo)] = renderer._static.vbo

    def alive(buf):
        try:
            buf.size
            return True
        except Exception:                           # noqa: BLE001 - released
            return False

    assert len(seen) > 1, "test did not exercise a reallocation"
    assert sum(alive(b) for b in seen.values()) == 1


def test_growing_dynamic_mesh_reallocates_sublinearly(renderer):
    """The dynamic slot reserves headroom, so it settles instead of thrashing."""
    reallocs, prev = 0, None
    for i in range(1, 200):
        renderer.set_dynamic_mesh(np.zeros((3 * i, 9), dtype=np.float32))
        if renderer._dynamic.vbo is not prev:
            reallocs += 1
            prev = renderer._dynamic.vbo
    assert reallocs < 20


def test_scene_target_falls_back_when_msaa_is_unavailable(ctx):
    from nclone.render3d.postfx import SceneTarget
    target = SceneTarget(ctx, 64, 64, samples=64)   # far above any driver max
    assert target.samples <= getattr(ctx, "max_samples", 0)
    target.release()


def test_scene_target_resize_keeps_the_blur_program(ctx):
    from nclone.render3d.postfx import SceneTarget
    target = SceneTarget(ctx, 64, 64, samples=0)
    prog = target._blur_prog
    target.resize(128, 96)
    assert (target.width, target.height) == (128, 96)
    assert target._blur_prog is prog
    target.release()
