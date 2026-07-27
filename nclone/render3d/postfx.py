"""Offscreen render target with MSAA resolve + optional edge motion blur.

The scene is drawn into a (multisampled) framebuffer, resolved to a sampleable
color texture, and then either read back directly (no blur) or composited to a
destination framebuffer through a fullscreen triangle running the radial
edge-blur shader. Used by the offscreen (record/headless) paths and by the
window path whenever motion blur is active.

Pipeline: render(MSAA) -> resolve -> [edge-blur] -> present.
"""

from __future__ import annotations

import moderngl

from .renderer import release_gl

# Fullscreen triangle generated from gl_VertexID (no vertex buffer needed).
_FS_VERTEX_SHADER = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Radial edge motion blur: sharp at screen center, smeared toward the edges,
# scaled by camera speed (blur_strength). N taps march from the fragment toward
# the screen center. blur_strength ~0 => a single tap (a plain copy).
_BLUR_FRAGMENT_SHADER = """
#version 330
uniform sampler2D scene;
uniform float blur_strength;   // 0 = off (plain copy)
in vec2 v_uv;
out vec4 f_color;
const int TAPS = 8;
void main() {
    vec2 uv = v_uv;
    if (blur_strength < 1e-4) { f_color = texture(scene, uv); return; }
    vec2 dir = uv - vec2(0.5);                    // fragment -> center
    float edge = smoothstep(0.0, 1.0, length(dir) * 2.0);
    edge *= edge;                                 // keep center extra crisp
    float radius = blur_strength * edge * 0.07;   // max ~7% of screen span
    if (radius < 1e-4) { f_color = texture(scene, uv); return; }
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < TAPS; i++) {
        float t = float(i) / float(TAPS - 1);     // 0..1, 0 = this fragment
        vec2 s = uv - dir * (radius * t);         // march toward center
        float w = 1.0 - 0.5 * t;                  // weight nearer taps more
        acc += texture(scene, s).rgb * w;
        wsum += w;
    }
    f_color = vec4(acc / wsum, 1.0);
}
"""


class SceneTarget:
    """Offscreen MSAA color/depth target that resolves to a color texture."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int, samples: int = 4):
        self.ctx = ctx
        # Clamp to what the driver actually supports (llvmpipe often caps at 4).
        req = samples if (samples and samples > 1) else 0
        self.samples = min(req, getattr(ctx, "max_samples", req)) if req else 0
        self._blur_prog = ctx.program(
            vertex_shader=_FS_VERTEX_SHADER, fragment_shader=_BLUR_FRAGMENT_SHADER
        )
        self._fs_vao = ctx.vertex_array(self._blur_prog, [])
        self.resolve_tex = None
        self.ms_fbo = None
        self.resolve_fbo = None
        self._ms_color = None
        self._ms_depth = None
        self._depth = None
        self._build(width, height)

    def _build(self, width: int, height: int) -> None:
        ctx = self.ctx
        self.width, self.height = width, height
        self.resolve_tex = ctx.texture((width, height), 3)
        self.resolve_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.resolve_tex.repeat_x = False
        self.resolve_tex.repeat_y = False
        # Step the sample count down until the driver accepts the renderbuffers,
        # releasing the rejected attachments as we go.
        while self.samples:
            color = depth = None
            try:
                color = ctx.renderbuffer((width, height), samples=self.samples)
                depth = ctx.depth_renderbuffer((width, height), samples=self.samples)
                self.ms_fbo = ctx.framebuffer(color_attachments=[color],
                                              depth_attachment=depth)
                self.resolve_fbo = ctx.framebuffer(color_attachments=[self.resolve_tex])
                self._ms_color, self._ms_depth = color, depth
                return
            except Exception:        # noqa: BLE001 - unsupported sample count
                release_gl(color, depth)
                self.samples = 0 if self.samples <= 2 else self.samples // 2
        # No MSAA: render straight into the resolve texture (with its own depth).
        self._depth = ctx.depth_renderbuffer((width, height))
        self.resolve_fbo = ctx.framebuffer(
            color_attachments=[self.resolve_tex], depth_attachment=self._depth
        )
        self.ms_fbo = self.resolve_fbo

    def release(self) -> None:
        """Free every GL object this target owns (including the blur program)."""
        self.release_buffers()
        release_gl(self._fs_vao, self._blur_prog)
        self._fs_vao = self._blur_prog = None

    def release_buffers(self) -> None:
        """Free just the size-dependent attachments, keeping the blur program."""
        if self.ms_fbo is not self.resolve_fbo:
            release_gl(self.ms_fbo)
        release_gl(self.resolve_fbo, self._ms_color, self._ms_depth,
                   self._depth, self.resolve_tex)
        self.ms_fbo = self.resolve_fbo = self.resolve_tex = None
        self._ms_color = self._ms_depth = self._depth = None

    def resize(self, width: int, height: int) -> None:
        """Rebuild the attachments at a new size, reusing the blur program.

        Cheaper than constructing a fresh SceneTarget on every resize event,
        which would recompile the shader each time.
        """
        if (width, height) == (self.width, self.height):
            return
        self.release_buffers()
        self._build(width, height)

    def use(self) -> None:
        self.ms_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

    def resolve(self) -> None:
        """Resolve MSAA samples into ``resolve_tex`` (no-op when samples==0)."""
        if self.samples:
            self.ctx.copy_framebuffer(self.resolve_fbo, self.ms_fbo)

    def present(self, dst_fbo: moderngl.Framebuffer, blur_strength: float = 0.0) -> None:
        """Composite the resolved scene into ``dst_fbo`` via the blur shader.

        With ``blur_strength <= 0`` this is a straight fullscreen copy.
        """
        dst_fbo.use()
        self.ctx.viewport = (0, 0, dst_fbo.width, dst_fbo.height)
        # Fullscreen pass needs no depth test; the scene renderer re-enables it
        # every frame, so leaving it off here is safe.
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.resolve_tex.use(location=0)
        self._blur_prog["scene"].value = 0
        self._blur_prog["blur_strength"].value = float(max(blur_strength, 0.0))
        self._fs_vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def read_rgb(self) -> bytes:
        """Read the resolved (un-blurred) scene as tightly-packed RGB bytes."""
        return self.resolve_fbo.read(components=3, alignment=1)
