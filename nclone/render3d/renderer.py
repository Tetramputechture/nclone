"""moderngl rendering of the extruded level and entity meshes.

One shader draws everything: per-vertex position, normal and colour, lit by a
key and fill direction. ``flat_mode`` picks between cel-banded tiles with an ink
outline and flat bright entities. The level mesh is uploaded once per level;
separate buffers hold the entity, trail, ghost and overlay geometry that is
rebuilt every frame.
"""

from __future__ import annotations

import moderngl
import numpy as np

from .geometry import DEPTH

# Length of the ninja footfall history the ground speed-imprint matches against.
# Shared by the GLSL uniform array, the fade window, and the caller-side deques.
TRAIL_MAX = 72

_VERTEX_SHADER = """
#version 330
uniform mat4 Mvp;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_worldpos;   // model matrix is identity here, so world == in_pos
void main() {
    gl_Position = Mvp * vec4(in_pos, 1.0);
    v_normal = in_normal;
    v_color = in_color;
    v_worldpos = in_pos;
}
"""

# Cel / toon shading: the diffuse term is quantized into a few flat bands (hard
# light steps) instead of a smooth gradient, and a thin view-angle (fresnel) term
# inks silhouette edges. Vertex colours are used as-is (entity vibrancy is baked
# into the mesh colours) so neutral tiles stay neutral.
_FRAGMENT_SHADER = """
#version 330
#define TRAIL_MAX @TRAIL_MAX@
#define PLANE_Z @PLANE_Z@
uniform vec3 key_dir;    // main directional light
uniform vec3 fill_dir;   // softer fill from the opposite side
uniform vec3 cam_pos;    // world-space camera position (for the rim outline)
uniform int flat_mode;   // 1 = entities/ninja: flat bright colour, no rim
uniform float alpha;     // < 1 for the translucent backplane
uniform vec4 u_trail[TRAIL_MAX];  // (worldx, surface_y, speed, _) ninja footfalls
uniform int u_ntrail;
uniform float u_fade;    // fade window in frames
in vec3 v_normal;
in vec3 v_color;
in vec3 v_worldpos;
out vec4 f_color;

// Ground speed imprint: starts from the tile's own colour, ramps to blue then
// magenta as the run-over speed increases.
vec3 floorSpeedColor(vec3 base, float s) {
    s = clamp(s, 0.0, 1.0);
    if (s < 0.5) return mix(base, vec3(0.20, 0.45, 1.0), s / 0.5);
    return mix(vec3(0.20, 0.45, 1.0), vec3(1.0, 0.30, 0.85), (s - 0.5) / 0.5);
}

float band(float x) {
    // Prefer flat lighting: a shallow cel ramp (0.72..1.0) so surfaces read as
    // near-flat bold colour with just a hint of banding for form.
    if (x > 0.72) return 1.00;
    if (x > 0.40) return 0.90;
    if (x > 0.15) return 0.80;
    return 0.72;
}

void main() {
    // Two-sided: flip the geometric normal to face the viewer so lighting is
    // correct regardless of triangle winding.
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    float key = max(dot(n, normalize(key_dir)), 0.0);

    // Entities & ninja: flat, fully-saturated colour (only a whisper of key so
    // spheres keep a little form) and no silhouette darkening, so gold reads as
    // bright yellow and mines as bright red regardless of viewing angle.
    if (flat_mode == 1) {
        f_color = vec4(v_color * (0.88 + 0.12 * key), alpha);
        return;
    }

    // Tiles: cel bands + a thin ink outline for the stylized diorama look.
    float fill = max(dot(n, normalize(fill_dir)), 0.0);
    vec3 col = v_color * band(key + 0.30 * fill);
    vec3 view = normalize(cam_pos - v_worldpos);
    float ndv = max(dot(n, view), 0.0);
    float edge = 1.0 - smoothstep(0.0, 0.10, ndv);
    col *= (1.0 - 0.55 * edge);

    // Speed imprint: where the ninja ran over a tile top, tint the grey tile
    // with the speed-colour it had at that moment, fading back to grey over the
    // trail window. A narrow band centred on the play plane with soft gradient
    // edges, so it doesn't cover the whole tile depth.
    // The cheap gates (up-facing plus play-plane z-band) are computed BEFORE the
    // trail loop so the TRAIL_MAX-iteration search only runs for the thin slice
    // of fragments that could be tinted; front/back faces and off-plane depth
    // skip it entirely.
    float topmask = smoothstep(0.35, 0.78, n.y);
    float zband = 1.0 - smoothstep(2.5, 6.5, abs(v_worldpos.z - PLANE_Z));
    if (topmask > 0.01 && zband > 0.01 && u_ntrail > 0) {
        float best = 0.0, bestS = 0.0;
        for (int i = 0; i < u_ntrail; i++) {
            vec4 tr = u_trail[i];
            if (abs(v_worldpos.x - tr.x) < 5.0 && abs(v_worldpos.y - tr.y) < 9.0) {
                float fade = clamp(1.0 - float(u_ntrail - 1 - i) / max(u_fade, 1.0),
                                   0.0, 1.0);
                if (fade > best) { best = fade; bestS = tr.z; }
            }
        }
        col = mix(col, floorSpeedColor(col, bestS), 0.85 * best * topmask * zband);
    }

    f_color = vec4(col, alpha);
}
"""

# --- animated geometric background (drawn behind everything, outside the level).
# A monochrome field of oblique-projected CUBES, each tumbling on its own random
# axis/speed, bobbing up and down, and pulsing in size. Cubes continually spawn
# (more of them, faster, the farther the agent has travelled) and fade out after
# a short life. Three parallax layers (slow far -> fast near, ~10x speed spread)
# sweep past each other. Blacks / whites / greys only. Fullscreen triangle
# generated from gl_VertexID (no vertex buffer).
_BG_VERTEX_SHADER = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;                       // 0..2
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_BG_FRAGMENT_SHADER = """
#version 330
uniform float u_time;
uniform vec2  u_res;
uniform float u_hspeed;  // 0..1 horizontal run speed (smoothed, continuous)
uniform float u_vspeed;  // 0..1 vertical speed (jump impulse, peak-hold decay)
uniform float u_netx;    // net X distance travelled (sim px): +right, -left
uniform float u_nety;    // net Y distance travelled (sim px): +down, -up
uniform float u_travelmax;  // high-water mark of |netx|+|nety| (never decreases)
in vec2 v_uv;
out vec4 f_color;

// One cheap hash -> two decorrelated values in [0,1). Replaces the six scalar
// hash() calls per cell the old cube field used (fewer fract/dot chains per
// pixel is the main performance win here).
vec2 hash22(vec2 p) {
    vec3 p3 = fract(p.xyx * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Is P inside the parallelogram A + s*u + t*v with s,t in [0,1]? (out s,t.)
bool inPara(vec2 P, vec2 A, vec2 u, vec2 v, out float s, out float t) {
    vec2 d = P - A;
    float det = u.x * v.y - u.y * v.x;
    if (abs(det) < 1e-6) { s = -1.0; t = -1.0; return false; }
    s = (d.x * v.y - d.y * v.x) / det;
    t = (u.x * d.y - u.y * d.x) / det;
    return s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0;
}

// One cell's cube, evaluated in this layer's grid space at `gp`. A cell holds a
// cube iff its stable activation hash is below `part` (which only rises with the
// agent's travel high-water mark), so the cube COUNT stays constant while the
// agent isn't going anywhere new, and grows (capped) as it ventures farther. Each
// cube fades IN as `part` climbs past its threshold, then stays solid; it never
// self-fades on a timer. `alpha` is that fade-in weight. Cheap: inactive cells
// bail after one hash.
bool cellShape(vec2 c, vec2 gp, float t, float part,
               float spv, float sizeMul, out float bshade, out float edge, out float alpha) {
    bshade = -1.0; edge = 0.0; alpha = 0.0;
    vec2 r1 = hash22(c);                                   // stable per cell
    float actHash = fract(r1.x * 1.7 + r1.y * 3.3);        // activation threshold
    if (actHash > part) return false;                     // not (yet) active
    alpha = smoothstep(0.0, 0.07, part - actHash);        // fade in as travel raises part

    vec2 r2 = hash22(c + 19.7);                            // stable cube identity
    float h = r2.x;

    float ph = t * (0.6 * (0.55 + 0.9 * r2.y)) + h * 6.2831;
    float a = 0.30 * sizeMul * (0.42 + 0.58 * h) * (0.66 + 0.34 * sin(ph));
    a += 0.05 * spv * step(0.5, fract(h * 7.0));           // pop on jump
    if (a < 1e-3) return false;

    float bob = (0.08 + 0.16 * fract(r2.y * 9.0)) * cos(ph);  // up/down bob
    vec2 fc = c + 0.5 + (r2 - 0.5) * 0.5 + vec2(0.0, bob);

    // Tumble: rotate the sample into the cube's local frame (random axis + speed).
    float theta = fract(r2.x * 13.0) * 6.2831 + t * (fract(h * 11.0) - 0.5) * 3.0;
    float ct = cos(theta), st = sin(theta);
    vec2 q = gp - fc;
    vec2 ql = vec2(ct * q.x + st * q.y, -st * q.x + ct * q.y);
    vec2 eL = vec2(0.30 * a, 0.42 * a);                    // oblique depth (local)

    // CUBE: front rectangle + top & right parallelograms.
    float s, tt;
    vec2 dd = abs(ql) - vec2(a);
    if (max(dd.x, dd.y) < 0.0) {
        bshade = 0.60; edge = max(edge, 1.0 - smoothstep(0.0, 0.02, -max(dd.x, dd.y)));
    }
    if (inPara(ql, vec2(-a, a), vec2(2.0 * a, 0.0), eL, s, tt)) {
        bshade = 0.93; edge = max(edge, 1.0 - smoothstep(0.0, 0.06, min(min(s, 1.0 - s), min(tt, 1.0 - tt))));
    }
    if (inPara(ql, vec2(a, a), vec2(0.0, -2.0 * a), eL, s, tt)) {
        bshade = 0.34; edge = max(edge, 1.0 - smoothstep(0.0, 0.06, min(min(s, 1.0 - s), min(tt, 1.0 - tt))));
    }
    return bshade >= 0.0;
}

// One parallax layer: a 3x3 neighbourhood of cells over this layer's grid. The
// front-most (down-left) cube covering the pixel wins; its shade/edge/alpha are
// returned so the caller can blend it (fade) over the layers behind.
bool layerEval(vec2 uv, float lscale, vec2 scroll, float t, float part,
               float spv, float sizeMul, out float shade, out float edge, out float alpha) {
    vec2 gp = uv * lscale + scroll;
    vec2 cell = floor(gp);
    float best = -1e9;
    bool hit = false;
    shade = 0.0; edge = 0.0; alpha = 0.0;
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        vec2 c = cell + vec2(dx, dy);
        float bs, ed, al;
        if (cellShape(c, gp, t, part, spv, sizeMul, bs, ed, al)) {
            float near = -(c.x + c.y);             // down-left = nearer
            if (near > best) { best = near; shade = bs; edge = ed; alpha = al; hit = true; }
        }
    }
    return hit;
}

void main() {
    vec2 uv = v_uv - 1.0;                          // -1..1
    uv.x *= u_res.x / max(u_res.y, 1.0);
    float sph = clamp(u_hspeed, 0.0, 1.0);
    float spv = clamp(u_vspeed, 0.0, 1.0);
    float t = u_time;

    // Net travel: moving LEFT grows the cubes (RIGHT shrinks), UP brightens the
    // whole field (DOWN darkens).
    float sizeMul = clamp(1.0 - u_netx / 650.0, 0.4, 2.0);
    float bright = clamp(-u_nety / 450.0, -0.7, 1.0);

    // How many cells hold a cube is driven by the TRAVEL HIGH-WATER MARK, so the
    // count holds steady when the agent isn't reaching anywhere new and only ever
    // grows (up to a sensible cap) as it ventures farther.
    float part = clamp(0.24 + u_travelmax / 620.0, 0.24, 0.66);

    // Base scroll eases DOWN as the agent speeds up so the backdrop stays calm.
    vec2 baseScroll = vec2(t * (0.055 - 0.035 * sph), t * 0.02);

    float shade = 0.95;                            // paper background
    float ink = 0.0;                               // outline accumulator

    // Three parallax layers, far -> near. Screen pan speed = panMul / lscale, so
    // the near layer sweeps ~10x faster than the far one. Each cube is blended by
    // its fade-in alpha over whatever is behind it.
    float s0, e0, a0;
    if (layerEval(uv, 13.0, baseScroll * 1.0, t, part, spv, sizeMul, s0, e0, a0)) {
        shade = mix(shade, s0 * 0.82, a0); ink = mix(ink, e0, a0);
    }
    float s1, e1, a1;
    if (layerEval(uv, 8.5, baseScroll * 2.4, t, part, spv, sizeMul, s1, e1, a1)) {
        shade = mix(shade, s1 * 0.92, a1); ink = mix(ink, e1, a1);
    }
    float s2, e2, a2;
    if (layerEval(uv, 5.0, baseScroll * 4.6, t, part * 0.85, spv, sizeMul, s2, e2, a2)) {
        shade = mix(shade, s2, a2); ink = mix(ink, e2, a2);
    }

    // Crisp dark outline on edges for graphic punch.
    shade = mix(shade, 0.06, ink * 0.9);

    // Subtle vignette; a touch darker at speed.
    shade -= (0.10 + 0.14 * max(sph, spv)) * smoothstep(0.7, 1.6, length(uv));

    // Net-Y brightness: upward travel lifts the solids toward white, downward
    // darkens them (kept greyscale).
    shade += bright * 0.35;
    f_color = vec4(vec3(clamp(shade, 0.02, 1.0)), 1.0);
}
"""


def release_gl(*objs) -> None:
    """Release GL objects, ignoring already-released ones.

    moderngl's default ``gc_mode`` is manual, so dropping the last Python
    reference to a Buffer/VertexArray does NOT free the GL object. Every
    replace-in-place and teardown path has to release explicitly or it leaks for
    the life of the process.
    """
    for o in objs:
        if o is None:
            continue
        try:
            o.release()
        except Exception:            # noqa: BLE001 - already released / dead ctx
            pass


class _MeshSlot:
    """A vertex buffer plus its VAO, replaced when the mesh outgrows it.

    ``write`` reuses the existing buffer whenever the new data fits, and
    releases the old buffer/VAO pair when it does not. Reallocations reserve
    headroom so a mesh that grows a little each frame (entities appearing, the
    trail filling up) settles instead of reallocating every frame.
    """

    GROWTH = 1.5

    __slots__ = ("_ctx", "_make_vao", "vbo", "vao", "count")

    def __init__(self, ctx: moderngl.Context, make_vao):
        self._ctx = ctx
        self._make_vao = make_vao
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.count = 0

    def write(self, verts) -> None:
        self.count = 0 if verts is None else len(verts)
        if self.count == 0:
            return
        # ascontiguousarray is a no-op for the float32 meshes the builders emit,
        # so the common path uploads with no host-side copy at all.
        data = np.ascontiguousarray(verts, dtype="f4")
        if self.vbo is None or self.vbo.size < data.nbytes:
            release_gl(self.vao, self.vbo)
            self.vbo = self._ctx.buffer(reserve=int(data.nbytes * self.GROWTH) + 1,
                                        dynamic=True)
            self.vao = self._make_vao(self.vbo)
        self.vbo.write(data)

    def render(self, mode, vertices: int | None = None, first: int = 0) -> None:
        self.vao.render(mode, vertices=self.count if vertices is None else vertices,
                        first=first)

    def __bool__(self) -> bool:
        return self.vao is not None and self.count > 0

    def release(self) -> None:
        release_gl(self.vao, self.vbo)
        self.vbo = self.vao = None
        self.count = 0


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)  # two-sided shader -> no face culling
        frag = (_FRAGMENT_SHADER
                .replace("@TRAIL_MAX@", str(TRAIL_MAX))
                .replace("@PLANE_Z@", f"{DEPTH / 2.0:.1f}"))
        self.prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=frag)
        self.prog["key_dir"].value = (0.5, 0.85, 0.55)
        self.prog["fill_dir"].value = (-0.5, 0.2, -0.4)
        self.prog["cam_pos"].value = (0.0, 0.0, 3000.0)
        self.prog["flat_mode"].value = 0
        self.prog["alpha"].value = 1.0
        self.prog["u_ntrail"].value = 0
        self.prog["u_fade"].value = float(TRAIL_MAX)
        # Animated geometric background (behind the level).
        self.bg_anim = True
        self._bg_prog = ctx.program(vertex_shader=_BG_VERTEX_SHADER,
                                    fragment_shader=_BG_FRAGMENT_SHADER)
        self._bg_vao = ctx.vertex_array(self._bg_prog, [])
        self._bg_prog["u_hspeed"].value = 0.0
        self._bg_prog["u_vspeed"].value = 0.0
        self._bg_prog["u_netx"].value = 0.0
        self._bg_prog["u_nety"].value = 0.0
        self._bg_prog["u_travelmax"].value = 0.0
        # Translucent backing plane (open cells tint but show the shader through).
        self.backplane_alpha = 0.42
        self._backplane = _MeshSlot(ctx, self._make_vao)
        self._static = _MeshSlot(ctx, self._make_vao)
        self._dynamic = _MeshSlot(ctx, self._make_vao)
        # Speed-coloured ninja trail (translucent, rebuilt per frame).
        self.trail_alpha = 0.8
        self._trail = _MeshSlot(ctx, self._make_vao)
        # Speed-driven ninja motion-blur echoes (translucent, rebuilt per frame).
        self._ghosts = _MeshSlot(ctx, self._make_vao)
        self._ghost_stride = 0            # verts per echo copy
        self._ghost_alphas: list[float] = []
        # Translucent neighbour "slabs" for stacked multi-level mode: the previous
        # and next levels shown as faint depth-offset panes around the live level.
        # Each is uploaded once (on level change) and drawn blended every frame.
        self._slabs: list[dict] = []
        # Teleport progress 0..1: dissolves the live level and fades the upcoming
        # (far) slab up to opaque as the camera dives through into it.
        self.teleport_k = 0.0
        # Always-on-top overlay (the warp streak): drawn last, depth test off.
        self._overlay = _MeshSlot(ctx, self._make_vao)
        self._trail_uniform = np.zeros((TRAIL_MAX, 4), dtype="f4")

    def _make_vao(self, vbo: moderngl.Buffer) -> moderngl.VertexArray:
        return self.ctx.vertex_array(
            self.prog,
            [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
        )

    def set_static_mesh(self, verts: np.ndarray) -> None:
        """Upload the static level mesh (interleaved pos/normal/color, Nx9)."""
        self._static.write(verts)

    def set_backplane(self, verts: np.ndarray) -> None:
        """Upload the translucent backing-plane quad (interleaved Nx9)."""
        if verts is None or len(verts) == 0:
            return
        self._backplane.write(verts)

    def set_dynamic_mesh(self, verts: np.ndarray) -> None:
        """Replace the per-frame dynamic mesh (entities + ninja)."""
        self._dynamic.write(verts)

    def set_dash_trail(self, pts) -> None:
        """Ninja footfall history for colouring road-dash lines it ran over.

        ``pts`` is an ordered list of (world_x, surface_y, speed) oldest->newest;
        the newest entries are freshest (least faded). Packed into the u_trail
        uniform array (padded to TRAIL_MAX).
        """
        pts = list(pts)
        n = min(len(pts), TRAIL_MAX)
        self.prog["u_ntrail"].value = n
        if n == 0:
            return
        arr = self._trail_uniform
        arr[:] = 0.0
        for i, (x, y, s) in enumerate(pts[-n:]):
            arr[i] = (x, y, s, 0.0)
        self.prog["u_trail"].write(arr)

    def set_trail(self, verts: np.ndarray) -> None:
        """Replace the per-frame ninja trail ribbon (interleaved Nx9)."""
        self._trail.write(verts)

    def set_ninja_ghosts(self, ghosts) -> None:
        """Per-frame ninja motion-blur echoes.

        ``ghosts`` is the ``(verts, verts_per_ghost, alphas)`` tuple from
        ``scene.build_ninja_ghosts`` (or ``None`` to clear). The copies are
        packed farthest-first so they blend back-to-front in one buffer.
        """
        if not ghosts:
            self._ghost_stride = 0
            self._ghost_alphas = []
            self._ghosts.count = 0
            return
        verts, stride, alphas = ghosts
        self._ghost_stride = int(stride)
        self._ghost_alphas = list(alphas)
        self._ghosts.write(verts)

    def set_ghost_slabs(self, slabs) -> None:
        """Register the translucent neighbour levels for stacked mode.

        ``slabs`` is a list of dicts, each with:
            ``static``  : Nx9 tile/decoration mesh (cel-shaded) or None
            ``dynamic`` : Nx9 entity/ninja mesh (flat) or None
            ``alpha``   : blend opacity in [0, 1]
            ``near``    : True if the slab sits between the camera and the live
                          level (drawn last, over everything); False if it sits
                          behind the live level (drawn first).
        The slab meshes carry their own baked z-offset, so they share the live
        level's MVP. Buffers and VAOs from a previous call are released.
        """
        self._release_slabs()
        for s in slabs or []:
            entry = {"alpha": float(s.get("alpha", 0.3)),
                     "near": bool(s.get("near", False)),
                     "static": None, "dynamic": None}
            for key in ("static", "dynamic"):
                verts = s.get(key)
                if verts is None or len(verts) == 0:
                    continue
                slot = _MeshSlot(self.ctx, self._make_vao)
                slot.write(verts)
                entry[key] = slot
            self._slabs.append(entry)

    def _release_slabs(self) -> None:
        for entry in self._slabs:
            for key in ("static", "dynamic"):
                if entry[key] is not None:
                    entry[key].release()
        self._slabs = []

    def _draw_slabs(self, near: bool) -> None:
        """Draw the registered neighbour slabs on the requested side, blended.

        Depth writes are off (translucent panes must not occlude one another or
        the live level) while the depth test stays on so foreground tiles of the
        live level still hide the panes behind them.
        """
        slabs = [s for s in self._slabs if s["near"] == near]
        if not slabs:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.depth_mask = False
        for s in slabs:
            a = float(s["alpha"])
            if not near:                          # upcoming level: fade up to solid
                a += (1.0 - a) * self.teleport_k
            self.prog["alpha"].value = a
            if s["static"]:
                self.prog["flat_mode"].value = 0     # tiles: cel + outline
                s["static"].render(moderngl.TRIANGLES)
            if s["dynamic"]:
                self.prog["flat_mode"].value = 1     # entities/ninja: flat bright
                s["dynamic"].render(moderngl.TRIANGLES)
        self.prog["alpha"].value = 1.0
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    def set_overlay_mesh(self, verts) -> None:
        """Always-on-top mesh (the teleport warp streak), drawn ignoring depth so
        it punches through the level it recedes behind. ``None`` clears it."""
        self._overlay.write(verts)

    def release(self) -> None:
        """Free every GL object this renderer owns.

        Call when tearing down a renderer whose context outlives it (the dev-tool
        panel, a resized window). moderngl will not do it on garbage collection.
        """
        self._release_slabs()
        for slot in (self._backplane, self._static, self._dynamic,
                     self._trail, self._ghosts, self._overlay):
            slot.release()
        release_gl(self._bg_vao, self._bg_prog, self.prog)

    def render(self, mvp: np.ndarray, clear=(1.0, 1.0, 1.0), cam_pos=None,
               time: float = 0.0, hspeed: float = 0.0, vspeed: float = 0.0,
               netx: float = 0.0, nety: float = 0.0, travelmax: float = 0.0,
               main_alpha: float = 1.0) -> None:
        self.ctx.clear(*clear, depth=1.0)
        # Animated background fills the frame first (depth test off, no depth
        # write); the level geometry + backplane then draw over it, so the shader
        # only shows around/outside the (translucent) level footprint.
        if self.bg_anim:
            vp = self.ctx.viewport
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._bg_prog["u_time"].value = float(time)
            self._bg_prog["u_res"].value = (float(vp[2]), float(vp[3]))
            self._bg_prog["u_hspeed"].value = float(hspeed)
            self._bg_prog["u_vspeed"].value = float(vspeed)
            self._bg_prog["u_netx"].value = float(netx)
            self._bg_prog["u_nety"].value = float(nety)
            self._bg_prog["u_travelmax"].value = float(travelmax)
            self._bg_vao.render(moderngl.TRIANGLES, vertices=3)
            self.ctx.enable(moderngl.DEPTH_TEST)
        if cam_pos is not None:
            self.prog["cam_pos"].value = tuple(float(c) for c in cam_pos)
        self.prog["Mvp"].write(np.ascontiguousarray(mvp, dtype="f4"))
        # Neighbour levels sitting BEHIND the live level (stacked mode): drawn
        # first so the opaque live level composites over them.
        self._draw_slabs(near=False)
        # Translucent backplane in the level footprint (over the shader, behind
        # the opaque tiles which overwrite it where they exist).
        if self._backplane:
            self.ctx.enable(moderngl.BLEND)
            self.prog["flat_mode"].value = 1
            self.prog["alpha"].value = float(self.backplane_alpha)
            self._backplane.render(moderngl.TRIANGLES)
            self.prog["alpha"].value = 1.0
            self.ctx.disable(moderngl.BLEND)
        main_blend = main_alpha < 0.999
        if main_blend:
            self.ctx.enable(moderngl.BLEND)
            self.prog["alpha"].value = float(main_alpha)
        if self._static:
            self.prog["flat_mode"].value = 0   # tiles: cel + outline
            self._static.render(moderngl.TRIANGLES)
        if main_blend:
            self.prog["alpha"].value = 1.0
            self.ctx.disable(moderngl.BLEND)
        # Speed trail: translucent, behind the ninja (drawn before the entities).
        if self._trail:
            self.ctx.enable(moderngl.BLEND)
            self.prog["flat_mode"].value = 1
            self.prog["alpha"].value = float(self.trail_alpha)
            self._trail.render(moderngl.TRIANGLES)
            self.prog["alpha"].value = 1.0
            self.ctx.disable(moderngl.BLEND)
        # Ninja motion-blur echoes: translucent copies trailing along velocity,
        # drawn just before the solid ninja. Depth writes are off so the echoes
        # neither occlude the solid ninja nor punch holes in each other, but the
        # depth test stays on so foreground tiles still hide them.
        if self._ghosts and self._ghost_stride and self._ghost_alphas:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.depth_mask = False
            self.prog["flat_mode"].value = 1
            for i, a in enumerate(self._ghost_alphas):
                self.prog["alpha"].value = float(a)
                self._ghosts.render(moderngl.TRIANGLES,
                                    vertices=self._ghost_stride,
                                    first=i * self._ghost_stride)
            self.prog["alpha"].value = 1.0
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)
        if self._dynamic:
            if main_blend:
                self.ctx.enable(moderngl.BLEND)
                self.prog["alpha"].value = float(main_alpha)
            self.prog["flat_mode"].value = 1   # entities/ninja: flat bright
            self._dynamic.render(moderngl.TRIANGLES)
            if main_blend:
                self.prog["alpha"].value = 1.0
                self.ctx.disable(moderngl.BLEND)
        # Neighbour levels sitting between the camera and the live level (stacked
        # mode): drawn last as faint panes over everything.
        self._draw_slabs(near=True)
        # Always-on-top warp streak (teleport): ignores depth so it stays visible
        # even as it recedes behind the (dissolving) live level into the next.
        if self._overlay:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog["flat_mode"].value = 1
            self._overlay.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.BLEND)
