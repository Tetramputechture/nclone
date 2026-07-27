"""Mesh construction: turn sim state (tiles, entities, ninja) into 3D geometry.

Coordinate mapping (sim -> world):
    world_x = sim_x
    world_y = -sim_y          # sim +y is DOWN; flip for a Y-up 3D scene
    world_z = extruded depth  # the play plane sits at z in [0, DEPTH]

Full tile rendering: each tile type is turned into its 2D footprint polygon and
extruded along z into a prism. Slopes become wedges; quarter circles and quarter
pipes become tessellated arcs. Winding is irrelevant, because the shader lights
two-sided (normals are flipped to face the viewer), so only the footprints have
to be right.

The footprints follow ``shared_tile_renderer.draw_complex_tile_to_cairo`` with
one deliberate difference: types 34-37 are the glitch tiles, which the sim
treats as empty (see ``tile_definitions``), so they render as nothing here. The
2D cairo path draws them as full squares.
"""

from __future__ import annotations

import functools
import math

import numpy as np

TILE = 24.0          # tile size in pixels (sim units)
DEPTH = 24.0         # how far tiles are extruded along +z
HALF = TILE / 2.0

# Translucent backing plane behind the level footprint: open cells get a soft
# grey tint but the animated background shader still shows through it.
BACKPLANE_Z = -2.0
BACKPLANE_COLOR = (0.86, 0.86, 0.90)


def build_backplane_mesh(color=BACKPLANE_COLOR) -> np.ndarray:
    """A single quad hugging the 44x25 level footprint (drawn translucent)."""
    bx0, bx1 = -0.5 * TILE, 44.5 * TILE
    by0, by1 = 0.5 * TILE, -(25.5 * TILE)
    mb = MeshBuilder()
    mb.add_quad((bx0, by1, BACKPLANE_Z), (bx1, by1, BACKPLANE_Z),
                (bx1, by0, BACKPLANE_Z), (bx0, by0, BACKPLANE_Z), color)
    return mb.build()

# Vertex format: position(3) + normal(3) + color(3) = 9 floats.
FLOATS_PER_VERTEX = 9

_ARC_SEGMENTS = 8


class MeshBuilder:
    """Accumulates lit, per-vertex-colored triangles.

    Normals are computed per triangle from its own geometry, so callers never
    have to reason about winding.

    Vertices accumulate into a flat Python float list and are converted to an
    ``(N, 9)`` float32 array once in :meth:`build`. Allocating a small numpy
    array per triangle instead costs roughly 35 us each, which dominated level
    load (seconds per level for a tile mesh of ~20k triangles).
    """

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        self._rows: list[float] = []

    def add_tri(self, p0, p1, p2, color) -> None:
        ax, ay, az = float(p0[0]), float(p0[1]), float(p0[2])
        bx, by, bz = float(p1[0]), float(p1[1]), float(p1[2])
        cx, cy, cz = float(p2[0]), float(p2[1]), float(p2[2])
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln > 1e-9:
            nx, ny, nz = nx / ln, ny / ln, nz / ln
        else:
            nx, ny, nz = 0.0, 0.0, 1.0
        r, g, b = color
        self._rows += [ax, ay, az, nx, ny, nz, r, g, b,
                       bx, by, bz, nx, ny, nz, r, g, b,
                       cx, cy, cz, nx, ny, nz, r, g, b]

    def add_quad(self, p0, p1, p2, p3, color) -> None:
        self.add_tri(p0, p1, p2, color)
        self.add_tri(p0, p2, p3, color)

    def add_quad_n(self, p0, p1, p2, p3, n0, n1, n2, n3, color) -> None:
        """Quad with explicit per-vertex normals (for smooth-shaded arc walls)."""
        r, g, b = color
        rows = self._rows
        for p, n in ((p0, n0), (p1, n1), (p2, n2),
                     (p0, n0), (p2, n2), (p3, n3)):
            rows += [float(p[0]), float(p[1]), float(p[2]),
                     float(n[0]), float(n[1]), float(n[2]), r, g, b]

    def add_box(self, x0, y0, z0, x1, y1, z1, color) -> None:
        """Axis-aligned box spanning [x0,x1] x [y0,y1] x [z0,z1]."""
        # 8 corners
        a = (x0, y0, z0); b = (x1, y0, z0); c = (x1, y1, z0); d = (x0, y1, z0)
        e = (x0, y0, z1); f = (x1, y0, z1); g = (x1, y1, z1); h = (x0, y1, z1)
        self.add_quad(e, f, g, h, color)  # +z
        self.add_quad(b, a, d, c, color)  # -z
        self.add_quad(f, b, c, g, color)  # +x
        self.add_quad(a, e, h, d, color)  # -x
        self.add_quad(h, g, c, d, color)  # +y
        self.add_quad(a, b, f, e, color)  # -y

    def add_sphere(self, center, radius, color, lat=10, lon=14) -> None:
        """UV sphere with smooth (radial) normals, centered at ``center``."""
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        r, g, b = color
        rad = float(radius)
        # Vertex grid first: (lat+1) x (lon+1) entries instead of four
        # recomputed trig pairs per quad.
        grid = []
        for i in range(lat + 1):
            theta = math.pi * i / lat          # 0..pi (pole to pole)
            st, ct = math.sin(theta), math.cos(theta)
            row = []
            for j in range(lon + 1):
                phi = 2.0 * math.pi * j / lon
                nx, ny, nz = st * math.cos(phi), ct, st * math.sin(phi)
                row.append((cx + rad * nx, cy + rad * ny, cz + rad * nz,
                            nx, ny, nz, r, g, b))
            grid.append(row)
        rows = self._rows
        for i in range(lat):
            gi, gj = grid[i], grid[i + 1]
            for j in range(lon):
                a, bb = gi[j], gj[j]
                c, d = gj[j + 1], gi[j + 1]
                rows += (*a, *bb, *c, *a, *c, *d)

    def add_segment(self, p0, p1, radius, color) -> None:
        """Thin square-section prism between two 3D points (a 'bone')."""
        ax, ay, az = float(p0[0]), float(p0[1]), float(p0[2])
        bx, by, bz = float(p1[0]), float(p1[1]), float(p1[2])
        dx, dy, dz = bx - ax, by - ay, bz - az
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return
        dx, dy, dz = dx / length, dy / length, dz / length
        # Any reference vector not parallel to the axis gives a valid frame.
        rx, ry, rz = (1.0, 0.0, 0.0) if abs(dz) > 0.9 else (0.0, 0.0, 1.0)
        ux, uy, uz = dy * rz - dz * ry, dz * rx - dx * rz, dx * ry - dy * rx
        ln = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / ln * radius, uy / ln * radius, uz / ln * radius
        vx, vy, vz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
        ln = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
        vx, vy, vz = vx / ln * radius, vy / ln * radius, vz / ln * radius
        corners = ((ux + vx, uy + vy, uz + vz), (-ux + vx, -uy + vy, -uz + vz),
                   (-ux - vx, -uy - vy, -uz - vz), (ux - vx, uy - vy, uz - vz))
        c0 = [(ax + ox, ay + oy, az + oz) for ox, oy, oz in corners]
        c1 = [(bx + ox, by + oy, bz + oz) for ox, oy, oz in corners]
        for i in range(4):
            j = (i + 1) % 4
            self.add_quad(c0[i], c1[i], c1[j], c0[j], color)
        self.add_quad(c0[0], c0[1], c0[2], c0[3], color)
        self.add_quad(c1[3], c1[2], c1[1], c1[0], color)

    def add_prism(self, poly_world, z0, z1, color) -> None:
        """Extrude a world-space 2D polygon [(x, y), ...] between z0 and z1.

        Front/back caps are fan-triangulated from vertex 0 (valid for every tile
        footprint here: triangles, quads, pie slices, and the concave quarter
        pipes are all star-shaped from their first vertex). Sides are quads.
        """
        n = len(poly_world)
        if n < 3:
            return
        x0, y0 = poly_world[0]
        for i in range(1, n - 1):
            ax, ay = poly_world[i]
            bx, by = poly_world[i + 1]
            self.add_tri((x0, y0, z1), (ax, ay, z1), (bx, by, z1), color)  # front cap
            self.add_tri((x0, y0, z0), (ax, ay, z0), (bx, by, z0), color)  # back cap
        for i in range(n):
            ax, ay = poly_world[i]
            bx, by = poly_world[(i + 1) % n]
            self.add_quad((ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1), color)

    def add_prism_smooth(self, poly_world, z0, z1, color, vnormals, arc_edge) -> None:
        """Like ``add_prism`` but arc side walls get smooth per-vertex normals.

        ``vnormals[i]`` is the outward (x, y, 0) world normal for polygon vertex
        i (or None if not on the arc); ``arc_edge[i]`` marks the side quad from
        vertex i to i+1 as a curved segment. Curved segments interpolate the
        radial normals of their two endpoints (killing the faceted look); every
        other wall and both caps stay flat.
        """
        n = len(poly_world)
        if n < 3:
            return
        x0, y0 = poly_world[0]
        for i in range(1, n - 1):
            ax, ay = poly_world[i]
            bx, by = poly_world[i + 1]
            self.add_tri((x0, y0, z1), (ax, ay, z1), (bx, by, z1), color)  # front cap
            self.add_tri((x0, y0, z0), (ax, ay, z0), (bx, by, z0), color)  # back cap
        for i in range(n):
            j = (i + 1) % n
            ax, ay = poly_world[i]
            bx, by = poly_world[j]
            if arc_edge[i] and vnormals[i] is not None and vnormals[j] is not None:
                ni, nj = vnormals[i], vnormals[j]
                self.add_quad_n((ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1),
                                ni, nj, nj, ni, color)
            else:
                self.add_quad((ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1), color)

    def build(self) -> np.ndarray:
        if not self._rows:
            return np.zeros((0, FLOATS_PER_VERTEX), dtype=np.float32)
        return np.asarray(self._rows, dtype=np.float32).reshape(
            -1, FLOATS_PER_VERTEX)


# --- tile type -> cell-local footprint polygon -----------------------------
# Points are in cell-local pixels with the sim convention (y DOWN, 0 = top of
# cell), following shared_tile_renderer.draw_complex_tile_to_cairo.

def _arc(cx, cy, r, a1, a2, segs=_ARC_SEGMENTS):
    return [
        (cx + r * math.cos(a1 + (a2 - a1) * i / segs),
         cy + r * math.sin(a1 + (a2 - a1) * i / segs))
        for i in range(segs + 1)
    ]


@functools.lru_cache(maxsize=64)
def tile_polygon(t: int):
    """Return the solid footprint polygon for a tile id, or None if empty.

    There are only 38 tile ids, so the result is cached and handed back as a
    read-only tuple of ``(x, y)`` points.
    """
    poly = _tile_polygon(t)
    return None if poly is None else tuple(poly)


def _tile_polygon(t: int):
    T = TILE
    if t == 0 or 34 <= t <= 37:
        return None
    if t == 1 or t > 33:
        return [(0, 0), (T, 0), (T, T), (0, T)]
    if t < 6:  # 2-5 half tiles
        dx = T / 2 if t == 3 else 0
        dy = T / 2 if t == 4 else 0
        w = T if t % 2 == 0 else T / 2
        h = T / 2 if t % 2 == 0 else T
        return [(dx, dy), (dx + w, dy), (dx + w, dy + h), (dx, dy + h)]
    if t < 10:  # 6-9 45-degree slope triangles
        dy1 = T if t == 8 else 0
        dx2 = 0 if t == 9 else T
        dy2 = T if t == 9 else 0
        dx3 = 0 if t == 6 else T
        return [(0, dy1), (dx2, dy2), (dx3, T)]
    if t < 14:  # 10-13 quarter circles (convex pie slice)
        dx = T if t in (11, 12) else 0
        dy = T if t in (12, 13) else 0
        a1 = (math.pi / 2) * (t - 10)
        a2 = (math.pi / 2) * (t - 9)
        return [(dx, dy)] + _arc(dx, dy, T, a1, a2)
    if t < 18:  # 14-17 quarter pipes (concave)
        dx1 = T if t in (15, 16) else 0
        dy1 = T if t in (16, 17) else 0
        dx2 = T if t in (14, 17) else 0
        dy2 = T if t in (14, 15) else 0
        a1 = math.pi + (math.pi / 2) * (t - 10)
        a2 = math.pi + (math.pi / 2) * (t - 9)
        return [(dx1, dy1)] + _arc(dx2, dy2, T, a1, a2)
    if t < 22:  # 18-21 mild slope triangles
        dy1 = T if t in (20, 21) else 0
        dy2 = T if t in (20, 21) else 0
        dx3 = T if t in (19, 20) else 0
        return [(0, dy1), (T, dy2), (dx3, T / 2)]
    if t < 26:  # 22-25 raised mild slope quads
        dy1 = T / 2 if t in (23, 24) else 0
        dx2 = 0 if t == 23 else T
        dy2 = T / 2 if t == 25 else 0
        dy3 = (T / 2 if t == 22 else 0) if t < 24 else T
        dx4 = T if t == 23 else 0
        return [(0, dy1), (dx2, dy2), (T, dy3), (dx4, T)]
    if t < 30:  # 26-29 steep slope triangles
        dy1 = T if t in (28, 29) else 0
        dx2 = T if t in (27, 28) else 0
        dx3 = T if t in (27, 28) else 0
        return [(T / 2, dy1), (dx2, 0), (dx3, T)]
    if t < 34:  # 30-33 raised steep slope quads
        dy1 = T if t in (30, 31) else 0
        dx2 = T if t in (31, 33) else 0
        dx3 = T if t in (31, 32) else 0
        dy3 = T if t in (32, 33) else 0
        dx4 = T if t in (30, 32) else 0
        return [(T / 2, dy1), (dx2, T), (dx3, dy3), (dx4, 0)]
    return None


def _arc_center(t: int):
    """(center_local, sign) for arc tiles 10-17; sign=+1 convex, -1 concave."""
    T = TILE
    if t < 14:  # 10-13 convex quarter circle: arc centered on the corner
        dx = T if t in (11, 12) else 0
        dy = T if t in (12, 13) else 0
        return (dx, dy), 1.0
    # 14-17 concave quarter pipe: arc carved around (dx2, dy2)
    dx2 = T if t in (15, 16) else 0
    dy2 = T if t in (16, 17) else 0
    return (dx2, dy2), -1.0


def _arc_side_data(t: int, poly_local):
    """Per-vertex world-space radial normals + arc-edge flags for an arc tile."""
    (clx, cly), sign = _arc_center(t)
    vnormals = []
    for (lx, ly) in poly_local:
        # World normal: local radial mapped through the y-flip (world_y = -sim_y).
        nx = (lx - clx) * sign
        ny = -(ly - cly) * sign
        ln = math.hypot(nx, ny)
        vnormals.append((nx / ln, ny / ln, 0.0) if ln > 1e-6 else None)
    n = len(poly_local)
    # poly = [corner] + arc(segs+1 pts); edges between arc vertices are 1..segs.
    arc_edge = [False] * n
    for i in range(1, n - 1):
        arc_edge[i] = True
    return vnormals, arc_edge


def _cell_hash01(x: int, y: int, seed: int) -> float:
    """Deterministic per-cell value in [0, 1) (stable across runs)."""
    h = (int(x) & 0xFFFF) | ((int(y) & 0xFFFF) << 16)
    h = (h ^ (seed & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = (h * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return (h & 0xFFFFFFFF) / 4294967295.0


@functools.lru_cache(maxsize=4)
def build_scatter_blocks(color=(0.42, 0.42, 0.48), reach: int = 34,
                         falloff: float = 7.5, seed: int = 0x5EED) -> np.ndarray:
    """Type-1 solid 3D blocks scattered around the level footprint.

    Blocks radiate outward from the 44x25 play area on the tile lattice, with
    placement probability decaying (``exp(-dist/falloff)``) so density thins with
    distance. Each block gets a small random height (z-extrusion) and shade for a
    lively 3D field.

    The result depends only on the arguments, not on the level, so it is cached
    and returned read-only. Callers concatenate or copy it rather than mutating.
    """
    mb = MeshBuilder()
    lx0, lx1, ly0, ly1 = 0, 43, 0, 24
    for cy in range(ly0 - reach, ly1 + reach + 1):
        for cx in range(lx0 - reach, lx1 + reach + 1):
            if lx0 - 1 <= cx <= lx1 + 1 and ly0 - 1 <= cy <= ly1 + 1:
                continue  # keep the play area (and a 1-cell gap) clear
            dx = max(lx0 - cx, cx - lx1, 0)
            dy = max(ly0 - cy, cy - ly1, 0)
            dist = math.hypot(dx, dy)
            if _cell_hash01(cx, cy, seed) >= math.exp(-dist / falloff):
                continue
            r1 = _cell_hash01(cx, cy, seed + 101)
            r2 = _cell_hash01(cx, cy, seed + 202)
            shade = 0.82 + 0.34 * r1
            col = (color[0] * shade, color[1] * shade, color[2] * shade)
            world = [(cx * TILE + lx, -(cy * TILE + ly))
                     for (lx, ly) in ((0, 0), (TILE, 0), (TILE, TILE), (0, TILE))]
            mb.add_prism(world, 0.0, DEPTH * (0.7 + 1.1 * r2), col)
    out = mb.build()
    out.flags.writeable = False
    return out


# --- Floating guardrails ---------------------------------------------------
# Segmented rails that hover a quarter-tile above the *exposed, up-facing* top
# surfaces of solid tiles (ground + slopes), offset along that surface's normal.
# Walls and ceilings get none; only surfaces whose world normal points up. The
# rails are broken into dashes so they stream past as motion markers, reading as
# a speed cue (like road lane markings). Built once as static geometry, phased in
# global tangent space so dashes run continuously across a row of floor tiles.

GUARDRAIL_COLOR = (0.27, 0.27, 0.31)   # the tile grey (0.42,0.42,0.48), a bit darker
GUARDRAIL_LIFT = TILE * 0.25           # hover a quarter tile above the surface
GUARDRAIL_RADIUS = 0.75                # rail bar half-thickness in px (thin)
GUARDRAIL_DASH = 8.0                   # lit dash length along the surface (px)
GUARDRAIL_GAP = 6.0                    # gap between dashes (px)
GUARDRAIL_Z = DEPTH * 0.12             # sit flush against the back edge (z=0 face)
_GUARDRAIL_UP_MIN = 0.30               # min world normal.y to count as ground/slope
_GUARDRAIL_PROBE = 2.0                 # how far past the edge to test for burial (px)


def _point_in_poly(px: float, py: float, poly) -> bool:
    """Even-odd point-in-polygon test (cell-local sim coords)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _sim_point_solid(tile_dic, gx: float, gy: float) -> bool:
    """True if global sim point (gx, gy) falls inside a solid tile footprint."""
    cx = int(math.floor(gx / TILE))
    cy = int(math.floor(gy / TILE))
    t = tile_dic.get((cx, cy))
    if t is None:
        return False
    poly = tile_polygon(t)
    if not poly:
        return False
    return _point_in_poly(gx - cx * TILE, gy - cy * TILE, poly)


def build_guardrail_mesh(tile_dic: dict[tuple[int, int], int],
                         color=GUARDRAIL_COLOR) -> np.ndarray:
    """Segmented rails floating above exposed ground/slope tile surfaces."""
    mb = MeshBuilder()
    period = GUARDRAIL_DASH + GUARDRAIL_GAP
    zc = GUARDRAIL_Z
    for (cx, cy), tile_id in tile_dic.items():
        poly = tile_polygon(tile_id)
        if not poly:
            continue
        ox, oy = cx * TILE, cy * TILE
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            length = math.hypot(ex, ey)
            if length < 1e-6:
                continue
            # Outward sim-space edge normal: pick the side that leaves the polygon.
            nx, ny = ey / length, -ex / length
            mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
            if _point_in_poly(mx + nx * 0.5, my + ny * 0.5, poly):
                nx, ny = -nx, -ny
            # Only up-facing surfaces (world_y = -sim_y, so up means sim ny < 0).
            if -ny < _GUARDRAIL_UP_MIN:
                continue
            # Skip surfaces buried against an adjacent solid tile (not exposed).
            if _sim_point_solid(tile_dic, ox + mx + nx * _GUARDRAIL_PROBE,
                                oy + my + ny * _GUARDRAIL_PROBE):
                continue
            # Lift the edge along the world normal and dash it in world space.
            wnx, wny = nx, -ny                       # world normal (unit, up-ish)
            aw = (ox + ax + wnx * GUARDRAIL_LIFT, -(oy + ay) + wny * GUARDRAIL_LIFT)
            twx, twy = ex / length, -ey / length     # unit tangent (world)
            g0 = aw[0] * twx + aw[1] * twy           # start pos along the infinite line
            k0 = int(math.floor(g0 / period))
            k1 = int(math.floor((g0 + length) / period))
            for k in range(k0, k1 + 1):
                d0 = max(g0, k * period)
                d1 = min(g0 + length, k * period + GUARDRAIL_DASH)
                if d1 - d0 < 0.5:
                    continue
                p0 = (aw[0] + twx * (d0 - g0), aw[1] + twy * (d0 - g0), zc)
                p1 = (aw[0] + twx * (d1 - g0), aw[1] + twy * (d1 - g0), zc)
                mb.add_segment(p0, p1, GUARDRAIL_RADIUS, color)
    return mb.build()


def build_level_mesh(tile_dic: dict[tuple[int, int], int],
                     scatter: bool = True, guardrails: bool = True) -> np.ndarray:
    """The full static mesh for a level: tiles, scatter blocks, guardrails.

    This is everything uploaded once per level, as a single interleaved Nx9
    array. Both the single-level app and stacked mode build slabs through here.
    """
    parts = [build_tile_mesh(tile_dic)]
    if scatter:
        parts.append(build_scatter_blocks())
    if guardrails:
        parts.append(build_guardrail_mesh(tile_dic))
    parts = [p for p in parts if len(p)] or parts[:1]
    return np.concatenate(parts, axis=0)


def build_tile_mesh(tile_dic: dict[tuple[int, int], int],
                    color=(0.42, 0.42, 0.48)) -> np.ndarray:
    """Build the static extruded level mesh from ``sim.tile_dic`` (44x25 grid)."""
    mb = MeshBuilder()
    for (cx, cy), tile_id in tile_dic.items():
        poly = tile_polygon(tile_id)
        if not poly:
            continue
        world = [(cx * TILE + lx, -(cy * TILE + ly)) for (lx, ly) in poly]
        if 10 <= tile_id <= 17:  # quarter circles / pipes -> smooth arc walls
            vnormals, arc_edge = _arc_side_data(tile_id, poly)
            mb.add_prism_smooth(world, 0.0, DEPTH, color, vnormals, arc_edge)
        else:
            mb.add_prism(world, 0.0, DEPTH, color)
    return mb.build()
