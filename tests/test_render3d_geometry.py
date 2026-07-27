"""render3d mesh construction: footprints, extrusion, and per-frame geometry.

These tests need no GL context. They cover the parts most likely to break
silently: the tile footprint table (which must stay in step with the 2D cairo
renderer), the vertex layout every buffer upload assumes, and the sim-reading
seam in ``scene``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from nclone.render3d import geometry as G
from nclone.render3d import scene as S

TILE = G.TILE


# --- tile footprints -------------------------------------------------------

def test_empty_and_glitch_tiles_have_no_footprint():
    """Type 0 is empty; 34-37 are the glitch tiles the sim treats as non-solid."""
    for t in (0, 34, 35, 36, 37):
        assert G.tile_polygon(t) is None


def test_full_tiles_cover_the_whole_cell():
    for t in (1, 38, 100):
        poly = G.tile_polygon(t)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert (min(xs), max(xs)) == (0, TILE)
        assert (min(ys), max(ys)) == (0, TILE)


@pytest.mark.parametrize("t", range(1, 34))
def test_every_solid_tile_has_a_usable_polygon(t):
    poly = G.tile_polygon(t)
    assert poly is not None and len(poly) >= 3
    for x, y in poly:
        assert -1e-9 <= x <= TILE + 1e-9
        assert -1e-9 <= y <= TILE + 1e-9


def test_half_tiles_cover_half_the_cell_area():
    for t in (2, 3, 4, 5):
        assert _poly_area(G.tile_polygon(t)) == pytest.approx(TILE * TILE / 2)


def test_slope_triangles_cover_half_the_cell_area():
    for t in (6, 7, 8, 9):
        assert _poly_area(G.tile_polygon(t)) == pytest.approx(TILE * TILE / 2)


def test_convex_and_concave_arcs_are_complementary():
    """A quarter circle plus its matching quarter pipe should tile the cell.

    Both are tessellated with the same segment count, so the two areas add up to
    the full cell area up to the shared polygonal approximation.
    """
    convex = _poly_area(G.tile_polygon(10))
    concave = _poly_area(G.tile_polygon(14))
    assert convex + concave == pytest.approx(TILE * TILE, rel=1e-6)
    # The convex pie slice is close to a true quarter disc.
    assert convex == pytest.approx(math.pi * TILE * TILE / 4, rel=0.02)


def test_tile_polygon_is_cached_and_read_only():
    a = G.tile_polygon(10)
    assert a is G.tile_polygon(10)
    assert isinstance(a, tuple)


# --- mesh building ---------------------------------------------------------

def test_mesh_builder_emits_the_declared_vertex_layout():
    mb = G.MeshBuilder()
    mb.add_tri((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.5, 0.75))
    verts = mb.build()
    assert verts.shape == (3, G.FLOATS_PER_VERTEX)
    assert verts.dtype == np.float32
    # Normal faces +z for a counter-clockwise triangle in the xy plane.
    assert verts[0, 3:6] == pytest.approx([0.0, 0.0, 1.0])
    assert verts[0, 6:9] == pytest.approx([0.25, 0.5, 0.75])


def test_empty_mesh_builder_has_the_right_shape():
    verts = G.MeshBuilder().build()
    assert verts.shape == (0, G.FLOATS_PER_VERTEX)
    assert verts.dtype == np.float32


def test_mesh_builder_normals_are_unit_length():
    mb = G.MeshBuilder()
    mb.add_box(0, 0, 0, 3, 5, 7, (1.0, 0.0, 0.0))
    mb.add_sphere((1, 2, 3), 4.0, (0.0, 1.0, 0.0))
    mb.add_segment((0, 0, 0), (10, 3, -2), 0.5, (0.0, 0.0, 1.0))
    verts = G.MeshBuilder.build(mb)
    lengths = np.linalg.norm(verts[:, 3:6], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_degenerate_segment_emits_nothing():
    mb = G.MeshBuilder()
    mb.add_segment((1, 1, 1), (1, 1, 1), 0.5, (0, 0, 0))
    assert len(mb.build()) == 0


def test_tile_mesh_extrudes_between_zero_and_depth():
    verts = G.build_tile_mesh({(0, 0): 1, (1, 0): 6, (2, 0): 10})
    assert len(verts) > 0
    assert verts[:, 2].min() == pytest.approx(0.0)
    assert verts[:, 2].max() == pytest.approx(G.DEPTH)


def test_tile_mesh_flips_y_into_a_y_up_world():
    """Sim +y is down, so cell (0, 1) must sit below cell (0, 0) in world y."""
    top = G.build_tile_mesh({(0, 0): 1})
    below = G.build_tile_mesh({(0, 1): 1})
    assert below[:, 1].max() <= top[:, 1].min() + 1e-6


def test_empty_tile_dict_builds_an_empty_mesh():
    assert len(G.build_tile_mesh({})) == 0


def test_arc_tiles_get_smooth_per_vertex_normals():
    """A quarter circle's curved wall varies its normal across the arc.

    A flat prism only has a handful of distinct normals (the caps share one,
    since both are wound the same way and the shader lights two-sided).
    """
    flat_normals = {tuple(n) for n in np.round(G.build_tile_mesh({(0, 0): 1})[:, 3:6], 4)}
    arc_normals = {tuple(n) for n in np.round(G.build_tile_mesh({(0, 0): 10})[:, 3:6], 4)}
    assert len(flat_normals) == 5           # 4 sides + one shared cap normal
    assert len(arc_normals) > len(flat_normals)
    assert len(arc_normals) >= G._ARC_SEGMENTS


def test_scatter_blocks_are_cached_and_immutable():
    a = G.build_scatter_blocks()
    b = G.build_scatter_blocks()
    assert a is b
    assert not a.flags.writeable


def test_level_mesh_composes_its_parts():
    tiles = {(c, 12): 1 for c in range(1, 43)}
    bare = G.build_level_mesh(tiles, scatter=False, guardrails=False)
    rails = G.build_level_mesh(tiles, scatter=False, guardrails=True)
    everything = G.build_level_mesh(tiles, scatter=True, guardrails=True)
    assert len(bare) == len(G.build_tile_mesh(tiles))
    assert len(rails) > len(bare)          # a floor row gets dash markings
    assert len(everything) == len(rails) + len(G.build_scatter_blocks())


def test_guardrails_only_hover_over_exposed_up_facing_surfaces():
    floor = {(c, 12): 1 for c in range(1, 43)}
    assert len(G.build_guardrail_mesh(floor)) > 0
    # A solid block has its top buried by the tile above it, so no rails.
    buried = {(5, 11): 1, (5, 12): 1}
    top_only = G.build_guardrail_mesh(buried)
    assert len(top_only) < len(G.build_guardrail_mesh({(5, 12): 1})) * 2


# --- per-frame scene geometry ---------------------------------------------

def _fake_sim(entities=(), ninja=None):
    by_type: dict[int, list] = {}
    for e in entities:
        by_type.setdefault(e.type, []).append(e)
    return SimpleNamespace(entity_dic=by_type, ninja=ninja, tile_dic={})


def _entity(type_, x=100.0, y=200.0, radius=None, semi=None, active=True):
    e = SimpleNamespace(type=type_, xpos=x, ypos=y, active=active)
    if radius is not None:
        e.RADIUS = radius
    if semi is not None:
        e.SEMI_SIDE = semi
    return e


def _ninja(x=50.0, y=60.0, xspeed=0.0, yspeed=0.0, state=1):
    return SimpleNamespace(xpos=x, ypos=y, xspeed=xspeed, yspeed=yspeed,
                           state=state, facing=1, bones=None)


def test_dynamic_mesh_skips_inactive_entities():
    live = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=6.0)]))
    dead = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=6.0, active=False)]))
    assert len(live) > 0
    assert len(dead) == 0


def test_dynamic_mesh_has_the_shared_vertex_layout():
    verts = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=6.0)]))
    assert verts.shape[1] == G.FLOATS_PER_VERTEX
    assert verts.dtype == np.float32


def test_exit_door_disappears_once_the_level_is_won():
    door = [_entity(3, radius=12.0)]
    playing = S.build_dynamic_mesh(_fake_sim(door, _ninja(state=1)))
    won = S.build_dynamic_mesh(_fake_sim(door, _ninja(state=8)))
    assert len(won) < len(playing)


def test_entities_render_at_their_physical_radius():
    small = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=4.0)]))
    large = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=8.0)]))
    assert len(small) == len(large)        # same template, different scale
    assert _extent(large) == pytest.approx(2 * _extent(small), rel=1e-3)


def test_square_entities_use_the_box_template():
    sphere = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=6.0)]))
    box = S.build_dynamic_mesh(_fake_sim([_entity(20, semi=9.0)]))
    assert len(box) == 36                  # 12 triangles
    assert len(sphere) != len(box)


def test_mines_use_the_spiky_template_and_read_red():
    mine = S.build_dynamic_mesh(_fake_sim([_entity(1, radius=4.0)]))
    gold = S.build_dynamic_mesh(_fake_sim([_entity(2, radius=6.0)]))
    assert len(mine) != len(gold)
    assert np.allclose(mine[0, 6:9], S._MINE_RED)
    assert np.allclose(gold[0, 6:9], S._VIBRANT[2])


def test_ninja_ghosts_absent_at_rest_and_present_at_speed():
    assert S.build_ninja_ghosts(_ninja(), speed01=0.0) is None
    assert S.build_ninja_ghosts(None, speed01=1.0) is None
    assert S.build_ninja_ghosts(_ninja(xspeed=3.0), speed01=1.0, strength=0.0) is None
    verts, stride, alphas = S.build_ninja_ghosts(_ninja(xspeed=3.0), speed01=1.0)
    assert len(alphas) == S.NINJA_GHOSTS
    assert len(verts) == stride * S.NINJA_GHOSTS
    # Packed farthest-first, so opacity rises toward the end of the buffer.
    assert alphas == sorted(alphas)


def test_ninja_ghosts_trail_opposite_the_velocity():
    nj = _ninja(xspeed=3.0)
    verts, stride, _ = S.build_ninja_ghosts(nj, speed01=1.0)
    solid_x = nj.xpos
    assert verts[:, 0].mean() < solid_x     # smear lags behind rightward motion


def test_trail_is_empty_until_pushed_and_grows_after():
    trail = S.NinjaTrail()
    assert len(trail.build()) == 0
    for _ in range(5):
        trail.push(_ninja(xspeed=2.0))
    assert len(trail.build()) > 0
    trail.reset()
    assert len(trail.build()) == 0


def test_trail_survives_a_degenerate_sample_count():
    trail = S.NinjaTrail(samples=1)
    trail.push(_ninja(xspeed=2.0))
    assert trail.samples >= 2
    assert len(trail.build()) > 0


# --- helpers ---------------------------------------------------------------

def _poly_area(poly) -> float:
    """Absolute shoelace area of a closed polygon."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1]
                   - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def _extent(verts) -> float:
    return float(verts[:, 0].max() - verts[:, 0].min())
