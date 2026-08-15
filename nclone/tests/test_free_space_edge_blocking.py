"""Free-space flood must not leak through solid tile collision segments.

Regression for the reported bug: cost-to-go fields / paths passing THROUGH solid
tiles, specifically the thin orthogonal grid-edge "walls" on the boundary of
slopes / steep slopes / quarter-pipes that the center-sampled node-free mask
cannot see (both adjacent cell centers are non-solid, yet a solid wall sits
between them).

These tests exercise the pure segment helpers and the ``_build_blocked_edges`` /
``_step_blocked`` machinery directly with a synthetic level, so they need neither
the native simulator nor torch.
"""

from __future__ import annotations

import numpy as np

from nclone.value_function import free_space_distance as fs
from nclone.value_function.free_space_distance import FreeSpaceDistanceSource


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_tile_segments_cover_all_types():
    assert fs._tile_local_solid_segments(0) == ()          # empty: no collision
    assert len(fs._tile_local_solid_segments(1)) == 8      # full solid: perimeter
    assert (0, 24, 24, 0) in fs._tile_local_solid_segments(6)  # slope diagonal
    # Quarter pipe 14: 4 FULL flat faces + the arc. Its other four faces are
    # PARTIAL (the arc meets the border there) and must NOT become straight
    # walls -- see test_partial_faces_are_not_emitted_as_walls.
    assert len(fs._tile_local_solid_segments(14)) == 4 + fs._ARC_CHORDS
    assert fs._tile_local_solid_segments(34) == ()         # glitched: empty (as sim)
    # Every real tile type 1..33 has at least one solid segment.
    for tt in range(1, 34):
        assert fs._tile_local_solid_segments(tt), tt


def test_straight_walls_are_exactly_the_full_flat_faces():
    """Straight grid-edge walls come from ternary ``2`` only.

    ``TILE_GRID_EDGE_MAP`` collapses the ternary table's ``1`` (a PARTIAL face,
    where a slope or curve meets the tile border) into the same ``1`` as a full
    flat face. Emitting the partial ones fabricated a solid 12px wall across the
    open part of every sloped/curved border -- for tiles 10-13 the collapsed map
    is byte-identical to tile 1, so quarter circles became sealed boxes.
    """
    from nclone.tile_definitions import TILE_EDGE_GRID_TERNARY

    for tt in range(1, 34):
        segs = set(fs._tile_local_solid_segments(tt))
        H, V = TILE_EDGE_GRID_TERNARY[tt]
        for idx, seg in enumerate(fs._GRID_EDGE_SEGS):
            axis, rem = divmod(idx, 6)
            outer, inner = divmod(rem, 2)
            expect_wall = (H, V)[axis][outer + 3 * inner] == 2
            assert (seg in segs) is expect_wall, (tt, idx, seg)


def test_partial_faces_are_not_emitted_as_walls():
    """No emitted straight wall may have free space on BOTH sides along its
    whole length -- such a wall is fabricated and seals a passable border.

    This is the property the quarter-circle regression violated. It is checked
    against the independent sub-cell solidity predicate, not the edge tables.
    """
    from nclone.graph.reachability.graph_builder import _check_subnode_validity_simple

    def solid_at(px, py):
        px = min(max(px, 0), 23)
        py = min(max(py, 0), 23)
        return not _check_subnode_validity_simple(tt, int(px), int(py))

    for tt in range(1, 34):
        for (x1, y1, x2, y2) in fs._tile_local_solid_segments(tt):
            if x1 != x2 and y1 != y2:
                continue                      # diagonal/arc chord, not a face
            for k in range(1, 8):             # sample along the wall
                t = k / 8.0
                mx, my = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                # Probe 3px either side, i.e. the neighbouring 6px sub-cells.
                if x1 == x2:
                    a, b = (mx - 3, my), (mx + 3, my)
                else:
                    a, b = (mx, my - 3), (mx, my + 3)
                # Skip samples that fall outside the tile on both probes.
                if not (0 <= a[0] < 24 and 0 <= a[1] < 24):
                    continue
                if not (0 <= b[0] < 24 and 0 <= b[1] < 24):
                    continue
                assert solid_at(*a) or solid_at(*b), (
                    f"tile {tt}: wall {(x1, y1, x2, y2)} is free on both sides "
                    f"at {(mx, my)} -- fabricated, seals a passable border"
                )


def test_tile8_node_leak_fixed():
    """Tile 8 (inverted 45 slope) is solid where x+y>24. The old predicate OR'd
    ``pixel_y == pixel_x``, wrongly marking the solid sub-cell centers (15,15) and
    (21,21) as free -> the cost-to-go leaked through the solid triangle."""
    from nclone.graph.reachability.graph_builder import _check_subnode_validity_simple
    assert _check_subnode_validity_simple(8, 15, 15) is False  # solid (30 > 24)
    assert _check_subnode_validity_simple(8, 21, 21) is False  # solid (42 > 24)
    # The traversable (top-left) side is unaffected.
    assert _check_subnode_validity_simple(8, 3, 3) is True     # free (6 < 24)


def test_seg_crosses():
    # Horizontal step through a vertical wall x=10 (y 0..20).
    assert fs._seg_crosses(5, 10, 15, 10, 10, 0, 10, 20) is True
    # Step that stops short of the wall.
    assert fs._seg_crosses(5, 10, 9, 10, 10, 0, 10, 20) is False
    # Parallel step offset from the wall (real cell-center geometry: centers at
    # odd*3, walls at multiples of 12, so steps never run collinear along a wall).
    assert fs._seg_crosses(3, 9, 3, 15, 12, 0, 12, 24) is False
    # A diagonal step that crosses the wall in its interior IS blocked.
    assert fs._seg_crosses(9, 9, 15, 15, 12, 0, 12, 24) is True


# --------------------------------------------------------------------------
# Blocked-edge construction on a synthetic level (one full-solid tile at (5,5))
# --------------------------------------------------------------------------
def _blocked_for_single_solid_tile(rtx=5, rty=5):
    src = FreeSpaceDistanceSource.__new__(FreeSpaceDistanceSource)
    tiles = [0] * (42 * 23)
    tiles[rtx + rty * 42] = 1  # full solid
    free = np.ones((fs._W, fs._H), bool)  # mark everything free to probe wall edges
    src._build_blocked_edges(tiles, free)
    return src


def test_no_leak_through_solid_tile_edges():
    src = _blocked_for_single_solid_tile()
    # Tile (5,5) origin in global px = ((5+1)*24, (5+1)*24) = (144,144); it spans
    # x in [144,168], y in [144,168]. Left wall x=144, top wall y=144.
    # Horizontal step crossing the LEFT wall: cell ix=23 (center 141) -> ix=24 (147),
    # blocked for iy whose center falls inside the tile's y-span (144..168) = iy 24..27.
    assert src._block_r[23, 25]      # crosses left wall of the solid tile
    assert not src._block_r[23, 10]  # far from the tile: open
    assert not src._block_r[23, 23]  # just above the wall span (y=141): open
    # Vertical step crossing the TOP wall: iy=23 (141) -> iy=24 (147), blocked for
    # ix whose center is inside the tile's x-span = ix 24..27.
    assert src._block_u[25, 23]
    assert not src._block_u[25, 10]


def test_diagonal_into_solid_corner_blocked():
    src = _blocked_for_single_solid_tile()
    # Diagonal step (23,23) -> (24,24) squeezes into the solid tile's top-left
    # corner; must be blocked (via the surrounding-ortho-wall squeeze fold).
    assert src._block_dr[23, 23]


def test_step_blocked_dispatch_is_symmetric():
    src = _blocked_for_single_solid_tile()
    # Forward right step blocked == reverse left step from the neighbour.
    assert src._step_blocked(23, 25, 1, 0) == bool(src._block_r[23, 25])
    assert src._step_blocked(24, 25, -1, 0) == bool(src._block_r[23, 25])
    # Diagonal DR forward (23,23)->(24,24) == reverse (24,24)->(23,23).
    assert src._step_blocked(23, 23, 1, 1) == bool(src._block_dr[23, 23])
    assert src._step_blocked(24, 24, -1, -1) == bool(src._block_dr[23, 23])


def test_open_level_blocks_nothing():
    src = FreeSpaceDistanceSource.__new__(FreeSpaceDistanceSource)
    tiles = [0] * (42 * 23)  # fully empty
    src._build_blocked_edges(tiles, np.ones((fs._W, fs._H), bool))
    assert not src._block_r.any()
    assert not src._block_u.any()
    assert not src._block_dr.any()
    assert not src._block_ur.any()
