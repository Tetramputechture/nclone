"""Attribute a segment by calling the REAL builder with a logging container.

The previous attempt reimplemented the side rule outside the factory and did not
reproduce the offending face, which only proved the reimplementation differed.
This drives `_build_segments_from_tables` itself.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.utils.tile_segment_factory import TileSegmentFactory  # noqa: E402
from nclone.constants.physics_constants import (  # noqa: E402
    FULL_MAP_WIDTH, FULL_MAP_HEIGHT,
)

demo = sys.argv[1]
tx, ty = float(sys.argv[2]), float(sys.argv[3])

dec = NppAttractDecoder()
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(
    list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
tiles = h.sim.tile_dic

current = {"cell": None}


class Logging(list):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def append(self, seg):
        px = getattr(seg, "px", None)
        if px is not None and seg.py == 0 and seg.y1 == ty:
            lo, hi = sorted((seg.x1, seg.x2))
            if hi >= tx - 30 and lo <= tx + 30:
                print("   cell %-9s tile %-3d -> (%g,%g)->(%g,%g) len=%g"
                      % (str(self.cell), tiles.get(self.cell, 0),
                         seg.x1, seg.y1, seg.x2, seg.y2, abs(seg.px)))
        super().append(seg)


segdic = {}
for x in range(FULL_MAP_WIDTH):
    for y in range(FULL_MAP_HEIGHT):
        segdic[(x, y)] = Logging((x, y))

print("horizontal emissions at y=%g near x=%g, by owning cell:" % (ty, tx))
TileSegmentFactory._build_segments_from_tables(
    TileSegmentFactory._tile_accessor(tiles), segdic)

cx, cy = int(tx // 24), int(ty // 24)
print("\nborder-aware tile ids around cell (%d,%d):" % (cx, cy))
acc = TileSegmentFactory._tile_accessor(tiles)
for yy in range(cy - 2, cy + 2):
    row = []
    for xx in range(cx - 2, cx + 2):
        row.append("%d/%d" % (
            tiles.get((xx, yy), 0),
            TileSegmentFactory._border_aware_tile(
                acc, xx, yy, FULL_MAP_WIDTH, FULL_MAP_HEIGHT)))
    print("   y=%-3d  %s" % (yy, "  ".join("%-7s" % r for r in row)))
