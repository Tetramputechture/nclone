"""Which tile and which side emits the segment at a given location?

Re-walks the table builder's own rule outside the factory so each emission can
be attributed, rather than inferred from the finished segment list.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.tile_definitions import TILE_EDGE_OCCUPANCY  # noqa: E402
from nclone.utils.tile_segment_factory import (  # noqa: E402
    SIDE_SEGMENT_TEMPLATES,
)

demo = sys.argv[1]
tx, ty = float(sys.argv[2]), float(sys.argv[3])
dec = NppAttractDecoder()
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(
    list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
tiles = h.sim.tile_dic

NAMES = ["left", "bottom", "right", "top"]
DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]

print("segments touching y=%g between x=%g and %g:" % (ty, tx - 30, tx + 30))
for (x, y), tile in sorted(tiles.items()):
    occ = TILE_EDGE_OCCUPANCY.get(tile, (0,) * 8)
    if not any(occ):
        continue
    ox, oy = 24 * x, 24 * y
    for k in range(4):
        dx, dy = DELTA[k]
        nb = tiles.get((x + dx, y + dy), 0)
        nocc = TILE_EDGE_OCCUPANCY.get(nb, (0,) * 8)
        exp_a = occ[2 * k] and not nocc[(2 * k + 4) & 7]
        exp_b = occ[2 * k + 1] and not nocc[(2 * k + 5) & 7]
        tpl = SIDE_SEGMENT_TEMPLATES[k]
        seg = tpl[2] if (exp_a and exp_b) else tpl[0] if exp_a else tpl[1] if exp_b else None
        if seg is None:
            continue
        p1 = (ox + seg[0][0], oy + seg[0][1])
        p2 = (ox + seg[1][0], oy + seg[1][1])
        if p1[1] != ty or p2[1] != ty:
            continue
        if max(p1[0], p2[0]) < tx - 30 or min(p1[0], p2[0]) > tx + 30:
            continue
        print("   tile %-3d at cell (%d,%d) side %-6s -> (%g,%g)->(%g,%g)  len=%g"
              % (tile, x, y, NAMES[k], p1[0], p1[1], p2[0], p2[1],
                 abs(p2[0] - p1[0])))
        print("        occ=%s  neighbour %d occ=%s  exp=(%s,%s)"
              % (occ, nb, nocc, bool(exp_a), bool(exp_b)))
