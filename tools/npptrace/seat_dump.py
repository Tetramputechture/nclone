"""Dump every candidate segment the seating search sees for one laser."""
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.physics import gather_segments_from_region  # noqa: E402
from nclone.entity_classes.entity_dual_laser import EntityDualLaser  # noqa: E402

_orig = EntityDualLaser._seat_on_surface


def _seat(self):
    self._spawn_xy = (self.xpos, self.ypos)
    return _orig(self)


EntityDualLaser._seat_on_surface = _seat

demo = sys.argv[1]
idxs = [int(a) for a in sys.argv[2].split(",")]
cap = Capture(os.path.join(os.environ.get("NCLONE_TRACES",
    "/home/tetra/projects/nclone/tools/npptrace/out9"), demo))
p = next(q for q in cap.pools if q["i"] == 21)
dec = NppAttractDecoder()
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(
    list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
sim = h.sim
lasers = [e for lst in sim.entity_dic.values() for e in lst
          if getattr(e, "type", None) == 23]
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
rec = cap.frame(1)

for i in idxs:
    e = lasers[i]
    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
    g = (struct.unpack_from("<d", b, 0x68)[0], struct.unpack_from("<d", b, 0x70)[0])
    ox, oy = e._spawn_xy
    print("=== laser %d  spawn=(%.2f, %.2f)  orientation=%s  port_n=(%.1f, %.1f)  game_n=(%.1f, %.1f)"
          % (i, ox, oy, getattr(e, "orientation", "?"),
             e.xnormal, e.ynormal, g[0], g[1]))
    print("    tiles around spawn (cell %d,%d):" % (ox // 24, oy // 24))
    cx, cy = int(ox // 24), int(oy // 24)
    for yy in range(cy - 1, cy + 2):
        print("      " + " ".join("%3d" % sim.tile_dic.get((xx, yy), 0)
                                  for xx in range(cx - 1, cx + 2)))
    segs = gather_segments_from_region(sim, ox - 24, oy - 24, ox + 24, oy + 24)
    print("    %-6s %-22s %-22s %-6s %-9s %s"
          % ("dist2", "p1", "p2", "back", "closest", "normal(-py,px)"))
    for s in segs:
        back, a, bb = s.get_closest_point(ox, oy)
        raw = (ox - a) ** 2 + (oy - bb) ** 2
        px, py = getattr(s, "px", 0.0), getattr(s, "py", 0.0)
        pl = math.hypot(px, py) or 1.0
        sg = -1.0 if back else 1.0
        n = (-py / pl * sg, px / pl * sg)
        if raw > 4.0:
            continue
        print("    %-6.2f (%7.1f,%7.1f)      (%7.1f,%7.1f)      %-6s (%.1f,%.1f)  (%+.1f,%+.1f)"
              % (raw, getattr(s, "x1", 0), getattr(s, "y1", 0),
                 getattr(s, "x2", 0), getattr(s, "y2", 0), back, a, bb, n[0], n[1]))
    print()
