"""For the seating misses, what anchor did the GAME choose?

`+0x58`/`+0x60` is the contact point the game seated on. Comparing it against
ours says which surface it picked, which is a far more direct question than
which selection rule reproduces the normal.
"""
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
TRACES = "/home/tetra/projects/nclone/tools/npptrace/out10"

for demo in sys.argv[1:]:
    cap = Capture(os.path.join(TRACES, demo))
    p = next(q for q in cap.pools if q["i"] == 21)
    dec = NppAttractDecoder()
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(
        list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
    lasers = [e for lst in h.sim.entity_dic.values() for e in lst
              if getattr(e, "type", None) == 23]
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    rec = cap.frame(1)
    for i in range(min(p["count"], len(lasers))):
        e = lasers[i]
        if e.mode != 1:
            continue
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gn = (struct.unpack_from("<d", b, 0x68)[0],
              struct.unpack_from("<d", b, 0x70)[0])
        ga = (struct.unpack_from("<d", b, 0x58)[0],
              struct.unpack_from("<d", b, 0x60)[0])
        gp = (struct.unpack_from("<d", b, 0x10)[0],
              struct.unpack_from("<d", b, 0x18)[0])
        if gn == (0.0, 0.0):
            continue
        if abs(e.xnormal - gn[0]) < 1e-9 and abs(e.ynormal - gn[1]) < 1e-9:
            continue
        sx, sy = e._spawn_xy
        print("%s laser %d  spawn=(%g, %g)" % (demo, i, sx, sy))
        print("   port anchor=(%.4f, %.4f) n=(%.4f, %.4f) body=(%.4f, %.4f)"
              % (e.xanchor, e.yanchor, e.xnormal, e.ynormal, e.xpos, e.ypos))
        print("   game anchor=(%.4f, %.4f) n=(%.4f, %.4f) body=(%.4f, %.4f)"
              % (ga[0], ga[1], gn[0], gn[1], gp[0], gp[1]))
        print("   segments within 26px of the spawn:")
        for s in gather_segments_from_region(h.sim, sx - 26, sy - 26, sx + 26, sy + 26):
            if getattr(s, "px", None) is None:
                continue
            back, a, bb = s.get_closest_point(sx, sy)
            d = ((sx - a) ** 2 + (sy - bb) ** 2) ** 0.5
            if d > 14:
                continue
            print("      (%7.1f,%7.1f)->(%7.1f,%7.1f) oriented=%-5s dist=%5.2f closest=(%.1f,%.1f)"
                  % (s.x1, s.y1, s.x2, s.y2, s.oriented, d, a, bb))
        print()
