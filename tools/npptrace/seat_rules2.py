"""Score flush tie-break orderings against every captured laser normal.

§15 resolved a flush mount by taking the first segment whose outward normal
points into an empty tile. On a convex corner BOTH candidates do, so the rule
degenerates to gather order -- and on `10139043` the game takes the vertical face
where this port takes the horizontal. Score the orderings rather than guess.
"""
import collections
import glob
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
POOL = 21


def outward(seg, a, b):
    px, py = getattr(seg, "px", None), getattr(seg, "py", None)
    if px is not None and py is not None:
        plen = math.sqrt(px * px + py * py)
        return None if plen == 0 else (-py / plen, px / plen)
    r = getattr(seg, "radius", 0.0)
    if not r:
        return None
    s = 1.0 if getattr(seg, "convex", True) else -1.0
    return ((a - seg.xpos) / r * s, (b - seg.ypos) / r * s)


def empty_at(sim, x, y):
    return sim.tile_dic.get((int(x // 24), int(y // 24)), 0) == 0


def candidates(sim, x, y, rad=24):
    out = []
    for s in gather_segments_from_region(sim, x - rad, y - rad, x + rad, y + rad):
        back, a, b = s.get_closest_point(x, y)
        raw = (x - a) ** 2 + (y - b) ** 2
        out.append((raw - (0.0 if back else 0.1), raw, s, a, b, back))
    return out


def is_vertical(seg):
    return getattr(seg, "px", None) == 0


def make_rule(order):
    def rule(sim, x, y):
        c = candidates(sim, x, y)
        if not c:
            return None
        best = min(c, key=lambda t: t[0])
        flush = [t for t in c if t[1] == 0.0]
        if len(flush) > 1:
            if order == "vertical":
                flush = sorted(flush, key=lambda t: not is_vertical(t[2]))
            elif order == "horizontal":
                flush = sorted(flush, key=lambda t: is_vertical(t[2]))
            for t in flush:
                n = outward(t[2], t[3], t[4])
                if n and empty_at(sim, t[3] + n[0] * 12.0, t[4] + n[1] * 12.0):
                    return n
        return outward(best[2], best[3], best[4])
    return rule


RULES = [(o, make_rule(o)) for o in ("gather", "vertical", "horizontal")]
score = collections.Counter()
misses = collections.defaultdict(list)
total = 0

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        p = next((q for q in cap.pools if q["i"] == POOL), None)
        if p is None or not p["count"]:
            continue
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
            g = (struct.unpack_from("<d", b, 0x68)[0],
                 struct.unpack_from("<d", b, 0x70)[0])
            if g == (0.0, 0.0):
                continue
            ox, oy = getattr(e, "_spawn_xy", (e.xpos, e.ypos))
            total += 1
            for name, fn in RULES:
                n = fn(h.sim, ox, oy)
                if n and abs(n[0] - g[0]) < 1e-9 and abs(n[1] - g[1]) < 1e-9:
                    score[name] += 1
                else:
                    misses[name].append((demo, i, n, g))

print("scored %d surface lasers across three capture batches" % total)
for name, _ in RULES:
    print("  %-11s %3d/%3d  (%.1f%%)" % (name, score[name], total,
                                         100.0 * score[name] / max(total, 1)))
best = max(score, key=lambda k: score[k])
print("\nmisses under '%s':" % best)
for m in misses[best][:12]:
    print("   %-10s idx %-3d port=%s game=%s" % m)
