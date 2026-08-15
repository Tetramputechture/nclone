"""Score the ENGINE's own seeded laser normals against the game.

Immune to reimplementation drift: reads `e.xnormal` / `e.ynormal` straight off
the constructed entity rather than re-deriving the rule.
"""
import collections
import glob
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

POOL = 21
tally = collections.Counter()
misses = []

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
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gn = (struct.unpack_from("<d", b, 0x68)[0],
                  struct.unpack_from("<d", b, 0x70)[0])
            if gn == (0.0, 0.0) or e.mode != 1:
                continue
            mn = (getattr(e, "xnormal", 0.0), getattr(e, "ynormal", 0.0))
            if abs(mn[0] - gn[0]) < 1e-9 and abs(mn[1] - gn[1]) < 1e-9:
                tally["match"] += 1
            elif abs(mn[0] + gn[0]) < 1e-9 and abs(mn[1] + gn[1]) < 1e-9:
                tally["negated"] += 1
                misses.append((demo, i, mn, gn))
            else:
                tally["other"] += 1
                misses.append((demo, i, mn, gn))

total = sum(tally.values())
print("engine seeded normals vs the game, %d surface lasers:" % total)
for k in ("match", "negated", "other"):
    print("   %-8s %3d  (%.1f%%)" % (k, tally[k], 100.0 * tally[k] / max(total, 1)))
print("\nfirst misses:")
for m in misses[:10]:
    print("   %-10s idx %-3d port=%s game=%s" % m)
