"""How far off-axis can the ninja be and still trigger a chase?

12 is too narrow (the game latches 5 frames before us on 1001030 drone 3) and 24
is too wide (49.24% vs 52.14% bit-exact chaser frames). Measure it: for every
latch, the ninja's offset PERPENDICULAR to the direction the drone commits to.
The largest offset that latches bounds the corridor from below; the smallest
that does not, from above.
"""
import collections
import glob
import json
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 13
PATROL, CHASE = 8 / 7, 16 / 7
perp_latch = []

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    prev = {}
    for f in range(1, min(cap.n, len(ninja)) + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            x, y = struct.unpack_from("<dd", b, 0x10)
            key = (stem, i)
            if key in prev:
                (px, py), (ppx, ppy) = prev[key]
                sx, sy = x - px, y - py
                osx, osy = px - ppx, py - ppy
                if (abs(math.hypot(osx, osy) - PATROL) < 1e-6
                        and abs(math.hypot(sx, sy) - CHASE) < 1e-6):
                    # The decision used the pose at the END of the previous
                    # frame, i.e. (px, py), and the ninja as of that frame.
                    n = ninja[f - 2] if f >= 2 else ninja[0]
                    dx, dy = n["xpos"] - px, n["ypos"] - py
                    # Perpendicular to the committed heading.
                    perp = min(abs(dx), abs(dy))
                    along = max(abs(dx), abs(dy))
                    perp_latch.append((round(perp, 3), round(along, 1),
                                       os.path.basename(stem), i, f))
                prev[key] = ((x, y), (px, py))
            else:
                prev[key] = ((x, y), (x, y))

perp_latch.sort()
print("%d latches\n" % len(perp_latch))
print("perpendicular offset at latch -- distribution:")
buckets = collections.Counter(int(r[0]) for r in perp_latch)
for k in sorted(buckets):
    print("   %2d..%2d px : %s" % (k, k + 1, "#" * buckets[k] + " %d" % buckets[k]))
print("\nmin %.3f   max %.3f" % (perp_latch[0][0], perp_latch[-1][0]))
print("\nlargest 10 perpendicular offsets that DID latch:")
for r in perp_latch[-10:]:
    print("   perp=%7.3f along=%7.1f  %s ent %d frame %d" % r)
