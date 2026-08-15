"""What ENDS a chase in the game? Capture-only, so it is fast.

Our model terminates at a goal cell snapped from the ninja at acquisition. The
game clearly does not: on 10138995 it runs straight past that cell at 16/7 while
the ninja moves the other way. Find every chase run and report how it ends.
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
runs = []
lens = collections.Counter()

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
    active = {}
    for f in range(1, min(cap.n, len(ninja)) + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            x, y = struct.unpack_from("<dd", b, 0x10)
            key = (stem, i)
            if key in prev:
                px, py = prev[key]
                spd = math.hypot(x - px, y - py)
                chasing = abs(spd - CHASE) < 1e-6
                patrol = abs(spd - PATROL) < 1e-6
                if chasing and key not in active:
                    active[key] = (f, px, py, x - px, y - py)
                elif patrol and key in active:
                    f0, x0, y0, sx, sy = active.pop(key)
                    n = ninja[f - 1]
                    dist = math.hypot(x - x0, y - y0)
                    runs.append((os.path.basename(stem), i, f0, f - 1, f - f0,
                                 round(dist, 1), round(x, 1), round(y, 1),
                                 round(n["xpos"], 1), round(n["ypos"], 1)))
                    lens[f - f0] += 1
            prev[key] = (x, y)

print("%d completed chase runs\n" % len(runs))
print("run length in frames -- most common:")
for k, v in lens.most_common(12):
    print("   %4d frames  x%d   (%.1f px travelled)" % (k, v, k * CHASE))
if runs:
    ls = sorted(r[4] for r in runs)
    print("\nmin %d  median %d  max %d" % (ls[0], ls[len(ls) // 2], ls[-1]))
    print("\ndistance travelled per run, in 24px cells:")
    cells = collections.Counter(round(r[5] / 24.0, 2) for r in runs)
    for k, v in cells.most_common(10):
        print("   %6.2f cells  x%d" % (k, v))
print("\nfirst 8 runs (stem, ent, start, end, frames, dist, endx, endy, ninjax, ninjay):")
for r in runs[:8]:
    print("   %s" % (r,))
