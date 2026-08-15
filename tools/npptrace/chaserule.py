"""When the game's chaser latches (speed 8/7 -> 16/7), which way does it go?

Capture-only, like the measurement that cracked the chaingun's sweep sign. Score
candidate rules against every latch in every captured chaser and PRINT THE
FAILURES rather than a single percentage.
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
hits = collections.Counter()
fails = []

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
        n = ninja[f - 1]
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            x, y = struct.unpack_from("<dd", b, 0x10)
            key = (stem, i)
            if key in prev:
                (px, py), (ppx, ppy) = prev[key]
                sx, sy = x - px, y - py
                osx, osy = px - ppx, py - ppy
                spd, ospd = math.hypot(sx, sy), math.hypot(osx, osy)
                # The latch frame: was at patrol speed, now at chase speed.
                if (abs(ospd - PATROL) < 1e-6 and abs(spd - CHASE) < 1e-6):
                    dx, dy = n["xpos"] - px, n["ypos"] - py
                    # Which axis did it actually take?
                    took = "x" if abs(sx) > abs(sy) else "y"
                    same_dir = (abs(osx) > abs(osy)) == (abs(sx) > abs(sy)) and (
                        (sx > 0) == (osx > 0) if took == "x" else (sy > 0) == (osy > 0))
                    dominant = "x" if abs(dx) > abs(dy) else "y"
                    if same_dir:
                        hits["keeps its current direction"] += 1
                    if took == dominant:
                        hits["takes the ninja's dominant axis"] += 1
                    if same_dir and took == dominant:
                        hits["both agree"] += 1
                    elif same_dir:
                        hits["keeps dir, NOT dominant"] += 1
                    elif took == dominant:
                        hits["dominant, NOT keeps dir"] += 1
                    else:
                        hits["neither"] += 1
                        if len(fails) < 8:
                            fails.append((os.path.basename(stem), i, f,
                                          round(dx, 1), round(dy, 1),
                                          round(sx, 3), round(sy, 3),
                                          round(osx, 3), round(osy, 3)))
                    hits["TOTAL latches"] += 1
                prev[key] = ((x, y), (px, py))
            else:
                prev[key] = ((x, y), (x, y))

tot = hits["TOTAL latches"]
print("chaser latches found: %d\n" % tot)
for k in ("keeps its current direction", "takes the ninja's dominant axis",
          "both agree", "keeps dir, NOT dominant", "dominant, NOT keeps dir",
          "neither"):
    v = hits[k]
    print("   %-32s %4d  %5.1f%%" % (k, v, 100.0 * v / max(tot, 1)))
print("\nfailures of BOTH rules (stem, ent, frame, dninja, step, prev step):")
for r in fails:
    print("   %s" % (r,))
