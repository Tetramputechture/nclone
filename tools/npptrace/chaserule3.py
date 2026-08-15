"""Candidate: keep going if the current direction closes on the ninja, else turn.

Every failure of "keeps its current direction" had the drone moving AWAY from the
ninja on its current axis, and turning onto an axis that moved toward it. Test
that as a rule, and print what is left over.
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
            if key not in prev:
                prev[key] = ((x, y), (x, y))
                continue
            (px, py), (ppx, ppy) = prev[key]
            sx, sy = x - px, y - py
            osx, osy = px - ppx, py - ppy
            if (abs(math.hypot(osx, osy) - PATROL) < 1e-6
                    and abs(math.hypot(sx, sy) - CHASE) < 1e-6):
                dx, dy = n["xpos"] - px, n["ypos"] - py
                on_x = abs(osx) > abs(osy)
                # Does the current heading close on the ninja?
                closing = (osx > 0) == (dx > 0) if on_x else (osy > 0) == (dy > 0)
                if True:
                    pred = (math.copysign(CHASE, dx), 0.0) if on_x else (0.0, math.copysign(CHASE, dy))
                else:  # unreachable
                    pred = (0.0, math.copysign(CHASE, dy)) if on_x else (math.copysign(CHASE, dx), 0.0)
                ok = abs(pred[0] - sx) < 1e-6 and abs(pred[1] - sy) < 1e-6
                hits["closing" if closing else "turning"] += 1
                hits["RIGHT" if ok else "WRONG"] += 1
                if not ok and len(fails) < 10:
                    fails.append((os.path.basename(stem), i, f,
                                  "closing" if closing else "turning",
                                  "d=(%.1f,%.1f)" % (dx, dy),
                                  "prev=(%.2f,%.2f)" % (osx, osy),
                                  "got=(%.2f,%.2f)" % (sx, sy),
                                  "pred=(%.2f,%.2f)" % pred))
            prev[key] = ((x, y), (px, py))

tot = hits["RIGHT"] + hits["WRONG"]
print("rule: STAY ON THE CURRENT AXIS, move toward the ninja along it")
print("   %d/%d latches = %.1f%%   (closing %d, turning %d)\n"
      % (hits["RIGHT"], tot, 100.0 * hits["RIGHT"] / max(tot, 1),
         hits["closing"], hits["turning"]))
print("failures:")
for r in fails:
    print("   %s" % (r,))
