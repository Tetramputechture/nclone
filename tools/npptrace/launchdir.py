"""Invert the rocket's launch direction: what was it actually aiming at?

The heading is wrong by 7.1e-4 on the first flying frame. Rather than guess
another formula, take the game's own initial direction and test it against a
grid of candidate (origin, target) pairs: the ninja at several frame offsets,
with and without a lead, from the launcher centre.
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

POOL = 16
hits = collections.Counter()
rows = []

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
    if p is None or p["stride"] < 0x88:
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    for i in range(p["count"]):
        off = base + i * p["stride"]
        prev_travel = None
        for f in range(2, min(cap.n, len(ninja)) + 1):
            b = cap.frame(f)[off: off + p["stride"]]
            tr = struct.unpack_from("<d", b, 0x80)[0]
            dx, dy = struct.unpack_from("<dd", b, 0x60)
            lx, ly = struct.unpack_from("<dd", b, 0x10)
            # A launch: odometer resets to (nearly) zero having been larger.
            px, py = struct.unpack_from("<dd", b, 0x50)
            airborne = math.hypot(px - lx, py - ly) > 1e-9
            launching = airborne and not prev_travel
            first = False
            if launching or first:
                best = None
                for k in range(-8, 3):
                    idx = f - 1 + k
                    if not (0 <= idx < len(ninja)):
                        continue
                    n = ninja[idx]
                    for lead in (0.0,):
                        tx = n["xpos"] + n["xspeed"] * lead
                        ty = n["ypos"] + n["yspeed"] * lead
                        vx, vy = tx - lx, ty - ly
                        norm = math.hypot(vx, vy)
                        if norm == 0:
                            continue
                        err = math.hypot(vx / norm - dx, vy / norm - dy)
                        if best is None or err < best[0]:
                            best = (err, k, lead)
                if best:
                    hits[(best[1], best[2])] += 1
                    if len(rows) < 8:
                        rows.append((os.path.basename(stem), i, f,
                                     round(best[0], 8), best[1], best[2]))
            prev_travel = airborne

tot = sum(hits.values())
print("launches identified: %d\n" % tot)
print("best (ninja frame offset, lead) per launch:")
for (k, lead), v in hits.most_common(10):
    print("   offset %+d frames, lead %.1f : %3d  (%.1f%%)"
          % (k, lead, v, 100.0 * v / max(tot, 1)))
print("\nsample launches (stem, ent, frame, best err, offset, lead):")
for r in rows:
    print("   %s" % (r,))
