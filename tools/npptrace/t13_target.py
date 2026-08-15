"""Recover the laser drone's charge target by inverting its 20% lerp.

Uses the EARLY steps only: dividing by 0.2 multiplies error fivefold, and the
late steps are 1e-5 where the recovered target is noise.
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

POOL = 11
LERP = 0.2
hits = collections.Counter()
rows = []


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out13/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    prev, run = {}, {}
    for f in range(1, min(cap.n, len(ninja)) + 1):
        rec = cap.frame(f)
        n = ninja[f - 1]
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gun = struct.unpack_from("<f", b, 0x50)[0]
            st = struct.unpack_from("<i", b, 0x7C)[0]
            gx = struct.unpack_from("<d", b, 0x10)[0]
            gy = struct.unpack_from("<d", b, 0x18)[0]
            key = (stem, i)
            if st == 1:
                run.setdefault(key, []).append(
                    (gun, math.atan2(n["ypos"] - gy, n["xpos"] - gx),
                     n["xpos"], n["ypos"], gx, gy, n["xspeed"], n["yspeed"]))
            elif key in run:
                seq = run.pop(key)
                # Steps 2..5 of the charge: past the initial jump, still large.
                tg = []
                for j in range(1, min(6, len(seq) - 1)):
                    step = wrap(seq[j + 1][0] - seq[j][0])
                    if abs(step) > 1e-3:
                        tg.append(seq[j][0] + step / LERP)
                if len(tg) < 2:
                    continue
                target = sum(tg) / len(tg)
                spread = max(tg) - min(tg)
                if spread > 0.05:
                    hits["target not stable"] += 1
                    continue
                g0, bear0, nx, ny, dx0, dy0, vx, vy = seq[1]
                # Candidate: the bearing to where the ninja WILL be, if it keeps
                # its current velocity for the rest of the charge.
                lead = math.atan2((ny + vy * 46) - dy0, (nx + vx * 46) - dx0)
                cands = {
                    "ninja bearing": bear0,
                    "lead 46 frames": lead,
                    "lead 23 frames": math.atan2((ny + vy * 23) - dy0,
                                                 (nx + vx * 23) - dx0),
                }
                named = None
                for nm, v in cands.items():
                    if abs(wrap(target - v)) < 0.03:
                        named = nm
                        break
                if named is None:
                    for k in range(4):
                        if abs(wrap(target - k * math.pi / 2)) < 0.03:
                            named = "axis %d" % k
                            break
                hits[named or "unexplained"] += 1
                if named is None and len(rows) < 10:
                    rows.append((os.path.basename(stem), round(target, 4),
                                 round(bear0, 4), round(wrap(target - bear0), 4),
                                 round(lead, 4)))
            prev[key] = (gun, st)

print("recovered charge target, %d charges:" % sum(hits.values()))
for k, v in hits.most_common():
    print("   %-22s %4d  %5.1f%%" % (k, v, 100.0 * v / max(sum(hits.values()), 1)))
print("\nunexplained (stem, target, bearing, target-bearing, lead46):")
for r in rows:
    print("   %s" % (r,))
