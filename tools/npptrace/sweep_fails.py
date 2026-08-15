"""The discharges where "sweep against the ninja's cross-velocity" is wrong.

Scoring whole-corpus percentages cannot separate correlated hypotheses. Print
the failures with everything that might distinguish them instead.
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

POOLS = (10, 11)
fails, wins = [], []


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        nj = os.path.join(d, demo + ".ninja.jsonl")
        if not os.path.exists(nj):
            continue
        ninja = [json.loads(l) for l in open(nj) if l.strip()]
        cap = Capture(os.path.join(d, demo))
        for pool in POOLS:
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            start, seqs, last = {}, collections.defaultdict(list), {}
            for f in range(1, min(cap.n, len(ninja)) + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    key = (demo, pool, i)
                    if st == 2:
                        if key not in start:
                            gx = struct.unpack_from("<d", b, 0x10)[0]
                            gy = struct.unpack_from("<d", b, 0x18)[0]
                            n1 = ninja[f - 1]
                            cross = (n1["xspeed"] * math.sin(gun)
                                     - n1["yspeed"] * math.cos(gun))
                            bear = math.atan2(n1["ypos"] - gy, n1["xpos"] - gx)
                            rng = math.hypot(n1["xpos"] - gx, n1["ypos"] - gy)
                            spd = math.hypot(n1["xspeed"], n1["yspeed"])
                            start[key] = (f, gun, cross, wrap(bear - gun), rng, spd)
                        if last.get(key) != gun:
                            seqs[key].append(gun)
                            last[key] = gun
                    else:
                        sq = seqs.pop(key, [])
                        st0 = start.pop(key, None)
                        last.pop(key, None)
                        if len(sq) >= 4 and st0 is not None:
                            steps = [wrap(sq[j + 1] - sq[j])
                                     for j in range(1, len(sq) - 1)]
                            mean = sum(steps) / len(steps)
                            f0, gun, cross, toward, rng, spd = st0
                            ok = (cross < 0) == (mean > 0)
                            row = (demo, pool + 2, f0, round(cross, 4),
                                   round(toward, 3), round(rng, 1), round(spd, 3),
                                   "+" if mean > 0 else "-")
                            (wins if ok else fails).append(row)

import collections as _c
per=_c.defaultdict(lambda:[0,0])
for r in wins: per[r[1]][0]+=1
for r in fails: per[r[1]][1]+=1
print("rule = sweep AGAINST the ninja's cross-velocity, split by drone type:")
for t in sorted(per):
    a,b=per[t]
    print("   type %-3d  agrees %-4d disagrees %-4d  -> %.0f%%" % (t,a,b,100.0*a/(a+b)))
print()
print("%d agree, %d disagree overall" % (len(wins), len(fails)))
print("\n%-10s %-5s %-6s %-9s %-8s %-7s %-7s %s"
      % ("demo", "type", "frame", "cross", "toward", "range", "speed", "sweep"))
for r in fails:
    print("%-10s %-5d %-6d %-9.4f %-8.3f %-7.1f %-7.3f %s" % r)


def stat(name, xs):
    xs = sorted(xs)
    if not xs:
        return
    print("  %-22s n=%-4d min %.3f  median %.3f  max %.3f"
          % (name, len(xs), xs[0], xs[len(xs) // 2], xs[-1]))


print("\n|cross| at the burst start:")
stat("agreeing", [abs(r[3]) for r in wins])
stat("disagreeing", [abs(r[3]) for r in fails])
print("ninja speed:")
stat("agreeing", [r[6] for r in wins])
stat("disagreeing", [r[6] for r in fails])
