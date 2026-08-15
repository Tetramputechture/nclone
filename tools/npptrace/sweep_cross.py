"""Does the discharge sweep turn TOWARD the ninja's bearing?

Capture-only: the ninja's position is in the `.ninja.jsonl`, so no replay is
needed and nothing depends on our own simulation being right.

For each discharge take the gun angle on the frame the burst starts, the ninja's
bearing from the drone on that frame, and the sign of the sweep. If the sweep
crosses the target, the sign is the short way round toward that bearing.
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
tab = collections.Counter()
rows = []


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
            start = {}
            seq = collections.defaultdict(list)
            last = {}
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
                            n = ninja[f - 1]
                            start[key] = (gun, math.atan2(n["ypos"] - gy,
                                                          n["xpos"] - gx))
                        if last.get(key) != gun:
                            seq[key].append(gun)
                            last[key] = gun
                    else:
                        sq = seq.pop(key, [])
                        st0 = start.pop(key, None)
                        last.pop(key, None)
                        if len(sq) >= 4 and st0 is not None:
                            steps = [wrap(sq[j + 1] - sq[j])
                                     for j in range(1, len(sq) - 1)]
                            mean = sum(steps) / len(steps)
                            gun0, bearing = st0
                            toward = wrap(bearing - gun0)
                            agree = (mean > 0) == (toward > 0)
                            tab[agree] += 1
                            rows.append((demo, round(gun0, 3), round(bearing, 3),
                                         round(toward, 3), round(mean, 4), agree))

import collections as _c
band=_c.defaultdict(lambda:[0,0])
for r in rows:
    t=abs(r[3])
    key='|gap| < 0.01' if t<0.01 else ('0.01-0.05' if t<0.05 else '> 0.05')
    band[key][0 if r[5] else 1]+=1
print("by size of the gap between the gun and the ninja's bearing:")
for k in ('|gap| < 0.01','0.01-0.05','> 0.05'):
    a,dis=band[k]
    tot=a+dis
    if tot: print("   %-14s agrees %-4d disagrees %-4d  %.0f%%" % (k,a,dis,100.0*a/tot))
print()
print("does the sweep turn toward the ninja's bearing at the start of the burst?")
print("   agrees    %d" % tab[True])
print("   disagrees %d" % tab[False])
tot = tab[True] + tab[False]
print("   -> %.0f%% consistent over %d discharges"
      % (100.0 * max(tab[True], tab[False]) / max(tot, 1), tot))
print("\n(demo, gun at burst start, ninja bearing, short way round, mean step, agrees)")
for r in rows[:12]:
    print("   %s" % (r,))
