"""Fit the discharge sweep: how does `+0x50` move from shot to shot in state 2?

Capture-only, no replay, so it runs in seconds. For every discharge, collect the
gun angle on each shot frame and report the per-shot deltas.
"""
import collections
import glob
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOLS = (10, 11)
first_step = collections.Counter()
later_step = collections.Counter()
per_burst = []


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
        cap = Capture(os.path.join(d, demo))
        for pool in POOLS:
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            cur = collections.defaultdict(list)
            last = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    key = (demo, pool, i)
                    if st == 2:
                        if last.get(key) != gun:
                            cur[key].append(gun)
                            last[key] = gun
                    else:
                        seq = cur.pop(key, [])
                        last.pop(key, None)
                        if len(seq) >= 3:
                            steps = [wrap(seq[j + 1] - seq[j])
                                     for j in range(len(seq) - 1)]
                            per_burst.append((demo, pool + 2, len(seq),
                                              round(steps[0], 4),
                                              round(sum(steps[1:]) / len(steps[1:]), 4)))
                            first_step[round(steps[0], 3)] += 1
                            for st2 in steps[1:]:
                                later_step[round(st2, 3)] += 1

import statistics
mags=[abs(k) for k,v in later_step.items() for _ in range(v) if abs(k)>1e-6]
print("%d discharges; %d non-zero later steps" % (len(per_burst), len(mags)))
print("mean |step| = %.6f   median = %.6f" % (statistics.mean(mags), statistics.median(mags)))
import math as _m
for name,val in (("pi/190",_m.pi/190),("pi/192",_m.pi/192),("pi/189",_m.pi/189),
                 ("0.0165",0.0165),("1/60",1/60.0),("pi/180",_m.pi/180)):
    print("   %-8s %.6f" % (name,val))
print("\nfirst step of the discharge, most common:")
for k, v in first_step.most_common(8):
    print("   %+.3f  x%d" % (k, v))
print("\nsteps after the first, most common:")
for k, v in later_step.most_common(10):
    print("   %+.3f  x%d" % (k, v))
print("\nper-burst (demo, type, distinct angles, first step, mean later step):")
for r in per_burst[:12]:
    print("   %s" % (r,))
