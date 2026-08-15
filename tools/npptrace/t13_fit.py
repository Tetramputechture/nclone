"""Fit the LASER drone's discharge, the way §28 fitted the chaingun's.

Capture-only. Batch 6 (`out13`) was chosen for maps containing a type 13, so
this is the first corpus with enough of them to fit anything.
"""
import collections
import glob
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 11
dwell = collections.defaultdict(collections.Counter)
shots_per = collections.Counter()
gaps = collections.Counter()
later = collections.Counter()
firsts = []


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out13/*.layout.json")):
    stem = s.replace(".layout.json", "")
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    st_run, seqs, lastg, lastf = {}, collections.defaultdict(list), {}, {}
    for f in range(1, cap.n + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            st = struct.unpack_from("<i", b, 0x7C)[0]
            gun = struct.unpack_from("<f", b, 0x50)[0]
            key = (stem, i)
            c = st_run.get(key)
            if c is None or c[0] != st:
                if c is not None:
                    dwell[c[0]][f - c[1]] += 1
                st_run[key] = (st, f)
            if st == 2:
                if lastg.get(key) != gun:
                    seqs[key].append(gun)
                    lastg[key] = gun
                    if key in lastf:
                        gaps[f - lastf[key]] += 1
                    lastf[key] = f
            else:
                sq = seqs.pop(key, [])
                lastg.pop(key, None)
                lastf.pop(key, None)
                if len(sq) >= 3:
                    shots_per[len(sq)] += 1
                    steps = [wrap(sq[j + 1] - sq[j]) for j in range(len(sq) - 1)]
                    firsts.append(round(steps[0], 4))
                    for x in steps[1:]:
                        later[round(x, 3)] += 1

print("type 13 state dwell lengths:")
for st in sorted(dwell):
    print("   state %-3d %s" % (st, dwell[st].most_common(3)))
print("\ndistinct gun angles per discharge: %s" % shots_per.most_common(5))
print("frames between gun updates:        %s" % gaps.most_common(5))
mags = [abs(k) for k, v in later.items() for _ in range(v) if abs(k) > 1e-6]
if mags:
    import statistics
    print("\n%d non-zero later steps; mean |step| = %.6f  median %.6f"
          % (len(mags), statistics.mean(mags), statistics.median(mags)))
    for name, val in (("1/60", 1 / 60.0), ("1/50", 0.02), ("1/40", 0.025),
                      ("1/80", 0.0125), ("1/100", 0.01)):
        print("   %-7s %.6f" % (name, val))
print("\nfirst steps (first 12): %s" % firsts[:12])
