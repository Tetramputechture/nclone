"""What IS the laser drone's gun doing during its charge?

It does not track the ninja (17%). Print the raw per-frame steps of `+0x50`
through some charges and look at the shape before guessing a rule.
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
shown = 0
steps_all = collections.Counter()


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
                if key not in run:
                    run[key] = []
                bearing = math.atan2(n["ypos"] - gy, n["xpos"] - gx)
                step = wrap(gun - prev[key][0]) if key in prev else 0.0
                run[key].append((f, gun, bearing, step))
                if abs(step) > 1e-6:
                    steps_all[round(abs(step), 4)] += 1
            elif key in run:
                seq = run.pop(key)
                if shown < 2 and len(seq) >= 40:
                    shown += 1
                    print("\n%s drone %d, charge of %d frames:"
                          % (os.path.basename(stem), i, len(seq)))
                    print("  %5s %10s %10s %10s %s"
                          % ("frame", "+0x50", "bearing", "step", "gap to bearing"))
                    for row in seq[:6] + seq[-6:]:
                        fr, g, be, sp = row
                        print("  %5d %10.5f %10.5f %10.5f %10.5f"
                              % (fr, g, be, sp, wrap(be - g)))
            prev[key] = (gun, st)

print("\nmost common non-zero |step| during a type-13 charge:")
for k, v in steps_all.most_common(10):
    print("   %.4f x%d" % (k, v))
import math as _m
for name, val in (("pi/300", _m.pi / 300), ("1/60", 1 / 60.0),
                  ("pi/180", _m.pi / 180), ("0.01", 0.01)):
    print("   ref %-8s %.5f" % (name, val))
