"""Solve for the rocket's launch ORIGIN instead of guessing a muzzle offset.

The residual is ~1px of lateral error at the target and no frame offset explains
it. Given the game's own launch direction and the ninja position it aimed at,
the ray back from the ninja along -dir passes through the true origin. Measure
the launcher's signed perpendicular offset from that ray: a real muzzle offset
would cluster, noise would not.
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
perps = []
alongs = []

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    if not os.path.basename(stem).isdigit():
        continue  # unresolved capNN -- §49 shows these carry the wild outliers
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
        prev_air = False
        for f in range(2, min(cap.n, len(ninja)) + 1):
            b = cap.frame(f)[off: off + p["stride"]]
            lx, ly = struct.unpack_from("<dd", b, 0x10)
            px, py = struct.unpack_from("<dd", b, 0x50)
            dx, dy = struct.unpack_from("<dd", b, 0x60)
            air = math.hypot(px - lx, py - ly) > 1e-9
            if air and not prev_air:
                idx = f - 3            # the -2 sample the inversion identified
                if 0 <= idx < len(ninja):
                    n = ninja[idx]
                    # Vector from launcher to the ninja it aimed at.
                    vx, vy = n["xpos"] - lx, n["ypos"] - ly
                    # Decompose along / perpendicular to the ACTUAL direction.
                    along = vx * dx + vy * dy
                    perp = vx * (-dy) + vy * dx
                    perps.append(round(perp, 4))
                    alongs.append(round(along, 1))
            prev_air = air

print("launches: %d\n" % len(perps))
if perps:
    print("signed perpendicular offset of the ninja from the actual aim ray:")
    print("   mean   %+.4f" % (sum(perps) / len(perps)))
    print("   min    %+.4f" % min(perps))
    print("   max    %+.4f" % max(perps))
    srt = sorted(perps)
    print("   median %+.4f" % srt[len(srt) // 2])
    buckets = collections.Counter(round(v * 2) / 2 for v in perps)
    print("\n   distribution (0.5px bins):")
    for k in sorted(buckets):
        print("      %+5.1f : %s %d" % (k, "#" * min(buckets[k], 40), buckets[k]))
    print("\n   range to target: min %.0f median %.0f max %.0f"
          % (min(alongs), sorted(alongs)[len(alongs) // 2], max(alongs)))
