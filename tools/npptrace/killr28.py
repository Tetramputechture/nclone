"""Read the drones' kill radius directly: it lives at +0x28.

`log_t14_15_26_zap.c` (FUN_104007f0, shared logical_collision for types 14/15/26)
passes `*(double *)(this + 0x28)` as the radius to `FUN_1050c650`, the circle
overlap test. So the kill radius is a FIELD, not a per-type constant, and the
captures hold it.

§54 inferred <=3.5 for the chaser from minimum observed separations. This reads
the number.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOLS = {12: ("zap (14)", 7.5), 13: ("chaser (15)", 7.5), 24: ("mini (26)", 4.0)}
seen = collections.defaultdict(collections.Counter)

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    try:
        cap = Capture(stem)
    except Exception:
        continue
    for pool, (name, _) in POOLS.items():
        p = next((q for q in cap.pools if q["i"] == pool and q["count"]), None)
        if p is None or p["stride"] < 0x30:
            continue
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        rec = cap.frame(1)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            r = struct.unpack_from("<d", b, 0x28)[0]
            seen[pool][round(r, 6)] += 1

print("kill radius read from +0x28 (frame 1, every captured entity):\n")
for pool, (name, ours) in POOLS.items():
    if not seen[pool]:
        continue
    print("   %-14s our RADIUS = %-5s" % (name, ours))
    for v, n in seen[pool].most_common(6):
        flag = "  <-- MATCHES" if abs(v - ours) < 1e-9 else "  <-- DIFFERS"
        print("        +0x28 = %-12s x%-4d%s" % (v, n, flag))
