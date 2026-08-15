"""Watch every entity in a pool across a frame window.

The shooter drone fires on a known frame; if the shot is a projectile it has to
show up somewhere. Every captured map carries exactly 8 entities of type 29,
which is the shape of a preallocated projectile pool.
"""
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

TRACES = os.environ.get("NCLONE_TRACES",
                        "/home/tetra/projects/nclone/tools/npptrace/out9")
demo, pool = sys.argv[1], int(sys.argv[2])
lo, hi = int(sys.argv[3]), int(sys.argv[4])

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == pool)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
print("%s pool %d (type %d): %d entities, stride 0x%X"
      % (demo, pool, pool + 2, p["count"], p["stride"]))

prev = {}
for f in range(lo, hi + 1):
    rec = cap.frame(f)
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        x = struct.unpack_from("<d", b, 0x10)[0]
        y = struct.unpack_from("<d", b, 0x18)[0]
        # Any non-zero byte at all means the slot is in use.
        live = any(b)
        key = i
        sig = (round(x, 3), round(y, 3), live)
        if prev.get(key) != sig:
            print("  f%-5d entity %d  pos=(%.3f, %.3f) live=%s" % (f, i, x, y, live))
            prev[key] = sig
