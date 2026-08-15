"""Interpret one entity slab at chosen frames, as doubles and as int32s."""
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

TRACES = os.environ.get("NCLONE_TRACES",
                        "/home/tetra/projects/nclone/tools/npptrace/out9")
demo, pool, idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
frames = [int(a) for a in sys.argv[4].split(",")]

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == pool)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]


def slab(f):
    o = base + idx * p["stride"]
    return cap.frame(f)[o:o + p["stride"]]


slabs = {f: slab(f) for f in frames}
print("%-7s %s" % ("offset", " ".join("%22s" % ("f%d" % f) for f in frames)))
for off in range(0, p["stride"] - 7, 8):
    vals = []
    for f in frames:
        d = struct.unpack_from("<d", slabs[f], off)[0]
        a, b = struct.unpack_from("<ii", slabs[f], off)
        if d != d or abs(d) > 1e12 or (d != 0.0 and abs(d) < 1e-8):
            vals.append("i32 %d, %d" % (a, b))
        else:
            vals.append("%.6f" % d)
    if len(set(vals)) == 1 and vals[0] in ("0.000000",):
        continue
    print("0x%-5X %s" % (off, " ".join("%22s" % v for v in vals)))
