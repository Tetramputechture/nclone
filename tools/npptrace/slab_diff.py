"""Show every offset in an entity's captured slab that changes across a window.

field_hunt matches nclone's own attributes against the game's doubles, which is
no help when the question is "what field decides a transition nclone does not
model". This just prints the slab's moving parts around the frame in question.
"""
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

TRACES = os.environ.get("NCLONE_TRACES",
                        "/home/tetra/projects/nclone/tools/npptrace/out9")
demo = sys.argv[1]
pool = int(sys.argv[2])
idx = int(sys.argv[3])
lo, hi = int(sys.argv[4]), int(sys.argv[5])

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == pool)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
stride = p["stride"]


def slab(f):
    r = cap.frame(f)
    o = base + idx * stride
    return r[o:o + stride]


frames = list(range(lo, hi + 1))
slabs = {f: slab(f) for f in frames}

# Which 4-byte-aligned offsets are not constant across the window?
moving = []
for off in range(0, stride - 7, 4):
    vals = {slabs[f][off:off + 4] for f in frames}
    if len(vals) > 1:
        moving.append(off)

print("%s pool %d entity %d, frames %d..%d, stride 0x%X" % (demo, pool, idx, lo, hi, stride))
print("offsets that change: %s" % " ".join("0x%X" % o for o in moving))
print()
hdr = "%5s" % "frame"
for o in moving:
    hdr += " %14s" % ("0x%X" % o)
print(hdr)
for f in frames:
    line = "%5d" % f
    for o in moving:
        raw = slabs[f][o:o + 4]
        i32 = struct.unpack("<i", raw)[0]
        f32 = struct.unpack("<f", raw)[0]
        # Show the int when it is small and plausible, else the float.
        if -100000 < i32 < 100000:
            line += " %14d" % i32
        else:
            line += " %14.4g" % f32
    print(line)
