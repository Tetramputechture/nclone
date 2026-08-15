"""Does the game's dual laser start with its beam already ON the surface normal?

Our `_seat_on_surface` sets `beam_angle = atan2(ynormal, xnormal)` at spawn, so
the beam is fully aimed on frame 1. If the game instead starts it somewhere else
and eases in at 20% a frame (`BEAM_LERP`), a laser whose normal points at the
ninja's spawn would NOT kill on frame 1 -- and 12 demos in the corpus die exactly
there.

Fields (pool 21, stride 0xd0): +0x10/+0x18 pos, +0x58/+0x60 anchor,
+0x68/+0x70 normal, +0xb0 beam angle.
"""
import glob
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

n = 0
onnormal = 0
rows = []
for stem in stems:
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == 21 and q["count"]), None)
    if p is None or p["stride"] < 0xB8:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    rec = cap.frame(1)
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        nx, ny = D(b, 0x68), D(b, 0x70)
        beam = D(b, 0xB0)
        if nx == 0.0 and ny == 0.0:
            continue                       # spinner: no seated normal
        want = math.atan2(ny, nx)
        d = abs(((beam - want) + math.pi) % (2 * math.pi) - math.pi)
        n += 1
        if d < 1e-9:
            onnormal += 1
        elif len(rows) < 12:
            rows.append((os.path.basename(stem), i, beam, want, d))

print("frame-1 surface lasers in the captures: %d" % n)
print("   beam angle already EXACTLY on the normal: %d" % onnormal)
print("   beam angle elsewhere:                    %d" % (n - onnormal))
if rows:
    print("\nexamples where it is NOT on the normal:")
    for stem, i, beam, want, d in rows:
        print("   %-10s laser %-2d beam=%-22.17g normal_angle=%-22.17g  off by %.6f rad"
              % (stem, i, beam, want, d))
