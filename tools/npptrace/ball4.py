"""All four death balls around frame 273. Did ball 2 bounce off another ball?

It reversed and roughly doubled its speed with no wall in the way and the ninja
280 px off. Ball-vs-ball collision is the obvious candidate; check separations.
"""
import glob
import json
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

DEMO, POOL, AT = "10132476", 23, 273
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]

for f in range(AT - 2, AT + 3):
    rec = cap.frame(f)
    print("\nframe %d" % f)
    balls = []
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        x, y = struct.unpack_from("<dd", b, 0x10)
        vx, vy = struct.unpack_from("<dd", b, 0x20)
        r = struct.unpack_from("<d", b, 0x00)[0]
        balls.append((i, x, y, vx, vy, r))
        print("   ball %d  (%9.4f, %9.4f)  v=(%+.4f, %+.4f)  +0x00=%.3f"
              % (i, x, y, vx, vy, r))
    for a in range(len(balls)):
        for c in range(a + 1, len(balls)):
            _, ax, ay, _, _, ar = balls[a]
            _, cx, cy, _, _, cr = balls[c]
            d = ((ax - cx) ** 2 + (ay - cy) ** 2) ** 0.5
            if d < 60:
                print("   >>> balls %d and %d are %.3f apart (radii %.1f+%.1f=%.1f)"
                      % (a, c, d, ar, cr, ar + cr))
