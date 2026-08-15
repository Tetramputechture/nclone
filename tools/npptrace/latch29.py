"""Exact geometry of the latch the game makes and we miss.

1001030 drone 3: the game commits on frame 29, we do not until 34. The decision
uses the pose at the end of the previous frame. Print full precision so the
acquisition predicate can be identified rather than guessed.
"""
import glob
import json
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

DEMO, POOL, ENT = "1001030", 13, 3
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"] + ENT * p["stride"]
ninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]

print("frame |    drone x,y      |     ninja x,y      |     dx |     dy | step")
prev = None
for f in range(24, 33):
    b = cap.frame(f)[base: base + p["stride"]]
    x, y = struct.unpack_from("<dd", b, 0x10)
    n = ninja[f - 1]
    step = "-" if prev is None else "%+.5f,%+.5f" % (x - prev[0], y - prev[1])
    print("%5d | %8.4f,%8.4f | %9.4f,%9.4f | %6.2f | %6.3f | %s"
          % (f, x, y, n["xpos"], n["ypos"], n["xpos"] - x, n["ypos"] - y, step))
    prev = (x, y)

print("\nThe latch on frame 29 used the pose at the end of frame 28.")
b = cap.frame(28)[base: base + p["stride"]]
x28, y28 = struct.unpack_from("<dd", b, 0x10)
n28 = ninja[27]
print("   drone (%.6f, %.6f)  ninja (%.6f, %.6f)" % (x28, y28, n28["xpos"], n28["ypos"]))
print("   dy = %.6f   (our CORRIDOR_HALF_WIDTH is 12.0)" % (n28["ypos"] - y28))
b = cap.frame(27)[base: base + p["stride"]]
x27, y27 = struct.unpack_from("<dd", b, 0x10)
n27 = ninja[26]
print("   frame 27 (did NOT latch): dy = %.6f" % (n27["ypos"] - y27))
print("\nso the threshold lies in (%.4f, %.4f]"
      % (n28["ypos"] - y28, n27["ypos"] - y27))
print("candidates: 12=%s 18=%s 19=%s 20=%s 24=%s"
      % tuple("yes" if (n28["ypos"] - y28) <= c else "no"
              for c in (12, 18, 19, 20, 24)))
