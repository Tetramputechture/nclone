"""Per-frame walk of the chaingun's shot fields through one whole discharge.

+0xC0/+0xC8 populate only while discharging, and +0xC8 hits exactly 120.000 --
which would be a range cap, something our hitscan does not have. Print them
against the drone and the ninja every frame so their meaning is read, not
guessed.
"""
import glob
import json
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 10
DEMO = "10131718"
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
ninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]

print("frame st |  drone x,y   |   +0xC0    +0xC8 |  +0xD0   +0xD8 |"
      "   gun   |   ninja x,y   | dist")
for f in range(86, 150):
    if f > cap.n or f > len(ninja):
        break
    b = cap.frame(f)[base: base + p["stride"]]
    st = struct.unpack_from("<i", b, 0x7C)[0]
    gx, gy = struct.unpack_from("<dd", b, 0x10)
    c0, c8 = struct.unpack_from("<dd", b, 0xC0)
    d0, d8 = struct.unpack_from("<dd", b, 0xD0)
    gun = struct.unpack_from("<f", b, 0x50)[0]
    n = ninja[f - 1]
    dist = math.hypot(n["xpos"] - gx, n["ypos"] - gy)
    print("%5d %2d | %6.1f,%6.1f | %8.3f %8.3f | %6.2f %6.2f | %7.4f |"
          " %6.1f,%6.1f | %6.1f"
          % (f, st, gx, gy, c0, c8, d0, d8, gun, n["xpos"], n["ypos"], dist))

print("\nIf +0xC0/+0xC8 were a projectile POSITION it would start at the drone")
print("(518.9, 300.0) and walk away. Check that against the numbers above.")
