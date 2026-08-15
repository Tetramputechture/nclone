"""Where is the type-13 drone's POSITION in its slab?

Every bearing computed from +0x10/+0x18 came out near pi, across unrelated maps.
That is not plausible, and §26 warns these two types do not share a struct.
Find the offset whose doubles match the layout's own spawn coords on frame 1.
"""
import glob
import json
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 11

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out13/*.layout.json"))[:3]:
    stem = s.replace(".layout.json", "")
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    lay = cap.layout
    spawns = [e for e in lay.get("entities", []) if e.get("type") == 13]
    print("\n%s  stride 0x%X  count %d  spawns %s"
          % (os.path.basename(stem), p["stride"], p["count"],
             [(round(e["x"], 1), round(e["y"], 1)) for e in spawns[:4]]))
    base = 4 + lay["ninjaSlab"] + p["offset"]
    rec = cap.frame(1)
    b = rec[base: base + p["stride"]]
    # Print every plausible double pair in the slab.
    for off in range(0, p["stride"] - 15, 4):
        try:
            x, y = struct.unpack_from("<dd", b, off)
        except struct.error:
            continue
        if not (all(map(math.isfinite, (x, y)))
                and 0 < x < 1200 and 0 < y < 800):
            continue
        near = [e for e in spawns
                if abs(e["x"] - x) < 2 and abs(e["y"] - y) < 2]
        print("   +0x%02X  (%8.2f, %8.2f) %s"
              % (off, x, y, "<-- MATCHES SPAWN" if near else ""))
