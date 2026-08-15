"""What is the game's ACTUAL kill distance for contact drones?

§53: both the chaingun and the chaser get more correct positionally and kill more
demos. Both kill via `logical_collision` overlap against a moving entity. If our
radius or our test point is wrong, the game will routinely allow separations we
treat as lethal.

Capture-only: over every frame where the ninja is ALIVE, measure the drone-ninja
separation. The minimum the game ever permits bounds the true kill distance from
above. Our test kills below DRONE_RADIUS + NINJA_RADIUS = 17.5.
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

# pool -> (type, our radius)
POOLS = {12: ("zap (14)", 7.5), 13: ("chaser (15)", 7.5), 24: ("mini (26)", 4.0)}
mins = collections.defaultdict(lambda: 1e9)
hist = collections.defaultdict(collections.Counter)

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    for pool, (name, _) in POOLS.items():
        p = next((q for q in cap.pools if q["i"] == pool and q["count"]), None)
        if p is None:
            continue
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        for f in range(1, min(cap.n, len(ninja)) + 1):
            n = ninja[f - 1]
            # state 6 is dead; skip the death frame and anything after it.
            if n.get("state") == 6:
                break
            rec = cap.frame(f)
            for i in range(p["count"]):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                x, y = struct.unpack_from("<dd", b, 0x10)
                d = math.hypot(n["xpos"] - x, n["ypos"] - y)
                if d < mins[pool]:
                    mins[pool] = d
                if d < 40:
                    hist[pool][int(d)] += 1

print("closest approach the game ALLOWS while the ninja is alive:\n")
for pool, (name, r) in POOLS.items():
    if mins[pool] > 1e8:
        continue
    print("   %-14s min separation %7.3f   our kill threshold %.1f + 10 = %.1f"
          % (name, mins[pool], r, r + 10))
print("\nseparation histogram (1px bins, alive frames only):")
for pool, (name, r) in POOLS.items():
    if not hist[pool]:
        continue
    print("   %s:" % name)
    for k in sorted(hist[pool])[:14]:
        print("      %2d..%2d px : %s %d"
              % (k, k + 1, "#" * min(hist[pool][k] // 4 + 1, 40), hist[pool][k]))
