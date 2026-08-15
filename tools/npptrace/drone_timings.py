"""How long does the game hold a shooter drone in each state?

CHARGE_DURATION, FIRE_DURATION and COOLDOWN_DURATION are all marked [I] in the
port -- no writer was ever found for +0x70/+0x74. The captures carry the state
field, so measure the dwell times instead of guessing them.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

SHOOTER_POOLS = (10, 11)
runs = collections.defaultdict(collections.Counter)
entry_timer = collections.defaultdict(collections.Counter)

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        for pool in SHOOTER_POOLS:
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            cur = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    tm = struct.unpack_from("<i", b, 0x78)[0]
                    key = (demo, pool, i)
                    c = cur.get(key)
                    if c is None or c[0] != st:
                        if c is not None and c[2]:
                            # Only count runs that both started and ended here.
                            runs[c[0]][f - c[1]] += 1
                        cur[key] = (st, f, True)
                        entry_timer[st][tm] += 1
                    # else: same state, keep the run open

for st in sorted(runs):
    tot = sum(runs[st].values())
    common = runs[st].most_common(6)
    print("state %d: %d completed runs; most common lengths %s"
          % (st, tot, ", ".join("%d frames x%d" % c for c in common)))
print()
for st in sorted(entry_timer):
    print("state %d entry timer values: %s"
          % (st, ", ".join("%d x%d" % c for c in entry_timer[st].most_common(6))))
