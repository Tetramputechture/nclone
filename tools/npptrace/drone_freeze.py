"""Does the game's shooter drone move while its state is non-zero?

10131718's chaingun drone stops dead at the frame it acquires and never moves
again through charge, fire and cooldown. Score that against every shooter drone
in the capture corpus before changing either engine.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

SHOOTER_POOLS = {10: 12, 11: 13}     # pool index -> entity type
tab = collections.Counter()
per_state = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        for pool, _typ in SHOOTER_POOLS.items():
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            prev = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    x = struct.unpack_from("<d", b, 0x10)[0]
                    y = struct.unpack_from("<d", b, 0x18)[0]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    key = (demo, pool, i)
                    if key in prev:
                        px, py, pst = prev[key]
                        moved = (x, y) != (px, py)
                        # Attribute the move to the state it was in when it moved.
                        tab[(pst != 0, moved)] += 1
                        per_state[(pst, moved)] += 1
                    prev[key] = (x, y, st)

print("game shooter drones: does it move this frame?")
print("  %-14s %-10s %s" % ("has target", "moved", "frames"))
for (busy, moved), n in sorted(tab.items()):
    print("  %-14s %-10s %d" % (busy, moved, n))
print("\nby state:")
for (st, moved), n in sorted(per_state.items()):
    print("  state %d  moved=%-6s %d" % (st, moved, n))
