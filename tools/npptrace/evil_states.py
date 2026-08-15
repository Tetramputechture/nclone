"""What state does the game keep its evil ninjas in, and do any ever leave it?

`+0x48` is the state and `+0x4C` the timer. The port's ctor starts them in
WAITING with a 120-frame countdown, then plays back the ninja's own path; if the
game keeps them in DORMANT the port has an enemy the game does not.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 20
states = collections.Counter()
first = collections.Counter()
moved = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        p = next((q for q in cap.pools if q["i"] == POOL), None)
        if p is None or not p["count"]:
            continue
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        start = {}
        for f in range(1, cap.n + 1):
            rec = cap.frame(f)
            for i in range(p["count"]):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                st = struct.unpack_from("<i", b, 0x48)[0]
                tm = struct.unpack_from("<i", b, 0x4C)[0]
                x = struct.unpack_from("<d", b, 0x10)[0]
                y = struct.unpack_from("<d", b, 0x18)[0]
                states[(st, tm)] += 1
                key = (demo, i)
                if key not in start:
                    start[key] = (x, y)
                    first[(st, tm)] += 1
                elif (x, y) != start[key]:
                    moved[key] += 1

print("(state, timer) on the FIRST captured frame of each evil ninja:")
for k, v in first.most_common(8):
    print("   state=%-3d timer=%-5d  %d" % (k[0], k[1], v))
print("\n(state, timer) over EVERY captured frame:")
for k, v in states.most_common(8):
    print("   state=%-3d timer=%-5d  %d" % (k[0], k[1], v))
print("\nevil ninjas that ever move: %d" % len(moved))
