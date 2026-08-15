"""How many frames into a discharge does the game's FIRST shot land?

At every captured false kill the game's +0xC0 is still (0,0) while our beam is
live: we fire earlier in the discharge than the game does. Measure the offset
over every discharge in the corpus.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 10
offsets = collections.Counter()
never = 0

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    for i in range(p["count"]):
        off = base + i * p["stride"]
        prev = None
        start = None
        for f in range(1, cap.n + 1):
            b = cap.frame(f)[off: off + p["stride"]]
            st = struct.unpack_from("<i", b, 0x7C)[0]
            xe, ye = struct.unpack_from("<dd", b, 0xC0)
            if st == 2 and prev != 2:
                start = f
            if st != 2:
                start = None
            if start is not None and (xe != 0.0 or ye != 0.0):
                offsets[f - start] += 1
                start = None       # only the first shot of each discharge
            prev = st

tot = sum(offsets.values())
print("discharges with an observed first shot: %d\n" % tot)
print("frames from state->2 until the first non-zero endpoint:")
for k in sorted(offsets):
    print("   +%-3d frames : %4d  %s" % (k, offsets[k], "#" * min(offsets[k], 50)))
if offsets:
    common = offsets.most_common(1)[0]
    print("\nmost common offset: +%d frames (%d of %d = %.1f%%)"
          % (common[0], common[1], tot, 100.0 * common[1] / tot))
    print("SHOT_INTERVAL is 4; our first shot lands at +%s"
          % "(check entity_drone_shooter_base.think)")
