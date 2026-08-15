"""Mode 2: what were the ~26 bad launches aiming at?

Their perpendicular offsets are 53-512px and some have the ninja BEHIND the aim
direction, so they are not aiming at the ninja's recent position. The rocket
charges while it can see the ninja; a target latched at acquisition would look
exactly like this. Search the ninja's whole recent history for the frame whose
bearing matches the launch direction.
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

POOL = 16
BACK = 200
best_offsets = []
unexplained = []

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None or p["stride"] < 0x88:
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    for i in range(p["count"]):
        off = base + i * p["stride"]
        prev_air = False
        for f in range(2, min(cap.n, len(ninja)) + 1):
            b = cap.frame(f)[off: off + p["stride"]]
            lx, ly = struct.unpack_from("<dd", b, 0x10)
            px, py = struct.unpack_from("<dd", b, 0x50)
            dx, dy = struct.unpack_from("<dd", b, 0x60)
            air = math.hypot(px - lx, py - ly) > 1e-9
            if air and not prev_air:
                best = None
                for k in range(0, BACK):
                    idx = f - 1 - k
                    if idx < 0:
                        break
                    n = ninja[idx]
                    vx, vy = n["xpos"] - lx, n["ypos"] - ly
                    nrm = math.hypot(vx, vy)
                    if nrm < 1e-9:
                        continue
                    err = math.hypot(vx / nrm - dx, vy / nrm - dy)
                    if best is None or err < best[0]:
                        best = (err, k)
                if best:
                    best_offsets.append((best[1], best[0]))
                    if best[0] > 1e-2:
                        unexplained.append((os.path.basename(stem), i, f,
                                            best[1], round(best[0], 5)))
            prev_air = air

print("launches: %d\n" % len(best_offsets))
good = [k for k, e in best_offsets if e <= 1e-2]
print("explained by SOME ninja frame within %d back (err<=1e-2): %d / %d"
      % (BACK, len(good), len(best_offsets)))
print("\nhow many frames back the best match sits:")
b = collections.Counter(good)
for k in sorted(b):
    if b[k] > 1:
        print("   %3d frames back : %s %d" % (k, "#" * min(b[k], 40), b[k]))
rare = sum(v for k, v in b.items() if v == 1)
print("   (%d singleton offsets)" % rare)
errs = sorted(e for k, e in best_offsets)
print("\nbest-fit error: median %.2e  p90 %.2e  max %.2e"
      % (errs[len(errs) // 2], errs[int(len(errs) * 0.9)], errs[-1]))
print("\nstill unexplained (%d):" % len(unexplained))
for r in unexplained[:8]:
    print("   %s" % (r,))
