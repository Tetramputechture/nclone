"""Which captures contain rockets / surface lasers / turrets / floor guards?

Pool index = type - 2. These four account for 24 of the remaining 62 losses and
none has had the position-diff treatment that cracked the death ball.
"""
import glob
import os
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

ROOT = "/home/tetra/projects/nclone/tools/npptrace"
WANT = {16: "rocket (18)", 21: "dual_laser (23)", 17: "turret (19)",
        14: "floor_guard (16)"}

for corpus in sorted(glob.glob(os.path.join(ROOT, "out*"))):
    if not os.path.isdir(corpus):
        continue
    found, n = {}, 0
    for s in sorted(glob.glob(os.path.join(corpus, "*.layout.json"))):
        stem = s.replace(".layout.json", "")
        try:
            cap = Capture(stem)
        except Exception:
            continue
        n += 1
        for p in cap.pools:
            if p["i"] in WANT and p["count"]:
                found.setdefault(p["i"], []).append(
                    (os.path.basename(stem), p["count"], cap.n))
    if not found:
        continue
    print("\n%-14s %d captures" % (os.path.basename(corpus), n))
    for i in sorted(found):
        files = found[i]
        print("   pool %-3d %-16s in %2d captures, %3d entities: %s"
              % (i, WANT[i], len(files), sum(f[1] for f in files),
                 ", ".join("%s(x%d,%df)" % f for f in files[:4])))
