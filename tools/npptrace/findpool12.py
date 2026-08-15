"""Which captured corpora contain ZAP drones (type 14, pool index 12)?

drone_zap is now the second-largest cause of loss (13 of 69) and has never been
worked on. If any existing capture has type 14 in it, its positions can be
diffed against ours with no new capture and no game running.
"""
import glob
import os
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

ROOT = "/home/tetra/projects/nclone/tools/npptrace"
WANT = {12: "zap (14)", 24: "mini (26)", 23: "death_ball (25)",
        10: "chaingun (12)", 11: "laser (13)", 13: "chaser (15)"}

for corpus in sorted(glob.glob(os.path.join(ROOT, "out*"))):
    if not os.path.isdir(corpus):
        continue
    found = {}
    n = 0
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
    if not n:
        continue
    print("\n%-14s %d captures" % (os.path.basename(corpus), n))
    for i in sorted(found):
        files = found[i]
        tot = sum(f[1] for f in files)
        print("   pool %-3d %-14s in %2d captures, %3d entities, e.g. %s"
              % (i, WANT[i], len(files), tot,
                 ", ".join("%s(x%d,%df)" % f for f in files[:3])))
    if not found:
        print("   (none of the drone/ball pools)")
