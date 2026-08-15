"""Three candidates for the discharge sweep's sign, scored on all 95 discharges.

  A  the ninja's bearing on the frame BEFORE the burst starts
  B  the latched bearing at +0xE0
  C  the sign of the ninja's velocity ACROSS the line of fire (lead the target)

Capture-only. The ninja's position and velocity come from the `.ninja.jsonl`.
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

POOLS = (10, 11)
score = collections.Counter()
band = collections.defaultdict(lambda: collections.Counter())


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        nj = os.path.join(d, demo + ".ninja.jsonl")
        if not os.path.exists(nj):
            continue
        ninja = [json.loads(l) for l in open(nj) if l.strip()]
        cap = Capture(os.path.join(d, demo))
        for pool in POOLS:
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            start, seqs, last = {}, collections.defaultdict(list), {}
            for f in range(1, min(cap.n, len(ninja)) + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    key = (demo, pool, i)
                    if st == 2:
                        if key not in start and f >= 2:
                            gx = struct.unpack_from("<d", b, 0x10)[0]
                            gy = struct.unpack_from("<d", b, 0x18)[0]
                            # +0xE0 exists only in the 0xE8 type-12 slab (§26).
                            e0 = (struct.unpack_from("<f", b, 0xE0)[0]
                                  if p["stride"] >= 0xE4 else None)
                            n0 = ninja[f - 2]          # the frame BEFORE
                            n1 = ninja[f - 1]
                            bear_prev = math.atan2(n0["ypos"] - gy, n0["xpos"] - gx)
                            # velocity across the line of fire
                            cross = (n1["xspeed"] * math.sin(gun)
                                     - n1["yspeed"] * math.cos(gun))
                            start[key] = (gun, bear_prev, e0, cross)
                        if last.get(key) != gun:
                            seqs[key].append(gun)
                            last[key] = gun
                    else:
                        sq = seqs.pop(key, [])
                        st0 = start.pop(key, None)
                        last.pop(key, None)
                        if len(sq) >= 4 and st0 is not None:
                            steps = [wrap(sq[j + 1] - sq[j])
                                     for j in range(1, len(sq) - 1)]
                            mean = sum(steps) / len(steps)
                            gun0, bear_prev, e0, cross = st0
                            pos = mean > 0
                            cands = {
                                "A prev-frame bearing": wrap(bear_prev - gun0) > 0,
                                "B latched +0xE0": (wrap(e0 - gun0) > 0
                                                    if e0 is not None else None),
                                "C ninja cross-velocity": cross > 0,
                                "D against cross-velocity": cross < 0,
                            }
                            for name, pred in cands.items():
                                if pred is None:
                                    continue
                                score[(name, pred == pos)] += 1
                                gap = abs(wrap(bear_prev - gun0))
                                if gap >= 0.01:
                                    band[name][pred == pos] += 1

names = ["A prev-frame bearing", "B latched +0xE0",
         "C ninja cross-velocity", "D against cross-velocity"]
print("%-26s %-18s %s" % ("candidate", "all 95", "excl. sub-0.01 gaps"))
for n in names:
    ok, bad = score[(n, True)], score[(n, False)]
    bok, bbad = band[n][True], band[n][False]
    print("%-26s %3d/%-3d %5.0f%%     %3d/%-3d %5.0f%%"
          % (n, ok, ok + bad, 100.0 * ok / max(ok + bad, 1),
             bok, bok + bbad, 100.0 * bok / max(bok + bbad, 1)))
