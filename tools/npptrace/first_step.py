"""The chaingun discharge's FIRST step: the offset from the charge's last angle.

Everything else about the type-12 discharge is now determined -- 20 shots, 4
frames apart, sweeping 1/60 rad per shot against the ninja's cross-velocity.
This is the last scalar.
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

STEP = 1.0 / 60.0
rows = []


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
        p = next((q for q in cap.pools if q["i"] == 10), None)   # type 12 only
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
                key = (demo, i)
                if st == 2:
                    if key not in start:
                        gx = struct.unpack_from("<d", b, 0x10)[0]
                        gy = struct.unpack_from("<d", b, 0x18)[0]
                        n1 = ninja[f - 1]
                        bear = math.atan2(n1["ypos"] - gy, n1["xpos"] - gx)
                        start[key] = (gun, wrap(bear - gun))
                    if last.get(key) != gun:
                        seqs[key].append(gun)
                        last[key] = gun
                else:
                    sq = seqs.pop(key, [])
                    st0 = start.pop(key, None)
                    last.pop(key, None)
                    if len(sq) >= 4 and st0 is not None:
                        gun0, toward = st0
                        first = wrap(sq[1] - sq[0])
                        later = [wrap(sq[j + 1] - sq[j])
                                 for j in range(1, len(sq) - 1)]
                        mean = sum(later) / len(later)
                        sign = 1.0 if mean > 0 else -1.0
                        rows.append((demo, round(first, 4), round(toward, 4),
                                     sign, round(first / (sign * STEP), 2),
                                     round(first - toward, 4)))

print("%d type-12 discharges" % len(rows))
print("\n%-10s %-9s %-9s %-6s %-9s %s"
      % ("demo", "first", "toward", "sign", "first/step", "first-toward"))
for r in rows[:16]:
    print("%-10s %-9.4f %-9.4f %-6.0f %-9.2f %.4f" % r)

mult = collections.Counter(round(r[4]) for r in rows)
print("\nfirst step as a multiple of the sweep step (1/60), rounded:")
for k, v in mult.most_common(10):
    print("   %+d x%d" % (k, v))
match = sum(1 for r in rows if abs(r[5]) < 0.02)
print("\nfirst step equals the gap to the ninja's bearing (within 0.02): %d / %d"
      % (match, len(rows)))
