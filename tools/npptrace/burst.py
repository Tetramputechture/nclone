"""Within one discharge, does the fire direction (+0x90/+0x98) change per shot?

Capture-reading only, no simulation. Also reports how many shots a discharge
contains and how far apart they are, from the +0x88 / +0x8C counters.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOLS = (10, 11)
dirs_per_burst = collections.Counter()
shots_per_burst = collections.Counter()
gap = collections.Counter()
b0 = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        for pool in POOLS:
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            cur = collections.defaultdict(list)
            last_state = {}
            last_shot_frame = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    key = (demo, pool, i)
                    if st == 2:
                        b0[struct.unpack_from("<d", b, 0xB0)[0]] += 1
                        sub = struct.unpack_from("<i", b, 0x8C)[0]
                        if sub == 0:
                            dv = (struct.unpack_from("<d", b, 0x90)[0],
                                  struct.unpack_from("<d", b, 0x98)[0])
                            cur[key].append(dv)
                            if key in last_shot_frame:
                                gap[f - last_shot_frame[key]] += 1
                            last_shot_frame[key] = f
                    elif last_state.get(key) == 2:
                        seq = cur.pop(key, [])
                        if seq:
                            shots_per_burst[len(seq)] += 1
                            dirs_per_burst[len(set(seq))] += 1
                        last_shot_frame.pop(key, None)
                    last_state[key] = st

print("shots per discharge:      %s" % shots_per_burst.most_common(6))
print("DISTINCT directions used: %s" % dirs_per_burst.most_common(6))
print("frames between shots:     %s" % gap.most_common(6))
print("+0xB0 during state 2:     %s" % b0.most_common(6))
