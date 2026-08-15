"""State dwell times, split by drone TYPE.

The earlier measurement pooled types 12 and 13 and concluded the durations are
not type-specific. That pooling is invalid: the two structs are different sizes
(0xE8 vs 0xD8), so reading `+0x7C` as the state on a type 13 is reading whatever
happens to live there.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

runs = collections.defaultdict(collections.Counter)
strides = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        for pool in (10, 11):
            p = next((q for q in cap.pools if q["i"] == pool), None)
            if p is None or not p["count"]:
                continue
            strides[(pool + 2, p["stride"])] += 1
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            cur = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    key = (demo, pool, i)
                    c = cur.get(key)
                    if c is None or c[0] != st:
                        if c is not None:
                            runs[(pool + 2, c[0])][f - c[1]] += 1
                        cur[key] = (st, f)

print("struct strides seen per type:")
for (t, stride), n in sorted(strides.items()):
    print("   type %-3d stride 0x%X  (%d captures)" % (t, stride, n))
print("\nstate dwell lengths, read at +0x7C, split by type:")
for key in sorted(runs):
    t, st = key
    tot = sum(runs[key].values())
    common = runs[key].most_common(4)
    print("   type %-3d state %-3d %4d runs   %s"
          % (t, st, tot, ", ".join("%d x%d" % c for c in common)))
