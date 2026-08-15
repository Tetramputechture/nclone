"""Is ninja byte +0x09 ever set? It gates target validity and we ignore it.

`full/ninja_10513C30.c` is `is_valid_target`:

    valid = state != 9 && state != 6
         && (this[0x218] == 0 || state != 8)
         && this[9] == 0

This port checks only `state not in (6, 8, 9)`. If `+0x09` is ever non-zero the
game refuses the kill and we do not -- which is §74's exact signature: every
component verified, far too many kills.

The ninja slab is captured, so read it.
"""
import collections
import glob
import os
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

vals = collections.Counter()
byoffset = collections.defaultdict(collections.Counter)
frames = 0
examples = []

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"))[:25]:
    stem = s.replace(".layout.json", "")
    try:
        cap = Capture(stem)
    except Exception:
        continue
    slab = cap.layout["ninjaSlab"]
    for f in range(1, min(cap.n, 4000) + 1):
        rec = cap.frame(f)
        b = rec[4: 4 + slab]
        frames += 1
        vals[b[9]] += 1
        if b[9] != 0 and len(examples) < 6:
            examples.append((os.path.basename(stem), f, b[9]))
        # also survey nearby single bytes so a mis-offset is visible
        for o in (8, 9, 10, 11):
            byoffset[o][b[o]] += 1

print("ninja byte +0x09 over %d captured frames:" % frames)
for v, n in vals.most_common(8):
    print("   value %3d : %8d  (%.2f%%)" % (v, n, 100.0 * n / max(frames, 1)))
print("\nexamples where it is non-zero:")
for e in examples:
    print("   %s" % (e,))
print("\nneighbouring bytes, for sanity:")
for o in sorted(byoffset):
    top = byoffset[o].most_common(4)
    print("   +0x%02X : %s" % (o, ", ".join("%d x%d" % kv for kv in top)))
