"""What state does the game start a toggle mine in, by entity TYPE?

Our loader decides from the type byte (1 vs 21). The captures can settle it:
pool 19 holds both, in map order, so frame 1 of every capture is a labelled
sample of (type -> initial +0x30).
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402

I = lambda b, o: struct.unpack_from("<i", b, o)[0]
D = lambda b, o: struct.unpack_from("<d", b, o)[0]

dec = NppAttractDecoder()
pairs = collections.Counter()
mismatched_counts = 0
seen = 0

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

for stem in stems:
    demo = os.path.basename(stem)
    if not demo.isdigit():
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == 19 and q["count"]), None)
    if p is None or p["stride"] < 0x34:
        continue
    # entity records in the decoded map: [type, x, y, orientation, mode] from 1230
    md = list(dec.create_nclone_map(path))
    mines = []
    i = 1230
    while i + 4 < len(md):
        t = int(md[i])
        if t in (1, 21):
            mines.append((t, int(md[i + 1]), int(md[i + 2]), int(md[i + 4])))
        i += 5
    if len(mines) != p["count"]:
        mismatched_counts += 1
        continue
    seen += 1
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    rec = cap.frame(1)
    for k, (t, mx, my, mode) in enumerate(mines):
        b = rec[base + k * p["stride"]: base + (k + 1) * p["stride"]]
        pairs[(t, mode, I(b, 0x30))] += 1

print("captures used: %d   (skipped for count mismatch: %d)\n" % (seen, mismatched_counts))
print("(map type, map mode byte) -> game's +0x30 on frame 1:")
for (t, mode, st), c in sorted(pairs.items()):
    print("   type %-3d mode %-3d  ->  state %d   x%d" % (t, mode, st, c))
