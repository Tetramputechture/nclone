"""Read the game's own toggle-mine radius (+0x28) and state (+0x30) from captures.

`FUN_10417F30` kills when `FUN_1050C650(x, y, +0x28, ninja_x, ninja_y, R)` and
`+0x30 == 0`, so those two fields ARE the hazard. Pool 19; types 1 and 21 share it.

Prints the radius observed for each state, so a state-dependent radius shows up
as more than one value per state.
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
I = lambda b, o: struct.unpack_from("<i", b, o)[0]

by_state = collections.defaultdict(collections.Counter)
transitions = collections.Counter()
seen = 0

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

for stem in stems:
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == 19 and q["count"]), None)
    if p is None or p["stride"] < 0x34:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    seen += 1
    prev = {}
    for f in range(1, cap.n + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            st, r = I(b, 0x30), D(b, 0x28)
            by_state[st][r] += 1
            if i in prev and prev[i] != st:
                transitions[(prev[i], st)] += 1
            prev[i] = st

print("captures with a mine pool: %d\n" % seen)
print("+0x28 radius observed, by +0x30 state:")
for st in sorted(by_state):
    for r, c in sorted(by_state[st].items()):
        print("   state %-3d  radius %-22.17g  x%d" % (st, r, c))
print("\nstate transitions observed: %s" % dict(sorted(transitions.items())))
