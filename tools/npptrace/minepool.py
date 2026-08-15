"""Which pool holds TYPE 1 mines? Match counts, then read frame-1 +0x30.

Pool 19 is type 21. `pool = type - 2` cannot address type 1, so find it by
looking for a pool whose count equals the map's type-1 count in captures where
that count is distinctive.
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
votes = collections.Counter()
rows = []

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
    md = list(dec.create_nclone_map(path))
    counts = collections.Counter()
    coords = collections.defaultdict(list)
    i = 1230
    while i + 4 < len(md):
        t = int(md[i])
        if t:
            counts[t] += 1
            coords[t].append((int(md[i + 1]) * 6, int(md[i + 2]) * 6))
        i += 5
    n1 = counts.get(1, 0)
    if not n1:
        continue
    pools = {p["i"]: p for p in cap.pools if p["count"]}
    cand = [i for i, p in pools.items() if p["count"] == n1]
    rows.append((demo, n1, counts.get(21, 0), sorted(cand),
                 sorted((i, p["count"]) for i, p in pools.items())))
    for c in cand:
        votes[c] += 1

print("captures whose map has type-1 mines: %d" % len(rows))
print("pools whose count matches the type-1 count (votes): %s\n" % votes.most_common(8))
for demo, n1, n21, cand, pools in rows[:12]:
    print("  %-10s type1=%-4d type21=%-4d  matching pools=%s   pools=%s"
          % (demo, n1, n21, cand, pools))

# For the best candidate, check that positions line up too.
if votes:
    best = votes.most_common(1)[0][0]
    print("\nchecking positions for pool %d:" % best)
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
        md = list(dec.create_nclone_map(path))
        mines = []
        i = 1230
        while i + 4 < len(md):
            if int(md[i]) == 1:
                mines.append((int(md[i + 1]) * 6, int(md[i + 2]) * 6))
            i += 5
        p = next((q for q in cap.pools if q["i"] == best and q["count"] == len(mines)), None)
        if p is None or not mines:
            continue
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        rec = cap.frame(1)
        ok = sts = 0
        stc = collections.Counter()
        for k, (mx, my) in enumerate(mines):
            b = rec[base + k * p["stride"]: base + (k + 1) * p["stride"]]
            gx, gy = D(b, 0x10), D(b, 0x18)
            if abs(gx - mx) < 1e-9 and abs(gy - my) < 1e-9:
                ok += 1
            stc[I(b, 0x30)] += 1
            sts += 1
        print("   %-10s %d/%d positions match; frame-1 states %s"
              % (demo, ok, sts, dict(stc)))
