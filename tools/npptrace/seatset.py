"""Every captured surface laser as a labelled seating sample.

For each laser in each capture: the map's spawn / orientation / mode byte, the
GAME's frame-1 anchor and normal (`+0x58`/`+0x60`, `+0x68`/`+0x70`), and OUR
seat. That is the whole training set for the seating rule, and it already exists
on disk -- no new captures needed.

Frame 1 is one tick in, so a crawling laser has moved up to 1.0 px; samples where
the game's laser is static across the first 8 frames are marked STATIC and are
the clean ones to fit against.
"""
import collections
import glob
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
dec = NppAttractDecoder()

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

rows = []
seen_demos = set()
for stem in stems:
    demo = os.path.basename(stem)
    if not demo.isdigit() or demo in seen_demos:
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == 21 and q["count"]), None)
    if p is None or p["stride"] < 0xB8:
        continue
    seen_demos.add(demo)
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]

    md = list(dec.create_nclone_map(path))
    spawns = []
    i = 1230
    while i + 4 < len(md):
        if md[i] == 23:
            spawns.append((md[i + 1] * 6, md[i + 2] * 6, md[i + 3], md[i + 4]))
        i += 5
    if len(spawns) != p["count"]:
        continue

    h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
    h.load_map_from_map_data(md)
    ours = h.sim.entity_dic.get(23, [])
    if len(ours) != p["count"]:
        continue

    r1 = cap.frame(1)
    r8 = cap.frame(min(8, cap.n))
    for i in range(p["count"]):
        b1 = r1[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        b8 = r8[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        nx, ny = D(b1, 0x68), D(b1, 0x70)
        if nx == 0.0 and ny == 0.0:
            continue                                   # spinner
        ax, ay = D(b1, 0x58), D(b1, 0x60)
        static = (D(b8, 0x58) == ax and D(b8, 0x60) == ay
                  and D(b8, 0x68) == nx and D(b8, 0x70) == ny)
        sx, sy, orient, mode = spawns[i]
        e = ours[i]
        rows.append(dict(demo=demo, i=i, sx=sx, sy=sy, orient=orient, mode=mode,
                         gax=ax, gay=ay, gnx=nx, gny=ny, static=static,
                         oax=e.xanchor, oay=e.yanchor, onx=e.xnormal, ony=e.ynormal,
                         omode=e.mode))

print("captured surface lasers: %d over %d demos" % (len(rows), len(seen_demos)))
static = [r for r in rows if r["static"]]
print("   static for the first 8 frames: %d" % len(static))

def close(a, b):
    return abs(a - b) < 1e-6

ok_n = [r for r in rows if close(r["gnx"], r["onx"]) and close(r["gny"], r["ony"])]
ok_a = [r for r in rows if close(r["gax"], r["oax"]) and close(r["gay"], r["oay"])]
print("\nour seat vs the game, frame 1:")
print("   normal matches: %d / %d" % (len(ok_n), len(rows)))
print("   anchor matches: %d / %d" % (len(ok_a), len(rows)))

# Hypothesis from 10251717: normal == orientation rotated +45 deg,
# anchor == spawn - 3*sqrt(2)*normal.
R = math.sqrt(0.5)
VEC = {0: (1.0, 0.0), 2: (0.0, 1.0), 4: (-1.0, 0.0), 6: (0.0, -1.0)}
hit = miss = nomap = 0
for r in rows:
    v = VEC.get(r["orient"])
    if v is None:
        nomap += 1
        continue
    # rotate +45: (x cos - y sin, x sin + y cos)
    hx = v[0] * R - v[1] * R
    hy = v[0] * R + v[1] * R
    if close(hx, r["gnx"]) and close(hy, r["gny"]):
        hit += 1
    else:
        miss += 1
print("\nhypothesis  normal == orientation rotated +45deg:")
print("   holds %d, fails %d (orientations off-axis: %d)" % (hit, miss, nomap))

print("\nsign-only errors in our normal (same axis, wrong sign): %d"
      % sum(1 for r in rows
            if close(r["gnx"], -r["onx"]) and close(r["gny"], -r["ony"])))

by = collections.Counter()
for r in rows:
    n = close(r["gnx"], r["onx"]) and close(r["gny"], r["ony"])
    by[(r["mode"], r["static"], n)] += 1
print("\n(map mode byte, game static, our normal correct) -> count:")
for k in sorted(by):
    print("   mode=%d static=%-5s normal_ok=%-5s  %d" % (k[0], k[1], k[2], by[k]))
