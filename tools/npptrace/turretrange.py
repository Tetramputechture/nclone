"""How far away will the GAME's turret acquire a target?

`hazardscore.py` shows our turret entering state 1 (TRACKING) on `10136880`
frames 129 and 145 while the game stays in state 0 -- with a clear line of sight
and the ninja 222 px away. Our `SEARCHING -> TRACKING` rule is pure line of
sight, with no range gate at all. If the game has one, every turret in the corpus
acquires too early and the 210 turret losses have a single cause.

Measured, not assumed: across every capture, take each frame the game's own
`+0xA4` goes 0 -> 1 and record how far the ninja was. The largest such distance
is an upper bound on the acquisition range; the distances the game DECLINED to
acquire at, while visible, bound it from below.
"""
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
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.physics import raycast_vs_player  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
I = lambda b, o: struct.unpack_from("<i", b, o)[0]
POOL, STATE_OFF = 17, 0xA4

dec = NppAttractDecoder()
stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

acquired = []     # distance at each observed 0 -> 1 transition
declined = []     # distance while state 0 AND we had line of sight
seen = set()

for stem in stems:
    demo = os.path.basename(stem)
    if not demo.isdigit() or demo in seen:
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None or p["stride"] < STATE_OFF + 4:
        continue
    try:
        md = list(dec.create_nclone_map(path))
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    except Exception:
        continue
    seen.add(demo)
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]

    h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
    h.load_map_from_map_data(md)
    n = h.sim.ninja

    prev = [None] * p["count"]
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        row = cap.frame(f)
        for i in range(p["count"]):
            b = row[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            tx, ty, st = D(b, 0x10), D(b, 0x18), I(b, STATE_OFF)
            d = math.hypot(n.xpos - tx, n.ypos - ty)
            if prev[i] == 0 and st == 1:
                acquired.append((d, demo, f))
            elif st == 0 and raycast_vs_player(h.sim, tx, ty, n.xpos, n.ypos):
                declined.append(d)
            prev[i] = st

print("captured turret demos: %d" % len(seen))
print("observed 0 -> 1 acquisitions: %d" % len(acquired))
if acquired:
    ds = sorted(d for d, _, _ in acquired)
    print("   acquisition distance  min %.1f  median %.1f  MAX %.1f"
          % (ds[0], ds[len(ds) // 2], ds[-1]))
    print("   farthest few:")
    for d, demo, f in sorted(acquired, reverse=True)[:6]:
        print("      %8.1f px   %s f=%d" % (d, demo, f))

print("\nframes the game stayed in state 0 WITH line of sight: %d" % len(declined))
if declined:
    ds = sorted(declined)
    print("   declined-at distance  min %.1f  median %.1f  max %.1f"
          % (ds[0], ds[len(ds) // 2], ds[-1]))
    # If a range gate exists, the smallest distance the game ever declines at
    # sits just above the largest it ever acquires at.
    if acquired:
        print("\n   largest ACQUIRED  %.1f" % max(d for d, _, _ in acquired))
        print("   smallest DECLINED %.1f" % ds[0])
        print("   -> a clean range gate would show these two well separated;"
              " overlapping means sight is not the only other rule")
