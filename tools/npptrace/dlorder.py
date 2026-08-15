"""Dual lasers 2 and 3 diverge on frame 1 -- that is spawn order, not behaviour.

At frame 1, match each of our entities to the nearest captured one and report the
permutation. If it is not the identity, our entity_dic order does not follow the
pool order and every later comparison for those slots is meaningless.
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

POOL, ETYPE = 21, 23
dec = NppAttractDecoder()
perm_counts = {}

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"))[:40]:
    stem = s.replace(".layout.json", "")
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
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    if not ours or not tape:
        continue
    hor, jmp = decode_input_to_controls(int(tape[0]))
    h.tick(hor, jmp)
    rec = cap.frame(1)
    game = []
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        game.append(struct.unpack_from("<dd", b, 0x10))
    perm = []
    for i, e in enumerate(ours[:len(game)]):
        best = min(range(len(game)),
                   key=lambda j: math.hypot(e.xpos - game[j][0], e.ypos - game[j][1]))
        d = math.hypot(e.xpos - game[best][0], e.ypos - game[best][1])
        perm.append(best if d < 1.0 else None)
    key = tuple(perm)
    perm_counts.setdefault(key, []).append(demo)

print("frame-1 permutation (our index -> nearest captured index):\n")
for k, demos in sorted(perm_counts.items(), key=lambda t: -len(t[1])):
    ident = tuple(range(len(k))) == k
    print("   %-28s x%-3d %s   e.g. %s"
          % (str(k), len(demos), "IDENTITY" if ident else "<-- PERMUTED",
             demos[0]))
