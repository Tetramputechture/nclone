"""Ninja state plus every entity within R px, over a frame window. No capture needed.

Usage: ctx2.py <demo> <first> <last> [radius]
"""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

demo = sys.argv[1]
FIRST, LAST = int(sys.argv[2]), int(sys.argv[3])
R = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
nj = h.sim.ninja
nj.kill = lambda *a, **k: None
allents = [(t, e) for t, lst in h.sim.entity_dic.items() for e in lst]

for f in range(1, min(LAST, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if f < FIRST:
        continue
    near = []
    for t, e in allents:
        if t in (2,):          # gold is noise
            continue
        d = math.hypot(nj.xpos - e.xpos, nj.ypos - e.ypos)
        if d <= R:
            near.append("t%d@(%.0f,%.0f)d=%.1f%s"
                        % (t, e.xpos, e.ypos, d,
                           ("st%s" % e.state) if hasattr(e, "state") else ""))
    print("f=%-5d in=%2d,%d pos=(%9.4f,%9.4f) v=(%+7.4f,%+7.4f) st=%d fl=(%.3f,%.3f) w=%s | %s"
          % (f, hor, jmp, nj.xpos, nj.ypos, nj.xspeed, nj.yspeed, nj.state,
             nj.floor_normalized_x, nj.floor_normalized_y,
             getattr(nj, "walled", None), " ".join(near)))
    if h.ninja_has_won():
        print("WON f=%d" % f)
        break
