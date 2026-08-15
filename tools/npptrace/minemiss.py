"""How close does our ninja actually pass the mine that kills it, if it cannot die?

Usage: minemiss.py <demo> [<demo> ...]

The game wins these demos, so its ninja never comes within `4 + 10` of a toggled
mine. Neutering `kill` and reporting our closest approach says whether we are
marginally off (a boundary case) or on a different path entirely.
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
from nclone.ninja import NINJA_RADIUS  # noqa: E402

dec = NppAttractDecoder()
for demo in sys.argv[1:]:
    path = "/home/tetra/projects/attracts/%s" % demo
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    nj = h.sim.ninja
    nj.kill = lambda *a, **k: None          # invulnerable
    mines = [e for t in (1, 21) for e in h.sim.entity_dic.get(t, [])]
    best = {}
    won_at = None
    for f in range(1, len(tape) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        pre = [(m.state, m.RADIUS) for m in mines]
        h.tick(hor, jmp)
        for m, (st, r) in zip(mines, pre):
            if st != 0:
                continue
            d = math.hypot(nj.xpos - m.xpos, nj.ypos - m.ypos)
            gap = d - (r + NINJA_RADIUS)
            k = (m.xpos, m.ypos)
            if k not in best or gap < best[k][0]:
                best[k] = (gap, d, f)
        if h.ninja_has_won():
            won_at = f
            break
    print("== %s  won=%s at f=%s (%d tape frames)" % (demo, won_at is not None, won_at, len(tape)))
    rows = sorted(best.items(), key=lambda kv: kv[1][0])[:5]
    for (mx, my), (gap, d, f) in rows:
        print("   mine (%7.2f,%7.2f)  closest dist=%8.4f on f=%-5d  gap vs 14.0 = %+8.4f%s"
              % (mx, my, d, f, gap, "   <-- we would die here" if gap < 0 else ""))
