"""Which toggle mine kills us, in what state, and by how much does it overlap?

Usage: minewhy.py <demo> [<demo> ...]   (no capture required)
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
    mines = [e for t in (1, 21) for e in h.sim.entity_dic.get(t, [])]
    hist = []
    for f in range(1, len(tape) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        prev = [(m.state, m.RADIUS) for m in mines]
        h.tick(hor, jmp)
        hist.append((f, nj.xpos, nj.ypos, prev))
        if h.ninja_has_died() or h.ninja_has_won():
            break
    f, nx, ny, prev = hist[-1]
    print("== %s  ended f=%d died=%s  ninja (%.4f, %.4f)"
          % (demo, f, h.ninja_has_died(), nx, ny))
    rows = []
    for m, (pst, prad) in zip(mines, prev):
        d = math.hypot(nx - m.xpos, ny - m.ypos)
        rows.append((d - (prad + NINJA_RADIUS), d, m, pst, prad))
    rows.sort(key=lambda r: r[0])
    print("   nearest mines at the death frame (state/radius as of the START of it):")
    for gap, d, m, pst, prad in rows[:6]:
        print("      (%7.2f,%7.2f) state %d r=%.2f  dist=%8.4f  overlap bound=%5.2f  slack=%+8.4f%s"
              % (m.xpos, m.ypos, pst, prad, d, prad + NINJA_RADIUS, -gap,
                 "   <-- KILLED" if gap <= 0 else ""))
    # replay the last few frames of the killer's state history
    killer = next((r[2] for r in rows if r[0] <= 0), None)
    if killer is None:
        print("   no mine overlapped -- the kill came from somewhere else")
        continue
    print("   killer state history (last 30 frames), gap = dist - (r + 10):")
    h2 = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h2.load_map_from_map_data(list(dec.create_nclone_map(path)))
    m2 = [e for t in (1, 21) for e in h2.sim.entity_dic.get(t, [])][mines.index(killer)]
    nj2 = h2.sim.ninja
    out = []
    for g in range(1, f + 1):
        hor, jmp = decode_input_to_controls(int(tape[g - 1]))
        st0, r0 = m2.state, m2.RADIUS
        h2.tick(hor, jmp)
        d = math.hypot(nj2.xpos - m2.xpos, nj2.ypos - m2.ypos)
        out.append("      f=%-5d pre(st=%d r=%.1f) post(st=%d r=%.1f) ninja(%8.3f,%8.3f) "
                   "dist=%8.4f gap=%+8.4f"
                   % (g, st0, r0, m2.state, m2.RADIUS, nj2.xpos, nj2.ypos, d,
                      d - (r0 + NINJA_RADIUS)))
    print("\n".join(out[-30:]))
    print()
