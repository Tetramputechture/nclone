"""The ball-ball kick exists and matches the game, but did not fire. Why?

Our balls are bit-exact with the game up to the event, so the 15.245 separation
was present in our sim too and `dist < 16` should have been true. Instrument the
guard: index, db_count, and the pairwise distances our sim actually sees.
"""
import glob
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

DEMO, AT = "10132476", 273
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
sim = h.sim
balls = sim.entity_dic[25]
print("our sim: %d death balls" % len(balls))
print("map_data[1200] (db_count) = %r" % sim.map_data[1200])
for b in balls:
    print("   ball: index=%r type=%r pos=(%.2f, %.2f)"
          % (getattr(b, "index", "MISSING"), b.type, b.xpos, b.ypos))

for f in range(1, AT + 2):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if f >= AT - 2:
        print("\nframe %d" % f)
        for i, b in enumerate(balls):
            print("   ball %d idx=%r (%9.4f,%9.4f) v=(%+.4f,%+.4f)"
                  % (i, getattr(b, "index", "?"), b.xpos, b.ypos,
                     b.xspeed, b.yspeed))
        for a in range(len(balls)):
            for c in range(a + 1, len(balls)):
                d = ((balls[a].xpos - balls[c].xpos) ** 2
                     + (balls[a].ypos - balls[c].ypos) ** 2) ** 0.5
                if d < 40:
                    ia = getattr(balls[a], "index", -1)
                    fires = (ia + 1) < sim.map_data[1200] and d < 16
                    print("   pair %d-%d dist %.4f   guard(index+1<db_count)=%s"
                          "  dist<16=%s  WOULD FIRE=%s"
                          % (a, c, d, (ia + 1) < sim.map_data[1200], d < 16, fires))
