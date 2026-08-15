"""Exact bits around the 1011495 frame-76 divergence, plus the segment we sit on.

Usage: ulp.py <demo> <frame>
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
from nclone.physics import gather_segments_from_region  # noqa: E402

demo, TARGET = sys.argv[1], int(sys.argv[2])
bits = lambda x: struct.unpack("<q", struct.pack("<d", float(x)))[0]

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
for s in glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json"):
    caps.setdefault(os.path.basename(s).replace(".layout.json", ""), s.replace(".layout.json", ""))
cap = Capture(caps[demo])

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
nj = h.sim.ninja

for f in range(1, TARGET + 2):
    pre = (nj.xpos, nj.ypos, nj.xspeed, nj.yspeed)
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if f < TARGET - 1:
        continue
    rec = cap.frame(f)
    gx = struct.unpack_from("<d", rec, 4 + 0x20)[0]
    gy = struct.unpack_from("<d", rec, 4 + 0x28)[0]
    gvx = struct.unpack_from("<d", rec, 4 + 0x30)[0]
    gvy = struct.unpack_from("<d", rec, 4 + 0x38)[0]
    print("f=%d  (pre: x=%.17g y=%.17g vx=%.17g vy=%.17g)" % ((f,) + pre))
    for name, ours, game in (("xpos", nj.xpos, gx), ("ypos", nj.ypos, gy),
                             ("xspeed", nj.xspeed, gvx), ("yspeed", nj.yspeed, gvy)):
        d = bits(ours) - bits(game)
        print("   %-7s ours %016x %-24.17g | game %016x %-24.17g | ulp %+d"
              % (name, bits(ours) & 0xFFFFFFFFFFFFFFFF, ours,
                 bits(game) & 0xFFFFFFFFFFFFFFFF, game, d))
    if f == TARGET:
        print("   floor normal ours (%.17g, %.17g)"
              % (nj.floor_normalized_x, nj.floor_normalized_y))
        segs = gather_segments_from_region(h.sim, nj.xpos - 12, nj.ypos - 12,
                                           nj.xpos + 12, nj.ypos + 12)
        print("   segments in region:")
        for s in segs:
            d = {k: v for k, v in s.__dict__.items() if isinstance(v, (int, float, bool))}
            back, a, b = s.get_closest_point(nj.xpos, nj.ypos)
            print("      %-20s %s  closest=(%.17g, %.17g) dist=%.17g back=%s"
                  % (type(s).__name__, d, a, b, math.hypot(nj.xpos - a, nj.ypos - b), back))
