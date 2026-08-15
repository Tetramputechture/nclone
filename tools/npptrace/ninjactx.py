"""Full ninja state, ours vs the game's, around a named frame.

Usage: ninjactx.py <demo> <frame> [<span>]
"""
import glob
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

F_XPOS, F_YPOS = 0x20, 0x28
F_XSPEED, F_YSPEED = 0x30, 0x38
F_STATE = 0xD0
F_FLOORX, F_FLOORY = 0x120, 0x128

demo = sys.argv[1]
target = int(sys.argv[2])
span = int(sys.argv[3]) if len(sys.argv) > 3 else 6

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

n = min(cap.n, len(tape))
print("%-6s %-4s | %-24s %-24s | %-12s %-12s | %-4s | %-10s %-10s"
      % ("frame", "inp", "x  (game/ours)", "y  (game/ours)", "vx", "vy", "st", "floor_x", "floor_y"))
for f in range(1, min(n, target + span) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if f < target - span:
        continue
    r = cap.frame(f)
    g = lambda off, t="d": struct.unpack_from("<" + t, r, 4 + off)[0]
    gx, gy, gvx, gvy = g(F_XPOS), g(F_YPOS), g(F_XSPEED), g(F_YSPEED)
    gst = g(F_STATE, "i")
    gfx, gfy = g(F_FLOORX), g(F_FLOORY)
    mark = lambda a, b: "" if a == b else " *"
    print("%-6d %2d,%-1d | %-24.17g %-24.17g | g %-12.6g %-12.6g | %-4d | %-10.6g %-10.6g"
          % (f, hor, jmp, gx, gy, gvx, gvy, gst, gfx, gfy))
    print("       %-4s | %-24.17g %-24.17g | o %-12.6g %-12.6g | %-4d | %-10.6g %-10.6g  %s%s"
          % ("", nj.xpos, nj.ypos, nj.xspeed, nj.yspeed, nj.state,
             nj.floor_normalized_x, nj.floor_normalized_y,
             mark(gx, nj.xpos) + mark(gy, nj.ypos),
             " STATE" if gst != nj.state else ""))
    if h.ninja_has_died() or h.ninja_has_won():
        print("   -> ended f=%d died=%s won=%s" % (f, h.ninja_has_died(), h.ninja_has_won()))
        break
