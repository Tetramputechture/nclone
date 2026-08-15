"""Diff our zap-drone / death-ball positions against the game's, on a demo we LOSE.

10138995 is one of the 13 drone_zap losers and was captured; 10132476 is one of
the 7 death_ball losers and was captured. §3's method applies directly: first
divergent frame, first divergent field.
"""
import glob
import json
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

DEMO = sys.argv[1]
POOL = int(sys.argv[2])
ETYPE = POOL + 2

stem = None
for corpus in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*"):
    c = os.path.join(corpus, DEMO)
    if os.path.exists(c + ".layout.json"):
        stem = c
        break
if stem is None:
    raise SystemExit("no capture for %s" % DEMO)
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
print("%s: pool %d (type %d), %d entities, stride 0x%X, %d captured frames"
      % (DEMO, POOL, ETYPE, p["count"], p["stride"], cap.n))

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
mapb = list(dec.create_nclone_map(path))
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(mapb)
sim = h.sim
ours = sim.entity_dic.get(ETYPE, [])
print("our sim has %d entities of type %d\n" % (len(ours), ETYPE))

first = None
for f in range(1, min(cap.n, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    for i in range(min(p["count"], len(ours))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        e = ours[i]
        dx, dy = e.xpos - gx, e.ypos - gy
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            if first is None:
                first = f
                print("FIRST DIVERGENCE frame %d, entity %d" % (f, i))
                print("   game (%.17g, %.17g)" % (gx, gy))
                print("   ours (%.17g, %.17g)" % (e.xpos, e.ypos))
                print("   d    (%.3e, %.3e)" % (dx, dy))
                for attr in ("state", "dir", "orientation", "mode", "speed"):
                    if hasattr(e, attr):
                        print("   ours.%-12s = %r" % (attr, getattr(e, attr)))
                # A few frames of context on this entity.
                print("\n   next 6 frames for entity %d:" % i)
    if first is not None and f > first + 5:
        break
    if h.ninja_has_died() or h.ninja_has_won():
        print("(run ended at frame %d)" % f)
        break

if first is None:
    print("no divergence in %d frames -- positions are bit-identical"
          % min(cap.n, len(tape)))
