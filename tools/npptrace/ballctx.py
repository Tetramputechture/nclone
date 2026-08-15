"""Both death-ball tracks around a divergence, with per-frame steps.

A 3.7 px gap opening in ONE frame from bit-identical is a discrete event -- a
bounce, a state change, a clamp -- not accumulated drift. Print the steps so the
event is visible.
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

DEMO, POOL, ENT, AT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ETYPE = POOL + 2
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"] + ENT * p["stride"]
gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
sim = h.sim
e = sim.entity_dic[ETYPE][ENT]
print("stride 0x%X -- dumping every float pair in the slab at the event\n"
      % p["stride"])

pg = po = None
print("frame |      game x,y      |      ours x,y      |  game step  |  ours step  "
      "|   ninja x,y  | ours xspeed,yspeed")
for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    b = cap.frame(f)[base: base + p["stride"]]
    gx, gy = struct.unpack_from("<dd", b, 0x10)
    if AT - 6 <= f <= AT + 6:
        n = gninja[f - 1]
        gs = ("%+.4f,%+.4f" % (gx - pg[0], gy - pg[1])) if pg else "-"
        os_ = ("%+.4f,%+.4f" % (e.xpos - po[0], e.ypos - po[1])) if po else "-"
        flag = " <<<" if (abs(e.xpos - gx) > 1e-9 or abs(e.ypos - gy) > 1e-9) else ""
        print("%5d | %9.4f,%9.4f | %9.4f,%9.4f | %s | %s | %6.1f,%6.1f | %+.4f,%+.4f%s"
              % (f, gx, gy, e.xpos, e.ypos, gs, os_, n["xpos"], n["ypos"],
                 getattr(e, "xspeed", 0.0), getattr(e, "yspeed", 0.0), flag))
        if f == AT:
            print("       game slab floats:", " ".join(
                "%s@0x%02X=%.4f" % ("d", o, struct.unpack_from("<d", b, o)[0])
                for o in range(0, p["stride"] - 7, 8)))
    pg, po = (gx, gy), (e.xpos, e.ypos)
    if f > AT + 6:
        break
    if h.ninja_has_died() or h.ninja_has_won():
        print("(run ended frame %d)" % f)
        break
