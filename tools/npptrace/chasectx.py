"""Both chaser tracks around the divergence, with the ninja, to read the rule.

Ours is at chase speed 16/7; the game moved 8/7 on the divergent frame. Either
the game drops to patrol speed there or ours latched onto the ninja when the
game had not.
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

DEMO, POOL, AT = "10138995", 13, 1197
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
sim = h.sim
e = sim.entity_dic[15][0]

print("frame |        game x,y        |        ours x,y        | game step |"
      " ours step | ninja x,y | ours dir/orient")
pg = po = None
for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if f < AT - 8:
        b = cap.frame(f)[base: base + p["stride"]]
        pg = struct.unpack_from("<dd", b, 0x10)
        po = (e.xpos, e.ypos)
        continue
    if f > AT + 10:
        break
    b = cap.frame(f)[base: base + p["stride"]]
    gx, gy = struct.unpack_from("<dd", b, 0x10)
    n = gninja[f - 1]
    gs = ("%+.5f,%+.5f" % (gx - pg[0], gy - pg[1])) if pg else "-"
    os_ = ("%+.5f,%+.5f" % (e.xpos - po[0], e.ypos - po[1])) if po else "-"
    flag = " <<<" if (abs(e.xpos - gx) > 1e-9 or abs(e.ypos - gy) > 1e-9) else ""
    print("%5d | %10.5f,%10.5f | %10.5f,%10.5f | %s | %s | %6.1f,%6.1f | %s/%s%s"
          % (f, gx, gy, e.xpos, e.ypos, gs, os_,
             n["xpos"], n["ypos"], e.dir, e.orientation, flag))
    pg, po = (gx, gy), (e.xpos, e.ypos)

print("\n8/7 = %.11f   16/7 = %.11f" % (8 / 7, 16 / 7))
