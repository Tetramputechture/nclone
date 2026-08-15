"""Generic per-entity divergence context: demo, pool, entity, frame."""
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
e = h.sim.entity_dic[ETYPE][ENT]

pg = po = None
print("frame |     game x,y     |     ours x,y     |  game step  |  ours step  "
      "|   ninja    | dir chasing")
for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    b = cap.frame(f)[base: base + p["stride"]]
    gx, gy = struct.unpack_from("<dd", b, 0x10)
    if AT - 6 <= f <= AT + 8:
        n = gninja[f - 1]
        gs = ("%+.4f,%+.4f" % (gx - pg[0], gy - pg[1])) if pg else "-"
        os_ = ("%+.4f,%+.4f" % (e.xpos - po[0], e.ypos - po[1])) if po else "-"
        flag = " <<<" if (abs(e.xpos - gx) > 1e-9 or abs(e.ypos - gy) > 1e-9) else ""
        print("%5d | %8.3f,%8.3f | %8.3f,%8.3f | %s | %s | %5.0f,%5.0f | %s %s%s"
              % (f, gx, gy, e.xpos, e.ypos, gs, os_, n["xpos"], n["ypos"],
                 e.dir, getattr(e, "chasing", "?"), flag))
    pg, po = (gx, gy), (e.xpos, e.ypos)
    if f > AT + 8:
        break
    if h.ninja_has_died() or h.ninja_has_won():
        break
print("\n8/7 = %.6f   16/7 = %.6f" % (8 / 7, 16 / 7))
