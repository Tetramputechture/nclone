"""Compare our evil ninjas (type 22) against the game's, frame by frame."""
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

TRACES = os.environ.get("NCLONE_TRACES",
                        "/home/tetra/projects/nclone/tools/npptrace/out9_batch1")
demo = sys.argv[1]
upto = int(sys.argv[2])
POOL, TYPE = 20, 22

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == POOL)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ev = [e for lst in h.sim.entity_dic.values() for e in lst
      if getattr(e, "type", None) == TYPE]
print("%s: game %d, port %d evil ninjas" % (demo, p["count"], len(ev)))

prev = {}
for f in range(1, min(upto, cap.n, len(inputs)) + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    for i in range(min(p["count"], len(ev))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx = struct.unpack_from("<d", b, 0x10)[0]
        gy = struct.unpack_from("<d", b, 0x18)[0]
        e = ev[i]
        sig = (round(gx, 3), round(gy, 3), round(e.xpos, 3), round(e.ypos, 3))
        if prev.get(i) != sig:
            d = ((e.xpos - gx) ** 2 + (e.ypos - gy) ** 2) ** 0.5
            print("  f%-5d ev %d  game=(%8.3f,%8.3f)  port=(%8.3f,%8.3f)  gap %.3f"
                  % (f, i, gx, gy, e.xpos, e.ypos, d))
            prev[i] = sig
    if h.ninja_has_died():
        print("  ninja dies at frame %d, cause=%s"
              % (f, getattr(h.sim.ninja, "death_cause", None)))
        break
