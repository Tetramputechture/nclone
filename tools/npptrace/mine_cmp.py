"""Compare our toggle mines' state against the game's, frame by frame.

Only type 21 is observable: the pool index is `type - 2`, so type 1 mines would
be pool -1 and are simply absent from the capture. Prints every frame either side
changes state, so a wrong transition shows as a mismatched frame.
"""
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
POOL, TYPE = 19, 21
OFF = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x30

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == POOL)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
mines = [e for lst in h.sim.entity_dic.values() for e in lst
         if getattr(e, "type", None) == TYPE]
print("%s: game %d type-%d mines, port %d" % (demo, p["count"], TYPE, len(mines)))
print("also in port: %s" % sorted({getattr(e, "type", None)
                                   for lst in h.sim.entity_dic.values() for e in lst}))

prev = {}
n = min(cap.n, len(inputs))
for f in range(1, n + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    for i in range(min(p["count"], len(mines))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gs = struct.unpack_from("<i", b, OFF)[0]
        ms = mines[i].state
        if prev.get(i) != (gs, ms):
            flag = "" if gs == ms else "   <== MISMATCH"
            print("  f%-5d mine %d  game=%s port=%s%s" % (f, i, gs, ms, flag))
            prev[i] = (gs, ms)
    if h.ninja_has_died():
        print("  ninja died at frame %d" % f)
        break
