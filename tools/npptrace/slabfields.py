"""What are +0x30 / +0x38 in the chaser slab?

`t15_Chaser_think.c` reads the drone position from +0x30/+0x38 while
`FUN_103fe0d0` uses +0x10/+0x18. §59: they are different fields and a correct
port needs to know which is which. Compare them against our xpos/ypos and
xtarget/ytarget.
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

DEMO, POOL, ETYPE = "1001030", 13, 15
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
e = h.sim.entity_dic[ETYPE][0]

print("chaser slab stride 0x%X\n" % p["stride"])
print("%5s | %-19s %-19s | %-19s %-19s"
      % ("frame", "game +0x10/+0x18", "game +0x30/+0x38", "our xpos,ypos",
         "our xtarget,ytarget"))
for f in range(1, 26):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    b = cap.frame(f)[base: base + p["stride"]]
    a10, a18 = struct.unpack_from("<dd", b, 0x10)
    a30, a38 = struct.unpack_from("<dd", b, 0x30)
    print("%5d | %8.2f,%8.2f  %8.2f,%8.2f | %8.2f,%8.2f  %8.2f,%8.2f"
          % (f, a10, a18, a30, a38, e.xpos, e.ypos,
             getattr(e, "xtarget", 0.0), getattr(e, "ytarget", 0.0)))
