"""State history of one shooter drone in both engines, side by side.

`entlock.py` says the two ports first disagree about a type-12 drone on frame 43
of `10244473`: Python idle, native firing. A shooter cannot reach state 2 on
frame 43 by the documented route -- acquire, then a 53-frame charge -- so one of
them is not taking that route at all. The history shows which.

    shooterstates.py <demo> <entity_type> <index> [frames]
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

import nclone_native  # noqa: E402

from nclone.nsim import Simulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402
from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402

demo = sys.argv[1]
etype = int(sys.argv[2])
idx = int(sys.argv[3])
frames = int(sys.argv[4]) if len(sys.argv) > 4 else 60

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = (dec.decode_npp_attract_file(path).get("demo_data") or [])[:frames]
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
nat = nclone_native.Simulator.load(md)

print("frame | python                                  | native")
print("      | state timer fire  pos                   | state pos")
prev = None
for f, b in enumerate(tape, 1):
    hor, jmp = decode_input_to_controls(int(b))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)
    e = py.entity_dic[etype][idx]
    ne = [x for x in nat.get_state_dynamic().entities if x.entity_type == etype][idx]
    row = (getattr(e, "state", None), getattr(e, "timer", None),
           getattr(e, "fire_timer", None), round(e.xpos, 3), round(e.ypos, 3),
           ne.state, round(ne.x, 3), round(ne.y, 3))
    mark = "  <-- STATE DIFFERS" if row[0] != row[5] else ""
    # Print every frame where either side changes state, plus the divergence.
    if prev is None or row[0] != prev[0] or row[5] != prev[5] or mark:
        print("%5d | %5s %5s %5s  (%8.3f,%8.3f) | %5s (%8.3f,%8.3f)%s"
              % (f, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], mark))
    prev = row
