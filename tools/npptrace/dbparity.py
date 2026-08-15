"""Death balls, python sim vs native sim, on a REAL attract demo with REAL input.

`diffcheck_entity_parity.py` reports 12/12 exact for type 25 -- but it uses
synthetic input on official maps, and the attract maps that fail have balls
spawning INSIDE solid tiles, which the official set does not exercise. This runs
the same comparison on a demo the two engines actually disagree about.

Usage: dbparity.py <demo> [max_frames]
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
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 400

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
nat = nclone_native.Simulator.load(md)

pyballs = py.entity_dic[25]
print("%s: %d death balls, %d tape frames" % (demo, len(pyballs), len(tape)))


def nat_balls(snap):
    return [e for e in snap if int(e.entity_type) == 25]


first = None
for f in range(1, min(MAXF, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)
    nb = nat_balls(nat.get_state_dynamic().entities)
    if len(nb) != len(pyballs):
        print("count mismatch: py %d nat %d" % (len(pyballs), len(nb)))
        break
    for i, (p, n) in enumerate(zip(pyballs, nb)):
        if abs(p.xpos - float(n.x)) > 1e-9 or abs(p.ypos - float(n.y)) > 1e-9:
            if first is None:
                first = (f, i, p.xpos, p.ypos, float(n.x), float(n.y))
            break
    if first:
        break

if first is None:
    print("no death-ball divergence in %d frames" % min(MAXF, len(tape)))
else:
    f, i, px, pypos, nx, ny = first
    print("\nFIRST DIVERGENCE  frame %d, ball %d" % (f, i))
    print("   python (%.17g, %.17g)" % (px, pypos))
    print("   native (%.17g, %.17g)" % (nx, ny))
    print("   delta  (%.6g, %.6g)" % (nx - px, ny - pypos))
