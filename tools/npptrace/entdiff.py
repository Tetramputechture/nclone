"""First frame the two engines disagree, on a real attract demo with real input.

    entdiff.py <demo> [type] [max_frames]

Reports the earliest of: an entity of `type` moving differently, or the ninja
moving differently. `diffcheck_entity_parity.py` runs synthetic input on official
maps and misses everything the attract corpus exercises; this is the same
comparison pointed at a demo the engines actually disagree about.
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
ETYPE = int(sys.argv[2]) if len(sys.argv) > 2 else None
MAXF = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
COMPARE_STATE = os.environ.get("ENTDIFF_STATE") == "1"

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
nat = nclone_native.Simulator.load(md)
pyents = py.entity_dic.get(ETYPE, []) if ETYPE else []
print("%s: %d frames, %d entities of type %s" % (demo, len(tape), len(pyents), ETYPE))

for f in range(1, min(MAXF, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)
    st = nat.get_state_dynamic()
    n = st.ninja

    if ETYPE:
        ne = [e for e in st.entities if int(e.entity_type) == ETYPE]
        if len(ne) == len(pyents):
            for i, (p, q) in enumerate(zip(pyents, ne)):
                dx = abs(p.xpos - float(q.x))
                dy = abs(p.ypos - float(q.y))
                ps = int(getattr(p, "state", 0) or 0)
                ns = int(q.state)
                if dx > 1e-9 or dy > 1e-9 or (COMPARE_STATE and ps != ns):
                    print("\nENTITY %d divergence at frame %d" % (ETYPE, f))
                    print("   idx %d  python (%.17g, %.17g) state=%s"
                          % (i, p.xpos, p.ypos, ps))
                    print("           native (%.17g, %.17g) state=%s"
                          % (float(q.x), float(q.y), int(q.state)))
                    print("   delta  (%.6g, %.6g)" % (float(q.x) - p.xpos,
                                                      float(q.y) - p.ypos))
                    sys.exit(0)

    for field in ("xpos", "ypos", "xspeed", "yspeed", "state"):
        pv = float(getattr(py.ninja, field))
        nv = float(getattr(n, field))
        if pv != nv:
            print("\nNINJA divergence at frame %d: %s python=%.17g native=%.17g"
                  % (f, field, pv, nv))
            print("   python pos (%.17g, %.17g) state=%d"
                  % (py.ninja.xpos, py.ninja.ypos, py.ninja.state))
            print("   native pos (%.17g, %.17g) state=%d"
                  % (float(n.xpos), float(n.ypos), int(n.state)))
            sys.exit(0)

print("no divergence in %d frames" % min(MAXF, len(tape)))
