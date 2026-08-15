"""Why did we die here, when the game did not?

Usage: killctx.py <demo> <pool> [span]

Replays to our death frame against the capture, then prints, for the frames
around it: our ninja vs the game's, and every entity of that pool as the game
records it beside ours. Pool index = entity type - 2.
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

demo, POOL = sys.argv[1], int(sys.argv[2])
SPAN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ETYPE = POOL + 2

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
for s in glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json"):
    caps.setdefault(os.path.basename(s).replace(".layout.json", ""), s.replace(".layout.json", ""))

cap = Capture(caps[demo])
p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
if p is None:
    print("no pool %d in %s; pools present: %s"
          % (POOL, demo, [(q["i"], q["count"]) for q in cap.pools if q["count"]]))
    sys.exit(1)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
nj = h.sim.ninja
ours = h.sim.entity_dic.get(ETYPE, [])
print("%s pool %d (type %d): game count %d, ours %d, stride 0x%x"
      % (demo, POOL, ETYPE, p["count"], len(ours), p["stride"]))

n = min(cap.n, len(tape))
hist = []
for f in range(1, n + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    gx = struct.unpack_from("<d", rec, 4 + 0x20)[0]
    gy = struct.unpack_from("<d", rec, 4 + 0x28)[0]
    ents = []
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        ex, ey = struct.unpack_from("<dd", b, 0x10)
        st = struct.unpack_from("<i", b, 0x30)[0] if p["stride"] >= 0x34 else None
        o = ours[i] if i < len(ours) else None
        ents.append((ex, ey, st, o))
    hist.append((f, gx, gy, nj.xpos, nj.ypos, nj.state, ents))
    if h.ninja_has_died() or h.ninja_has_won():
        print("ended f=%d died=%s won=%s\n" % (f, h.ninja_has_died(), h.ninja_has_won()))
        break

for f, gx, gy, ox, oy, ost, ents in hist[-(2 * SPAN + 1):]:
    print("f=%-5d ninja game (%9.4f,%9.4f)  ours (%9.4f,%9.4f)  d=%.4f  st=%d"
          % (f, gx, gy, ox, oy, math.hypot(ox - gx, oy - gy), ost))
    for i, (ex, ey, st, o) in enumerate(ents):
        if o is None:
            print("      e%-2d game (%9.3f,%9.3f) st=%s   <no local counterpart>" % (i, ex, ey, st))
            continue
        dg = math.hypot(gx - ex, gy - ey)
        do = math.hypot(ox - o.xpos, oy - o.ypos)
        flag = "  <-- MISPLACED" if math.hypot(ex - o.xpos, ey - o.ypos) > 1e-9 else ""
        print("      e%-2d game (%9.3f,%9.3f) st=%-4s d_gg=%8.3f | ours (%9.3f,%9.3f) st=%-4s d_oo=%8.3f%s"
              % (i, ex, ey, st, dg, o.xpos, o.ypos, getattr(o, "state", None), do, flag))
