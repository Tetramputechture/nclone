"""Compare our rocket projectile against the game's, frame by frame.

The launcher is static at `+0x10`/`+0x18`; the projectile it fires lives in the
same slab at `+0x50`/`+0x58` with its direction at `+0x60`/`+0x68` and a state
at `+0x90`.
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
                        "/home/tetra/projects/nclone/tools/npptrace/out11")
demo = sys.argv[1]
lo, hi = int(sys.argv[2]), int(sys.argv[3])
POOL, TYPE = 16, 18

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == POOL)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
rockets = [e for lst in h.sim.entity_dic.values() for e in lst
           if getattr(e, "type", None) == TYPE]
print("%s: game %d rockets, port %d" % (demo, p["count"], len(rockets)))
print("port rocket attrs: %s"
      % sorted(k for k in vars(rockets[0]) if not k.startswith("_"))[:24])

print("\n%5s %4s %10s %10s %8s %8s %6s" %
      ("frame", "who", "px", "py", "dx", "dy", "state"))
for f in range(1, min(hi, cap.n, len(inputs)) + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    if f < lo:
        continue
    rec = cap.frame(f)
    for i in range(min(p["count"], len(rockets))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        e = rockets[i]
        print("%5d %2dG %10.3f %10.3f %8.4f %8.4f %6d"
              % (f, i,
                 struct.unpack_from("<d", b, 0x50)[0],
                 struct.unpack_from("<d", b, 0x58)[0],
                 struct.unpack_from("<d", b, 0x60)[0],
                 struct.unpack_from("<d", b, 0x68)[0],
                 struct.unpack_from("<i", b, 0x90)[0]))
        print("%5d %2dP %10.3f %10.3f %8s %8s %6s"
              % (f, i,
                 e.projectile.xpos,
                 e.projectile.ypos,
                 round(getattr(e.projectile, "xdir", 0.0), 4),
                 round(getattr(e.projectile, "ydir", 0.0), 4),
                 getattr(e, "state", "-")))
    if h.ninja_has_died():
        print("  ninja dies at %d cause=%s"
              % (f, getattr(h.sim.ninja, "death_cause", None)))
        break
