"""Compare our chaser drones (type 15) against the game's, position by position.

Reports the first frame each one leaves the game's path, so the earliest wrong
turn is visible rather than the one that happens to kill.
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
POOL, TYPE = 13, 15

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == POOL)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
drones = [e for lst in h.sim.entity_dic.values() for e in lst
          if getattr(e, "type", None) == TYPE]
print("%s: game %d chasers, port %d" % (demo, p["count"], len(drones)))

first = {}
died = None
n = min(cap.n, len(inputs))
for f in range(1, n + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    if h.ninja_has_died() and died is None:
        died = f
    rec = cap.frame(f)
    for i in range(min(p["count"], len(drones))):
        if i in first:
            continue
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx = struct.unpack_from("<d", b, 0x10)[0]
        gy = struct.unpack_from("<d", b, 0x18)[0]
        e = drones[i]
        if (e.xpos, e.ypos) != (gx, gy):
            first[i] = (f, (e.xpos, e.ypos), (gx, gy))
    if died:
        break

print("port ninja dies at %s" % died)
print("%-6s %-8s %-24s %-24s %s" % ("drone", "diverges", "port", "game", "gap"))
for i in range(min(p["count"], len(drones))):
    if i not in first:
        print("%-6d %-8s matches to the kill" % (i, "never"))
        continue
    f, s, g = first[i]
    d = ((s[0] - g[0]) ** 2 + (s[1] - g[1]) ** 2) ** 0.5
    print("%-6d %-8d (%9.3f,%9.3f) (%9.3f,%9.3f) %.3f"
          % (i, f, s[0], s[1], g[0], g[1], d))
