"""Closed-loop: our live `aim_angle` against the capture's `+0x50`, every frame.

The forward model matches 98.1% of charge frames when driven from the game's own
previous angle. This runs it from OUR own, which is what the simulation does, and
prints where the two part company.
"""
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["NCLONE_AB_DRONE_FULL_SHOT"] = "1"
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

demo = sys.argv[1] if len(sys.argv) > 1 else "10131718"
sub = sys.argv[2] if len(sys.argv) > 2 else "out9"
pool = int(sys.argv[3]) if len(sys.argv) > 3 else 10
lo = int(sys.argv[4]) if len(sys.argv) > 4 else 35
hi = int(sys.argv[5]) if len(sys.argv) > 5 else 175

cap = Capture("/home/tetra/projects/nclone/tools/npptrace/%s/%s" % (sub, demo))
p = next(q for q in cap.pools if q["i"] == pool)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ds = [e for lst in h.sim.entity_dic.values() for e in lst
      if getattr(e, "type", None) == pool + 2]
e = ds[0]


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


print("%5s %6s %12s %12s %10s %s" %
      ("frame", "state", "port aim", "game +0x50", "error", "note"))
shown = 0
for f in range(1, min(hi, cap.n, len(inputs)) + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    if f < lo:
        continue
    b = cap.frame(f)[base: base + p["stride"]]
    gun = struct.unpack_from("<f", b, 0x50)[0]
    gst = struct.unpack_from("<i", b, 0x7C)[0]
    err = abs(wrap(e.aim_angle - gun))
    note = ""
    if err > 0.05:
        note = "<== diverged"
    if f <= lo + 6 or err > 0.01 or f % 20 == 0:
        print("%5d %6d %12.5f %12.5f %10.5f %s"
              % (f, gst, e.aim_angle, gun, err, note))
        shown += 1
    if shown > 44:
        break
