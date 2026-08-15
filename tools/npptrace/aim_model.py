"""Does the gun angle (+0x50) lerp toward the ninja's CURRENT bearing, or toward
the bearing latched at acquisition (+0xE0)?

Both models predict the first few frames identically; they separate over a
53-frame charge while the ninja moves. Score each against every charge frame.
"""
import math
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

TRACES = "/home/tetra/projects/nclone/tools/npptrace/out9"
demo, pool, idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lo, hi = int(sys.argv[4]), int(sys.argv[5])
LERP = 0.2

cap = Capture(os.path.join(TRACES, demo))
p = next(q for q in cap.pools if q["i"] == pool)
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


track = latch = None
print("%5s %10s %10s %10s %10s %10s"
      % ("frame", "game 0x50", "0xE0", "bearing", "track", "latch"))
for f in range(1, hi + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    if f < lo:
        continue
    b = cap.frame(f)[base + idx * p["stride"]: base + (idx + 1) * p["stride"]]
    gun = struct.unpack_from("<f", b, 0x50)[0]
    e0 = struct.unpack_from("<f", b, 0xE0)[0]
    gx = struct.unpack_from("<d", b, 0x10)[0]
    gy = struct.unpack_from("<d", b, 0x18)[0]
    n = h.sim.ninja
    bearing = math.atan2(n.ypos - gy, n.xpos - gx)
    if track is None:
        track = latch = gun
    print("%5d %10.4f %10.4f %10.4f %10.4f %10.4f"
          % (f, gun, e0, bearing, track, latch))
    track += wrap(bearing - track) * LERP
    latch += wrap(e0 - latch) * LERP
