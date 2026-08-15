"""Score the tracking-gun model against `+0x50` over EVERY captured frame.

§17 verified the 20%-per-frame lerp over a single 53-frame charge on one demo.
The full model then killed 186 times across 271 demos where replaying the game's
own shots kills 9, which means either the gun is not right in general or our
drone fires at the wrong times. This settles the first half.

Driven open-loop from the game's own drone position and the game's previous gun
angle each frame, so a single bad frame cannot cascade.
"""
import collections
import glob
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

LERP = 0.2
POOLS = (10, 11)
by_state = collections.defaultdict(lambda: [0, 0])
worst = []


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        pools = [q for q in cap.pools if q["i"] in POOLS and q["count"]]
        if not pools:
            continue
        dec = NppAttractDecoder()
        path = "/home/tetra/projects/attracts/" + demo
        inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
        n = min(cap.n, len(inputs))
        prev = {}
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            ninja = h.sim.ninja
            if ninja is None:
                continue
            rec = cap.frame(f)
            for p in pools:
                base = 4 + cap.layout["ninjaSlab"] + p["offset"]
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gx = struct.unpack_from("<d", b, 0x10)[0]
                    gy = struct.unpack_from("<d", b, 0x18)[0]
                    key = (demo, p["i"], i)
                    if key in prev:
                        pgun, pst = prev[key]
                        bearing = math.atan2(ninja.ypos - gy, ninja.xpos - gx)
                        pred = pgun + wrap(bearing - pgun) * LERP
                        err = abs(wrap(pred - gun))
                        ok = err < 2e-4
                        by_state[pst][0 if ok else 1] += 1
                        if not ok and len(worst) < 8:
                            worst.append((demo, f, pst, round(pgun, 4),
                                          round(pred, 4), round(gun, 4)))
                    prev[key] = (gun, st)

print("one-frame gun-angle prediction vs `+0x50`, by the state it was IN:")
print("  %-8s %-10s %-10s %s" % ("state", "match", "miss", "accuracy"))
for st in sorted(by_state):
    ok, bad = by_state[st]
    tot = ok + bad
    print("  %-8d %-10d %-10d %.1f%%" % (st, ok, bad, 100.0 * ok / max(tot, 1)))
print("\nfirst misses (demo, frame, state, prev, predicted, actual):")
for w in worst:
    print("   %s" % (w,))
