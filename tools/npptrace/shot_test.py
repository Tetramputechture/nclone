"""Would our lethality test kill the ninja on the game's OWN shots?

Takes the drone position (+0x10/+0x18) and fire direction (+0x90/+0x98) straight
out of the capture on every frame a shot leaves (state 2, +0x8C == 0), traces it
with our raycast and runs our `beam_hits_ninja` against our ninja -- which is
bit-exact on the demos the port now wins.

The game's answer on all of these is "no kill", because every attract demo is a
win. Any hit this reports is our lethality model being wrong, with the aim and
the geometry taken from the game rather than modelled.
"""
import collections
import glob
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
from nclone.physics import (  # noqa: E402
    beam_hits_ninja,
    get_raycast_distance,
    raycast_vs_player,
)

POOLS = (10, 11)
shots = hits = ray_ok = 0
per_demo = collections.Counter()
examples = []

for d in ["/home/tetra/projects/nclone/tools/npptrace/out12"]:
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
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            ninja = h.sim.ninja
            if ninja is None or not ninja.is_valid_target():
                # Once OUR ninja dies its position freezes, so every later shot
                # is tested against a ninja that is not where the game's is.
                # Counting those frames made the false-kill rate look ten times
                # better than it is.
                break
            rec = cap.frame(f)
            for p in pools:
                base = 4 + cap.layout["ninjaSlab"] + p["offset"]
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    if p["stride"] < 0xE4:
                        continue          # type 13: different layout (§26)
                    if struct.unpack_from("<i", b, 0x7C)[0] != 2:
                        continue
                    if struct.unpack_from("<i", b, 0x8C)[0] != 0:
                        continue          # only the frame a shot leaves
                    gx = struct.unpack_from("<d", b, 0x10)[0]
                    gy = struct.unpack_from("<d", b, 0x18)[0]
                    import math as _m2
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    dx, dy = _m2.cos(gun), _m2.sin(gun)
                    shots += 1
                    length = get_raycast_distance(h.sim, gx, gy, dx, dy)
                    if length:
                        xe, ye = gx + dx * length, gy + dy * length
                    else:
                        xe, ye, length = gx, gy, None
                    if raycast_vs_player(h.sim, gx, gy, ninja.xpos, ninja.ypos):
                        ray_ok += 1
                    if beam_hits_ninja(gx, gy, xe, ye, ninja, length):
                        hits += 1
                        per_demo[demo] += 1
                        if len(examples) < 6:
                            examples.append((demo, f, i, round(gx, 1), round(gy, 1),
                                             round(dx, 3), round(dy, 3)))

print("shots the game fired: %d" % shots)
print("  our raycast_vs_player says the ninja is visible: %d" % ray_ok)
print("  our beam_hits_ninja says LETHAL:                 %d" % hits)
print("  the game killed the ninja on:                    0  (every demo is a win)")
if per_demo:
    print("\nby demo: %s" % per_demo.most_common())
for e in examples:
    print("  e.g. %s f%d drone %d at (%.1f, %.1f) dir (%.3f, %.3f)" % e)
