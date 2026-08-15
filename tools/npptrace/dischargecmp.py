"""Does our chaingun discharge when the game's does not?

endcmp2.py only compares frames where BOTH drones are in state 2, so it is blind
to the case that actually kills: ours firing while the real one is idle. Count
the state agreement directly.
"""
import collections
import glob
import os
import struct
import sys

os.environ["NCLONE_AB_DRONE_FULL_SHOT"] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

POOL, ETYPE = 10, 12
dec = NppAttractDecoder()
pair = collections.Counter()
per_demo = collections.Counter()

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    demo = os.path.basename(stem)
    if not demo.isdigit():
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    if not ours:
        continue
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(min(p["count"], len(ours))):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gst = struct.unpack_from("<i", b, 0x7C)[0]
            ost = getattr(ours[i], "state", -1)
            pair[(gst, ost)] += 1
            if ost == 2 and gst != 2:
                per_demo[demo] += 1
        if h.ninja_has_died() or h.ninja_has_won():
            break

tot = sum(pair.values())
same = sum(v for (g, o), v in pair.items() if g == o)
print("chaingun state frames compared: %d" % tot)
print("states agree: %d = %.1f%%\n" % (same, 100.0 * same / max(tot, 1)))

ours_fire_game_not = sum(v for (g, o), v in pair.items() if o == 2 and g != 2)
game_fire_ours_not = sum(v for (g, o), v in pair.items() if g == 2 and o != 2)
print("WE discharge, game does NOT : %d frames = %.2f%%"
      % (ours_fire_game_not, 100.0 * ours_fire_game_not / max(tot, 1)))
print("game discharges, we do NOT  : %d frames = %.2f%%"
      % (game_fire_ours_not, 100.0 * game_fire_ours_not / max(tot, 1)))

print("\nfull (game_state, our_state) breakdown, most common:")
for (g, o), v in pair.most_common(10):
    tag = "  <-- FALSE DISCHARGE" if (o == 2 and g != 2) else ""
    print("   game=%d ours=%d : %6d%s" % (g, o, v, tag))

if per_demo:
    print("\ndemos with false-discharge frames:")
    for d, v in per_demo.most_common(10):
        print("   %s : %d" % (d, v))
