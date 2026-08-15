"""Score toggle-mine STATE against the game.

Mines never move, so entscore's position test is vacuous for them. What decides a
death is the toggled/untoggled/toggling state, which our port drives from ninja
contact. Compare it frame by frame wherever a capture has the pool.

Type 21 -> pool 19; type 1 mines share the same pool in the game's layout.
"""
import collections
import glob
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

dec = NppAttractDecoder()
found_any = False

for POOL, ETYPE in ((19, 21), (-1, 1)):
    if POOL < 0:
        continue
    pair = collections.Counter()
    demos = 0
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
        ours = h.sim.entity_dic.get(21, [])
        if not ours:
            continue
        demos += 1
        found_any = True
        for f in range(1, min(cap.n, len(tape)) + 1):
            hor, jmp = decode_input_to_controls(int(tape[f - 1]))
            h.tick(hor, jmp)
            rec = cap.frame(f)
            for i in range(min(p["count"], len(ours))):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                # State lives near the end of the slab for most pools; probe the
                # first int that only ever holds 0/1/2.
                gst = struct.unpack_from("<i", b, 0x30)[0]
                pair[(gst, getattr(ours[i], "state", -1))] += 1
            if h.ninja_has_died() or h.ninja_has_won():
                break
    tot = sum(pair.values())
    print("\npool %d (type %d): %d demos, %d mine-frames" % (POOL, ETYPE, demos, tot))
    if tot:
        agree = sum(v for (g, o), v in pair.items() if g == o)
        print("   states agree: %d = %.1f%%" % (agree, 100.0 * agree / tot))
        for (g, o), v in pair.most_common(8):
            print("      game=%-3s ours=%-3s : %6d" % (g, o, v))

if not found_any:
    print("\nNo capture contains a toggle-mine pool -- mines were never captured.")
    print("That is why this category has stayed untouched: there is no ground")
    print("truth for it on disk.")
