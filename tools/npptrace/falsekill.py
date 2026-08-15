"""Six of the 88 FULL_SHOT regressions are captured. Find the false kill.

Replay each with the model on, stop at the frame our chaingun kills the ninja,
and print everything the kill depends on next to the game's own values: drone
state, drone position, beam endpoint, ninja position. The game did not kill here,
so whatever differs at this frame is the bug.
"""
import glob
import json
import math
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

DEMOS = ["10132175", "10133346", "10134101", "10134102", "10136116", "10138394"]
POOL, ETYPE = 10, 12
dec = NppAttractDecoder()

for demo in DEMOS:
    stem = None
    for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*"):
        if os.path.exists(os.path.join(c, demo) + ".layout.json"):
            stem = os.path.join(c, demo)
            break
    if stem is None:
        continue
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        print("\n%s: no chaingun pool" % demo)
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]
    path = "/home/tetra/projects/attracts/%s" % demo
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    nj = h.sim.ninja
    killed = None
    for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        if h.ninja_has_died():
            killed = f
            break
        if h.ninja_has_won():
            break
    if killed is None:
        print("\n%s: no death within the capture (%d frames)" % (demo, cap.n))
        continue
    f = killed
    rec = cap.frame(f)
    g = gninja[f - 1]
    print("\n=== %s  WE KILL at frame %d (cause=%s)"
          % (demo, f, getattr(nj, "death_cause", "?")))
    print("   ninja  ours (%8.2f,%8.2f)   game (%8.2f,%8.2f)   d=%.3f"
          % (nj.xpos, nj.ypos, g["xpos"], g["ypos"],
             math.hypot(nj.xpos - g["xpos"], nj.ypos - g["ypos"])))
    for i in range(min(p["count"], len(ours))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gst = struct.unpack_from("<i", b, 0x7C)[0]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        gxe, gye = struct.unpack_from("<dd", b, 0xC0)
        e = ours[i]
        print("   drone %d  state ours=%s game=%d | pos ours (%.2f,%.2f) game (%.2f,%.2f)"
              % (i, getattr(e, "state", "?"), gst, e.xpos, e.ypos, gx, gy))
        print("            beam end ours (%.2f,%.2f)  game (%.2f,%.2f)"
              % (getattr(e, "xend", 0), getattr(e, "yend", 0), gxe, gye))
