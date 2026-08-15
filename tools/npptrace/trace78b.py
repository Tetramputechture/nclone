"""Bit-exact diff of one trace against the game, on the six fields verify uses.

The float comparison earlier said "all match" because the difference is below
1e-12 -- verify compares BIT PATTERNS. Print the bits so a 1-ulp divergence is
visible, which is exactly the shape §19's floor_x bug had.
"""
import json
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

FIELDS = ("xpos", "ypos", "xspeed", "yspeed", "floor_x", "floor_y")
DEMO = sys.argv[1] if len(sys.argv) > 1 else "10138288"
AT = int(sys.argv[2]) if len(sys.argv) > 2 else 78


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


game = {r["frame"]: r for r in
        (json.loads(l) for l in
         open("/home/tetra/projects/nclone/docs/game_traces/%s.jsonl" % DEMO)
         if l.strip())}
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
mapb = list(dec.create_nclone_map(path))
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(mapb)

print("%s: verify reports first divergence at frame %d\n" % (DEMO, AT))
for i, b in enumerate(tape, 1):
    hor, jmp = decode_input_to_controls(int(b))
    h.tick(hor, jmp)
    n = h.sim.ninja
    mine = {"xpos": float(n.xpos), "ypos": float(n.ypos),
            "xspeed": float(n.xspeed), "yspeed": float(n.yspeed),
            "floor_x": float(n.floor_normalized_x),
            "floor_y": float(n.floor_normalized_y),
            "state": int(n.state)}
    if i not in game:
        break
    g = game[i]
    diff = [k for k in FIELDS if bits(g[k]) != bits(mine[k])]
    if g["state"] != mine["state"]:
        diff.append("state")
    if AT - 2 <= i <= AT + 2 or diff:
        print("frame %d  h=%d j=%d  state game=%d ours=%d %s"
              % (i, hor, jmp, g["state"], mine["state"],
                 ("DIFF: " + ",".join(diff)) if diff else "(bit-exact)"))
        for k in diff:
            if k == "state":
                continue
            gb, mb = bits(g[k]), bits(mine[k])
            print("     %-8s game=%.17g  ours=%.17g  ulp=%d"
                  % (k, g[k], mine[k], mb - gb))
    if diff:
        print("\n  -- stopping at the first divergent frame --")
        break
    if h.ninja_has_won() or h.ninja_has_died():
        break
