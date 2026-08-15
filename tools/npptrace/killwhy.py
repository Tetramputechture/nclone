"""Why do we die to a positionally-perfect entity when the game does not?

Generalises zapwhy.py. For each losing demo that has a capture, replay to the
death, then report: where our ninja is vs the game's, and the separation from
every entity of the killing type in both engines. If the ninja tracks the game
exactly and the entity is where the game has it, the kill TEST is wrong.

Usage: killwhy.py <pool> <cause>
"""
import glob
import json
import math
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
from nclone.ninja import NINJA_RADIUS  # noqa: E402

POOL = int(sys.argv[1])
CAUSE = sys.argv[2]
ETYPE = POOL + 2

losers = [json.loads(l) for l in open("/tmp/shipped2.jsonl")]
want = [r["f"] for r in losers
        if r.get("outcome") == "die" and r.get("cause") == CAUSE]
caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
have = [d for d in want if d in caps]
print("%d demos lose to %s; %d have a capture: %s\n" % (len(want), CAUSE, len(have), have))

dec = NppAttractDecoder()
for demo in have[:4]:
    stem = caps[demo]
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        print("%s: no pool %d\n" % (demo, POOL))
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]
    path = "/home/tetra/projects/attracts/%s" % demo
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    nj = h.sim.ninja
    ours = h.sim.entity_dic.get(ETYPE, [])
    assert p["count"] == len(ours), (
        "%s: pool %d has %d, we have %d -- indices would not correspond"
        % (demo, POOL, p["count"], len(ours)))
    dead = None
    njdiv = None
    for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        g = gninja[f - 1]
        if njdiv is None and (abs(nj.xpos - g["xpos"]) > 1e-9
                              or abs(nj.ypos - g["ypos"]) > 1e-9):
            njdiv = f
        if h.ninja_has_died():
            dead = f
            break
        if h.ninja_has_won():
            break
    if dead is None:
        print("%s: no death inside the capture\n" % demo)
        continue
    g = gninja[dead - 1]
    rec = cap.frame(dead)
    print("=== %s  die frame %d   ninja first left the game's track at %s"
          % (demo, dead, njdiv))
    print("    ninja ours (%.3f, %.3f)  game (%.3f, %.3f)  d=%.4f"
          % (nj.xpos, nj.ypos, g["xpos"], g["ypos"],
             math.hypot(nj.xpos - g["xpos"], nj.ypos - g["ypos"])))
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        e = ours[i]
        print("    ent %d game (%.2f, %.2f) ours (%.2f, %.2f) | dist to our ninja"
              " %.3f, to game ninja %.3f  (NINJA_RADIUS=%s)"
              % (i, gx, gy, e.xpos, e.ypos,
                 math.hypot(nj.xpos - e.xpos, nj.ypos - e.ypos),
                 math.hypot(g["xpos"] - gx, g["ypos"] - gy), NINJA_RADIUS))
    print()
