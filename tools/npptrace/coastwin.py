"""How many `timeout` demos win if the sim keeps running after the inputs end?

Ground truth from a fresh capture of `10172495`: our ninja is **bit-identical to
the game for all 221 frames of the tape**, and the game's ninja reaches state 8
on frame **222** -- one frame after the last input. The remaining 238 captured
frames are the win celebration.

So the demo's last input launches the ninja INTO the exit and the win registers
during the coast. Our harness runs exactly `len(tape)` ticks and stops, which
scores that as a timeout. The game does not stop.

This measures the effect: for each timeout demo, keep ticking with no input and
record how many extra frames it takes to win (if it ever does).

    coastwin.py <gate.jsonl> [max_extra]
"""
import collections
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

import nclone_native  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402

gate = sys.argv[1]
MAX_EXTRA = int(sys.argv[2]) if len(sys.argv) > 2 else 120

demos = []
for line in open(gate):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("outcome") == "timeout":
        demos.append(r["f"])

dec = NppAttractDecoder()
hist = collections.Counter()
won_at = []
outcomes = collections.Counter()

for demo in demos:
    path = "/home/tetra/projects/attracts/%s" % demo
    try:
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
        md = list(dec.create_nclone_map(path))
    except Exception:
        outcomes["decode_error"] += 1
        continue
    sim = nclone_native.Simulator.load(md)
    st = 0
    for b in tape:
        hor, jmp = decode_input_to_controls(int(b))
        sim.tick(hor, jmp)
        st = int(sim.get_state_dynamic().ninja.state)
        if st in (6, 8):
            break
    if st == 8:
        outcomes["won_within_tape"] += 1
        continue
    if st == 6:
        outcomes["died_within_tape"] += 1
        continue
    # Coast: no input, exactly as the game does once the demo's inputs run out.
    res = "still_timeout"
    for k in range(1, MAX_EXTRA + 1):
        sim.tick(0, 0)
        st = int(sim.get_state_dynamic().ninja.state)
        if st == 8:
            res = "win"
            hist[k] += 1
            won_at.append((k, demo))
            break
        if st == 6:
            res = "died_coasting"
            break
    outcomes[res] += 1

print("timeout demos examined: %d\n" % len(demos))
for k, v in outcomes.most_common():
    print("   %-18s %d" % (k, v))

if hist:
    ks = sorted(hist)
    print("\nextra frames needed to win:")
    run = 0
    for k in ks:
        run += hist[k]
        print("   +%-4d frames  x%-4d  (cumulative %d)" % (k, hist[k], run))
    print("\n   a budget of +%d frames recovers %d demos" % (ks[-1], run))
