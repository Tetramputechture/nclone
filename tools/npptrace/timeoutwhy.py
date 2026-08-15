"""Why do the `timeout` demos run out of tape?

472 demos end with the tape exhausted and the ninja alive. Two very different
explanations, and they are distinguishable without any capture:

* **the ninja is AT the exit when the inputs run out** -- we are decoding one or
  two frames short, or the win condition is fractionally too strict. That is a
  container/decoder bug, and cheap to fix.
* **the ninja is nowhere near the exit** -- it diverged from the game's
  trajectory much earlier. That needs ground truth.

Reports, for each timeout demo: whether the exit switch was collected, how far
the ninja finished from the exit door, and how that distance is distributed.

    timeoutwhy.py <gate.jsonl> [limit]
"""
import collections
import json
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

import nclone_native  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402

gate = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

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
if limit:
    demos = demos[:limit]

dec = NppAttractDecoder()
buckets = collections.Counter()
rows = []
for demo in demos:
    path = "/home/tetra/projects/attracts/%s" % demo
    try:
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
        md = list(dec.create_nclone_map(path))
    except Exception:
        buckets["decode_error"] += 1
        continue
    sim = nclone_native.Simulator.load(md)
    for b in tape:
        hor, jmp = decode_input_to_controls(int(b))
        sim.tick(hor, jmp)
    st = sim.get_state_dynamic()
    n = st.ninja
    # Exit door is type 3. On native's snapshot `active` IS `switch_hit`, i.e.
    # true means the door is OPEN -- not the other way round. Reading it as
    # "still uncollected" inverted the statistic and briefly made a demo that
    # never opened its exit look like a 5 px near-miss win.
    doors = [e for e in st.entities if e.entity_type == 3]
    if not doors:
        buckets["no_exit_door"] += 1
        continue
    d = min(math.hypot(e.x - n.xpos, e.y - n.ypos) for e in doors)
    switch_done = any(e.active for e in doors)
    rows.append((demo, d, switch_done, len(tape)))
    if d < 12:
        buckets["at_the_door (<12px)"] += 1
    elif d < 50:
        buckets["near (<50px)"] += 1
    elif d < 200:
        buckets["mid (<200px)"] += 1
    else:
        buckets["far (>=200px)"] += 1
    buckets["switch_collected" if switch_done else "switch_MISSED"] += 1

print("timeout demos examined: %d\n" % len(rows))
print("distance from the exit door when the tape ran out:")
for k in ("at_the_door (<12px)", "near (<50px)", "mid (<200px)", "far (>=200px)"):
    if buckets[k]:
        print("   %-22s %4d" % (k, buckets[k]))
print("\nexit switch:")
for k in ("switch_collected", "switch_MISSED"):
    if buckets[k]:
        print("   %-22s %4d" % (k, buckets[k]))
for k, v in buckets.items():
    if k.startswith(("decode", "no_exit")):
        print("   %-22s %4d" % (k, v))

# A decoder or win-radius bug would look like: finished ON the exit, with the
# door already open, and still not a win. Anything with the switch uncollected
# diverged earlier and is not evidence about the win condition.
rows.sort(key=lambda r: r[1])
print("\nclosest finishers WITH THE EXIT OPEN (a win-condition bug lives here):")
opened = [r for r in rows if r[2]]
for demo, d, sw, n in opened[:12]:
    print("   %-10s dist=%8.2f frames=%d" % (demo, d, n))
if not opened:
    print("   (none)")
