"""Reclassify all 69 losers by what actually KILLS us, not by what is on the map.

The entity-presence bucketing that has steered this whole effort is misleading:
a demo counted as "death_ball" turns out to die of plain physics on a map that
merely contains a death ball. Cause is the number that should drive the work.
"""
import collections
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

ORDER = [("drone_shot", {12, 13}), ("drone", {14, 15, 26}),
         ("rocket", {18, 29}), ("laser", {23}), ("death_ball", {25}),
         ("turret", {19}), ("mine", {1, 21}), ("floor_guard", {16})]
ATTRACT = "/home/tetra/projects/attracts"


def bucket(types):
    for name, ts in ORDER:
        if ts & set(types):
            return name
    return "physics"


gate = [json.loads(l) for l in open("/tmp/gate15.jsonl")]
losers = [r for r in gate if r.get("outcome") in ("die", "timeout")]
print("reclassifying %d losers\n" % len(losers))

dec = NppAttractDecoder()
rows = []
by_cause = collections.Counter()
cross = collections.Counter()
for r in sorted(losers, key=lambda r: r["f"]):
    path = os.path.join(ATTRACT, r["f"])
    b = bucket(r["types"])
    try:
        res = dec.decode_npp_attract_file(path)
        demo = res.get("demo_data", []) or []
        mapb = list(dec.create_nclone_map(path))
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(mapb)
        outcome, frames = "timeout", 0
        for raw in demo:
            hor, jmp = decode_input_to_controls(int(raw))
            h.tick(hor, jmp)
            frames += 1
            if h.ninja_has_won():
                outcome = "win"
                break
            if h.ninja_has_died():
                outcome = "die"
                break
        nj = getattr(getattr(h, "sim", None), "ninja", None)
        cause = ((getattr(nj, "death_cause", None) or "physics")
                 if outcome == "die" else outcome)
        frac = frames / max(len(demo), 1)
        rows.append((b, str(cause), r["f"], frames, len(demo), round(frac, 3)))
        by_cause[str(cause)] += 1
        cross[(b, str(cause))] += 1
    except Exception as e:
        rows.append((b, "ERR:%s" % type(e).__name__, r["f"], -1, -1, -1.0))
        by_cause["ERR"] += 1

print("WHAT ACTUALLY KILLS US (69 losers):")
for c, v in by_cause.most_common():
    print("   %-14s %d" % (c, v))

print("\nentity-presence bucket  ->  actual cause:")
for (b, c), v in sorted(cross.items(), key=lambda t: -t[1]):
    flag = "  <-- MISLABELLED" if b != c and c not in ("timeout",) else ""
    print("   %-12s -> %-14s %2d%s" % (b, c, v, flag))

ok = [r for r in rows if r[5] >= 0]
late = [r for r in ok if r[5] > 0.9]
print("\n%d/%d end in the last 10%% of the tape, %d earlier"
      % (len(late), len(ok), len(ok) - len(late)))

with open("/tmp/cause69.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("wrote /tmp/cause69.json")
