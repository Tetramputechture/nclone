"""Triage the losers whose only suspect entity is death_ball / turret / mine.

No game needed. Replay each, record how far it got, what killed it, and where --
so that when a capture batch is possible it targets the right frames instead of
whole demos.
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
WANT = {"death_ball", "turret", "mine", "floor_guard"}
ATTRACT = "/home/tetra/projects/attracts"


def bucket(types):
    for name, ts in ORDER:
        if ts & set(types):
            return name
    return "physics"


gate = [json.loads(l) for l in open("/tmp/gate15.jsonl")]
losers = [r for r in gate if r.get("outcome") in ("die", "timeout")]
for r in losers:
    r["bucket"] = bucket(r["types"])
sel = [r for r in losers if r["bucket"] in WANT]
print("triaging %d\n" % len(sel))

dec = NppAttractDecoder()
rows, by_cause = [], collections.Counter()
for r in sorted(sel, key=lambda r: (r["bucket"], r["f"])):
    path = os.path.join(ATTRACT, r["f"])
    try:
        res = dec.decode_npp_attract_file(path)
        demo = res.get("demo_data", []) or []
        mapb = list(dec.create_nclone_map(path))
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(mapb)
        outcome, frames = "timeout", 0
        for b in demo:
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            frames += 1
            if h.ninja_has_won():
                outcome = "win"
                break
            if h.ninja_has_died():
                outcome = "die"
                break
        nj = getattr(getattr(h, "sim", None), "ninja", None)
        cause = (getattr(nj, "death_cause", None) or "physics") if outcome == "die" else outcome
        frac = frames / max(len(demo), 1)
        rows.append((r["bucket"], r["f"], frames, len(demo), round(frac, 3),
                     str(cause), round(getattr(nj, "xpos", 0), 1),
                     round(getattr(nj, "ypos", 0), 1), r["types"]))
        by_cause[(r["bucket"], str(cause))] += 1
    except Exception as e:
        rows.append((r["bucket"], r["f"], -1, -1, -1.0,
                     "ERR %s: %s" % (type(e).__name__, e), 0, 0, []))

print("%-11s %-9s %6s %6s %6s  %-12s %s" % ("bucket", "id", "frame", "of", "frac", "cause", "pos"))
for b, i, f, n, fr, c, x, y, ty in rows:
    print("%-11s %-9s %6d %6d %6.3f  %-12s (%s,%s)" % (b, i, f, n, fr, c, x, y))

print("\nby cause:")
for (b, c), v in by_cause.most_common():
    print("   %-11s %-22s %d" % (b, c, v))
ok = [r for r in rows if r[4] >= 0]
late = [r for r in ok if r[4] > 0.9]
print("\n%d/%d end in the last 10%% (late drift), %d earlier (hard divergence)"
      % (len(late), len(ok), len(ok) - len(late)))
