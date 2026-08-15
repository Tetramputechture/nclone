"""How many death-ball kills come from a ball that is FROZEN in place?

`EntityDeathBall.think` returns early when the depenetration query reports
`dist == 0` -- the ball's centre lying exactly on a segment. That `return` skips
the bounce, the ball-vs-ball pass and `grid_move`, and crucially it happens
BEFORE the ball has been pushed off the segment, so the next frame reproduces it
exactly. The ball is stranded on the surface forever, still lethal.

For each losing demo: find the ball that kills, and report whether it ever moved.
"""
import collections
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

demos = [l.strip().split("/")[-1] for l in open(sys.argv[1]) if l.strip()]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
demos = demos[:LIMIT]

dec = NppAttractDecoder()
tally = collections.Counter()
froze_at = collections.Counter()
examples = []

for demo in demos:
    path = "/home/tetra/projects/attracts/%s" % demo
    try:
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
        h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    except Exception:
        tally["error"] += 1
        continue
    balls = h.sim.entity_dic.get(25, [])
    if not balls:
        continue
    nj = h.sim.ninja
    spawns = [(b.xpos, b.ypos) for b in balls]
    moved_after = [None] * len(balls)   # last frame each ball moved
    prev = list(spawns)
    killer = None
    for f in range(1, len(tape) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        for i, b in enumerate(balls):
            if (b.xpos, b.ypos) != prev[i]:
                moved_after[i] = f
                prev[i] = (b.xpos, b.ypos)
        if h.ninja_has_died():
            killer = min(range(len(balls)),
                         key=lambda i: math.hypot(nj.xpos - balls[i].xpos,
                                                  nj.ypos - balls[i].ypos))
            break
        if h.ninja_has_won():
            break
    if killer is None:
        tally["no death"] += 1
        continue
    last = moved_after[killer]
    b = balls[killer]
    d = math.hypot(nj.xpos - b.xpos, nj.ypos - b.ypos)
    if last is None:
        tally["killer NEVER moved"] += 1
    elif last <= 3:
        tally["killer froze by frame 3"] += 1
        froze_at[last] += 1
    elif last < f - 30:
        tally["killer froze >30f before the kill"] += 1
    else:
        tally["killer was still moving"] += 1
    if len(examples) < 8 and (last is None or last <= 3):
        sx, sy = spawns[killer]
        examples.append((demo, killer, sx, sy, b.xpos, b.ypos, last, d))

print("%d demos sampled\n" % len(demos))
for k, v in tally.most_common():
    print("   %-34s %4d" % (k, v))
print("\nfroze on frame: %s" % dict(sorted(froze_at.items())))
print("\nexamples (ball stranded almost immediately):")
for demo, i, sx, sy, x, y, last, d in examples:
    print("   %-10s ball %-3d spawn (%8.2f,%8.2f) -> stuck (%11.5f,%11.5f) "
          "last moved f=%s, killed at dist %.4f" % (demo, i, sx, sy, x, y, last, d))
