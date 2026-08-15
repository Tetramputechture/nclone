"""Find which arithmetic ordering reproduces the game's grounded xspeed bit-exactly."""
import itertools
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
game = [json.loads(x) for x in open(os.path.join(HERE, "final", "10131738.jsonl"))]
gi = {r["frame"]: r for r in game}

ACCEL = 0.06666666666666665
DRAG_REG = 0.9933221725495059
FRICTION_GROUND = 0.9459290248226102
DRAG_SLOW = 0.8617738760127536


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def ulps(a, b):
    return bits(a) - bits(b) if (a >= 0) == (b >= 0) else None


# frames 2..20 are a clean grounded left-run: ypos constant, yspeed 0
print("frame  state  xpos                     xspeed")
for f in range(1, 22):
    r = gi[f]
    print("%5d  %5d  %-22.17g  %.17g" % (f, r["state"], r["xpos"], r["xspeed"]))

print("\n=== candidate updates, prev -> next xspeed (input = left, -1) ===")
cands = {
    "(v - A) * D_reg": lambda v: (v - ACCEL) * DRAG_REG,
    "v * D_reg - A": lambda v: v * DRAG_REG - ACCEL,
    "(v - A) * FRIC": lambda v: (v - ACCEL) * FRICTION_GROUND,
    "v * FRIC - A": lambda v: v * FRICTION_GROUND - ACCEL,
    "v - A (no drag)": lambda v: v - ACCEL,
    "(v * D_reg) - A*D_reg": lambda v: (v * DRAG_REG) - ACCEL * DRAG_REG,
}
score = {k: [0, 0] for k in cands}
for f in range(2, 20):
    prev, cur = gi[f - 1], gi[f]
    if cur["state"] != 1 or prev["state"] != 1:
        continue
    v0, v1 = prev["xspeed"], cur["xspeed"]
    for name, fn in cands.items():
        got = fn(v0)
        score[name][1] += 1
        if bits(got) == bits(v1):
            score[name][0] += 1
for name, (hit, tot) in sorted(score.items(), key=lambda kv: -kv[1][0]):
    print("  %-24s exact %d/%d" % (name, hit, tot))

print("\n=== per-frame check for the best candidate ===")
best = max(score, key=lambda k: score[k][0])
print("best = %s" % best)
for f in range(2, 20):
    prev, cur = gi[f - 1], gi[f]
    if cur["state"] != 1:
        continue
    got = cands[best](prev["xspeed"])
    d = bits(got) - bits(cur["xspeed"])
    print("  f%-4d got=%.17g  game=%.17g  ulp_delta=%+d"
          % (f, got, cur["xspeed"], d))

print("\n=== divergence profile vs nclone ===")
ref = r"D:\nppdata\sim_trace.json"
if os.path.exists(ref):
    port = json.load(open(ref))
    worst = 0.0
    firstpos = None
    for pr in port:
        gr = gi.get(pr["frame"])
        if gr is None:
            break
        dx = abs(pr["xpos"] - gr["xpos"])
        dy = abs(pr["ypos"] - gr["ypos"])
        d = max(dx, dy)
        if d > 0 and firstpos is None:
            firstpos = (pr["frame"], dx, dy)
        if d > worst:
            worst = d
        if pr["frame"] in (10, 20, 50, 100, 200, 300, 400, 500, 587):
            print("  f%-4d port=(%.10f, %.10f) game=(%.10f, %.10f) dx=%.3e dy=%.3e"
                  % (pr["frame"], pr["xpos"], pr["ypos"], gr["xpos"], gr["ypos"], dx, dy))
    print("  first POSITION difference: frame %s (dx=%.3e dy=%.3e)"
          % (firstpos[0], firstpos[1], firstpos[2]) if firstpos else "  none")
    print("  max position error over port's 587 frames: %.6g px" % worst)
