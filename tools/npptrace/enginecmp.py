"""Per-demo outcome disagreement between the two engines.

The Python and Rust ports are independent implementations of the same spec, so
any demo where they disagree is a place where at least one of them is provably
wrong -- and unlike a capture, this evidence costs nothing to collect. A demo
both engines lose is a shared misunderstanding; a demo they split on is a
porting bug in one of them.

    enginecmp.py /tmp/losers_corner.jsonl /tmp/losers_py.jsonl
"""
import collections
import json
import sys


def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["f"]] = r
    return out


nat, py = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(nat) & set(py))
print("demos in both runs: %d" % len(common))

pair = collections.Counter()
split = []
for f in common:
    a, b = nat[f].get("outcome"), py[f].get("outcome")
    pair[(a, b)] += 1
    if (a == "win") != (b == "win"):
        split.append((f, a, b, nat[f].get("frames"), py[f].get("frames")))

print("\n(native, python) -> count")
for k, v in pair.most_common():
    print("   %-10s %-10s  %d" % (k[0], k[1], v))

print("\ndemos where exactly one engine wins: %d" % len(split))
for f, a, b, fa, fb in split[:40]:
    print("   %-10s native=%-8s(%s)  python=%-8s(%s)" % (f, a, fa, b, fb))

# Same losing outcome but a different death frame is also a divergence, and a
# much more common one -- it means both engines lose but not for the same
# reason, so at least one is not merely "the ninja drifted".
diff_frame = [f for f in common
              if nat[f].get("outcome") == py[f].get("outcome") != "win"
              and nat[f].get("frames") != py[f].get("frames")]
print("\nboth lose but on a different frame: %d" % len(diff_frame))
for f in diff_frame[:25]:
    print("   %-10s %s  native f=%s  python f=%s"
          % (f, nat[f].get("outcome"), nat[f].get("frames"), py[f].get("frames")))
