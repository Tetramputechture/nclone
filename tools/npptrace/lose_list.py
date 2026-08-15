"""Write the attract paths of every losing demo in a gate JSONL.

Re-running the whole 246k corpus to score a one-constant change wastes an hour;
re-running only the demos that were already losing answers "did this recover
any" directly. It cannot detect a regression -- that is what the 8000-demo gate
is for -- so always run both.

    lose_list.py /tmp/gate_full3.jsonl /tmp/losers.txt
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
ATTRACTS = "/home/tetra/projects/attracts"

lost = []
outcomes = {}
for line in open(src):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    o = r.get("outcome", "")
    outcomes[o] = outcomes.get(o, 0) + 1
    if o in ("die", "timeout"):
        lost.append(r["f"])

with open(dst, "w") as fh:
    for f in lost:
        fh.write("%s/%s\n" % (ATTRACTS, f))

print("outcomes: %s" % dict(sorted(outcomes.items(), key=lambda kv: -kv[1])))
print("wrote %d losing demos to %s" % (len(lost), dst))
