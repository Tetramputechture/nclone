"""Deduplicate a gate JSONL, keeping the LAST record for each demo.

`attract_gate_report.py --out` appends, so a run that is stopped and restarted
against the same path leaves the stopped run's records in front of the new
ones -- and those were produced by a different binary. Keeping the last
occurrence keeps the newest, and a truncated final line from the killed process
is dropped rather than crashing the summariser.

    gate_clean.py <in.jsonl> <out.jsonl>
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]

seen = {}
order = []
bad = 0
for line in open(src):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except ValueError:
        bad += 1
        continue
    f = r.get("f")
    if f not in seen:
        order.append(f)
    seen[f] = line

with open(dst, "w") as fh:
    for f in order:
        fh.write(seen[f] + "\n")

print("read %d unique demos (%d unparseable lines dropped)" % (len(order), bad))
print("wrote %s" % dst)
