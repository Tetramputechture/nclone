"""Which of the remaining losses have ground truth on disk?

Diagnosis needs a capture. Anything without one is blocked on a capture batch,
which needs N++ running. Split the remaining losers accordingly so effort goes
where it can actually land.
"""
import collections
import glob
import json
import os

losers = [json.loads(l) for l in open("/tmp/shipped2.jsonl")]
bad = [r for r in losers if r.get("outcome") in ("die", "timeout")]
caps = {os.path.basename(s).replace(".layout.json", "")
        for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")}

have = collections.Counter()
lack = collections.Counter()
have_ids = collections.defaultdict(list)
for r in bad:
    cause = r.get("cause", "timeout")
    if r["f"] in caps:
        have[cause] += 1
        have_ids[cause].append(r["f"])
    else:
        lack[cause] += 1

print("%-16s %8s %8s" % ("cause", "captured", "blocked"))
for c in sorted(set(have) | set(lack), key=lambda c: -(have[c] + lack[c])):
    print("%-16s %8d %8d" % (c, have[c], lack[c]))
print("%-16s %8d %8d" % ("TOTAL", sum(have.values()), sum(lack.values())))

print("\nWORKABLE NOW (have a capture):")
for c, ids in sorted(have_ids.items()):
    print("   %-16s %s" % (c, " ".join(ids)))
