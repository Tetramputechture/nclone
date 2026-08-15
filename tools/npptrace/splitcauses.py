"""Group the native-only losses by the cause native reports.

A demo the Python engine wins and native loses is a native bug by construction --
these are attract replays the real game completes. Grouping them by native's own
death cause says whether the 74 are one bug or several, which decides whether to
chase them one at a time or look for a shared mechanism.

    splitcauses.py <native.jsonl> <python.jsonl>
"""
import collections
import json
import sys


def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[r["f"]] = r
    return out


nat, py = load(sys.argv[1]), load(sys.argv[2])
split = [f for f in nat
         if f in py and nat[f].get("outcome") != "win" and py[f].get("outcome") == "win"]

by = collections.Counter(nat[f].get("cause") or "?" for f in split)
print("native-only losses: %d\n" % len(split))
print("by native's death cause:")
for cause, n in by.most_common():
    print("   %-16s %4d" % (cause, n))

print("\nshortest demo per cause (native death frame), best to diagnose first:")
for cause, _ in by.most_common():
    rows = sorted((nat[f].get("frames") or 0, f) for f in split
                  if (nat[f].get("cause") or "?") == cause)
    show = ", ".join("%s(f=%d)" % (f, fr) for fr, f in rows[:4])
    print("   %-16s %s" % (cause, show))
