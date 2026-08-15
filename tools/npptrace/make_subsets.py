#!/usr/bin/env python3
"""Rebuild the fast gate subsets from a full 8000-demo gate run.

    python scripts/attract_gate_report.py --in-dir /home/tetra/projects/attracts \
        --out /tmp/gate.jsonl --limit 8000 --workers 8 --engine native
    python tools/npptrace/make_subsets.py /tmp/gate.jsonl

Writes `tools/npptrace/subsets/`, which `--files-from` reads:

    losers.txt   every current non-win.  ~0.8 s.  Baseline is 0/N by
                 construction, so any fix shows up immediately as a positive
                 count. It can ONLY show improvement -- it contains no demo
                 that currently passes, so it can never report a regression.
    smoke.txt    the losers plus up to 120 winners per entity type. ~6 s, and
                 it is the one that catches a fix breaking something else.

The full 8000 still costs ~150 s and is what a result gets reported on; these
are for the edit/measure loop in between. Regenerate them after landing a fix,
or `losers.txt` slowly turns into a list of demos that already pass.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "subsets")
ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")
#: Enough winners per type to catch a hazard regression without paying for the
#: whole corpus. Raise it if a fix lands that `smoke.txt` failed to flag.
PER_TYPE = 120


def write(name, names):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        for n in sorted(names):
            fh.write(os.path.join(ATTRACTS, n) + "\n")
    print("  %-12s %5d demos" % (name, len(names)))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gate.jsonl"
    rows = [json.loads(line) for line in open(src)]
    os.makedirs(OUT, exist_ok=True)

    losers = {
        r["f"] for r in rows
        if r["outcome"] not in ("win", "coop", "no_demo")
        and not r["outcome"].startswith("err")
    }

    by_type = collections.defaultdict(list)
    for r in rows:
        if r["outcome"] == "win":
            for t in r.get("types", []):
                by_type[t].append(r["f"])
    guard = set()
    for fs in by_type.values():
        guard.update(fs[:PER_TYPE])

    print("from %s: %d rows, %d losers, %d types" % (src, len(rows), len(losers), len(by_type)))
    write("losers.txt", losers)
    write("smoke.txt", losers | guard)


if __name__ == "__main__":
    main()
