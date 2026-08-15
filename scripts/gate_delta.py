#!/usr/bin/env python3
"""Compare two attract-gate reports demo-by-demo.

`attract_gate_report.py --summarize` gives you the headline; this gives you the
*delta*, which is the only thing a physics change can be judged on. Aggregates
move by tenths of a percent while hundreds of demos flip in both directions, and
a change that fixes 40 and breaks 45 looks identical to one that does nothing.

Also splits out the **no-hazard subset** -- demos whose maps carry no entity that
can kill. That isolates the ninja from entity behaviour, so it is the signal
Phase A of `docs/NCLONE_INTEGRATION_PLAN.md` should move.

Usage:
  python scripts/gate_delta.py BASE.jsonl NEW.jsonl [--list-flips N]
  python scripts/gate_delta.py BASE.jsonl                 # subset breakdown only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: Entity types that cannot kill the ninja: gold, the exit pair, the three door
#: families, launch pad, one-way, bounce block and boost pad. Everything else in
#: the corpus is a hazard or carries one.
HARMLESS = frozenset({2, 3, 4, 5, 6, 8, 10, 11, 17, 24})


def load(path: str) -> dict[str, dict]:
    return {r["f"]: r
            for r in (json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip())}


def rate(rows) -> tuple[int, int]:
    playable = [r for r in rows if r["outcome"] in ("win", "die", "timeout")]
    return sum(1 for r in playable if r["outcome"] == "win"), len(playable)


def report(tag: str, rows: list[dict]) -> None:
    w, n = rate(rows)
    nh = [r for r in rows if not (set(r["types"]) - HARMLESS)]
    wn, nn = rate(nh)
    print(f"{tag:10s} overall {w}/{n} = {100 * w / max(n, 1):6.2f}%    "
          f"no-hazard {wn}/{nn} = {100 * wn / max(nn, 1):6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--list-flips", type=int, default=0,
                    help="Print up to N demo ids that regressed (win -> not win).")
    args = ap.parse_args()

    base = load(args.base)
    report("BASE", list(base.values()))
    if not args.new:
        return 0

    new = load(args.new)
    report("NEW", list(new.values()))

    shared = sorted(set(base) & set(new))
    gained = [f for f in shared
              if base[f]["outcome"] != "win" and new[f]["outcome"] == "win"]
    lost = [f for f in shared
            if base[f]["outcome"] == "win" and new[f]["outcome"] != "win"]
    print(f"\n  {len(shared)} demos compared: +{len(gained)} gained, "
          f"-{len(lost)} lost, net {len(gained) - len(lost):+d}")

    # Which cause each regression now dies of -- a physics change that trades
    # mine deaths for impact deaths is a different animal from one that trades
    # them for timeouts.
    if lost:
        print("  regressions now end as: "
              + ", ".join(f"{k} {v}" for k, v in
                          Counter(new[f].get("cause") or new[f]["outcome"]
                                  for f in lost).most_common()))
    if gained:
        print("  gains previously ended as: "
              + ", ".join(f"{k} {v}" for k, v in
                          Counter(base[f].get("cause") or base[f]["outcome"]
                                  for f in gained).most_common()))
    if args.list_flips:
        print("  regressed ids:", " ".join(lost[: args.list_flips]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
