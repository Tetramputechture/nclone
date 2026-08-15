"""Break a full-corpus gate run down into actionable buckets.

    python tools/npptrace/losses.py /tmp/gate_full.jsonl [--families] [--write-subsets]

`attract_gate_report.py --summarize` gives per-cause totals over the whole run;
this answers the next question, which is *which* demos and *which maps*. A cause
count says "37 laser deaths"; a map-family count says "37 laser deaths on four
maps", and the second is what tells you whether you are looking at one bug or
four.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/tetra/projects/nclone")

ATTRACTS = "/home/tetra/projects/attracts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--families", action="store_true",
                    help="hash each losing map's tiles and entities (slower)")
    ap.add_argument("--write-subsets", action="store_true")
    ap.add_argument("--limit-families", type=int, default=40)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    outcomes = collections.Counter(r["outcome"] for r in rows)
    # Same definition the gate report uses: co-op / no-demo / error are not
    # replayable, so they are neither wins nor losses.
    playable = [r for r in rows if r["outcome"] in ("win", "die", "timeout")]
    losses = [r for r in playable if r["outcome"] != "win"]
    wins = len(playable) - len(losses)

    print("%d rows   outcomes: %s" % (len(rows), dict(outcomes)))
    print("replayable: %d    win %d    LOSE %d    = %.4f%% win"
          % (len(playable), wins, len(losses), 100.0 * wins / max(len(playable), 1)))
    if not losses:
        print("\nno losses.")
        return

    print("\nlosses by cause:")
    for cause, n in collections.Counter(
            r.get("cause") or r["outcome"] for r in losses).most_common():
        print("   %-18s %5d" % (cause, n))

    print("\nlosses by entity types present on the map:")
    tc = collections.Counter()
    for r in losses:
        for t in r.get("types", []):
            tc[t] += 1
    for t, n in tc.most_common(12):
        print("   type %-3d present on %5d losing maps" % (t, n))

    if a.write_subsets:
        with open("/tmp/losers_full.txt", "w") as fh:
            for r in sorted(losses, key=lambda r: r["f"]):
                fh.write("%s/%s\n" % (ATTRACTS, r["f"]))
        print("\nwrote /tmp/losers_full.txt (%d demos)" % len(losses))

    if not a.families:
        return

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    dec = NppAttractDecoder()
    fam = collections.defaultdict(list)
    for r in losses:
        try:
            md = bytes(dec.create_nclone_map(os.path.join(ATTRACTS, r["f"])))
        except Exception:
            fam[("?", "?")].append(r)
            continue
        fam[(hashlib.sha1(md[:1230]).hexdigest()[:8],
             hashlib.sha1(md[1230:]).hexdigest()[:8])].append(r)

    print("\n%d losing demos on %d distinct maps (tiles+entities):"
          % (len(losses), len(fam)))
    for (t, e), rs in sorted(fam.items(), key=lambda kv: -len(kv[1]))[:a.limit_families]:
        causes = collections.Counter(x.get("cause") or x["outcome"] for x in rs)
        print("   tiles=%s ents=%s  %4d demos  %-40s  e.g. %s"
              % (t, e, len(rs), dict(causes),
                 " ".join(x["f"] for x in sorted(rs, key=lambda x: x["f"])[:3])))


if __name__ == "__main__":
    main()
