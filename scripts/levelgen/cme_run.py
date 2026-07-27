"""cme_run.py — the first real Constrained MAP-Elites experiment (spec §8).

Seeds the (mechanic-set × difficulty) archive from the real demo'd corpus, with:

* the **native faithful gate** = replay each level's winning demo on the native sim and
  keep it only if it reaches a win (the strongest, fastest solvability cert we have —
  stronger than the geometric route gate, no planner search needed);
* the **trace-derived required-mechanics mask** (``cme.trace_mechanics``) so each level
  lands in its true compositional cell (what its solution *uses*, not what it contains).

Then reports composition coverage (filled cells, mechanics-per-cell histogram, the actual
mechanic combinations found) and materializes the feasible elites to a packed pool that
drops straight into the RL ``GeneratedLevelSource``.

Optional illumination on top of the seed archive, two modes:

* ``--illuminate-faithful N`` (**the trustworthy hole-filler**) — *demo-transfer*
  mutation: route-preserving edits (spawn/switch/door kept) gated by **replaying the
  parent's winning demo** on the mutant. A win is the gold-standard native solvability
  cert AND yields a faithful trace mask, so the mutant lands in its true compositional
  cell and ships its own (transferred) demo. Fills holes *correctly*.
* ``--illuminate N`` — the cheap variant: mutated geometry with no demo, so its cell uses
  the **presence** mask and the *geometric* necessary-condition gate (``geometric_only_gate``
  — necessary, not sufficient). Fast/broad but not trustworthy.

The two compose (faithful runs first); seed-only remains the most trustworthy coverage.

Usage
-----
    python -m scripts.levelgen.cme_run --corpus datasets/levelgen/corpus_subset \\
        --out datasets/levelgen/pool_cme --bands 5 --workers 8 \\
        --report-out datasets/levelgen/pool_cme/cme_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import random
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("levelgen.cme_run")


def _iter_corpus(corpus_dir: str):
    for shard in sorted(glob.glob(os.path.join(corpus_dir, "shard_*.pkl"))):
        for r in pickle.load(open(shard, "rb"))["records"]:
            if r.get("demo_inputs") and r.get("demo_len", 0) > 0:
                yield list(r["map_data"]), list(r["demo_inputs"])


def _verify_one(args_tuple) -> Optional[Tuple[List[int], List[int]]]:
    """Native gate: keep the (map_data, demo) iff the demo replays to a win."""
    from scripts.levelgen.augment_corpus import replay_won

    md, demo = args_tuple
    return (md, demo) if replay_won(md, demo) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="decoded corpus dir (shard_*.pkl, with demos)")
    ap.add_argument("--out", required=True, help="output packed-pool dir for feasible elites")
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap source levels (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the native demo-replay gate (trust the corpus demos)")
    ap.add_argument("--illuminate", type=int, default=0,
                    help="mutation iterations after seeding (0 = seed-only; mutants use the "
                         "geometric gate + presence mask — see module docstring)")
    ap.add_argument("--illuminate-faithful", type=int, default=0,
                    help="FAITHFUL demo-transfer illumination iterations: route-preserving "
                         "mutants gated by replaying the parent's winning demo (gold-standard "
                         "native solvability + trace mask). The trustworthy hole-filler.")
    ap.add_argument("--grammar-fill", type=int, default=0,
                    help="plan-first GRAMMAR iterations: CONSTRUCT geometry that forces a "
                         "target mechanic set (one beat per bit) for the archive's remaining "
                         "holes — reaches the curve-compositions / difficulty extremes mutation "
                         "can't (corpus had no curve seeds). Contract-gated (geometric tier, "
                         "no demo), descriptor = presence mask. See nclone.levelgen.grammar.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curation-log", default=None,
                    help="npp-rl learnability_levels.json snapshot (content_id -> solve-rate). "
                         "Blends SFL p(1-p) learnability into the cell fitness so the archive "
                         "keeps + mutates the agent's frontier levels (phase-2 ACCEL).")
    ap.add_argument("--w-learn", type=float, default=4.0,
                    help="learnability weight (scales p(1-p); 0 = pure static phase-1 fitness)")
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nclone.levelgen import cme

    pairs = list(_iter_corpus(args.corpus))
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info("loaded %d demo'd corpus levels from %s", len(pairs), args.corpus)

    # ---- native faithful gate: keep only demo-verified-winning seeds -------------
    if args.no_verify:
        verified = pairs
    else:
        verified = []
        if args.workers > 1:
            import multiprocessing as mp

            with mp.Pool(args.workers) as pool:
                for i, res in enumerate(pool.imap(_verify_one, pairs, chunksize=16)):
                    if res is not None:
                        verified.append(res)
                    if (i + 1) % 2000 == 0:
                        logger.info("verified %d/%d (%d winning)", i + 1, len(pairs), len(verified))
        else:
            verified = [r for r in (_verify_one(p) for p in pairs) if r is not None]
    logger.info("native demo-replay gate: %d/%d levels win → seeds", len(verified), len(pairs))
    if not verified:
        raise SystemExit("no winning seeds — nothing to illuminate")

    mds = [md for md, _ in verified]
    demos = [demo for _, demo in verified]

    # ---- fit + freeze the difficulty edges, seed the archive (trace-mask) --------
    edges = cme.fit_difficulty_edges(mds, n_bands=args.bands)
    if not edges:
        raise SystemExit("could not fit difficulty edges (too few finite-route seeds)")
    logger.info("frozen difficulty edges (route-cost quantiles): %s",
                [round(e, 1) for e in edges])

    # ---- optional agent-learnability fitness (phase-2 / ACCEL) -------------------
    solve_rates = {}
    if args.curation_log:
        solve_rates = cme.load_solve_rates(args.curation_log)
        logger.info("loaded %d agent solve-rates from %s (w_learn=%.1f) → SFL p(1-p) fitness",
                    len(solve_rates), args.curation_log, args.w_learn)
    archive = cme.CMEArchive(edges=edges, w_learn=args.w_learn, solve_rates=solve_rates)
    n_seed = cme.seed_archive(archive, mds, gate=False, demos=demos)
    if solve_rates:
        n_known = sum(1 for e in archive.feasible.values()
                      if cme.content_id(e.map_data) in solve_rates)
        learns = [archive.learnability(e.map_data) for e in archive.feasible.values()]
        learns = [x for x in learns if x is not None]
        logger.info("learnability: %d/%d feasible elites have a solve-rate (mean p(1-p)=%.3f)",
                    n_known, archive.n_feasible, sum(learns) / max(len(learns), 1))
    logger.info("seeded %d feasible elites → %d filled cells (%.1f%% of %d)",
                n_seed, archive.n_feasible, 100 * archive.coverage, cme.N_CELLS)

    # ---- optional FAITHFUL demo-transfer illumination (gold-standard mutant gate) ----
    if args.illuminate_faithful > 0:
        rng = random.Random(args.seed)
        before = archive.n_feasible
        stats = cme.illuminate_faithful(archive, rng, budget=args.illuminate_faithful)
        logger.info("illuminate_faithful(%d): +%d cells (%d→%d); native_replays=%d "
                    "(ratio %.3f), feasible_admits=%d, demo_broke=%d, unknown_band=%d",
                    args.illuminate_faithful, archive.n_feasible - before, before,
                    archive.n_feasible, stats.physics_calls, stats.physics_call_ratio,
                    stats.feasible_admits, stats.demo_broke, stats.unknown_band)

    # ---- optional mutation illumination (geometric gate; see docstring) ----------
    if args.illuminate > 0:
        rng = random.Random(args.seed + 1)
        before = archive.n_feasible
        stats = cme.illuminate(archive, cme.geometric_only_gate, rng, budget=args.illuminate)
        logger.warning("illumination used the GEOMETRIC gate (necessary, not sufficient) + "
                       "presence masks for mutants — see module docstring")
        logger.info("illuminate(%d): +%d cells (%d→%d); physics_calls=%d (ratio %.3f), "
                    "feasible_admits=%d", args.illuminate, archive.n_feasible - before,
                    before, archive.n_feasible, stats.physics_calls,
                    stats.physics_call_ratio, stats.feasible_admits)

    # ---- optional plan-first GRAMMAR fill (construct geometry for the holes) ------
    if args.grammar_fill > 0:
        from nclone.levelgen import grammar

        rng = random.Random(args.seed + 2)
        before = archive.n_feasible
        gstats = grammar.grammar_fill(archive, rng, budget=args.grammar_fill)
        logger.info("grammar_fill(%d): +%d cells (%d→%d); built=%d, admits=%d, "
                    "cells_filled=%d, on_target=%d (match %.3f), unhostable=%d, unknown_band=%d",
                    args.grammar_fill, archive.n_feasible - before, before, archive.n_feasible,
                    gstats.built, gstats.admits, gstats.cells_filled, gstats.on_target,
                    gstats.target_match_ratio, gstats.unhostable, gstats.unknown_band)

    # ---- report + materialize ----------------------------------------------------
    rep = archive.coverage_report()
    combos = {}
    for (mask, _b) in archive.feasible:
        combos[cme.mask_label(mask)] = combos.get(cme.mask_label(mask), 0) + 1
    rep["mechanic_combos"] = dict(sorted(combos.items(), key=lambda kv: -kv[1]))
    rep["seeds_in"] = len(pairs)
    rep["seeds_verified"] = len(verified)
    logger.info("coverage: %d/%d cells (%.1f%%); mechanics-per-cell histogram %s",
                rep["n_feasible"], rep["n_cells"], 100 * rep["coverage"],
                rep["popcount_histogram"])
    logger.info("top mechanic combinations: %s",
                dict(list(rep["mechanic_combos"].items())[:12]))

    n_written = archive.materialize(args.out, writer_id="cme")
    logger.info("materialized %d feasible elites → %s", n_written, args.out)
    rep["materialized"] = n_written
    rep["out"] = args.out
    if args.report_out:
        json.dump(rep, open(args.report_out, "w"), indent=2)
        logger.info("wrote report → %s", args.report_out)


if __name__ == "__main__":
    main()
