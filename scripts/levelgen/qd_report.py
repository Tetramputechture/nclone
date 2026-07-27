"""QD coverage report + stratified held-out benchmark builder (WS-D).

Reads a level pool (packed pool or mechanic-benchmark dir), builds a
:class:`nclone.levelgen.qd.QDArchive` over behaviour descriptors, and:

* prints / writes a **coverage report** — full-product cell coverage, per-axis
  marginals, and the (difficulty × structure) **holes** the generator never fills (the
  bias signal for the next mutation round);
* optionally builds a **stratified held-out benchmark** (``--holdout-out``): up to
  ``--k-per-cell`` levels per (difficulty × structure) stratum, written to a separate
  packed pool, disjoint by ``level_id`` from a written train pool.

Example::

    python -m scripts.levelgen.qd_report --pool datasets/levelgen/pool \\
        --report-out datasets/levelgen/pool/qd_report.json \\
        --holdout-out datasets/levelgen/heldout --k-per-cell 5 \\
        --train-out datasets/levelgen/pool_train
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from nclone.levelgen.pool_store import PackedPool, PoolWriter, is_packed_pool
from nclone.levelgen.qd import QDArchive, descriptor_from_record, descriptor_from_map_data

logger = logging.getLogger("levelgen.qd_report")


def _iter_pool(pool_dir: str, with_map_data: bool):
    """Yield ``(level_id, record, map_data_or_None)`` for a packed pool or benchmark dir."""
    if is_packed_pool(pool_dir):
        pool = PackedPool(pool_dir)
        for rec in pool.records():
            lid = int(rec["id"])
            md = pool.map_data(lid) if with_map_data else None
            yield lid, rec, md
    else:
        from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader

        ld = MechanicBenchmarkLoader(pool_dir, splits=("train",))
        for i in range(len(ld)):
            payload = ld.load_payload(i)
            lid = int(payload["level_id"])
            md = list(payload["map_data"])
            meta = payload.get("metadata", {}) or {}
            rec = {
                "id": lid,
                "band": meta.get("difficulty_band", "unknown"),
                "n_mines": meta.get("n_mines", 0),
                "struct": meta.get("structure_score", 0.0),
            }
            yield lid, rec, (md if with_map_data else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="QD coverage report + held-out benchmark")
    ap.add_argument("--pool", required=True, help="packed pool dir or benchmark dir")
    ap.add_argument("--report-out", default=None, help="write coverage report JSON here")
    ap.add_argument("--fast", action="store_true",
                    help="skip reading map_data for the mechanic axis (faster; mechanic='flat')")
    ap.add_argument("--holdout-out", default=None, help="write a stratified held-out packed pool here")
    ap.add_argument("--train-out", default=None, help="write the train remainder packed pool here")
    ap.add_argument("--k-per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with_md = (not args.fast) or bool(args.holdout_out) or bool(args.train_out)
    arch = QDArchive()
    # keep map_data only if we need to re-emit a held-out / train pool
    keep_md = bool(args.holdout_out) or bool(args.train_out)
    md_by_id: Dict[int, List[int]] = {}
    rec_by_id: Dict[int, dict] = {}
    n = 0
    for lid, rec, md in _iter_pool(args.pool, with_map_data=with_md):
        if md is not None and not args.fast:
            d = descriptor_from_record(rec, map_data=md)
        else:
            d = descriptor_from_record(rec, map_data=None)
        arch.add(lid, d)
        rec_by_id[lid] = rec
        if keep_md and md is not None:
            md_by_id[lid] = md
        n += 1
    logger.info("indexed %d levels", n)

    report = arch.coverage_report()
    logger.info("coverage: %d/%d cells (%.1f%%), occupied; %d difficulty×structure holes",
                report["n_occupied_cells"], report["total_cells"],
                100 * report["cell_coverage"], len(report["difficulty_x_structure_holes"]))
    logger.info("marginals: %s", json.dumps(report["marginals"]))
    if report["difficulty_x_structure_holes"]:
        logger.info("HOLES (difficulty×structure): %s",
                    json.dumps(report["difficulty_x_structure_holes"]))
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2))
        logger.info("wrote report -> %s", args.report_out)

    if args.holdout_out:
        rng = random.Random(args.seed)
        holdout_ids = set(arch.stratified_sample(args.k_per_cell, rng))
        logger.info("held-out: %d levels (stratified, k_per_cell=%d)", len(holdout_ids), args.k_per_cell)
        _write_pool(args.holdout_out, "holdout", holdout_ids, md_by_id, rec_by_id, source="heldout")
        if args.train_out:
            train_ids = set(md_by_id) - holdout_ids
            _write_pool(args.train_out, "train", train_ids, md_by_id, rec_by_id, source=None)
            logger.info("train remainder: %d levels (disjoint from held-out)", len(train_ids))
            assert not (holdout_ids & train_ids), "held-out / train overlap!"


def _write_pool(out_dir, writer_id, ids, md_by_id, rec_by_id, source):
    with PoolWriter(out_dir, writer_id=writer_id) as w:
        for lid in ids:
            md = md_by_id.get(lid)
            if md is None:
                continue
            rec = rec_by_id.get(lid, {})
            w.add(
                md,
                bucket=str(rec.get("bucket", "GENERATED")),
                tier=int(rec.get("tier", 0)),
                source=source or rec.get("source", "generated"),
                parent=rec.get("parent"),
                fs_cost=rec.get("fs_cost"),
                difficulty_band=rec.get("band", "unknown"),
                structure_score=rec.get("struct"),
                mechanic=rec.get("mech"),
            )


if __name__ == "__main__":
    main()
