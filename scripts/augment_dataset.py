"""Extend a level dataset with physics-preserving map augmentations.

This applies the horizontal-mirror augmentation (the one isometry that preserves
N++ gravity physics; see :mod:`nclone.map_augmentation`) to existing levels and
writes the variants as new dataset records. It works on two source kinds:

* ``--procgen <dataset_dir>``  -- an existing mechanic-benchmark dataset produced
  by ``scripts/generate_benchmark_dataset.py``. Every ``*.pkl`` level is mirrored
  and written back into the same ``split/bucket/Ttier`` layout (or into
  ``--output_dir``), preserving bucket/tier so curriculum sampling is unchanged.

* ``--official <root>``        -- the official N++ corpus under
  ``nclone/maps/official/<dir>/``. Selected directories are read from their raw
  binary form, optionally filtered to the native-safe subset, and emitted both as
  the original and its mirror in dataset-record form under ``--output_dir``.

Symmetric levels (whose mirror is byte-identical) are skipped so augmentation
never produces duplicates. Generation is deterministic: a variant's ``level_id``
is a stable hash of ``(parent_level_id, augmentation)``.

Lattices
--------
The generalized agent trains with the surface-lattice d* potential, which is
computed per level and cached on disk keyed by the level's ``map_data`` hash
(:class:`nclone.value_function.lattice_distance.LatticeDistanceSource`). The
gym env builds it lazily on first use, but an augmented level is a *new* map and
needs its own lattice. Pass ``--warm-lattice`` to precompute and cache the
lattice for every emitted variant up front (parallelised with
``--lattice-workers``), so training does not pay the cold-build cost. Without it,
the env will still build each lattice lazily at first encounter.

Examples
--------
Mirror an existing procgen dataset in place and pre-bake lattices::

    python -m scripts.augment_dataset --procgen ./datasets/mechanic_benchmark \\
        --warm-lattice --lattice-workers 8

Build an augmented native-safe official benchmark::

    python -m scripts.augment_dataset --official nclone/maps/official \\
        --official-dirs S S2 SL --native-safe \\
        --output_dir ./datasets/official_augmented --warm-lattice
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nclone.map_augmentation import NONTRIVIAL_AUGMENTATIONS, iter_augmented


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hash_map_data(map_data: List[int]) -> str:
    h = hashlib.sha1()
    h.update(b",".join(str(int(v)).encode("ascii") for v in map_data))
    return h.hexdigest()


def _augmented_level_id(parent_id: int, kind: str) -> int:
    """Deterministic 63-bit id for a variant, stable across runs."""
    payload = f"aug:{parent_id}:{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _official_level_id(rel_path: str, kind: str) -> int:
    payload = f"official:{rel_path}:{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _write_record(output_dir: Path, record: Dict[str, Any]) -> Path:
    """Write a level record to ``<output>/<split>/<bucket>/T<tier>/<id>.pkl``."""
    split = record["metadata"].get("split", "train")
    bucket = record["bucket"]
    tier = record["tier"]
    tier_dir = output_dir / split / bucket / f"T{tier}"
    tier_dir.mkdir(parents=True, exist_ok=True)
    path = tier_dir / f"{record['level_id']}.pkl"
    with open(path, "wb") as f:
        pickle.dump(record, f)
    return path


def _update_split_metadata(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Merge new records into each split's ``<split>_metadata.json``.

    ``MechanicBenchmarkLoader`` discovers levels via the ``level_ids`` list in
    this file, so augmented pkls are invisible until they are registered here.
    Existing entries are preserved (in-place extension); new level_ids are
    appended, de-duplicated, and per-bucket/tier counts recomputed -- matching
    the schema written by ``scripts/generate_benchmark_dataset.py``.
    """
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_split.setdefault(r["metadata"].get("split", "train"), []).append(r)

    for split, recs in by_split.items():
        meta_path = output_dir / split / f"{split}_metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"split": split, "buckets": {}, "level_ids": []}

        buckets = meta.setdefault("buckets", {})
        flat: List[int] = list(meta.get("level_ids") or [])
        flat_seen = set(flat)

        for r in recs:
            bucket, tier, lid = r["bucket"], r["tier"], r["level_id"]
            b = buckets.setdefault(bucket, {"total": 0, "tiers": {}})
            t = b["tiers"].setdefault(
                str(tier),
                {"count": 0, "level_ids": [], "path": f"{split}/{bucket}/T{tier}"},
            )
            if lid not in t["level_ids"]:
                t["level_ids"].append(lid)
            if lid not in flat_seen:
                flat.append(lid)
                flat_seen.add(lid)

        # Recompute counts/totals from the (now-merged) tier lists.
        for b in buckets.values():
            for t in b["tiers"].values():
                t["count"] = len(t["level_ids"])
            b["total"] = sum(t["count"] for t in b["tiers"].values())
        meta["level_ids"] = flat
        meta["total_levels"] = len(flat)

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[metadata] {meta_path}: {meta['total_levels']} total level_ids")


# ---------------------------------------------------------------------------
# Procgen source: mirror an existing benchmark dataset
# ---------------------------------------------------------------------------

def _augment_procgen(
    dataset_dir: Path,
    output_dir: Path,
    kinds: Tuple[str, ...],
    seen_hashes: set,
) -> List[Dict[str, Any]]:
    """Mirror every level pkl under ``dataset_dir``. Returns the new records."""
    new_records: List[Dict[str, Any]] = []
    pkls = sorted(dataset_dir.glob("**/*.pkl"))
    print(f"[procgen] {dataset_dir}: {len(pkls)} source levels")

    n_skipped_symmetric = 0
    n_skipped_already_aug = 0
    n_dup = 0
    for pkl in pkls:
        with open(pkl, "rb") as f:
            parent = pickle.load(f)
        # Don't augment augmentations (idempotent re-runs add nothing).
        if parent.get("metadata", {}).get("source") == "augmented":
            n_skipped_already_aug += 1
            continue

        parent_map = parent["map_data"]
        parent_id = parent["level_id"]
        for kind, variant in iter_augmented(parent_map, kinds, dedup=True):
            map_hash = _hash_map_data(variant)
            if map_hash in seen_hashes:
                n_dup += 1
                continue
            seen_hashes.add(map_hash)

            record = dict(parent)  # shallow copy of all parent fields
            record["level_id"] = _augmented_level_id(parent_id, kind)
            record["map_data"] = list(variant)
            meta = dict(parent.get("metadata", {}))
            meta = {
                **meta,
                "source": "augmented",
                "augmentation": kind,
                "parent_level_id": parent_id,
                "parent_source": meta.get("source", "procedural"),
                "map_hash": map_hash,
                "description": f"{meta.get('description', '')} [{kind}]".strip(),
            }
            record["metadata"] = meta
            new_records.append(record)

    for r in new_records:
        _write_record(output_dir, r)
    print(
        f"[procgen] wrote {len(new_records)} variants "
        f"(skipped {n_skipped_already_aug} already-augmented, {n_dup} duplicates)"
    )
    return new_records


# ---------------------------------------------------------------------------
# Official source: read binary maps, emit original + mirror as records
# ---------------------------------------------------------------------------

def _read_official(path: Path) -> List[int]:
    with open(path, "rb") as f:
        return [int(b) for b in f.read()]


def _augment_official(
    root: Path,
    dirs: List[str],
    output_dir: Path,
    kinds: Tuple[str, ...],
    native_safe: bool,
    include_original: bool,
    seen_hashes: set,
) -> List[Dict[str, Any]]:
    """Emit dataset records for official maps and their augmentations."""
    classify = None
    if native_safe:
        from nclone.maps.official_native_safe_filter import classify_map_data

        classify = classify_map_data

    records: List[Dict[str, Any]] = []
    n_files = n_filtered = n_dup = 0
    for d in dirs:
        ddir = root / d
        if not ddir.is_dir():
            print(f"[official] WARNING: missing dir {ddir}", file=sys.stderr)
            continue
        for path in sorted(p for p in ddir.iterdir() if p.is_file()):
            n_files += 1
            try:
                base_map = _read_official(path)
            except Exception as exc:  # pragma: no cover - corrupt file guard
                print(f"[official] skip {path}: {exc}", file=sys.stderr)
                continue
            if classify is not None:
                ok, _reason, _hist = classify(base_map)
                if not ok:
                    n_filtered += 1
                    continue

            rel = f"{d}/{path.name}"
            # Original ("identity") plus the requested non-trivial augmentations.
            variants: List[Tuple[str, List[int]]] = []
            if include_original:
                variants.append(("identity", list(base_map)))
            variants.extend(iter_augmented(base_map, kinds, dedup=True))

            for kind, vmap in variants:
                map_hash = _hash_map_data(vmap)
                if map_hash in seen_hashes:
                    n_dup += 1
                    continue
                seen_hashes.add(map_hash)
                is_aug = kind != "identity"
                records.append(
                    {
                        "level_id": _official_level_id(rel, kind),
                        "seed": None,
                        "bucket": f"OFFICIAL_{d}",
                        "tier": 1,
                        "generator": "official",
                        "preset": rel,
                        "map_data": list(vmap),
                        "secondary": [],
                        "metadata": {
                            "split": "train",
                            "source": "augmented" if is_aug else "official",
                            "augmentation": kind if is_aug else None,
                            "official_path": rel,
                            "native_safe": bool(native_safe),
                            "map_hash": map_hash,
                            "description": f"official {rel}"
                            + (f" [{kind}]" if is_aug else ""),
                        },
                    }
                )

    for r in records:
        _write_record(output_dir, r)
    print(
        f"[official] {n_files} files -> wrote {len(records)} records "
        f"(filtered {n_filtered} non-native-safe, {n_dup} duplicates)"
    )
    return records


# ---------------------------------------------------------------------------
# Lattice warming
# ---------------------------------------------------------------------------

def _warm_one_lattice(map_data: List[int]) -> Tuple[bool, bool]:
    """Build+cache the lattice for one map. Returns (built_ok, available)."""
    try:
        from nclone.value_function.lattice_distance import LatticeDistanceSource

        src = LatticeDistanceSource(map_data, use_cache=True)
        return True, bool(src.available)
    except Exception:  # pragma: no cover - defensive; lattice is best-effort
        return False, False


def _warm_lattices(records: List[Dict[str, Any]], workers: int) -> None:
    maps = [r["map_data"] for r in records]
    print(f"[lattice] warming {len(maps)} lattices with {workers} worker(s)...")
    n_ok = n_unavail = n_err = 0
    if workers <= 1:
        results = (_warm_one_lattice(m) for m in maps)
        for built, avail in results:
            n_ok += int(built and avail)
            n_unavail += int(built and not avail)
            n_err += int(not built)
    else:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            for built, avail in pool.imap_unordered(_warm_one_lattice, maps, chunksize=1):
                n_ok += int(built and avail)
                n_unavail += int(built and not avail)
                n_err += int(not built)
    print(
        f"[lattice] done: {n_ok} cached & available, "
        f"{n_unavail} built-but-unavailable (geometric fallback), {n_err} errored"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _ensure_manifest(output_dir: Path) -> None:
    """Write a minimal ``manifest.json`` if one is absent.

    ``MechanicBenchmarkLoader`` refuses a directory without a top-level
    ``manifest.json``. In-place procgen extension keeps the source manifest; a
    fresh (e.g. official-only) output needs a stub so it loads as a dataset.
    """
    manifest = output_dir / "manifest.json"
    if manifest.is_file():
        return
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "source": "scripts/augment_dataset.py",
                "schema": "mechanic_primary_v1",
                "note": "augmentation-built dataset; see augmentation_manifest.json",
            },
            f,
            indent=2,
        )


def _write_summary(output_dir: Path, records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    by_split: Dict[str, int] = {}
    by_bucket: Dict[str, int] = {}
    for r in records:
        by_split[r["metadata"].get("split", "train")] = (
            by_split.get(r["metadata"].get("split", "train"), 0) + 1
        )
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    summary = {
        "version": 1,
        "source_script": "scripts/augment_dataset.py",
        "augmentations": list(args.kinds),
        "procgen_source": str(args.procgen) if args.procgen else None,
        "official_root": str(args.official) if args.official else None,
        "official_dirs": args.official_dirs if args.official else None,
        "native_safe": args.native_safe,
        "warm_lattice": args.warm_lattice,
        "total_variants": len(records),
        "by_split": by_split,
        "by_bucket": by_bucket,
    }
    out = output_dir / "augmentation_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("source (choose at least one)")
    src.add_argument(
        "--procgen", type=Path, default=None,
        help="Existing benchmark dataset dir to mirror (reads **/*.pkl).",
    )
    src.add_argument(
        "--official", type=Path, default=None,
        help="Root of the official corpus (e.g. nclone/maps/official).",
    )
    parser.add_argument(
        "--official-dirs", nargs="+", default=["S"],
        help="Official subdirectories to read (default: S).",
    )
    parser.add_argument(
        "--native-safe", action="store_true",
        help="For --official: keep only native-backend-safe maps "
        "(mines/gold/single exit; no doors/drones/etc).",
    )
    parser.add_argument(
        "--no-original", action="store_true",
        help="For --official: emit only mirrors, not the original maps.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Where to write variant pkls. Defaults to --procgen dir (in-place "
        "extension). Required when using --official.",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=list(NONTRIVIAL_AUGMENTATIONS),
        help=f"Augmentations to apply (default: {list(NONTRIVIAL_AUGMENTATIONS)}).",
    )
    parser.add_argument(
        "--warm-lattice", action="store_true",
        help="Precompute & cache the surface-lattice d* potential for every "
        "emitted variant (needed for the generalized agent; otherwise built "
        "lazily in-env).",
    )
    parser.add_argument(
        "--lattice-workers", type=int, default=4,
        help="Parallel workers for --warm-lattice (default: 4).",
    )
    args = parser.parse_args()

    if not args.procgen and not args.official:
        parser.error("provide at least one of --procgen or --official")
    if args.official and not args.output_dir:
        parser.error("--official requires --output_dir")

    kinds = tuple(args.kinds)
    seen_hashes: set = set()
    all_records: List[Dict[str, Any]] = []

    if args.procgen:
        out = args.output_dir or args.procgen
        out.mkdir(parents=True, exist_ok=True)
        procgen_records = _augment_procgen(args.procgen, out, kinds, seen_hashes)
        _update_split_metadata(out, procgen_records)
        all_records += procgen_records

    if args.official:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        official_records = _augment_official(
            args.official,
            args.official_dirs,
            args.output_dir,
            kinds,
            native_safe=args.native_safe,
            include_original=not args.no_original,
            seen_hashes=seen_hashes,
        )
        _update_split_metadata(args.output_dir, official_records)
        all_records += official_records

    if args.warm_lattice and all_records:
        _warm_lattices(all_records, args.lattice_workers)

    summary_dir = args.output_dir or args.procgen
    _ensure_manifest(summary_dir)
    _write_summary(summary_dir, all_records, args)

    print(f"\n=== Augmentation complete: {len(all_records)} variants ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
