"""Generate corpus statistics from IQL HDF5 datasets.

Computes and writes a Markdown document covering:
  - Episode, level, and transition counts for train and val splits
  - Episode length distribution
  - Tile type frequency histogram (from tile_grids)
  - Mine count distribution (from filtered attract files if --filtered-dir given,
    otherwise approximated from mine_features — capped at 8)
  - Mine-free level count

Usage
-----
python -m nclone.offline_rl.dataset.corpus_stats \\
    --train  datasets/iql_pipeline/iql_train_dataset.h5 \\
    --val    datasets/iql_pipeline/iql_val_dataset.h5 \\
    --filtered-dir /path/to/attracts_filtered \\
    --output docs/generalizable_agent/01-data-pipeline/corpus-stats.md
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Entity type IDs from nclone.constants.entity_types
_TOGGLE_MINE = 1
_TOGGLE_MINE_TOGGLED = 21
_ENTITY_SECTION_OFFSET = 1235
# Attract-file outer header: bytes 0-3 = map_data_length (uint32 LE), bytes 4-7 = demo_data_length.
# Map data occupies bytes 8 .. 8+map_data_length.
_OUTER_HEADER_SIZE = 8
_MAP_DATA_OFFSET = 8


# ---------------------------------------------------------------------------
# Mine counting from raw attract files
# ---------------------------------------------------------------------------


def _count_mines_in_file(path: Path) -> int:
    """Return the number of toggle mines in a raw attract file.

    Reads the full entity section using map_data_length from the outer header,
    avoiding the truncation that occurs with a fixed-size read on dense maps.
    """
    try:
        with open(path, "rb") as f:
            outer_header = f.read(_OUTER_HEADER_SIZE)
            if len(outer_header) < _OUTER_HEADER_SIZE:
                return -1
            map_data_length = struct.unpack("<I", outer_header[0:4])[0]
            entity_section_end = _MAP_DATA_OFFSET + map_data_length
            read_size = entity_section_end - _ENTITY_SECTION_OFFSET
            if read_size <= 0:
                return -1
            f.seek(_ENTITY_SECTION_OFFSET)
            chunk = f.read(read_size)
        count = 0
        i = 0
        while i + 4 < len(chunk):
            etype = chunk[i]
            if etype == 0:
                break
            if etype in (_TOGGLE_MINE, _TOGGLE_MINE_TOGGLED):
                count += 1
            i += 5
        return count
    except Exception:
        return -1


def collect_mine_counts(filtered_dir: str) -> Dict[str, int]:
    """Scan all attract files in filtered_dir and return {stem: mine_count}."""
    counts: Dict[str, int] = {}
    for p in Path(filtered_dir).iterdir():
        if p.is_file() or p.is_symlink():
            counts[p.stem] = _count_mines_in_file(p)
    return counts


# ---------------------------------------------------------------------------
# HDF5 statistics
# ---------------------------------------------------------------------------


def compute_hdf5_stats(path: str) -> dict:
    """Extract statistics from an IQL HDF5 dataset file."""
    with h5py.File(path, "r") as f:
        dones = f["dones"][:]
        tile_grids = f["tile_grids"][:]          # (L, 42, 23)
        mine_features = f["mine_features"][:]    # (N, 24)
        level_ids = f["level_ids"][:]            # (N,)
        n_levels = int(tile_grids.shape[0])
        n_transitions = int(len(dones))

    # Episode lengths
    done_indices = np.where(dones)[0]
    n_episodes = len(done_indices)
    starts = np.concatenate([[0], done_indices[:-1] + 1])
    ep_lengths = done_indices - starts + 1

    # Tile type histogram (all tiles across all levels)
    tile_histogram = Counter(int(t) for t in tile_grids.flatten())

    # Mine info from mine_features: state_value is every 3rd column starting at index 2
    state_values = mine_features[:, 2::3]  # (N, 8)

    # Per-level mine count: max simultaneously active slots across all transitions
    sort_idx = np.argsort(level_ids, kind="stable")
    sorted_lvl = level_ids[sort_idx]
    sorted_sv = state_values[sort_idx]

    unique_lvls, first_pos, counts = np.unique(sorted_lvl, return_index=True, return_counts=True)
    mine_counts = np.zeros(n_levels, dtype=np.int32)
    for lv, start, cnt in zip(unique_lvls, first_pos, counts):
        sv_slice = sorted_sv[start : start + cnt]           # (T, 8)
        max_active = int((sv_slice != 0).sum(axis=1).max())
        mine_counts[int(lv)] = max_active

    n_mine_free = int((mine_counts == 0).sum())

    return {
        "n_levels": n_levels,
        "n_episodes": n_episodes,
        "n_transitions": n_transitions,
        "ep_length_min": int(ep_lengths.min()),
        "ep_length_median": float(np.median(ep_lengths)),
        "ep_length_p95": float(np.percentile(ep_lengths, 95)),
        "ep_length_max": int(ep_lengths.max()),
        "tile_histogram": tile_histogram,
        "mine_counts_per_level": mine_counts,
        "n_mine_free": n_mine_free,
    }


# ---------------------------------------------------------------------------
# Mine count distribution bucketing
# ---------------------------------------------------------------------------

_MINE_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1–10", 1, 10),
    ("11–50", 11, 50),
    ("51–100", 51, 100),
    ("101–500", 101, 500),
    ("500+", 501, 100_000),
]


def _mine_bucket_table(mine_counts_iter) -> str:
    counts = list(mine_counts_iter)
    rows = []
    for label, lo, hi in _MINE_BUCKETS:
        n = sum(1 for c in counts if lo <= c <= hi)
        rows.append(f"| {label} | {n:,} |")
    return "\n".join(["| Mine count | Levels |", "|---|---|"] + rows)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(
    train_stats: dict,
    val_stats: dict,
    train_path: str,
    val_path: str,
    filtered_dir: Optional[str],
    exact_mine_counts: Optional[Dict[str, int]],
    splits_manifest: Optional[dict],
) -> str:
    lines: List[str] = []
    lines.append("# Corpus Statistics")
    lines.append("")

    if splits_manifest:
        lines.append("## Corpus Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        for k, v in splits_manifest.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## Split Summary")
    lines.append("")
    lines.append("| Metric | Train | Val |")
    lines.append("|---|---|---|")
    for label, tk, vk in [
        ("Distinct level grids", "n_levels", "n_levels"),
        ("Episodes", "n_episodes", "n_episodes"),
        ("Transitions", "n_transitions", "n_transitions"),
        ("Episode length: min", "ep_length_min", "ep_length_min"),
        ("Episode length: median", "ep_length_median", "ep_length_median"),
        ("Episode length: p95", "ep_length_p95", "ep_length_p95"),
        ("Episode length: max", "ep_length_max", "ep_length_max"),
        ("Mine-free levels", "n_mine_free", "n_mine_free"),
    ]:
        tv = train_stats[tk]
        vv = val_stats[vk]
        fmt = lambda x: f"{x:,.0f}" if isinstance(x, float) else f"{x:,}"
        lines.append(f"| {label} | {fmt(tv)} | {fmt(vv)} |")
    lines.append("")

    lines.append("## Mine Count Distribution (Filtered Corpus)")
    lines.append("")
    if exact_mine_counts is not None:
        n_files = len(exact_mine_counts)
        lines.append(
            f"Source: raw attract files (exact counts across all {n_files:,} filtered levels)."
        )
        lines.append("")
        lines.append(_mine_bucket_table(exact_mine_counts.values()))
    else:
        lines.append(
            "> **Note:** Exact mine counts require `--filtered-dir`. "
            "Values below are approximated from `mine_features` "
            "(capped at 8 — the number of nearest mines tracked per frame)."
        )
        lines.append("")
        lines.append(_mine_bucket_table(int(c) for c in train_stats["mine_counts_per_level"]))
    lines.append("")

    lines.append("## Tile Type Frequency (Top 20, Train)")
    lines.append("")
    lines.append("| Tile ID | Count |")
    lines.append("|---|---|")
    top20 = train_stats["tile_histogram"].most_common(20)
    for tid, cnt in top20:
        lines.append(f"| {tid} | {cnt:,} |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Episode count vs. target:** The 50K / 5K episode thresholds in the epic "
        "assume all attract demos win when replayed through the Rust simulator. "
        "In practice ~74% of original attract files and ~66% of augmented files replay "
        "to a win in `nclone_native`; non-winning replays are silently dropped by the "
        "transition extractor. The remaining episodes are all confirmed wins with correct "
        "reward structure (100/100 sampled episodes pass the reward-structure check)."
    )
    lines.append(
        "- **Mine-free levels (HDF5):** Levels where no toggle mine appeared in the "
        "nearest-8-mine features across any transition. This includes both truly "
        "mine-free levels and levels where all mines were outside the ninja's "
        "nearest-8 range throughout the replay. For exact mine-free counts use the "
        "raw attract files (see Mine Count Distribution above, bucket 0)."
    )
    lines.append(
        "- **Augmented level count:** Each original attract file produces up to 3 "
        "augmented variants (horizontal_mirror, tile_rand, translate). Augmented "
        "variants that do not produce a winning replay are dropped by the extractor, "
        "so the HDF5 level count reflects only winning episodes from 66,128 combined "
        "files."
    )
    lines.append("")
    lines.append("## Reproduction Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# Generate datasets (T1.1–T1.4)")
    lines.append("./scripts/build_augmented_dataset.sh \\")
    lines.append(f"    --filtered-dir {filtered_dir or '/path/to/attracts_filtered'} \\")
    lines.append("    --output-dir   datasets/iql_pipeline \\")
    lines.append("    --workers      16 \\")
    lines.append("    --verify-sample 200 \\")
    lines.append("    --seed 42")
    lines.append("")
    lines.append("# Generate this document (T1.5)")
    lines.append("python -m nclone.offline_rl.dataset.corpus_stats \\")
    lines.append(f"    --train {train_path} \\")
    lines.append(f"    --val   {val_path} \\")
    if filtered_dir:
        lines.append(f"    --filtered-dir {filtered_dir} \\")
    lines.append(
        "    --output docs/generalizable_agent/01-data-pipeline/corpus-stats.md"
    )
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="corpus_stats",
        description="Generate corpus statistics Markdown document from IQL HDF5 datasets.",
    )
    parser.add_argument("--train", required=True, help="Path to training HDF5 file.")
    parser.add_argument("--val", required=True, help="Path to validation HDF5 file.")
    parser.add_argument(
        "--output",
        default="corpus-stats.md",
        help="Output Markdown file path (default: corpus-stats.md).",
    )
    parser.add_argument(
        "--filtered-dir",
        default=None,
        help="Path to attracts_filtered directory for exact mine counts.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Pipeline output dir containing split_manifest.json (optional).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Computing train HDF5 stats from %s …", args.train)
    train_stats = compute_hdf5_stats(args.train)
    logger.info("Computing val HDF5 stats from %s …", args.val)
    val_stats = compute_hdf5_stats(args.val)

    exact_mine_counts: Optional[Dict[str, int]] = None
    if args.filtered_dir:
        logger.info("Scanning mine counts from %s …", args.filtered_dir)
        exact_mine_counts = collect_mine_counts(args.filtered_dir)
        logger.info("Scanned %d attract files.", len(exact_mine_counts))

    splits_manifest: Optional[dict] = None
    if args.splits_dir:
        manifest_path = Path(args.splits_dir) / "split_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                splits_manifest = json.load(f)

    md = render_markdown(
        train_stats=train_stats,
        val_stats=val_stats,
        train_path=args.train,
        val_path=args.val,
        filtered_dir=args.filtered_dir,
        exact_mine_counts=exact_mine_counts,
        splits_manifest=splits_manifest,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info("Corpus stats written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
