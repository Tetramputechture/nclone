# Corpus Statistics

## Corpus Overview

| Metric | Value |
|---|---|
| source | /home/tetra/projects/attracts_filtered |
| output | /home/tetra/projects/nclone/datasets/iql_pipeline_v2 |
| seed | 42 |
| val_fraction | 0.1 |
| n_total_files | 22596 |
| n_train_files | 20336 |
| n_val_files | 2260 |
| n_total_levels | 22596 |
| n_train_levels | 20336 |
| n_val_levels | 2260 |

## Split Summary

| Metric | Train | Val |
|---|---|---|
| Distinct level grids | 51,611 | 5,634 |
| Episodes | 51,611 | 5,634 |
| Transitions | 41,972,410 | 4,385,947 |
| Episode length: min | 12 | 60 |
| Episode length: median | 621 | 623 |
| Episode length: p95 | 2,083 | 1,968 |
| Episode length: max | 6,788 | 4,212 |
| Mine-free levels | 6,932 | 708 |

## Mine Count Distribution (Filtered Corpus)

Source: raw attract files (exact counts across all 22,596 filtered levels).

| Mine count | Levels |
|---|---|
| 0 | 2,829 |
| 1–10 | 1,715 |
| 11–50 | 8,564 |
| 51–100 | 6,548 |
| 101–500 | 2,833 |
| 500+ | 107 |

## Tile Type Frequency (Top 20, Train)

| Tile ID | Count |
|---|---|
| 0 | 21,558,951 |
| 1 | 16,978,259 |
| 9 | 504,354 |
| 8 | 504,259 |
| 6 | 475,467 |
| 7 | 474,279 |
| 3 | 454,381 |
| 5 | 450,100 |
| 4 | 415,202 |
| 2 | 399,989 |
| 20 | 351,439 |
| 21 | 347,581 |
| 24 | 344,707 |
| 25 | 340,088 |
| 18 | 337,911 |
| 19 | 335,686 |
| 22 | 333,323 |
| 23 | 330,899 |
| 32 | 329,751 |
| 28 | 328,238 |

## Notes

- **Episode count vs. target:** The 50K / 5K episode thresholds in the epic assume all attract demos win when replayed through the Rust simulator. In practice ~74% of original attract files and ~66% of augmented files replay to a win in `nclone_native`; non-winning replays are silently dropped by the transition extractor. The remaining episodes are all confirmed wins with correct reward structure (100/100 sampled episodes pass the reward-structure check).
- **Mine-free levels (HDF5):** Levels where no toggle mine appeared in the nearest-8-mine features across any transition. This includes both truly mine-free levels and levels where all mines were outside the ninja's nearest-8 range throughout the replay. For exact mine-free counts use the raw attract files (see Mine Count Distribution above, bucket 0).
- **Augmented level count:** Each original attract file produces up to 3 augmented variants (horizontal_mirror, tile_rand, translate). Augmented variants that do not produce a winning replay are dropped by the extractor, so the HDF5 level count reflects only winning episodes from 66,128 combined files.

## Reproduction Commands

```bash
# Generate datasets (T1.1–T1.4)
./scripts/build_augmented_dataset.sh \
    --filtered-dir /home/tetra/projects/attracts_filtered \
    --output-dir   datasets/iql_pipeline \
    --workers      16 \
    --verify-sample 200 \
    --seed 42

# Generate this document (T1.5)
python -m nclone.offline_rl.dataset.corpus_stats \
    --train datasets/iql_pipeline_v2/iql_train_dataset.h5 \
    --val   datasets/iql_pipeline_v2/iql_val_dataset.h5 \
    --filtered-dir /home/tetra/projects/attracts_filtered \
    --output docs/generalizable_agent/01-data-pipeline/corpus-stats.md
```
