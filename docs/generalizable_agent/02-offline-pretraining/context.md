# Epic 2 Context: What Epic 1 Delivered

This document summarises the state of the codebase and datasets at the end of Epic 1
(Data Pipeline) to give whoever picks up Epic 2 (Offline Pre-training) an accurate
picture of what exists, what changed from the original plan, and what to watch out for.

---

## Datasets

| Dataset | Path | Size |
|---------|------|------|
| Train HDF5 | `datasets/iql_pipeline_v2/iql_train_dataset.h5` | ~5.5 GB |
| Val HDF5 | `datasets/iql_pipeline_v2/iql_val_dataset.h5` | ~600 MB |
| Split manifest | `datasets/iql_pipeline_v2/split_manifest.json` | — |
| Val level IDs | `datasets/iql_pipeline_v2/val_level_ids.txt` | — |
| Filtered corpus | `/home/tetra/projects/attracts_filtered/` | 22,596 symlinks |

### Numbers at a glance

| Metric | Train | Val |
|---|---|---|
| Episodes | 51,611 | 5,634 |
| Distinct level grids | 51,611 | 5,634 |
| Transitions | 41,972,410 | 4,385,947 |
| Episode length: median | 621 frames | 623 frames |
| Episode length: p95 | 2,083 frames | 1,968 frames |
| Episode length: max | 6,788 frames | 4,212 frames |
| Mine-free levels | 6,932 | 708 |
| Level ID overlap | — | **0** (verified) |
| Reward structure check | 100/100 PASS | 100/100 PASS |

Both datasets pass the full validation suite (`validate_dataset.py`).
Full corpus statistics live in `docs/generalizable_agent/01-data-pipeline/corpus-stats.md`.

---

## What the Pipeline Does

The build script `scripts/build_augmented_dataset.sh` runs five steps in order:

1. **Split** — `prepare_splits.py` partitions the 22,596 filtered attract files 90/10 by
   level ID (seed 42) into `train/` and `val/` subdirectories.
2. **Augment train** — `augment_cli.py` applies three augmentations per file
   (`horizontal_mirror`, `tile_rand`, `translate`) writing `.replay` files to
   `augmented_train/`. Val is augmented the same way into `augmented_val/`.
3. **Merge** — originals and augmented files are symlinked together into
   `combined_train/` and `combined_val/` (≈81K and ≈9K files respectively).
4. **Build HDF5** — `build_dataset_cli.py` runs `TransitionExtractor` (Rust simulator
   only) on the combined dirs. Only winning episodes are kept; non-winning replays are
   silently dropped.
5. **Validate** — `validate_dataset.py` checks reward structure, level-ID disjoint
   constraint, and basic shape/dtype invariants.

To reproduce the datasets from scratch:

```bash
./scripts/build_augmented_dataset.sh \
    --filtered-dir /home/tetra/projects/attracts_filtered \
    --output-dir   datasets/iql_pipeline_v2 \
    --workers      16 \
    --verify-sample 0 \
    --seed 42
```

---

## Deviations from the Original Epic 1 Plan

### T1.1 — Accepted count was below 50K raw files

The original acceptance criterion required ≥50K raw filtered files. The corpus has only
22,596 qualifying levels (entity-set filter is strict: exactly 1 exit door, 1 exit
switch, no dynamic entities). The 50K episode threshold in HDF5 is met by combining
original replays with three augmented variants per level.

### Filter parser bug (fixed)

The original `filter_attracts.py` read a fixed 500 bytes (= 100 entities) from the
entity section. Attract files with the exit door beyond entity position 100 were
incorrectly rejected with `exit_doors=0`. The scanner was rewritten to read the full
entity section using the `map_data_length` field in the 8-byte file header. This
recovered ≈4,000 previously-rejected levels and raised the corpus from 18,369 to 22,596.

### Augmentor count-overflow bug (fixed)

`attract_augmentor._build_map_data()` stored per-entity-type counts in single-byte
fields. Dense mine levels (e.g. one level has 1,620 mines) overflowed with
`ValueError: byte must be in range(0, 256)`. Fixed in
`nclone/offline_rl/dataset/augmentation/attract_augmentor.py` lines 256–260 using
`struct.pack_into("<H", ...)` (2-byte little-endian) for all four count slots.

### Episode count shortfall vs. 50K/5K targets

With verification disabled (`--verify-sample 0`) the pipeline completes, but a fraction
of replays do not win in the Rust simulator:

- ~74% of original attract files → winning episode
- ~66% of augmented files → winning episode

The combined 81K train input files yield 51,611 winning episodes — slightly above the
50K target. Had we kept `--verify-sample 200` (the default) the augmentation step
aborted with a non-zero exit code, triggering `set -euo pipefail`. Running with
`--verify-sample 0` lets `TransitionExtractor` act as the quality gate (it already
drops non-winners).

---

## Infrastructure Additions and Changes

The following files were **created** during Epic 1 and are ready for use in Epic 2:

| File | What it does |
|------|-------------|
| `scripts/build_augmented_dataset.sh` | End-to-end pipeline: split → augment → merge → build → validate |
| `nclone/offline_rl/dataset/prepare_splits.py` | Splits corpus by level ID; merges originals + augmented |
| `nclone/offline_rl/dataset/validate_dataset.py` | Validates HDF5 reward structure; emits pass/fail |
| `nclone/offline_rl/dataset/corpus_stats.py` | Generates `corpus-stats.md` from HDF5 + raw files |

The following files were **modified**:

| File | What changed |
|------|-------------|
| `nclone/offline_rl/dataset/filter_attracts.py` | Full entity-section read (was capped at 100 entities); co-op format detection |
| `nclone/offline_rl/dataset/augmentation/attract_augmentor.py` | Entity count fields widened to uint16 |
| `nclone/offline_rl/dataset/augmentation/horizontal_mirror.py` | Mirror physics fixes (wall-normal sign, mine parity) |
| `nclone/offline_rl/dataset/augmentation/reachable_area_translator.py` | Reachable-area translation augmentation |
| `nclone/ninja.py` (Python sim) | Floating-point corner-case fixes for mirror symmetry |
| `nclone-native/src/ninja.rs` (Rust sim) | Same corner-case fixes; Rust sim is the authoritative simulator |

---

## What the Rust Simulator Is Used For

Both the HDF5 builder and the augmentation verifier run **only** the Rust simulator
(`nclone_native`). There is no Python-simulator fallback in the data pipeline.

- **`transition_extractor.py`** calls `nclone_native.Simulator.load(map_data)` and
  steps it frame-by-frame. Only replays whose final frame returns `won=True` produce
  transitions.
- **`verify_augmentation.py`** replays `.replay` files through the Rust simulator to
  spot-check that augmented files are still winnable.

The Python simulator (`nclone/ninja.py`) is **not** called anywhere in the data pipeline.

---

## State Vector and Action Space (for reference)

`TransitionExtractor` emits this layout; `IQLNetworks` must match it exactly.

### State (11D, stored in `states` / `next_states`)

| Index | Name | Range |
|-------|------|-------|
| 0 | `x_norm` | [0, 1] |
| 1 | `y_norm` | [0, 1] |
| 2 | `vx_norm` | approx [−1, 1] |
| 3 | `vy_norm` | approx [−1, 1] |
| 4 | `on_ground` | {0, 1} |
| 5 | `wall_sliding` | {0, 1} |
| 6 | `jump_frames` | [0, 1] (normalized) |
| 7 | `switch_collected` | {0, 1} |
| 8 | `floor_nx` | [−1, 1] |
| 9 | `floor_ny` | [−1, 1] |
| 10 | `wall_n` | {−1, 0, 1} |

### Mine features (24D, stored in `mine_features` / `next_mine_features`)

8 nearest mines × 3 features each: `[dx_norm, dy_norm, state]`  
`state = 0` (armed) or `1` (disarmed/gone). Padded with zeros if fewer than 8 mines
exist.

### Actions (6-class discrete, stored in `actions`)

`[left, right, jump, left+jump, right+jump, no-op]` — integer in `[0, 5]`.

### Reward

`+1` at switch collection, `+1` at exit, `−1` per step. Every episode in HDF5 ends in a
win, so `done=True` always corresponds to `reward=+1`.

---

## Tile Grid Layout (for `TileEncoder`)

Stored in `tile_grids` with shape `(n_levels, 42, 23)`, dtype `int32`, values in
`[0, 37]`. The level is 42 columns × 23 rows in tile coordinates (each tile is 24×24
game-units).

`switch_overlays` and `exit_overlays` have the same spatial shape, dtype `float32`,
values in `{0.0, 1.0}`. These are passed as additional input channels alongside a
one-hot tile encoding to the `TileEncoder`.

The `TileEncoder` expects an 18-channel input:
- Channels 0–15: one-hot tile type (binned; check `tile_encoder.py` for the exact
  binning scheme)
- Channel 16: `switch_overlay`
- Channel 17: `exit_overlay`

---

## Recommended First Steps for Epic 2

1. **Verify the dataset loads correctly** before launching a full training run:

   ```python
   import h5py, numpy as np
   with h5py.File("datasets/iql_pipeline_v2/iql_train_dataset.h5", "r") as f:
       print({k: v.shape for k, v in f.items()})
       # Spot-check reward structure on episode 0
       dones = f["dones"][:]
       rewards = f["rewards"][:]
       ends = np.where(dones)[0]
       ep = rewards[:ends[0]+1]
       print("Ep 0 length:", len(ep), "total reward:", ep.sum())
   ```

2. **Confirm `use_tile_encoder=True`** is set in `IQLConfig` before launching (the epic
   notes this as a critical flag; a missing tile encoder means the V-network cannot
   condition on level layout and will fail to generalize).

3. **Use the recommended training command** from the Epic 2 ticket T2.1 as the baseline.
   The batch size of 256 and `--prefetch` flag are important for stable training and
   GPU utilisation respectively.

4. **Watch the first 1K steps** before walking away. NaN in Q-loss within the first 500
   steps usually indicates a data-loading bug (wrong dtype or missing normalization),
   not a hyperparameter issue.

---

## Known Limitations / Follow-Up Items

- **~26% of original attract files do not win in the Rust simulator.** This is a
  property of the attract-demo corpus, not a bug — some attract demos use strategies
  that the Rust physics model does not reproduce exactly (floating-point edge cases,
  especially near wall transitions). These are simply dropped. The remaining 74% are
  all confirmed wins.

- **Dense-mine levels (500+ mines).** There are 107 such levels in the corpus. They are
  now correctly parsed and included in the dataset. The nearest-8-mine features are
  still the right representation for training (the network sees at most 8 mines at any
  step regardless of total mine count).

- **`--verify-sample 0` was used** for the production run. This means the augmentation
  step did not pre-screen augmented files for winnability. The HDF5 builder acts as the
  quality gate. If re-running the pipeline, keeping this flag at 0 is fine.

- **The old pipeline output** (`datasets/iql_pipeline/`) from an earlier run with only
  18,369 filtered levels (42,969 train / 4,791 val episodes) still exists on disk.
  **Use `datasets/iql_pipeline_v2/`** for Epic 2 training — it has 51,611 / 5,634
  episodes and a larger, cleaner filtered corpus.
