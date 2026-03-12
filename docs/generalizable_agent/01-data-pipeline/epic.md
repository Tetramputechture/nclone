# Epic 1: Data Pipeline

## Goal

Produce clean, validated HDF5 datasets (train and validation splits) from the raw attract demo corpus, filtered to levels containing only the target entity set: 1 exit door, 1 exit switch, up to 256 static toggle mines, and any tile layout.

The datasets feed directly into Epic 2 (IQL offline pre-training). Without a high-quality dataset the V-network cannot generalize.

## Background

The full attract demo corpus contains replays across many level configurations, including levels with dynamic enemies, multiple exits, launch pads, and other entities outside our scope. The `filter_attracts.py` script already implements entity-set filtering. The `transition_extractor.py` script extracts `(s, a, r, s′, done)` transitions and has already been updated to emit the correct two-phase sparse rewards (`+1` at switch collection, `+1` at exit, `−1` per step).

The IQL training pipeline expects an HDF5 file with datasets: `states`, `next_states`, `actions`, `rewards`, `dones`, `mine_features`, `next_mine_features`, and per-level `tile_grids`, `switch_overlays`, `exit_overlays`.

## Success Criteria

- Training HDF5 with ≥50K qualified episodes across ≥500 distinct level IDs
- Validation HDF5 with ≥5K episodes across ≥50 held-out level IDs (disjoint from training)
- Reward sequence sanity check passes: each successful episode has exactly one `+1` near the middle (switch) and one `+1` at the final frame (exit)
- No transition with `done=True` and `reward=-1.0` in the training split (all dones are wins)
- Dataset statistics documented for reproducibility

## Key Files

| File | Role |
|------|------|
| `nclone/offline_rl/dataset/filter_attracts.py` | Filters corpus by entity set; `ALLOWED_ENTITY_TYPES`, `check_file()` |
| `nclone/offline_rl/dataset/transition_extractor.py` | Extracts transitions; 11D state + 24D mine features |
| `nclone/offline_rl/dataset/build_dataset_cli.py` | Builds HDF5 from filtered replays; `build_dataset_parallel()` |
| `nclone/offline_rl/dataset/replay_dataset.py` | `OfflineRLDataset` HDF5 reader; `StreamingDatasetBuilder` |

---

## Tickets

### T1.1 — Run corpus filter and audit output

**What:** Execute `filter_attracts.py` against the full attract corpus to produce a filtered symlink directory of qualified replay files.

**Why:** The IQL training pipeline must only see levels within the target entity scope. Running the filter is a prerequisite for every downstream task.

**Acceptance criteria:**
- `filter_attracts.py` runs to completion without errors
- Output log reports the total number of files examined, accepted, and rejected (with rejection reasons)
- Accepted count ≥50K files
- Distinct level ID count ≥500 (use `--verbose` to log unique IDs or post-process the output directory)
- A 10% random sample of accepted files is manually spot-checked in the simulator to confirm entity set compliance

**How to run:**
```bash
python -m nclone.offline_rl.dataset.filter_attracts \
    --source /path/to/attracts \
    --output /path/to/attracts_filtered \
    --workers 16 \
    --dry-run   # count first, then remove flag to produce output
```

**Key implementation notes:**
- `ALLOWED_ENTITY_TYPES` must include `TOGGLE_MINE` (type 1), `TOGGLE_MINE_TOGGLED` (type 21), `EXIT_DOOR` (type 3), `EXIT_SWITCH` (type 4); verify gold (type 2) is also listed as gold does not affect our target scope
- The scanner reads 500 bytes at offset `ENTITY_SECTION_OFFSET = 1235`; verify this offset is correct against the map format spec
- If the corpus is organized in subdirectories (not flat), add recursive directory walking before running at full scale

**Relevant files:** `nclone/offline_rl/dataset/filter_attracts.py`

---

### T1.2 — Build training HDF5 dataset

**What:** Run `build_dataset_cli.py` on the filtered corpus (from T1.1) to produce `iql_train_dataset.h5`, reserving 10% of level IDs for validation.

**Why:** The IQL trainer reads from HDF5 for fast random-access batching. The dataset must encode the full transition format: 11D state, 24D mine features, per-level tile grids, and two-phase sparse rewards.

**Acceptance criteria:**
- `iql_train_dataset.h5` is produced without errors
- HDF5 metadata shows `n_levels ≥ 500` and `n_transitions ≥ 1M`
- `n_transitions / n_episodes` is in the range 300–5000 (reasonable episode lengths in frames)
- `tile_grids` dataset has shape `(n_levels, 42, 23)` with integer tile IDs in `[0, 37]`
- `switch_overlays` and `exit_overlays` have shape `(n_levels, 42, 23)` with float values in `{0.0, 1.0}`
- No `NaN` or `Inf` values in any numerical array

**How to run:**
```bash
python -m nclone.offline_rl.dataset.build_dataset_cli \
    --demo-dir /path/to/attracts_filtered \
    --output iql_train_dataset.h5 \
    --exclude-level-ids /path/to/val_level_ids.txt \
    --workers 8
```

**Key implementation notes:**
- The 10% validation split must be done by level ID, not by transition, to ensure true cross-level generalization measurement; save the held-out level ID list to `val_level_ids.txt` before running
- `StreamingDatasetBuilder` writes per-worker partial files and merges; verify the merge step handles duplicate level IDs across workers correctly
- The extractor only keeps episodes that end in a win (`won=True`); confirm that death episodes are silently dropped, not emitted with `done=True` and negative reward

**Relevant files:** `nclone/offline_rl/dataset/build_dataset_cli.py`, `nclone/offline_rl/dataset/transition_extractor.py`, `nclone/offline_rl/dataset/replay_dataset.py`

---

### T1.3 — Build validation HDF5 dataset

**What:** Run `build_dataset_cli.py` on the held-out level IDs (10% split from T1.2) to produce `iql_val_dataset.h5`.

**Why:** IQL training needs a held-out validation set to detect cross-level overfitting. The validation split must use disjoint level IDs from the training set so that validation loss measures true generalization, not memorization.

**Acceptance criteria:**
- `iql_val_dataset.h5` is produced using only the held-out level IDs from T1.2
- `n_levels ≥ 50` distinct levels in the validation HDF5
- `n_transitions ≥ 50K` in the validation set
- Zero level ID overlap between training and validation datasets (verify with a set intersection check)

**How to run:**
```bash
python -m nclone.offline_rl.dataset.build_dataset_cli \
    --demo-dir /path/to/attracts_filtered \
    --output iql_val_dataset.h5 \
    --include-level-ids /path/to/val_level_ids.txt \
    --workers 4
```

**Key implementation notes:**
- `build_dataset_cli.py` may need `--include-level-ids` / `--exclude-level-ids` flags added if they do not already exist
- Alternatively, split the filtered corpus directory into `attracts_train/` and `attracts_val/` subdirectories before running the CLI

**Relevant files:** `nclone/offline_rl/dataset/build_dataset_cli.py`, `nclone/offline_rl/dataset/replay_dataset.py`

---

### T1.4 — Validate dataset reward sequences

**What:** Write and run a validation script that loads both HDF5 files and checks the reward structure of a representative sample of episodes.

**Why:** The reward function was recently changed (`+1` at switch, `+1` at exit, `-1` per step). Incorrect rewards will silently cause IQL to learn the wrong value function. This must be verified before any training begins.

**Acceptance criteria:**
- For 100 randomly sampled episodes from the training set:
  - Each episode has exactly 1 frame with `reward = +1.0` and `done = False` (the switch event) or 0 (if switch already on at spawn — a known edge case)
  - Each episode has exactly 1 frame with `reward = +1.0` and `done = True` (the win)
  - All other frames have `reward = -1.0`
  - `R_total = -T + R_switch + R_exit` where T is episode length; verify T is in a reasonable range
- Validation set passes the same checks
- No episode with `done = True` and `reward = -1.0` exists (deaths were already filtered by the extractor)

**Validation script location:** `nclone/offline_rl/dataset/validate_dataset.py` (new file)

**Key implementation notes:**
- Use `h5py` to open the file and numpy to load episode boundaries from the `dones` array
- Edge case: if the exit switch is already collected at spawn (ninja spawns on top of it), the switch event occurs on frame 0 or 1; this produces two consecutive `+1` rewards and should be flagged but not fail the check

**Relevant files:** `nclone/offline_rl/dataset/replay_dataset.py`

---

### T1.5 — Document corpus statistics

**What:** Write `docs/generalizable_agent_v2/01-data-pipeline/corpus-stats.md` capturing key statistics about the datasets produced in T1.1–T1.3.

**Why:** Reproducibility. Future work (adding entity types, scaling data) needs a baseline to compare against. Statistics also reveal corpus biases (e.g., certain tile configurations being over-represented).

**Acceptance criteria:**
- Document includes:
  - Total replay files in raw corpus, accepted, rejected (with rejection breakdown by reason)
  - Number of distinct level IDs in training and validation sets
  - Total transitions and episodes in each split
  - Episode length distribution (min, median, max, 95th percentile) in frames
  - Mine count distribution (0, 1–10, 11–50, 51–100, 100+) across levels
  - Tile type frequency histogram across training levels
  - Number of levels where mine count = 0 (mine-free levels, useful as a curriculum starting point)
- Document references the commands used to produce the datasets for reproducibility

**Relevant files:** `nclone/offline_rl/dataset/replay_dataset.py`, `nclone/offline_rl/dataset/filter_attracts.py`
