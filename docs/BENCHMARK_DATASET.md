# Mechanic-Primary Benchmark Dataset

This document describes the procedural, mechanic-bucket benchmark
dataset produced by
[`scripts/generate_benchmark_dataset.py`](../scripts/generate_benchmark_dataset.py)
and consumed by the npp-rl training / evaluation stack
(`npp_rl.benchmarks.*`, BSRS, learned value function).

## Goals

1. **Single-mechanic baselines.** Every generated level isolates one
   primary mechanic from `BucketType` so progression (T1 → T4) is an
   almost-pure difficulty signal, not a confound with other mechanics.
2. **Reproducible difficulty curve.** Tier semantics and seed math are
   deterministic so the same commit produces the same dataset bit-for-bit
   every run.
3. **Value-based PBRS ready.** Levels carry stable integer `level_id`s
   so BSRS's `V(s)` / `V(s')` caches and `AStarDistanceMapCache` can key
   into them without any external bookkeeping.

## On-disk layout

The generator emits a self-contained dataset root:

```
<dataset_root>/
├── manifest.json
├── train/
│   ├── train_metadata.json
│   ├── <BUCKET>/
│   │   └── T<tier>/
│   │       └── <level_id>.pkl
│   └── ...
└── test/
    ├── test_metadata.json
    └── <BUCKET>/T<tier>/<level_id>.pkl
```

### `manifest.json` (top-level)

Schema `mechanic_primary_v1`. Captures the full generation run:

| Key                           | Meaning                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `version`                     | Manifest schema version (`1`).                            |
| `schema`                      | `"mechanic_primary_v1"`.                                  |
| `source`                      | Script path that produced the dataset.                    |
| `buckets`                     | Buckets included (subset of `BUCKET_CATALOG`).            |
| `tiers`                       | Tiers included (subset of `BUCKET_TIERS`).                |
| `seed_bases`                  | `{"train": int, "test": int}` — see seed math below.      |
| `counts_per_bucket_per_tier`  | `{"train": N_train, "test": N_test}`.                     |
| `splits`                      | Per-split summary: total levels, per-bucket totals.       |
| `catalog`                     | Snapshot of `BUCKET_CATALOG` with dataclass configs.      |
| `generation_info`             | Dedup retries, skipped entries, unique-hash count.        |

### Per-split metadata (`train_metadata.json` / `test_metadata.json`)

```jsonc
{
  "split": "train",
  "total_levels": 100,
  "buckets": {
    "BASIC_JUMP": {
      "total": 25,
      "tiers": {
        "1": {"count": 7, "level_ids": [...], "path": "train/BASIC_JUMP/T1"},
        ...
      }
    },
    ...
  },
  "level_ids": [...]   // flat ordered schedule
}
```

### Level pickles (`<level_id>.pkl`)

Each pickle is a Python `dict`:

| Key         | Type                 | Notes                                        |
| ----------- | -------------------- | -------------------------------------------- |
| `level_id`  | `int` (63-bit)       | SHA-1 of `(split, bucket, tier, index)`.     |
| `seed`      | `int`                | Generator seed (see seed math below).        |
| `bucket`    | `str`                | e.g. `"BASIC_JUMP"`.                         |
| `tier`      | `int`                | 1–4.                                         |
| `generator` | `str`                | Registry key (`basic_jump`, `mine_path`, …). |
| `preset`    | `str`                | Preset name in `GENERATOR_PRESETS`.          |
| `map_data`  | `List[int \| float]` | Full N++ map array (tile + entity).          |
| `secondary` | `list`               | Reserved for future secondary-mechanic tags. |
| `metadata`  | `dict`               | `description`, `split`, `source`, `map_hash`, `index`. |

`map_data` follows the exact same convention as
`Map.map_data()` (header 0–183, tile grid 184–1149, entity section from
byte 1235 onward). Dedicated generators emit only integer coordinates;
legacy generators (`horizontal_corridor`, `single_chamber`) occasionally
emit sub-pixel floats, which the npp-rl runner rounds to the nearest
integer when materializing a temporary binary map file.

## Mechanic-primary taxonomy (`BUCKET_CATALOG`)

Defined in
[`nclone/map_generation/generator_configs.py`](../nclone/map_generation/generator_configs.py).
Phase-1 buckets (Layer 1 = toggle mines + exit switch + exit door):

| Bucket              | Generators                                   | Focus                                          |
| ------------------- | -------------------------------------------- | ---------------------------------------------- |
| `FLAT_RUN`          | `horizontal_corridor`, `single_chamber`      | Flat horizontal locomotion; no jumps/mines.    |
| `BASIC_JUMP`        | `basic_jump`                                 | Single-mechanic jump (gap / drop / step-up).   |
| `WALL_JUMP`         | `wall_jump`                                  | Vertical shafts; wall-jump only.               |
| `MINE_NAVIGATION`   | `mine_path`                                  | Flat corridor + toggle mines on the floor.     |

Out-of-scope buckets (`SLOPES`, `SWITCH_PUZZLE`, `LAUNCH_PAD`, `THWUMP`,
`DRONE`, `DEATH_BALL`) are present in the catalog but intentionally
empty until their generators come online. See
[`docs/MECHANIC_SUPPORT_MATRIX.md`](MECHANIC_SUPPORT_MATRIX.md).

### Tier semantics

Tiers are catalog-level *intent*, realized by per-preset parameter
envelopes in `GENERATOR_PRESETS`:

| Tier | Intent                                                                   |
| ---- | ------------------------------------------------------------------------ |
| T1   | Smallest valid chamber; minimum-difficulty instance of the mechanic.     |
| T2   | One additional axis of variation (e.g. alternate variant, extra size).   |
| T3   | Wider/taller, larger obstacle magnitudes, multiple sub-variants.         |
| T4   | Combined variations (e.g. double gap, shaft + ledge, tighter mine spacing). |

Each `BUCKET_CATALOG[bucket][tier]` is a list of
`(generator_type, preset, weight)` tuples. Weights drive deterministic
round-robin expansion inside a tier (`weight=1` = one slot per entry);
the index-based selector is reproducible across runs.

## Seeding model

Seeds are fully deterministic. Given a `(bucket, tier, index, split)`
quadruple:

```
bucket_idx  = position of bucket in BUCKET_CATALOG (0-based)
seed_base   = manifest.seed_bases[split]
seed        = seed_base
            + bucket_idx * BUCKET_STRIDE       (10_000_000)
            + tier        * TIER_STRIDE        (1_000_000)
            + index                             (0..N-1)
```

Collision-free offsets: with `BUCKET_STRIDE` = 10 M and `TIER_STRIDE` =
1 M, neighboring buckets/tiers never overlap for plausible dataset
sizes (N ≤ 1 M per bucket-tier-split).

**Dedup retry.** If the generated `map_data` hashes to an already-seen
level, the script retries with `seed + 1_000_000` up to `max_retry`
(default 10) times. Retries are counted in
`manifest.generation_info.duplicates_retried`; exhausted slots are
reported in `skipped`.

**`level_id`.** `SHA1(f"{split}:{bucket}:{tier}:{index}")[:8]` masked
to 63 bits. Stable across runs; split-disambiguated so train/test never
collide.

`BSRS` and `AStarDistanceMapCache` key into these `level_id`s directly.

## Default counts

The defaults below are intended for the full Phase-1 benchmark:

| Setting                                          | Default     |
| ------------------------------------------------ | ----------- |
| `--per_bucket_per_tier_train`                    | 25          |
| `--per_bucket_per_tier_test`                     | 10          |
| `--seed_base_train`                              | 1 000 000   |
| `--seed_base_test`                               | 100 000     |
| `--max_retry`                                    | 10          |

With 4 Phase-1 buckets × 4 tiers that yields 400 train / 160 test
unique levels, dominated by the minimal/dedicated generators.

## Regenerating the dataset

Full default run (400 train / 160 test):

```bash
python -m scripts.generate_benchmark_dataset \
    --output_dir ./datasets/mechanic_benchmark
```

Smoke run (small, fast — matches the smoke test fixture):

```bash
python -m scripts.generate_benchmark_dataset \
    --output_dir ./datasets/mechanic_benchmark_smoke \
    --per_bucket_per_tier_train 2 \
    --per_bucket_per_tier_test 1
```

Generate only a subset of buckets and tiers:

```bash
python -m scripts.generate_benchmark_dataset \
    --buckets BASIC_JUMP,WALL_JUMP \
    --tiers 1,2 \
    --mode train
```

The script validates `BUCKET_CATALOG` before producing any output;
if any entry references a missing preset or has a non-positive weight,
it exits with status 2 and prints the offending entries to stderr.

## Loading the dataset

### From nclone

The dataset format is intentionally kept close to the existing
`TestSuiteLoader` layout so light wrappers are enough. Each pickle
contains `map_data` directly, which any nclone environment can ingest via
`nplay_headless.load_map_from_map_data`.

### From npp-rl

```python
from npp_rl.benchmarks.benchmark_suite import BenchmarkSuite
from npp_rl.benchmarks.mechanic_buckets import procedural_levels, BucketType

# Full procedural suite
suite = BenchmarkSuite(
    procedural_dataset="./datasets/mechanic_benchmark",
    procedural_only=True,
)

# Filter by bucket/tier/split
mine_t2 = procedural_levels(
    "./datasets/mechanic_benchmark",
    bucket=BucketType.MINE_NAVIGATION,
    tier=2,
    splits=("train",),
)
```

`BenchmarkSuite(procedural_dataset=..., procedural_only=False)` appends
procedural levels to the hand-crafted registry instead of replacing it.
`npp_rl.benchmarks.run_benchmark` exposes matching
`--procedural_dataset` / `--procedural_only` flags for CLI use.

## Tests

The pipeline has two fast test suites:

- `tests/test_procedural_bucket_generators.py`: reproducibility and
  mechanic-purity checks for the three dedicated Phase-1 generators
  (`basic_jump`, `wall_jump`, `mine_path`).
- `tests/test_benchmark_dataset_script.py`: `BUCKET_CATALOG` validity,
  deterministic level-id / seed math, and an end-to-end smoke run of
  `BenchmarkGenerator` that round-trips the pickles on disk.

A cross-repo integration test lives in npp-rl at
`tests/test_procedural_benchmark_loader.py`; it regenerates a tiny
procedural dataset and verifies the `procedural_levels` / `BenchmarkSuite`
loaders consume it correctly.

## Notes & future work

- Validation is currently **hash-dedup only** (fast path). A future
  iteration may add optional A\* solvability / baseline-playout
  verification behind a flag.
- Only Phase-1 buckets have generators; the empty buckets in the
  catalog are intentional placeholders and will be populated alongside
  their Layer-2+ support code.
- Legacy `horizontal_corridor` / `single_chamber` presets for
  `FLAT_RUN` can emit non-mine secondary entities (locked doors, trap
  doors) that are outside the Layer-1 simulator's scope. If strict
  mechanic purity is required end-to-end, restrict to the dedicated
  `basic_jump` / `wall_jump` / `mine_path` generators until `FLAT_RUN`
  gets its own dedicated generator.
