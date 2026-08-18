# `gamma_benchmark_phase2` — the packed training corpus

**This artifact is NOT regenerable from anything in git.** It is preserved as
bytes, in `gamma_benchmark_phase2.tar.gz`, because that is the only way it can be
preserved. Read the provenance section before assuming otherwise.

It is the training corpus for the gamma-5 / round-2 run: the level set every
recorded number was measured against. Losing it does not cost a rebuild, it costs
the comparability of every result on record.

## Restore

```sh
tar -xzf datasets/gamma_benchmark_phase2.tar.gz -C datasets/
```

Extracts to `datasets/gamma_benchmark_phase2/`, which is what `--benchmark-root`
and `launch_gamma5.sh`'s `BENCH` default point at. The extracted tree is gitignored
(`datasets/*`); only the tarball and this file are tracked.

## Contents — verify a restore against these

| | |
|---|---|
| tarball sha256 | `90cd60d132df0c5336d6b75937a3c4750bb24fc08988b42eb99dabfce19a58ff` |
| tarball size | 6,609,750 B (6.6 MB) |
| extracted size | 90 MB |
| files | 21,278 (21,271 `.pkl`, 4 `.json`, 3 `.bak`) |
| train / test levels (actual `.pkl` members) | **16,937 / 4,334** |
| buckets (actual loader view) | 28 train / 28 test |
| tiers | 4 |
| merged bucket | `HAZARD_FIELD` |
| official corpus | `phase1` |

The tarball is built deterministically (`--sort=name --owner=0 --group=0
--numeric-owner --mtime='UTC 2026-08-17'`, `gzip -9n`), so a rebuild from an
identical tree reproduces the same bytes. Round-trip was verified by sha256 over
all 21,278 files, not by spot check.

The embedded `manifest.json` still reports `split_totals_after_merge` as
16,206/3,902 and lists 19 buckets. Those counters predate the later hazard,
ATTRACT, phase-2 OFFICIAL and OFFICIAL_DOORS additions; they are stale metadata,
not the archive's contents. Measured directly from the tar on 2026-08-17:
21,271 `.pkl` members = 16,937 train + 4,334 test, byte-identical to the
extracted tree. Consumers scan the split directories and therefore see the
actual counts above.

## Provenance — and why it is not enough

`manifest.json` records exactly this much:

```json
"source": "scripts.levelgen.pack_benchmark",
"generation_info": {"generator": "corpus", "note": "execution-curriculum experiment"}
```

No seed. No source-corpus SHA. No command line. No generator version. The packing
script is named but its **inputs are not**, and level generation is seeded, so
re-running `pack_benchmark` produces *a* corpus, not *this* corpus. The train/test
split is the load-bearing part: the frozen split is what every contamination audit
(`audit_demo_test_contamination.py`, 0/308) and every held-out eval number is
defined against. A regenerated corpus would silently re-cut it.

**If you build a replacement corpus, it is a new corpus.** Give it a new name,
record the seed and inputs this one failed to, and re-run the contamination audits
— do not overwrite this one and do not compare across the two.

## Why this file exists

The previous GPU instance was recycled and took every checkpoint, deploy SHA and
derived corpus with it. Afterwards this corpus survived in exactly one place: one
workstation disk, under a blanket `datasets/` ignore rule, in no git ref. That is
the same failure, staged and waiting. `npp-rl/docs/REMOTE_INSTANCE_READINESS.md`
§0 sets the bar as "the box can be rebuilt from git in one scripted pass and
destroyed without losing anything" — this artifact was the one thing that made
that claim false.

Sibling artifacts, for contrast, are genuinely regenerable and are deliberately
NOT stored here: `demos_official_v2` and `v30_curricula` rebuild on CPU in ~40
minutes (readiness §2.1), and the 15 GB global-potential cache rebuilds in ~10
minutes (readiness §5).
