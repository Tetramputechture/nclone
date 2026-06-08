# Remote level-generator training (GH200 / arm64)

`scripts/levelgen/remote_train.sh` drives the synthetic level-generator pipeline on
a remote GH200 (arm64) box, mirroring the npp-rl remote PPO convention: tarball
code sync, detached `screen` jobs, and a Lambda-Stack `--system-site-packages`
venv on aarch64 (system PyTorch + pip `h5py`/`numpy<2`).

It covers three things the request asked for:

| Stage | What it does | Session |
|---|---|---|
| `data` | builds `tiles_full.h5` + `tiles_subset.h5` | `levelgen_data` |
| `train` | tile pretrain → tile fine-tune → entity | `levelgen_train` |
| `pipeline` | `data` then `train`, one detached job | `levelgen_pipeline` |

## Quick start

```bash
export INSTANCE_IP=1.2.3.4          # or pass --ip 1.2.3.4 to any command
export SSH_KEY=~/.ssh/npp-rl.pem    # default

./scripts/levelgen/remote_train.sh setup       # sync code + install deps (incl tensorboard)
./scripts/levelgen/remote_train.sh pipeline     # data + train, end to end
FOLLOW=1 ./scripts/levelgen/remote_train.sh logs pipeline
./scripts/levelgen/remote_train.sh tensorboard  # live training curves (see below)
./scripts/levelgen/remote_train.sh status
./scripts/levelgen/remote_train.sh download     # pull ckpts + run configs + tb/ back to ./ckpt/
```

Run the stages separately if you prefer: `setup` → `data` → `train`.

## Observability (TensorBoard)

All three training runs write per-step scalars to remote `~/experiments/tb/<run>/`
(`tile_pretrain`, `tile_finetune`, `entity`):

- `train/loss`, `train/lr`, `train/grad_norm`, `train/samples_per_s` (every `--log-every`
  steps, default 20)
- `epoch/train_loss` and `epoch/val_loss` — each stage holds out a deterministic 5%
  validation split (`--val-frac`, `--split-seed`); the best-by-val weights are saved
  alongside the final checkpoint as `<name>.best.pt`.

Each checkpoint gets a `<name>.run_config.json` sidecar capturing the full hparams,
the git SHA, and the model config for reproducibility.

**View the curves live** — launch the server on the remote and SSH-tunnel to it:

```bash
./scripts/levelgen/remote_train.sh tensorboard          # screen session 'levelgen_tb', port 6006
ssh -i ~/.ssh/npp-rl.pem -N -L 6006:localhost:6006 ubuntu@$INSTANCE_IP
# open http://localhost:6006
```

Or pull the event files back and view locally: `download` copies `tb/` into
`./ckpt/tb/`, then `tensorboard --logdir ./ckpt/tb`.

## Data source

By default `data`/`pipeline` **rsync the already-decoded corpus shards**
(`datasets/levelgen/corpus_full` + `corpus_subset`, ~1.4 GB) and run only
`build_dataset` on the remote (~3 min) — the "salvage" path from
`docs/LEVELGEN_REMAINING_WORK.md` (avoids re-decoding 250k attracts on Grace cores).

To re-decode from scratch instead, pass `--from-attracts` (syncs `../attracts`
and runs `filter_attracts → ingest → label → build` for both subset and full).
`LABEL_FULL=1` additionally labels the full corpus (otherwise difficulty is
`unknown` on full — fine for the masked tile pretrain).

## What `train` runs (doc-target 37.9M tile model)

```
train_tile  --data tiles_full.h5     --mode maskgit --d-model 512 --n-layers 12 --epochs 30 --amp <imb>  -> tile_pretrain.pt
train_tile  --data tiles_finetune.h5 --init-from tile_pretrain.pt              --epochs 20 --amp <imb>  -> tile.pt
train_entity --data tiles_subset.h5  --tile-ckpt tile.pt                       --epochs 20 --amp        -> entity.pt
```

`<imb>` = the token-imbalance flags `--focal-gamma 2.0 --empty-weight 0.5 --mask-nonempty-bias 1.5`
(env `FOCAL_GAMMA`/`EMPTY_WEIGHT`/`MASK_NONEMPTY_BIAS`). The corpus is ~84% empty tile 0;
without these the tile stage under-predicts structure and generations come out sparse and
unsolvable (Stage-B fix — see below).

**Dataset note (Stage B).** The two tile stages now also train on `maps/official/`
geometry: the `data` stage runs `ingest_corpus --official ... --official-all-geometry`
(3,518 of 3,568 maps — the native-safe entity filter is bypassed since the tile stage only
needs geometry) and folds it into `tiles_full.h5` (pretrain) and a new `tiles_finetune.h5`
(= subset + official, mirrored). The **entity** stage still trains on the pure, subset-valid
`tiles_subset.h5` so its constrained decode stays correct. Disable with `OFFICIAL=0`.

The entity stage trains its tile encoder end-to-end (no `--freeze-encoder`): only the
encoder *embeddings* can warm-start from `tile.pt`, and only when `ENTITY_DMODEL`
matches `TILE_DMODEL` — its transformer blocks are always trained from scratch, so
freezing would pin random weights.

Checkpoints land in remote `~/levelgen_ckpt/` (final `*.pt`, best-by-val `*.best.pt`, and
`*.run_config.json` sidecars); `download` brings them all back along with the `tb/` events.

## Overrides (env vars)

`TILE_DMODEL` `TILE_LAYERS` `TILE_HEADS` `TILE_MODE` (maskgit|gpt) `PRETRAIN_EPOCHS`
`PRETRAIN_BATCH` `FINETUNE_EPOCHS` `FINETUNE_BATCH` `ENTITY_DMODEL` `ENTITY_LAYERS`
`ENTITY_EPOCHS` `ENTITY_BATCH` `LR` `AMP=0` (disable bf16) `WORKERS` (attract decode)
`DL_WORKERS` (torch DataLoader workers, default 8).

Stage-B knobs: `OFFICIAL=0` (skip official-maps geometry) · `FOCAL_GAMMA` (default 2.0) ·
`EMPTY_WEIGHT` (default 0.5, CE weight on the empty tile) · `MASK_NONEMPTY_BIAS` (default 1.5).

Efficiency/quality knobs: `COMPILE=1` (torch.compile forward) · `LR_SCHEDULE=cosine` (warmup+cosine) ·
`WARMUP_FRAC` (0.03) · `EMA_DECAY` (0.999; sampling uses EMA weights). Batch defaults bumped to
512 + LR 4e-4 for the GH200. TF32 is auto-enabled. (Flash attention: `nn.MultiheadAttention` already
hits the fused SDPA kernel via `need_weights=False`, so no manual rewrite — preserves ckpt compat.)

After the run, measure objectively with `eval_generator.py` (now also reports **discriminator AUC**
— ~0.5 = indistinguishable from human — and **novelty / memorized_frac** vs the corpus), e.g.
`python -m scripts.levelgen.eval_generator --human h5:.../tiles_finetune.h5 --candidate h5:<generated> --discriminator`.
Conditioning now carries real signal: the mechanic tagger is salience-based (was ~96% `composite`,
now ~3%); validate difficulty with `label_corpus --dstar-sample N`.

## Generation (after `download`) — Stage-A findings

The two free generate-time fixes that took solvability 0% → 62% with no retraining are now
the defaults in `generate_levels.py`:
- **Spawn-on-surface** (`place_and_gate`): the ninja is placed on a reachable floor cell
  instead of the old hard-coded mid-air `(86,42)`. Disable with `--no-place-spawn`.
- **`--cfg-scale` default 2.0** (was 1.5): the model's *unconditional* prior collapsed to
  near-empty, so **higher** CFG raises fill+diversity here (the opposite of the usual
  text-to-image guidance/diversity tradeoff). cfg≈2 is the sweet spot; cfg≥3 over-fills and
  hurts solvability. Re-sweep cfg after the Stage-B retrain (the density loss should let a
  lower cfg suffice). Validate with `scripts/levelgen/visualize_generations.py --place-and-gate`.

SSH: `SSH_KEY` `SSH_USER` (default `ubuntu`). Remote layout: code at
`~/nclone-levelgen/nclone`, datasets at `~/datasets/levelgen`, logs at
`~/experiments/<stage>-<ts>.log`.

## Other commands

`status` (screen + GPU + remote files + TB server state) · `tensorboard` (launch TB on
`TB_PORT`, default 6006) · `logs [data|train|pipeline]` (`FOLLOW=1` to `-f`) ·
`stop [session]` (default all, incl `levelgen_tb`) · `download` (`WITH_DATA=1` also pulls
`*.h5`) · `shell` (interactive ssh).
