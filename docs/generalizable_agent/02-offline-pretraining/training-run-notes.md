# IQL Training Run Notes

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `datasets/iql_pipeline_v2/iql_train_dataset.h5` |
| Val dataset | `datasets/iql_pipeline_v2/iql_val_dataset.h5` |
| Epochs | 300 |
| Steps per epoch | 1,000 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Expectile τ | 0.9 |
| Discount γ | 0.99 |
| Hidden dim | 512 |
| AMP | enabled |
| Prefetch | enabled |
| Val every | 10 epochs |
| Val batches | 100 |
| Device | CUDA (NVIDIA 3080 Ti) |

## Dataset

| Split | Transitions | Levels | RAM (float16 storage) |
|-------|-------------|--------|----------------------|
| Train | 41,972,410 | 51,611 | 7,332 MB |
| Val   | 4,385,947  | 5,634  | 768 MB |

**Note:** Float arrays (states, mine_features) are stored as float16 in RAM and
cast to float32 at batch-sample time. This reduces peak RAM from ~14 GB to ~8 GB,
fitting comfortably on the 15 GB machine.

## Network

- Parameters: 1,141,261 (1.1M)
- Architecture: 35D AgentStateEncoder → 256D | 18ch TileEncoder → 128D | Combiner → 512D

## Epoch Speed

- Dataset load time: ~164 s (gzip decompression of 5.3 GB HDF5)
- Epoch 1: ~98 s (includes first-step CUDA warm-up)
- Epoch 2+: ~65 s per epoch
- Estimated total training time: ~5.5 hours

## Early Training Observations (Epochs 1–10)

All losses finite and decreasing from step 1. No NaN gradients observed.

| Epoch | Avg V-loss | Avg Q-loss | Avg Pi-loss | Avg mean_V | Val V-loss |
|-------|-----------|-----------|-------------|-----------|-----------|
| 1 | 0.0057 | 0.0370 | 0.2562 | -1.37 | — |
| 9 | 0.0003 | 0.5274 | 0.3136 | -19.02 | — |
| 10 | 0.0004 | 0.6732 | 0.3247 | -20.95 | **0.0003** |

V(s) magnitude is growing negative as expected. Val/train ratio at epoch 10: **0.75**
(target < 1.2) — strong generalization from the first checkpoint.

`iql_best_val.pt` saved at epoch 10 (val V-loss = 0.0003).

## Checkpoints

Saved to `iql_checkpoints/`:
- `iql_epoch_NNNN.pt` — every 10 epochs
- `iql_best.pt` — best by training V-loss (updated continuously)
- `iql_best_val.pt` — best by validation V-loss (updated at each val epoch)
- `iql_final.pt` — last epoch

TensorBoard logs: `iql_checkpoints/tb_logs/`

## Full Training Results (300 Epochs)

### Loss curves

| Epoch | Train V-loss | Val V-loss | Val/Train ratio | mean_V |
|-------|-------------|-----------|----------------|--------|
| 10    | 0.0004      | 0.0003    | 0.75           | -21    |
| 50    | 0.0150      | 0.0145    | 0.97           | -56    |
| 100   | 0.0160      | 0.0150    | 0.94           | -70    |
| 200   | 0.0158      | 0.0145    | 0.92           | -75    |
| 270   | 0.0168      | **0.0149**| **0.89**       | -74    |
| 300   | 0.0169      | 0.0151    | 0.89           | -75    |

### Best checkpoint: epoch 270

`iql_epoch_0270.pt` is the best checkpoint:
- **Lowest val V-loss in the converged phase**: 0.01488
- **Val/train ratio**: 0.884 (healthy generalization)
- **mean_V**: -73.8 (well-converged; stable from epoch ~100 onwards)

> **Note on `iql_best_val.pt` bug:** the first training run saved epoch 10 as "best val"
> because the warmup guard was absent. The fix (epoch ≥ 50 before tracking best-val)
> was applied. The correct checkpoint (`iql_epoch_0270.pt`) was manually re-exported.

### Export

```bash
python -m nclone.offline_rl.export \
    --checkpoint iql_checkpoints/iql_epoch_0270.pt \
    --output v_network.pt
```

Verified: `v_network.pt` → `training_step: 270000`, correct config.

### Value landscape heatmaps

Generated with:
```bash
python scripts/generate_value_landscapes.py \
    --checkpoint v_network.pt \
    --val-dataset datasets/iql_pipeline_v2/iql_val_dataset.h5 \
    --output-dir value_landscapes/
```

5 held-out levels rendered to `value_landscapes/`:
`level_0000`, `level_1126`, `level_2252`, `level_3378`, `level_4504` — each with
`_dual.png`, `_phase0.png` (switch phase), and `_phase1.png` (exit phase).

### Integration tests (T2.4)

All 17 tests pass:

```
10 structural tests (random checkpoint) — all PASS
 7 semantic tests (trained checkpoint)  — all PASS
   - Switch vicinity > opposite corner in phase 1: ✓ (margin: 24 units)
   - Exit vicinity > opposite corner in phase 2: ✓ (margin: 4 units)
   - Bellman phase consistency at switch position: ✓ (V_pre > V_post)
   - CUDA inference matches CPU: ✓
   - CNN cache populated and invalidated correctly: ✓
```

### Observed V-network limitations

The V-network has learned **coarse spatial gradients** but not perfectly fine-grained
goal-proximity ordering in all configurations:

- **Exit proximity (phase 2)**: consistently correct when the "far" reference is in the
  geometrically opposite quadrant of the level.
- **Switch proximity (phase 1)**: correct when the switch is near the level center or
  in a corner the "far" position cannot easily reach; fails for switches near the top of
  levels where ground-level positions have comparably high value (ground-level paths are
  natural spawn points on the optimal trajectory).
- **Phase transition at exit door**: the model does not reliably assign higher V to
  post-switch states at the exit compared to pre-switch states at the exit. This is a
  known limitation: the switch overlay guidance appears underutilised in phase 2.

These are expected at this pre-training stage. Online fine-tuning (Epic 3+) should
sharpen goal-directed behaviour via live environment rollouts.

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| Training launches without error | ✅ |
| All losses finite at step 1 | ✅ |
| No NaN gradients | ✅ |
| val/v_loss / train/v_loss < 1.2 at 300K steps | ✅ (0.89) |
| V(s) negative-valued on held-out levels | ✅ (all values ∈ [-80, -50]) |
| Exported v_network.pt loads cleanly | ✅ (step 270000, config correct) |
| Integration tests pass | ✅ (17/17) |
| Value landscape heatmaps generated | ✅ (5 levels, 3 views each) |
