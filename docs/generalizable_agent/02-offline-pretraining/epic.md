# Epic 2: Offline Pre-training (IQL)

## Goal

Train Implicit Q-Learning on the HDF5 datasets from Epic 1 to produce a V-network that generalizes across arbitrary tile layouts and mine configurations. Export the V-network as a PBRS potential for Epic 3 and as a policy prior for Epic 5 (MCTS).

## Background

The IQL infrastructure is fully built and was partially validated in the previous `add-sac` branch work (Phase 1 progress). The key fixes already applied:
- Reward function corrected: `+1` at switch, `+1` at exit, `−1` per step
- `batch_size` corrected from 1024 to 256 for better cross-level generalization signal
- Validation monitoring added to `train_cli.py` (`--val-dataset`, `--val-every`, `--val-batches`)
- `NeuralValuePBRS.get_potential()` fast path: CNN output cached per level, only MLP runs per step

The IQL architecture maps directly to the target entity set:
- 11D ninja state + 24D nearest-8-mine features → `AgentStateEncoder` (256D)
- 42×23 tile grid + switch overlay + exit overlay → `TileEncoder` (128D via global avg pool)
- Combined 384D → 512D combiner → V head / Q head / policy head

## Success Criteria

- IQL trains to convergence without NaN or instability
- Val V-loss tracks train V-loss with <20% gap at 300K steps (no cross-level overfitting)
- V(s) is negative-valued and magnitude correlates with estimated steps-to-completion on a held-out level
- Exported `v_network.pt` loads cleanly via `NeuralValuePBRS` and returns sensible potentials
- Value landscape visualization shows higher values (closer to 0) near the objective and lower values far from it

## Key Files

| File | Role |
|------|------|
| `nclone/offline_rl/algorithms/iql.py` | `IQLTrainer`, `IQLConfig` (τ=0.9, γ=0.99, lr=3e-4, batch_size=256) |
| `nclone/offline_rl/networks/value_networks.py` | `IQLNetworks`, `AgentStateEncoder` (35D→256D), combiner (384D→512D) |
| `nclone/offline_rl/networks/tile_encoder.py` | `TileEncoder` (18ch input, 3 conv layers, global avg pool → 128D) |
| `nclone/offline_rl/train_cli.py` | Training loop, TensorBoard, AMP, checkpoint, validation |
| `nclone/offline_rl/export.py` | `export_v_network()` → `v_network.pt` |
| `nclone/value_function/neural_value_pbrs.py` | `NeuralValuePBRS` loader and inference |
| `nclone/value_function/two_phase_pbrs.py` | `create_neural_pbrs()` factory for two-phase PBRS |
| `nclone/value_function/visualize_value_function.py` | Value landscape visualization |

---

## Tickets

### T2.1 — Configure and launch IQL training run

**What:** Prepare the training configuration, launch the IQL training job, and ensure it is stable for the first 10K steps.

**Why:** IQL training is the single most compute-intensive step in the pipeline. Verifying stability early avoids wasted GPU time.

**Acceptance criteria:**
- IQL training launches without error on the training HDF5 from Epic 1
- Loss values (Q-loss, V-loss, policy loss) are finite and decreasing over the first 1K steps
- No NaN gradients (monitor via TensorBoard `grad_norm` or `torch.nn.utils.clip_grad_norm_` output)
- AMP (mixed-precision) is enabled and the `GradScaler` does not skip steps excessively (>10% skip rate indicates numerical instability)
- Checkpoint saves correctly every 10 epochs

**Recommended training command:**
```bash
python -m nclone.offline_rl.train_cli \
    --dataset iql_train_dataset.h5 \
    --val-dataset iql_val_dataset.h5 \
    --output-dir iql_checkpoints/ \
    --epochs 300 \
    --steps-per-epoch 1000 \
    --batch-size 256 \
    --lr 3e-4 \
    --expectile-tau 0.9 \
    --discount 0.99 \
    --hidden-dim 512 \
    --device cuda \
    --amp \
    --prefetch \
    --save-every 10 \
    --val-every 10 \
    --val-batches 100
```

**Key implementation notes:**
- `use_tile_encoder = True` must be set — without the tile encoder the V-network cannot condition on level layout
- `IQLConfig.expectile_tau = 0.9` targets near-optimal behavior; if the demos are noisy, consider lowering to 0.7 and measuring the effect on downstream PBRS quality
- The `clip_advantage = 100.0` in `IQLConfig` clips `exp(β * A)` to prevent policy loss explosions on outlier transitions; this is safe to leave as-is

**Relevant files:** `nclone/offline_rl/algorithms/iql.py`, `nclone/offline_rl/train_cli.py`

---

### T2.2 — Monitor cross-level generalization during training

**What:** Track train vs. validation V-loss throughout the 300K-step training run and flag divergence.

**Why:** IQL can memorize level-specific patterns if the network capacity is too high or the training set is too small. The validation curve must track the training curve to confirm the V-network is learning generalizable features, not level-specific lookup tables.

**Acceptance criteria:**
- TensorBoard shows both `train/v_loss` and `val/v_loss` curves
- At training end (300K steps): `val/v_loss / train/v_loss < 1.2` (less than 20% gap)
- If divergence is detected (val loss rising while train loss falls): training is stopped early and the checkpoint at the best val loss is saved as the final checkpoint
- The best checkpoint epoch is recorded in `docs/generalizable_agent_v2/02-offline-pretraining/training-run-notes.md`

**Key implementation notes:**
- Validation runs every 10 epochs on 100 batches sampled from the validation HDF5; this adds ~5 seconds per validation pass and is negligible relative to training time
- Watch `val/mean_q`: if mean Q-value diverges to very large positive values, the Q-network is over-estimating; lower `beta` or increase `clip_advantage` to correct
- A secondary signal is the policy loss on val set: if the policy distribution concentrates on a single action across diverse states, it has memorized the mode rather than learning a generalizable prior

**Relevant files:** `nclone/offline_rl/train_cli.py`, `nclone/offline_rl/algorithms/iql.py`

---

### T2.3 — Export V-network checkpoint

**What:** Export the best IQL checkpoint (by val V-loss) as a standalone `v_network.pt` file loadable by `NeuralValuePBRS`.

**Why:** The training checkpoint contains optimizer state, target networks, and other training artifacts not needed for inference. `export.py` produces a clean, minimal file with only the components needed for `NeuralValuePBRS` and MCTS.

**Acceptance criteria:**
- `export_v_network()` produces `v_network.pt` without error
- The exported file contains keys: `v_network` (dict with `agent_encoder`, `combiner`, `v_head`, `tile_encoder`), `config` (IQLConfig), `training_step`
- File size is reasonable (< 100MB for a 512-hidden-dim network)
- The exported file can be loaded with `torch.load(..., weights_only=True)` without deprecation warnings

**How to run:**
```bash
python -m nclone.offline_rl.export \
    --checkpoint iql_checkpoints/best_checkpoint.pt \
    --output v_network.pt
```

**Key implementation notes:**
- `export.py` saves `state_dict` entries for `v_network.tile_encoder`, `v_network.agent_encoder`, `v_network.combiner`, `v_network.v_head` — verify the key names match what `NeuralValuePBRS.__init__()` expects when loading `data["v_network"]`
- The config is stored as a dataclass dict; `IQLConfig` fields must be JSON-serializable (no PyTorch tensors in config)

**Relevant files:** `nclone/offline_rl/export.py`, `nclone/value_function/neural_value_pbrs.py`

---

### T2.4 — Validate NeuralValuePBRS integration

**What:** Write and run an integration test that loads the exported `v_network.pt` via `NeuralValuePBRS`, calls `set_level()` with a real level, and verifies `get_potential()` returns sensible values.

**Why:** The export format and the loader must match exactly. A silent shape mismatch or key name error will produce all-zero or NaN potentials without raising an exception, breaking the reward shaping invisibly.

**Acceptance criteria:**
- `create_neural_pbrs(checkpoint_path, tile_grid, switch_pos, exit_pos, mine_positions)` succeeds
- `get_potential(state_near_switch)` returns a value closer to 0 than `get_potential(state_at_spawn)` (spawn is farther from completion)
- `get_potential(state_with_switch_collected=True)` > `get_potential(state_with_switch_collected=False)` for the same spatial position (switch collection should increase value estimate)
- CNN is not re-run on repeated `get_potential()` calls (verify via profiling or by checking that `_cached_level_repr` is set after `set_level()`)
- Calling `set_level()` with a different level produces a different cached level repr (confirm cache invalidation)

**Integration test location:** `nclone/value_function/tests/test_neural_pbrs_integration.py` (new file)

**Key implementation notes:**
- Use a real tile grid from the dataset for the test, not a zeros grid; a zeros grid (all empty tiles) is not a valid level and may produce misleading V-values
- Test both `device="cpu"` and `device="cuda"` if a GPU is available
- The two-phase factory `create_neural_pbrs()` wraps `NeuralValuePBRS` in `TwoPhaseValuePBRS`; test the two-phase wrapper directly since that is what the env will use

**Relevant files:** `nclone/value_function/neural_value_pbrs.py`, `nclone/value_function/two_phase_pbrs.py`

---

### T2.5 — Visualize value landscapes on sample levels

**What:** Use `visualize_value_function.py` to render V(s) as a spatial heatmap over the level grid for 5 representative held-out levels.

**Why:** Quantitative metrics (V-loss) can hide systematic failure modes. Visualizing the value landscape reveals whether the V-network has learned level geometry, whether there are discontinuities near mine boundaries, and whether the two-phase structure (switch before exit) is correctly encoded.

**Acceptance criteria:**
- 5 heatmap images are produced and saved to `docs/generalizable_agent_v2/02-offline-pretraining/value-landscapes/`
- Each image shows: (a) the tile layout as background, (b) V(s) as a color overlay at a grid of spatial positions, (c) mine positions marked, (d) switch and exit positions marked, and (e) separate visualizations for phase 0 (switch not collected) and phase 1 (switch collected)
- The phase-0 heatmap shows higher values near the switch; the phase-1 heatmap shows higher values near the exit door
- At least one of the 5 levels is a mine-heavy level (>50 mines) to verify that mine proximity reduces value as expected

**Key implementation notes:**
- Sample states are constructed by setting `switch_collected=0` or `1`, sweeping `(x, y)` positions across the level grid at a 12px resolution, and using `vx=vy=0`, `on_ground=1`
- `NeuralValuePBRS.get_potential()` is called for each grid position; the result is reshaped to the grid and plotted with `matplotlib.pyplot.imshow`
- Tiles that are solid (type 1) should be masked out of the visualization

**Relevant files:** `nclone/value_function/visualize_value_function.py`, `nclone/value_function/neural_value_pbrs.py`
