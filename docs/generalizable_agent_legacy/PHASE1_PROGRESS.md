# Phase 1 Progress Report — Offline Value Function Pre-training

**Date:** 2026-03-08  
**Branch:** `add-sac`  
**Status:** All code changes complete. Dataset runs and IQL training remain to be executed manually.

---

## Summary

Phase 1 tasks 1.1–1.4 have been audited and all identified gaps have been fixed in code. The infrastructure was largely in place; four concrete issues were resolved. Task 1.5 (QRL) was intentionally skipped per the plan.

---

## Task 1.1 — Filter and Audit Attract Demo Corpus

**Status: Complete (no code changes needed)**

`nclone/offline_rl/dataset/filter_attracts.py` is fully implemented and correct:

- `ALLOWED_ENTITY_TYPES` correctly includes `TOGGLE_MINE`, `EXIT_DOOR`, `EXIT_SWITCH`, `TOGGLE_MINE_TOGGLED`, `GOLD`.
- `check_file()` enforces exactly 1 exit door + 1 exit switch + no disallowed entities.
- Parallel entity-byte scanning via `multiprocessing.Pool` using a 500-byte read at offset `ENTITY_SECTION_OFFSET = 1235`, skipping full-file parsing.
- CLI supports `--source`, `--output`, `--workers`, `--dry-run`, `--verbose`.

**To run (manual):**
```bash
python -m nclone.offline_rl.dataset.filter_attracts \
    --source /path/to/attracts \
    --output /path/to/attracts_filtered \
    --workers 16 \
    --dry-run   # verify counts first
```

**Verification criteria:** The output log should show ≥50K files passed and ≥500 distinct level IDs. If counts are lower, the corpus needs expanding before proceeding to Task 1.2.

**Note on subdirectory support:** Both `filter_attracts.py` and `build_dataset_cli.py` scan a single flat directory via `iterdir()`. If the attract corpus is organized in subdirectories by level group, a recursive walk will need to be added. For now this is assumed to be a flat directory.

---

## Task 1.2 — Build Cross-Level HDF5 Dataset

**Status: Infrastructure existed. Reward function fixed.**

### Gap found and fixed: Reward assignment

The original `transition_extractor.py` used a simplified reward scheme:
```python
# OLD (incorrect)
rewards[frame_idx] = 0.0 if won else -1.0
```

This missed two things:
1. No `+1` reward when the exit switch is collected (intermediate signal the V-network needs to learn the two-phase structure).
2. Terminal win reward was `0.0` instead of `+1.0`.

**Fix applied** in `nclone/offline_rl/dataset/transition_extractor.py`:
```python
# NEW: sparse two-stage rewards
switch_just_collected = cur_switch_collected and not prev_switch_collected
if won:
    reward = 1.0
elif switch_just_collected:
    reward = 1.0
else:
    reward = -1.0
```

The switch transition is detected by comparing `_is_switch_collected()` between the previous and current frames, emitting `+1` exactly once at the collection frame. This creates a reward structure of:

```
R_total = -T + R_switch + R_exit
        = -T + 1 + 1    (for a successful episode of T frames)
```

Where `V(s) ≈ -(time_to_completion) + (expected_bonuses)` — a valid negative potential that correctly reflects level difficulty.

### Two-goal conditioning

The state vector `state[7] = switch_collected` (0.0 or 1.0) already provides phase conditioning to the IQL network. Because the tile encoder sees both `switch_overlay` and `exit_overlay` as input channels, and the state vector includes `switch_collected`, a single V-network can learn phase-appropriate values without explicit two-goal relabeling (i.e. the network implicitly learns "I'm in switch-phase" vs. "I'm in exit-phase" from state[7]).

True HER-style two-goal relabeling (truncate each episode at the switch event, emit two training trajectories with goal-masked overlays) would further improve data efficiency but is left as a future enhancement. It requires modifying `StreamingDatasetBuilder` to store a `goal_phase` field per transition and masking switch/exit overlays accordingly.

**To run (manual):**
```bash
python -m nclone.offline_rl.dataset.build_dataset_cli \
    --demo-dir /path/to/attracts_filtered \
    --output iql_dataset.h5 \
    --workers 8
```

**Verification criteria:**
- `n_levels ≥ 500` in the HDF5 metadata.
- `n_transitions / n_episodes` should be roughly 500–3000 (typical level length in frames).
- Spot-check: load 10 random episodes and verify reward sequence contains exactly one `+1` near the middle (switch event) and one `+1` at the final frame.

---

## Task 1.3 — Verify and Extend IQL for Multi-Level Goal Conditioning

**Status: Config mismatch fixed. Validation monitoring added.**

### Gap 1: Batch size

`IQLConfig.batch_size` defaulted to `1024`. The research spec calls for `256` to improve cross-level generalization signal — smaller batches sample from more diverse level IDs per step relative to the dataset size.

**Fix applied** in `nclone/offline_rl/algorithms/iql.py`:
```python
# OLD
batch_size: int = 1024

# NEW
batch_size: int = 256
```

All other config values already matched the research spec:
- `expectile_tau = 0.9` ✓ (τ=0.9 for near-optimal expert data)
- `learning_rate = 3e-4` ✓
- `discount = 0.99` ✓ (matches online PPO γ)
- `use_tile_encoder = True` default ✓ (required for cross-level transfer)

### Gap 2: No held-out validation

`train_cli.py` only tracked training loss. The research spec says "300K gradient steps with held-out demo validation loss monitoring." There was no mechanism to detect overfitting to the training levels.

**Fix applied** in `nclone/offline_rl/train_cli.py`:
- Added `--val-dataset` argument (path to a separate held-out HDF5 file).
- Added `--val-every N` (default: every 10 epochs) and `--val-batches N` (default: 100 batches).
- Validation runs the expectile V-loss on the held-out dataset in `torch.no_grad()` mode.
- Validation metrics are logged to TensorBoard under `val/v_loss` and `val/mean_q`.

**To run (manual) — recommended training command:**
```bash
python -m nclone.offline_rl.train_cli \
    --dataset iql_dataset.h5 \
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

This runs 300K gradient steps (300 epochs × 1000 steps). The validation curve should track the training curve; divergence indicates cross-level overfitting.

**Note on validation split:** The `--val-dataset` expects a separately built HDF5 file (e.g., filter 10% of levels out of the filtered attract corpus and run `build_dataset_cli` on them separately). A `--train-val-split` ratio for automatic splitting within `OfflineRLDataset` was not added because splitting by *levels* (not by transitions) requires knowing which transitions belong to which level — that information is available from the `level_ids` array and can be done as a pre-processing step.

---

## Task 1.4 — Export V-network for PBRS

**Status: CNN cache bypass bug fixed. Export format verified.**

### Gap found and fixed: CNN cache not used

`NeuralValuePBRS.set_level()` correctly precomputed `self._cached_level_repr` (the `(1, 128)` CNN output) once per level. However, `get_potential()` was not using it. It instead rebuilt the tile-grid batch dict and called `self._networks.value(batch)`, which calls `IQLNetworks._encode()`, which re-ran the CNN on the full `42×23` tile grid every single step:

```python
# OLD: cache computed but ignored, CNN re-ran every step
batch["tile_grid"] = self._cached_tile_grid       # passed to network
batch["switch_overlay"] = self._cached_switch_overlay
batch["exit_overlay"] = self._cached_exit_overlay
v_value = self._networks.value(batch)  # → runs TileEncoder again inside
```

**Fix applied** in `nclone/value_function/neural_value_pbrs.py`. When the cache is available, the fast path calls only the agent-state MLP + combiner + V-head, bypassing the CNN entirely:

```python
# NEW: fast path — CNN skipped, only MLP + combiner + V-head
agent_repr = self._networks.agent_encoder(state_tensor, mine_tensor)
combined = torch.cat([agent_repr, self._cached_level_repr], dim=-1)
h = self._networks.combiner(combined)
v_value = self._networks.v_head(h)
```

The CNN (`TileEncoder`) runs 3 conv layers over a `42×23` grid. On CPU this is ~0.5–2ms per call. With 60fps simulation and 4 parallel environments, removing this from the per-step path eliminates ~120–480 redundant CNN calls per second.

The slow path (no cache, or `use_tile_encoder=False`) remains for fallback.

### Export format verification

The format chain is correct end-to-end:

| Step | File | Key | Shape |
|---|---|---|---|
| Export | `export.py → export_v_network()` | `v_network.tile_encoder` | state dict |
| Load | `NeuralValuePBRS.__init__()` | `data["v_network"]["tile_encoder"]` | state dict |
| Cache | `set_level()` → `tile_encoder(grid, sw, ex)` | `_cached_level_repr` | `(1, 128)` |
| Inference | `get_potential()` fast path | `_cached_level_repr` | used directly |

No format mismatch found.

**Integration test to run (manual):**
```python
from nclone.value_function.two_phase_pbrs import create_neural_pbrs
import numpy as np

pbrs = create_neural_pbrs(
    checkpoint_path="iql_checkpoints/v_network.pt",
    tile_grid=np.zeros((42, 23), dtype=np.int64),
    switch_pos=(300.0, 200.0),
    exit_pos=(600.0, 400.0),
    mine_positions=[],
    device="cpu",
)

state = {"player_x": 100.0, "player_y": 100.0, "player_xspeed": 0.0,
         "player_yspeed": 0.0, "switch_activated": False}
phi = pbrs.get_potential(state)
print(f"Phi(s) = {phi:.4f}")  # Expected: negative float, roughly in [-100, 0]
```

---

## Remaining Work (Requires Dataset Results)

The following are gated on manual dataset runs:

| Step | Action | Success Criterion |
|---|---|---|
| 1.1 run | `filter_attracts` on corpus | ≥50K files, ≥500 levels |
| 1.2 run | `build_dataset_cli` on filtered corpus | ≥500 levels in HDF5; reward sequence sanity-check |
| 1.2 val | Build 10% held-out val dataset | Separate `iql_val_dataset.h5` |
| 1.3 run | IQL training (300 epochs, batch=256, AMP) | Val V-loss tracks train loss; no divergence |
| 1.4 run | Export best checkpoint | `v_network.pt` loads cleanly; `get_potential()` returns reasonable values |

---

## File Change Summary

| File | Change |
|---|---|
| `nclone/offline_rl/dataset/transition_extractor.py` | Reward: `0@win,-1/step` → `+1@switch,+1@exit,-1/step`; cleaned unused `Dict`/`Tuple` imports |
| `nclone/offline_rl/algorithms/iql.py` | `batch_size` default: `1024` → `256` |
| `nclone/offline_rl/train_cli.py` | Added `--val-dataset`, `--val-every`, `--val-batches`; validation loop with TensorBoard logging |
| `nclone/value_function/neural_value_pbrs.py` | `get_potential()` fast path: use `_cached_level_repr` directly, skip CNN forward |
