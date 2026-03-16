---
name: Pivot from Neural to Per-Level PBRS
overview: The neural IQL V-network produces gradients too coarse for PBRS (only ~7-15 V-units of dynamic range across a level, unreliable phase transitions). Rather than pursuing expensive and uncertain fixes (simulator rollout augmentation, architecture changes), we should leverage the already-implemented SurfaceGraphPBRS -- which gives frame-exact, velocity-aware, physics-based cost-to-go per level -- and proceed to multi-level online training.
todos:
  - id: batch-precompute
    content: Write a batch precomputation script that processes training levels through precompute_cli.py, adapting it to accept attract file input (decode attract -> map_data -> precompute_surface_pbrs)
    status: pending
  - id: simplify-epic3
    content: "Update Epic 3 document: mark T3.1/T3.2 as SKIP, adapt T3.3-T3.5 for SurfaceGraphPBRS instead of NeuralValuePBRS"
    status: pending
  - id: verify-multilevel-pbrs
    content: "Integration test: load 10 training levels with precomputed SurfaceGraphPBRS, run 500 steps each, verify rewards are finite and phase transitions work"
    status: pending
  - id: update-master-plan
    content: Update 00-master-plan.md to reflect the pivot from neural to per-level precomputed PBRS, documenting the IQL V-network finding
    status: pending
isProject: false
---

# Pivot from Neural V(s) to Per-Level Precomputed PBRS

## Diagnosis: Why the Neural V-Network Failed for PBRS

Looking at the value landscapes, the problems are clear:

- **Narrow dynamic range**: V(s) spans only ~7-15 units within a level (e.g., -77 to -70 for level 3378). For PBRS where the shaped reward is `gamma * Phi(s') - Phi(s)`, this yields shaping signals of ~0.01 per step -- too weak to guide learning.
- **Phase transition failure**: The model does not reliably assign higher V to post-switch states at the exit vs. pre-switch states. The switch overlay channel is underutilized.
- **Coarse spatial gradients**: The V-network learns vaguely correct quadrant-level preferences but not the fine-grained goal-proximity ordering PBRS demands.

These are **architectural / signal problems**, not data quantity problems. We have 46M transitions across 57K episodes -- adding more data (via simulator rollouts) won't fix the narrow dynamic range or phase discrimination. The IQL expectile regression with gamma=0.99 and a per-step reward of -1 compresses all values into a narrow band where spatial differences are lost.

## The Existing Solution: SurfaceGraphPBRS

The codebase already has a **superior per-level PBRS system** that is fully implemented and integrated:

- [nclone/value_function/precompute_cli.py](nclone/value_function/precompute_cli.py) -- batch precomputation pipeline
- [nclone/value_function/precomputed_pbrs.py](nclone/value_function/precomputed_pbrs.py) -- `SurfaceGraphPBRS` runtime lookups
- [nclone/value_function/two_phase_pbrs.py](nclone/value_function/two_phase_pbrs.py) -- two-phase wrapper

This gives:

- **Exact cost-to-go**: Dijkstra on velocity-augmented graph = frame-accurate shortest paths
- **Velocity-aware**: 15 velocity bins on floors, 12 on walls (momentum routing)
- **Two-phase**: Separate d* arrays for switch and exit goals
- **O(1) runtime**: ~5 microseconds per lookup (vs. ~0.15ms for path-based)
- **Airborne handling**: Arc cache for O(1) lookup; pessimistic fallback otherwise

The precomputation cost per level is 1-5 minutes on 8 cores, producing ~5 MB `.npz` files.

## Recommended Path Forward

### Step 1: Batch-precompute SurfaceGraphPBRS for all training levels

Write a batch script that iterates over the ~500+ training levels (from the filtered attract corpus) and runs `precompute_surface_pbrs()` for each.

**Input:** Attract files in `/home/tetra/projects/attracts_filtered/` (22,596 files, ~20K train + ~2K val)
**Output:** `value_functions/{level_id}_surface_pbrs.npz` for each level
**Estimated time:** ~500 levels x 3 min = 25 hours serial; ~3-4 hours with 8 parallel workers
**Estimated storage:** ~500 x 5 MB = 2.5 GB

The script needs to:

1. Decode each attract file using `NppAttractDecoder` to get `map_data`
2. Write temporary `.map` files (or adapt `precompute_surface_pbrs()` to accept `map_data` directly)
3. Call `precompute_surface_pbrs()` with parallel workers
4. Track failures and skip levels where precomputation fails (degenerate geometry)

We should focus precomputation on the training-level subset (not all 22K levels, just the ~500 we plan to train multi-level PPO on).

### Step 2: Simplify Epic 3 (Environment Integration)

Epic 3 was designed around wiring up `NeuralValuePBRS`. With per-level precomputed PBRS, most of Epic 3 is already done:

- **T3.1 (neural_pbrs_checkpoint config) -- SKIP**: Not needed; use existing `value_function_dir` + `use_precomputed_pbrs=True`
- **T3.2 (wire NeuralValuePBRS) -- SKIP**: `_try_load_value_function_pbrs()` already loads `SurfaceGraphPBRS` from `{level_id}_surface_pbrs.npz`
- **T3.3 (terminal potential handling) -- KEEP**: Verify Phi(terminal) = 0 with surface graph PBRS
- **T3.4 (shaping weight annealing) -- KEEP**: Alpha annealing still needed; `PBRSCalculator` already has `iql_alpha` / `iql_alpha_anneal_steps` wiring
- **T3.5 (integration test) -- ADAPT**: Test with precomputed surface graph PBRS across 10 random levels

### Step 3: Proceed to Epic 4 (Multi-Level FiGAR-PPO)

With per-level precomputed PBRS ready, multi-level training can start. The environment loads the right `.npz` per level on `env.reset()`.

### What About New / Held-Out Levels?

- **During training:** Only train on levels with precomputed PBRS (the ~500 training levels)
- **At test time (Epic 5 MCTS):** The Rust simulator IS the world model -- MCTS does not need precomputed PBRS. For held-out levels, the 3-min precomputation is negligible vs. the 10-min MCTS budget
- **Fallback:** The path-based PBRS in `PBRSCalculator` already works on any level without precomputation (just slower at ~0.15ms/step)

### What About the IQL Policy Prior for MCTS?

The IQL policy network (6-action softmax) from the existing training may still be useful as an MCTS prior even though V(s) is bad for PBRS. The policy learned coarse action preferences from 46M expert transitions. Alternatively, train a simple BC policy as the MCTS prior -- this is pure supervised learning on the same data and should be straightforward.

## Why NOT Pursue Simulator-Augmented IQL

The original conversation suggested generating synthetic rollouts to expand the dataset. This is not the right lever because:

1. **Data quantity is not the bottleneck**: 46M transitions is plenty for a 1.1M parameter network
2. **The V-range compression is inherent to the reward/discount structure**: With r=-1/step and gamma=0.99, V(s) values cluster in a narrow band regardless of data volume
3. **Random rollouts are low-quality**: IQL's expectile regression (tau=0.9) focuses on the upper tail, so bad rollouts contribute little. But they don't fix the dynamic range issue.
4. **We already have a better solution**: SurfaceGraphPBRS gives exact cost-to-go without any learning

The simulator's real value is for:

- Edge enumeration during surface graph precomputation (already used)
- MCTS tree search at test time (Epic 5)
- Potentially generating BC training data (if needed later)

## Updated Master Plan Critical Path

```
Epic 1 (Data Pipeline) -- DONE
    |
Epic 2 (IQL Training) -- DONE (V-network unsuitable for PBRS, policy prior usable)
    |
[NEW] Batch SurfaceGraphPBRS Precomputation (~3-4 hours)
    |
Epic 3 (Env Integration) -- SIMPLIFIED (just verify existing precomputed PBRS works multi-level)
    |
Epic 4 (Multi-Level FiGAR-PPO) -- unchanged
    |
Epic 5 (MCTS Planning) -- unchanged (uses simulator, not V-network)
    |
Epic 6 (Evaluation) -- unchanged
```

