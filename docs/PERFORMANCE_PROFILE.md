# Performance Profile: N++ Environment Step

**Profile date:** Feb 2026 — 500 frames, native physics (post-Optimization 1-5, profiling plan)  
**Command:** `python nclone/test_environment.py --headless --native --profile-frames 500 --map nclone/test_maps/<map>`

---

## Executive Summary

### Multi-Map Baseline (500 frames each, post-Optimizations 6-8)

| Map | Step Time | Steps/s | Physics | Observation | Reward |
|-----|-----------|---------|---------|-------------|--------|
| **simple-walk** | 0.67 ms | 1504 | 15.0% | 37.8% | 15.7% |
| **long-walk** | 0.79 ms | 1268 | 11.0% | 35.4% | 30.3% |
| **switch-simple** | 0.69 ms | 1456 | 12.7% | 31.3% | 29.5% |

*simple-walk uses `--profile-frames` which skips privileged features; baseline was 0.83 ms pre-optimizations.*

**Optimizations 6-8:** Route `find_ninja_node` through step cache, share cache obs↔reward, lazy privileged features. **~20% improvement** (0.83→0.67 ms on simple-walk).

---

## 1. High-Level Component Breakdown (Native Physics, simple-walk)

```
TOTAL STEP                                 0.83 ms | 100%
├── Physics (tick + sync)                   0.09 ms |  10.7%   ← OPTIMIZED (Rust)
├── Get observation                        0.30 ms |  36.5%   ← BOTTLENECK
│   └── Feature computation                1.67 ms | 201.3%*  (*includes sub-calls)
├── Reward calculation                     0.25 ms |  30.0%   ← BOTTLENECK
├── Process observation                    0.09 ms |  10.7%
└── Build episode info                     0.03 ms |   3.5%
```

### 1.1 Feature Computation Breakdown (Reachability Sections)

When feature-level profiling is enabled (`--profile-frames`), the report includes per-section timings:

| Section                       | Avg ms | Notes                           |
|------------------------------|--------|----------------------------------|
| `surface_area`               | 0.88   | get_cached_surface_area          |
| `exit_reachable_and_platform`| 0.46   | find_closest_node + platform     |
| `path_distances`             | 0.25   | get_path_cost x2 (switch, exit)  |
| `ninja_node_and_path_direction` | 0.01 | find_ninja_node + get_next_hop   |
| `extract_spatial_lookups`    | 0.00   | O(1) lookup                      |

*Note: simple-walk has low ninja movement, so feature computation cache hits heavily; sample counts (n) may be low when graph/ninja cell rarely changes.*

### 1.2 Directional Connectivity Cache (Platform Distances)

Platform distances (features 8-9) now use `get_cached_directional_connectivity` with LRU caching. Cache key: `(ninja_cell_x, ninja_cell_y, level_id)`. Max size: 500 entries. Profiling report prints hits, misses, and hit rate when `--profile-frames` is used. On maps with ninja movement across cells, hit rate should be high (ninja moves ~6px/sec, cell = 24px).

---

## 2. cProfile Hot Path (Cumulative Time, Top 15)

*source: profiling_stats.txt, ~0.42s total for 500 steps*

| % Total | Cumulative (ms) | Calls | Function |
|---------|-----------------|-------|----------|
| **18.8%** | **79** | 1006 | `get_path_cost` (path_distance_calculator.py:763) |
| **19.2%** | 81 | 500 | `calculate_combined_potential` (pbrs_potentials.py:1582) |
| **14.3%** | 60 | 1006 | `find_ninja_node` (node_lookup.py:414) |
| **16.2%** | 68 | 500 | `_get_cached_or_compute_potential` (pbrs_potentials.py:1350) |
| **14.7%** | 62 | 500 | `objective_distance_potential` (pbrs_potentials.py:189) |
| **35.6%** | 150 | 501 | `_get_observation` (base_environment.py:2879) |
| **9.3%** | 39 | 501 | `_compute_privileged_features` (npp_environment.py:1430) |
| **8.6%** | 36 | 2002 | `_get_cached_or_find_ninja_node` (path_distance_calculator.py:1991) |
| **7.0%** | 29 | 501 | `_get_ninja_node_position_for_cache` (npp_environment.py:2309) |
| **5.9%** | 25 | 500 | `_compute_spatial_context` (npp_environment.py:2266) |
| **7.8%** | 33 | 500 | `tick` (native_headless.py:94) |
| **1.9%** | 8 | 1002 | `LevelCache.get_distance` (Rust) |
| **1.2%** | 5 | 500 | `compute_local_tile_grid` (spatial_context.py:113) |

---

## 3. Bottleneck Analysis

### 3.1 `get_path_cost` — **#1 bottleneck (~30%)**
- **Calls:** 1006 over 500 steps ≈ **2 calls/step** (spawn→switch, switch→exit, etc.)
- **Time:** 89ms total → **~89μs/call**
- **Cause:** Path distance calculation. When Rust level cache hits: fast (`get_distance` ≈ 9μs for 1002 calls). Misses trigger Python A* or extra work.
- **Opportunity:** Ensure Rust level cache hits more often; reduce number of distinct `get_path_cost` calls per step.

### 3.2 `find_ninja_node` — **#2 bottleneck (~14%, reduced from ~27%)**
- **Calls:** 1006 over 500 steps ≈ **2 calls/step** (down from ~5/step after Optimization 1)
- **Time:** 68ms total → **~68μs/call**
- **Cause:** Python spatial lookup to find the graph node nearest to ninja position. Used by `get_path_cost` and action mask logic.
- **Opportunity:** Port to Rust or add a Rust-backed spatial index; cache result within step.

### 3.3 PBRS reward chain — **#3–5 (~50% combined)**
- `calculate_combined_potential` → `_get_cached_or_compute_potential` → `objective_distance_potential`
- **Cause:** PBRS requires path distances for potential shaping. Each call cascades to `get_path_cost` and `find_ninja_node`.
- **Opportunity:** Batch path lookups; consolidate potential computation; reuse ninja node within step.

### 3.4 Observation / action mask
- `_get_action_mask_with_path_update`, `_detect_straight_path_direction`, `_compute_privileged_features`
- **Cause:** Path-based heuristics and privileged features again trigger node lookups and path queries.
- **Opportunity:** Reuse `find_ninja_node` result; share path data between observation and reward.

---

## 4. What’s Already Optimized

| Component | Status |
|-----------|--------|
| Physics tick | Rust native (`tick` ~36ms for 500 steps) |
| Level cache `get_distance` | Rust (1002 calls ≈ 9ms) |
| Level cache `get_next_hop` | Rust (1002 calls ≈ 5ms) |
| Tile sync | Cached; `build_tile_dic` only on state change |
| Step cache | Deduplicates path lookups within step |

---

## 5. Remaining Optimization Opportunities (by impact)

1. **`find_ninja_node`** — Most expensive single function. Options:
   - Rust spatial hash / nearest-neighbor
   - Cache ninja node per step (invalidate when position moves across threshold)
   - Reduce call count via shared lookups

2. **Reduce `get_path_cost` calls** — Currently ~2/step. Options:
   - Broader use of step cache
   - Batch PBRS potential computation
   - Reuse path results across observation and reward

3. **`get_phi` (Python)** — 500 calls, ~5ms. Options:
   - Delegate to Rust `LevelCache.get_phi` when Rust cache is used

4. **`get_geometric_distance`** — 4508 calls (via `path_distance_cache`). Options:
   - Route through Rust level cache where possible
   - Audit call sites to reduce redundancy

5. **Feature computation** — Single expensive call (~1.66ms) during observation. See §1.1 for section breakdown. Options:
   - Feature-level profiling now available via `--profile-frames`
   - `surface_area` and `exit_reachable_and_platform` dominate; move heavy parts to Rust (e.g. mine SDF already has Rust variant)

---

## 6. Target vs Current

| Metric | Plan Target | Current (simple-walk, profiling) |
|--------|-------------|-----------------------------------|
| Step time | 0.3–0.5 ms | 0.67 ms |
| Gap | — | ~1.3–2× |

Post-Optimizations 6-8: **find_ninja_node** calls reduced ~50% (step cache + position-only sharing). **PBRS** reward calc ~50% faster (obs↔reward cache sharing). **Privileged features** skipped when profiling. Remaining bottleneck: **feature computation** (~1.67 ms when uncached) and **observation** pipeline.
