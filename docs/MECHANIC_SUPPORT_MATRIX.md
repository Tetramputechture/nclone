# Mechanic support matrix

Single source of truth for **what is simulated**, **what the observation exposes**, **what the offline planner models**, and **what the hybrid / non-ML baseline supports**. Used by curriculum regret metrics (NPP-26, NPP-27), expert routes (NPP-32), and observation-completeness audits (Linear NEW-D).

**Related:** `npp_rl/benchmarks/mechanic_buckets.py`, `npp_rl/benchmarks/SUPPORT_BOUNDARY.md`, `npp_rl/benchmarks/HYBRID_FIRST_CRITERIA.md`, `docs/sim_mechanics_doc.md`.

## Legend

| Column | Meaning |
|--------|---------|
| **Python sim** | Full `nclone` Gym / training environment. |
| **Rust sim (Phase 1)** | `nclone_native` entities used by non-ML precompute: ToggleMine, ExitSwitch, ExitDoor only. |
| **Obs (current)** | What the live observation dict exposes **today** (subject to change as Layer 1 lands). |
| **Planner (tile graph)** | Velocity-augmented surface graph / Dijkstra in `nclone/value_function/` — **approximate** for many mechanics. |
| **Baseline Tier** | From `SUPPORT_BOUNDARY.md`: Tier 1 = plan-only strong; Tier 2 = hybrid; Tier 3 = weak without ML. |
| **Bucket** | Primary `BucketType` in `mechanic_buckets.py` (if applicable). |

## Matrix (high level)

| Mechanic / feature | Python sim | Rust sim (Phase 1) | Obs (current — verify in code) | Planner modeled | Baseline tier | Benchmark bucket | **Layer 1 scope** |
|-------------------|------------|--------------------|--------------------------------|-----------------|---------------|------------------|-------------------|
| Flat run / walk | Yes | Yes | `tile_grid`, `game_state` | Yes (Tier 1) | 1 | FLAT_RUN | **in_scope** |
| Basic jump / gap / fall | Yes | Yes | `tile_grid`, `game_state` | Yes (Tier 1) | 1 | BASIC_JUMP | **in_scope** |
| Wall jump | Yes | Yes* | `tile_grid`, `game_state` | Yes* | 1 | WALL_JUMP | **in_scope** |
| Slopes / ramps | Yes | Varies by map | `tile_grid`, `game_state` | Yes (Tier 1) | 1 | SLOPES | **in_scope** |
| Toggle mines | Yes | Yes | `entity_overlays[mine_deadly_density, mine_safe_density]`, `hazard_map[deadly_sdf]` (Layer 1) | Partial (static deadly gap) | 1 | MINE_NAVIGATION | **in_scope** |
| Exit switch / exit door | Yes | Yes | `entity_overlays[switch_pos, exit_pos]`, `game_state.switch_activated` | Yes (goals) | 1 | (cross-cutting) | **in_scope** |
| Locked / regular doors | Yes | No | level-dependent | Tier 2 (phase Dijkstra) | 2 | SWITCH_PUZZLE | waived (out of scope, Layer 1 toggle-mines chunk) |
| Trap doors | Yes | No | level-dependent | Partial | 2–3 | SWITCH_PUZZLE | waived (out of scope) |
| One-way platforms | Yes | No | level-dependent | Needs directed edges | 2 | (often SWITCH_PUZZLE) | waived (out of scope) |
| Launch pads | Yes | No | overlays (planned L1) | Partial (NPP-32 optional) | 2 | LAUNCH_PAD | waived (out of scope) |
| Bounce blocks | Yes | No | overlays / hazard (planned) | Partial (NPP-32 optional) | 2 | — | waived (out of scope) |
| Thwumps | Yes | No | overlays / hazard (planned) | Weak static | 2–3 | THWUMP | waived (out of scope) |
| Drones | Yes | No | overlays / hazard (planned) | Weak | 3 | DRONE | waived (out of scope) |
| Death balls | Yes | No | overlays / hazard (planned) | Weak | 3 | DEATH_BALL | waived (out of scope) |
| Moving platforms | Yes | No | — | — | 3 | — | waived (out of scope) |

\*Wall-jump levels with **airborne goals** may fail precompute until virtual-goal support lands; see notes in `mechanic_buckets.py`.

### Layer 1 scope notes

Layer 1 is scoped to **toggle mines only** for entities; see [LAYER1_MARKOV_AUDIT.md](LAYER1_MARKOV_AUDIT.md) for the Markov gate and per-row observation-key mapping, and [LINEAR_PLAN_AUDIT.md](LINEAR_PLAN_AUDIT.md) for the scope rationale. The "waived" rows are **not gaps in the observation**; they are explicit non-goals of this chunk. Future chunks that bring any of these mechanics back must extend this matrix and refresh the Markov audit before new observation channels are added.

## Per-level support flag

For PLR / expert-route issues:

- Each benchmark level entry in `BUCKET_LEVELS` already has `rust_compatible` and `secondary` mechanics.
- For curriculum levels not in the registry, maintain an explicit **`planner_support: full | partial | none`** (or reuse tier) so “expert regret” is never labeled **true oracle** when the planner omits dynamics.

## Observation completeness (NEW-D)

For each row above where **Python sim = Yes**, NEW-D requires either:

1. The variable appears in `game_state`, `tile_grid`, `entity_overlays`, or `hazard_map` (once L1 lands), **or**
2. A documented proof that the policy Markov state does not need it (rare).

Update this matrix when observation keys change (e.g. `mine_threat_map` → `hazard_map`).
