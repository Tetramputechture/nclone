# Epic 6: Evaluation and Benchmarking

## Goal

Build evaluation infrastructure to measure generalization on held-out levels and produce a clear, reproducible comparison of the system components: IQL policy zero-shot, FiGAR-PPO online RL, Gumbel MCTS planning, and the combined pipeline. This epic answers the question: did we achieve the first goal?

## Background

Evaluation in the multi-level setting is qualitatively different from single-level training. The key question is not "does the agent solve this training level?" but "does the agent generalize to levels it has never seen?". This requires:
1. A held-out benchmark set of levels that the agent has never trained on
2. An evaluation harness that runs each approach deterministically and measures solve rate
3. Metrics that capture both success and effort (steps to completion, not just binary success)
4. A comparison that isolates each component's contribution

The benchmark levels must be from the same entity scope (1 exit, 1 switch, toggle mines, any tile layout) but must be genuinely unseen — not just unseen during PPO training, but also absent from the IQL training corpus (Epic 1) to test true generalization.

## Success Criteria

- Benchmark suite contains ≥100 held-out levels spanning a range of difficulties
- Evaluation harness runs reproducibly (fixed random seed, deterministic sim)
- All four approaches are benchmarked on the same 100 levels
- Results table is produced and committed to `docs/generalizable_agent_v2/06-evaluation-benchmarking/results.md`
- At least one of the four approaches meets its success criterion from the master plan

## Key Files

| File | Role |
|------|------|
| `nclone/gym_environment/npp_environment.py` | Environment for policy evaluation |
| `nclone/value_function/neural_value_pbrs.py` | Value function for MCTS leaf eval |
| `nclone/planning/mpc_planner.py` | MCTS planner from Epic 5 |
| New: `nclone/evaluation/benchmark_suite.py` | Benchmark level collection and loader |
| New: `nclone/evaluation/evaluator.py` | Evaluation harness |
| New: `nclone/evaluation/metrics.py` | Metric computation |
| New: `scripts/run_evaluation.py` | CLI entry point |

---

## Tickets

### T6.1 — Build held-out benchmark level suite

**What:** Curate or generate a set of ≥100 benchmark levels that are guaranteed to be disjoint from the IQL training corpus (Epic 1's filtered attract dataset), then store them in `datasets/benchmark/`.

**Why:** Levels used in the attract corpus have, in some sense, been "seen" by the IQL V-network — the V-network was trained on transitions from those levels. To measure true generalization, the benchmark must contain genuinely new level configurations.

**Acceptance criteria:**
- ≥100 levels in `datasets/benchmark/` as standard level files (`.npp` format or binary map data files)
- Zero level ID overlap with the training and validation HDF5 datasets from Epic 1
- Levels span difficulty tiers (documented by at least one metric per level, e.g., expert completion time from a reference policy or shortest path length):
  - Easy: ≤100 expert frames to complete, 0–10 mines, simple tile layout
  - Medium: 100–500 expert frames, 10–100 mines, some platforming
  - Hard: 500+ expert frames, 50–256 mines, complex tile geometry requiring momentum exploitation
- A `datasets/benchmark/manifest.json` enumerates all levels with: level_id, difficulty_tier, mine_count, tile_complexity_metric, expert_completion_frames (if known)

**Key implementation notes:**
- Source options: (a) N++ community levels from a separate corpus not included in the attract dataset, (b) procedurally generated levels using the existing map generation tools in `nclone/map_generation/`, or (c) hand-curated levels covering known difficulty patterns
- Tile complexity metric: one simple proxy is the number of non-empty, non-solid tile types in the level (count tile types 2–33 which are slopes and curves)
- Procedural generation is the most reliable way to guarantee level ID non-overlap; use a separate random seed range not used in training data generation
- Verify each level is solvable by running the Python simulator manually or with a brute-force search; unsolvable levels must be excluded

**Relevant files:** `nclone/map_generation/`, `nclone/offline_rl/dataset/filter_attracts.py` (for entity-set compliance check)

---

### T6.2 — Implement evaluation harness

**What:** Create `nclone/evaluation/evaluator.py` with a `Evaluator` class that runs a given policy or planner on the full benchmark suite and returns structured results.

**Why:** A consistent, reusable harness ensures that all comparisons (IQL zero-shot, FiGAR-PPO, MCTS) are evaluated under identical conditions. Without a harness, ad-hoc evaluation scripts tend to introduce subtle differences (different episode time limits, different random seeds, different level ordering) that make comparisons unreliable.

**Acceptance criteria:**
- `Evaluator.run(policy_fn, benchmark_dir, n_trials=5, max_steps=10000, seed=42) -> EvaluationResult` evaluates `policy_fn` on all levels in `benchmark_dir`
- `policy_fn(obs: dict) -> int` takes an observation and returns an action index (0–5); the harness handles environment stepping
- Each level is run `n_trials` times with different seeds (for stochastic policies); solve rate per level = solved_trials / n_trials
- `EvaluationResult` contains: per-level solve rate, per-level mean steps to completion (for solved trials), per-level mean final reward, global metrics
- The harness is deterministic when `seed` is fixed: same results on repeated runs
- Evaluation is single-threaded for reproducibility; parallelism is left to the calling script via multiprocessing

**Key implementation notes:**
- `policy_fn` interface is intentionally simple (maps obs to action) to support all four baselines: IQL (argmax on policy logits), FiGAR-PPO (argmax on action head), MCTS (run planner, return first action), combined (MCTS with PPO prior)
- The MCTS planner from Epic 5 has a different interface (takes environment reference, not per-step obs); wrap it in an adapter that matches `policy_fn` signature, or provide a separate `Evaluator.run_planner()` method
- Steps to completion for failed trials: use `max_steps` as the imputed value (for averaging purposes, penalizing failures)

**Relevant files:** new `nclone/evaluation/evaluator.py`, `nclone/gym_environment/npp_environment.py`

---

### T6.3 — Define and implement evaluation metrics

**What:** Implement `nclone/evaluation/metrics.py` with the full set of metrics used to compare approaches, and a `MetricsReport` class that formats results as a readable table.

**Why:** Binary solve rate alone is insufficient. Two approaches with identical solve rates may differ substantially in efficiency (steps to completion), reliability (variance across trials), and difficulty sensitivity (easy vs. hard levels). A richer metric set allows targeted diagnosis of where each approach fails.

**Acceptance criteria:**
- `MetricsReport` computes and reports:
  - **Solve rate** — fraction of (level, trial) pairs solved; overall and by difficulty tier
  - **Steps to completion** — mean and std across solved trials; overall and by difficulty tier
  - **Time to first solve** — for MCTS, total simulator steps consumed until first solution
  - **Normalized score** — `(max_steps - steps_to_completion) / max_steps`; 0 for failures, positive for successes; aggregated as mean
  - **MCTS improvement** — for MCTS-enabled approaches, measure Δsolve_rate vs. policy-only baseline
- `MetricsReport.to_markdown()` produces a formatted Markdown table suitable for committing to the results doc
- `MetricsReport.to_json()` produces a machine-readable summary for future automated comparison

**Key implementation notes:**
- "Solve rate by difficulty tier" requires the manifest from T6.1 to map level IDs to tiers
- MCTS-specific metric "simulator steps consumed" is logged by `MPCPlanner` from T5.5; aggregate per level and per trial
- Normalized score is preferable to raw step count for aggregation across levels of different lengths; it penalizes failures uniformly regardless of level difficulty

**Relevant files:** new `nclone/evaluation/metrics.py`, `nclone/evaluation/evaluator.py`

---

### T6.4 — Benchmark all four approaches

**What:** Run the evaluation harness on all four approaches and record results:
1. **IQL zero-shot**: IQL policy (argmax on 6-action logits), no online RL, no MCTS
2. **FiGAR-PPO**: trained FiGAR-PPO policy from Epic 4, no MCTS
3. **Gumbel MCTS (IQL prior)**: MCTS with IQL policy prior and value function, no online RL
4. **Combined**: FiGAR-PPO as MCTS prior + IQL value as leaf evaluator

**Why:** Isolating each component reveals where the value comes from. If IQL zero-shot already achieves 40% and MCTS lifts it to 55%, the MCTS contribution is clear. If FiGAR-PPO reaches 70%, the online training contribution is clear. This attribution is important for deciding where to invest in future work.

**Acceptance criteria:**
- All four approaches are evaluated on all 100 benchmark levels with n_trials=5
- Results table in `docs/generalizable_agent_v2/06-evaluation-benchmarking/results.md` with columns: approach, overall solve rate, solve rate by tier, mean steps (solved), MCTS sim steps (where applicable)
- The evaluation is run from a single script `scripts/run_evaluation.py` with approach selection via `--approach {iql,ppo,mcts,combined}` flags
- Runtimes are recorded: total evaluation time per approach (wall-clock) and per-level mean planning time for MCTS approaches
- The script is idempotent: re-running does not overwrite existing results, it appends with a timestamp

**Key implementation notes:**
- For IQL zero-shot: load `v_network.pt`; call `IQLNetworks.policy(batch)` to get 6-action logits; take argmax; use a fixed duration of 4 frames per action (same as MCTS default)
- For FiGAR-PPO: load the checkpoint from Epic 4; use `FiGARActorCriticPolicy.predict()` to get (action, duration) pairs; the policy handles duration internally
- For MCTS: use the best hyperparameter configuration from T5.6
- For Combined: replace the IQL prior in MCTS with the FiGAR-PPO policy's action head; value function remains IQL V(s)

**Relevant files:** `scripts/run_evaluation.py` (new), `nclone/evaluation/evaluator.py`, `nclone/planning/mpc_planner.py`

---

### T6.5 — Document results and findings

**What:** Write `docs/generalizable_agent_v2/06-evaluation-benchmarking/results.md` with the full results table, analysis of findings, and recommended next steps.

**Why:** Results without interpretation are not actionable. The findings document serves as the record of what was learned from the first goal, which components worked, and what the failure modes are. This directly informs the scope expansion to dynamic entities.

**Acceptance criteria:**
- Results document includes:
  - Full metrics table from T6.4 for all four approaches across all difficulty tiers
  - Analysis section: which component contributed the most lift? What is the failure mode on hard levels?
  - Representative failure cases: 2–3 specific hard levels where all approaches fail, with explanation of why (e.g., requires long momentum setup that PBRS does not capture)
  - Representative success cases: 2–3 levels where MCTS succeeds but policy-only fails, showing the value of planning
  - Comparison to master plan success criteria (see `00-master-plan.md`): which criteria were met, which were not
- Next steps section: what would need to change to solve the hard levels? What is the minimum viable extension to support dynamic entities?
- Document is ≤2000 words (concise and actionable)

**Relevant files:** `docs/generalizable_agent_v2/06-evaluation-benchmarking/results.md` (new), `docs/generalizable_agent_v2/00-master-plan.md`
