# Epic 5: Test-Time MCTS Planning

## Goal

Implement Gumbel AlphaZero tree search that uses the Rust simulator as a perfect world model. At inference time, given a level and an initial state, the planner searches the action space using the IQL policy as a prior and the IQL value function as a leaf evaluator — then executes actions one at a time in a Model Predictive Control loop.

## Background

The critical architectural insight from the blueprint: N++'s deterministic Rust simulator (`nclone_native`) is a perfect world model. Unlike MuZero which learns a dynamics model that fails at depth, our planner gets zero-error physics at ~0.1ms per step. At that speed, a 10-minute planning budget yields ~6M simulator steps — enough for deep search across every decision point in a level.

**Why Gumbel AlphaZero over standard MCTS:** Gumbel MuZero (Danihelka et al., 2022) uses the Gumbel-Top-k trick to sample candidate actions without replacement from the policy prior, then applies sequential halving to allocate simulation budget. This guarantees policy improvement with as few as 2 simulations and handles the 6-action N++ branching factor efficiently. With k=6 (full action coverage) and a modest sequential halving budget of 50–200 simulations per decision step, the planner reliably improves over the raw IQL policy.

**Action space for MCTS:** The MCTS tree uses the 6 base N++ actions (NOOP, LEFT, RIGHT, JUMP, JUMP+LEFT, JUMP+RIGHT). Each tree edge represents committing to one action for a fixed duration (e.g., 4 frames per step — a hyperparameter). This avoids the 13-bin duration branching that would increase tree width to 78 and makes the search tractable.

**MPC loop:** Plan H=15–25 macro-actions ahead at each decision point. Execute only the first action (for its chosen duration). Replan from the new state. This rolling-horizon approach provides robustness through continuous replanning while the IQL value function handles beyond-horizon estimation.

## Success Criteria

- Gumbel MCTS planner runs end-to-end on a fresh level with no level-specific configuration
- Planner completes 50 simulations per decision step in <500ms on CPU (fast enough for practical use)
- On 10 held-out levels: MCTS achieves strictly higher solve rate than the raw IQL policy-only baseline
- MPC loop successfully completes at least one full level (spawn → switch → exit) on a held-out level
- ReZero subtree reuse reduces redundant simulation by at least 30% compared to rebuilding the tree from scratch each step (measured by simulation call count)

## Key Files

| File | Role |
|------|------|
| `nclone/native_headless.py` | `NativePlayHeadless` — Rust sim wrapper; `tick()`, `fast_reset()`, `get_ninja_state()` |
| `nclone/value_function/neural_value_pbrs.py` | `NeuralValuePBRS.get_potential()` — leaf value evaluation |
| `nclone/offline_rl/algorithms/iql.py` | `IQLTrainer` — policy prior `P(a|s)` for MCTS |
| `nclone/offline_rl/networks/value_networks.py` | `IQLNetworks.policy()` — 6-action softmax prior |
| New: `nclone/planning/__init__.py` | New module |
| New: `nclone/planning/gumbel_mcts.py` | Gumbel-Top-k MCTS implementation |
| New: `nclone/planning/search_node.py` | Tree node data structure |
| New: `nclone/planning/sim_fork.py` | Simulator state forking for tree search |
| New: `nclone/planning/mpc_planner.py` | MPC loop: plan → execute first action → replan |

---

## Tickets

### T5.1 — Define tree node and search tree data structures

**What:** Implement `nclone/planning/search_node.py` with the data structures for MCTS tree nodes and edges, and `nclone/planning/sim_fork.py` for forking simulator state at each tree node.

**Why:** Correct, cache-friendly data structures are the foundation of a maintainable MCTS. These must be defined before any search logic is written.

**Acceptance criteria:**
- `SearchNode` dataclass has fields:
  - `state_snapshot`: simulator state at this node (bytes or dict — whatever `nclone_native` returns from `get_state()`)
  - `prior: np.ndarray` — shape (6,), the IQL policy prior at this node
  - `value_estimate: float` — IQL V(s) at this node (set at node creation, used for leaf evaluation)
  - `visit_count: int` — N(s)
  - `value_sum: float` — W(s) for computing Q(s, a) = W(s, a) / N(s, a)
  - `children: Dict[int, SearchNode]` — action → child node (empty until expanded)
  - `action_visit_counts: np.ndarray` — shape (6,), N(s, a)
  - `action_value_sums: np.ndarray` — shape (6,), W(s, a)
  - `is_terminal: bool` — True if this node is a win or death state
  - `terminal_value: float` — 0.0 (death) or 1.0 (win) if terminal
- `SimFork.fork(sim_state)` creates an independent `NativePlayHeadless` from a snapshot, ready to `tick()`
- `SimFork` implements `__enter__` / `__exit__` for context manager use (ensures the forked sim is properly cleaned up)
- All data structures are serializable (no references to live simulator objects stored in `SearchNode`)

**Key implementation notes:**
- `nclone_native` must expose a `serialize_state()` / `deserialize_state()` interface for forking; check the Rust bindings in `nclone/native_headless.py` to see what state snapshot APIs exist
- If serialization is not available, use `fast_reset()` + state injection as an alternative fork strategy
- Keep `SearchNode` lightweight — avoid storing the full observation tensor; only store what is needed to reconstruct it

**Relevant files:** `nclone/native_headless.py`, new: `nclone/planning/search_node.py`, `nclone/planning/sim_fork.py`

---

### T5.2 — Implement Gumbel-Top-k action sampling

**What:** Implement the Gumbel-Top-k sampling and sequential halving logic in `nclone/planning/gumbel_mcts.py`.

**Why:** Standard UCB1 MCTS explores actions uniformly, ignoring the policy prior. Gumbel-Top-k samples k candidate actions from the prior without replacement using Gumbel noise, ensuring high-probability actions are explored while maintaining coverage. Sequential halving eliminates low-performing candidates in rounds, concentrating the remaining budget on the best candidates.

**Acceptance criteria:**
- `gumbel_top_k(logits: np.ndarray, k: int) -> np.ndarray` — returns the top-k action indices sampled via `logits + Gumbel(0,1)`, without replacement, shape `(k,)`
- `sequential_halving(candidates: List[int], budget: int, node: SearchNode, sim: NativePlayHeadless, value_fn, prior_fn) -> int` — returns the best action after halving rounds
  - Round 1: simulate each candidate `budget // (k * ceil(log2(k)))` times
  - Eliminate bottom half by Q-value + prior bonus
  - Repeat until 1 candidate remains
- Both functions are pure Python + numpy with no side effects on the tree or simulator
- Unit tests: `gumbel_top_k` with a uniform prior produces near-uniform distribution over many samples; with a peaked prior it concentrates on high-probability actions

**Key implementation notes:**
- The Gumbel noise trick: sample `g ~ Gumbel(0,1)` by `g = -log(-log(U))` where `U ~ Uniform(0,1)`. Then rank `logits + g`
- Sequential halving budget formula: if k=6 candidates and total budget B simulations, round r uses B / (k * ceil(log2(k))) simulations per surviving candidate, eliminating half each round
- For N++ with k=6 and B=50: 3 halving rounds (6→3→2→1), ~8 simulations per candidate in the first round
- Keep `gumbel_top_k` and `sequential_halving` independent of the simulator and tree — they should operate only on numpy arrays and a callable value function

**Relevant files:** new `nclone/planning/gumbel_mcts.py`

---

### T5.3 — Wire simulator integration for MCTS rollouts

**What:** Build the simulation execution layer that forks the simulator state at each tree node, executes a candidate action for the configured duration, and returns the resulting state for value evaluation.

**Why:** The simulator must be forked cleanly at each node to allow parallel exploration of different action branches. This is the layer that translates the abstract MCTS operation ("apply action A for D frames") into concrete physics simulation.

**Acceptance criteria:**
- `simulate_action(node: SearchNode, action: int, duration: int) -> SearchNode` — forks `node`'s simulator state, runs `action` for `duration` frames, returns a new `SearchNode` with the resulting state
- Win/death detection: after each `tick()`, check `sim.ninja_has_won()` and `sim.ninja_has_died()`; set `is_terminal` and `terminal_value` accordingly
- `duration` parameter is configurable per search call (default 4 frames); a contact-interrupt variant is also supported: stop early if the ninja's movement regime changes
- State extraction: after simulation, call a provided `extract_state(sim) -> np.ndarray` function to get the 11D + 24D feature vector for the value function
- Simulation steps are counted globally and returned; the caller uses this count for budget tracking

**Key implementation notes:**
- The `NativePlayHeadless` interface exposes `tick(action_hoz, action_jump)` where inputs are `(-1, 0, 1)` for horizontal and `(0, 1)` for jump; map the 6 base action indices to these pairs
- Action mapping: `0=NOOP=(0,0)`, `1=LEFT=(-1,0)`, `2=RIGHT=(1,0)`, `3=JUMP=(0,1)`, `4=JUMP+LEFT=(-1,1)`, `5=JUMP+RIGHT=(1,1)`
- Contact interrupt: compare ninja movement regime before and after each frame via `get_ninja_state()[4:6]` (on_ground, wall_sliding flags at indices 4–5 of the 11D state vector)
- Avoid deep copying the full simulator on every node expansion; instead, use the serialized `state_snapshot` to reconstruct the state only when actually simulating that branch

**Relevant files:** `nclone/native_headless.py`, new `nclone/planning/sim_fork.py`, `nclone/planning/gumbel_mcts.py`

---

### T5.4 — Integrate IQL policy prior and value function as MCTS components

**What:** Wire the IQL policy (6-action softmax) and value function (V(s)) into the MCTS search as the prior distribution and leaf evaluator, respectively.

**Why:** These are the two components that give the MCTS search its intelligence. Without a prior, Gumbel sampling degenerates to random action selection. Without a value function, leaf nodes require expensive simulation rollouts to estimate returns.

**Acceptance criteria:**
- `PolicyPrior` class wraps the IQL policy head: `get_prior(state: np.ndarray, mine_features: np.ndarray, level_repr: np.ndarray) -> np.ndarray` returns a 6-element softmax array
- `ValueEvaluator` class wraps `NeuralValuePBRS`: `get_value(state, mine_features) -> float` returns V(s) (a scalar negative float representing estimated time-to-completion)
- Both classes cache the level encoding (CNN output) per level and do not re-run the CNN between calls
- Both classes support batched evaluation: `get_prior_batch(states) -> np.ndarray` shape `(B, 6)` for evaluating multiple nodes in one forward pass
- `PolicyPrior` and `ValueEvaluator` are initialized with the same `v_network.pt` checkpoint; they share the same `IQLNetworks` instance to avoid loading the weights twice

**Key implementation notes:**
- The IQL policy head outputs unnormalized logits; apply softmax before returning; add a small temperature parameter `T=1.0` (lower T sharpens the prior, higher T flattens it toward uniform)
- Temperature can be swept during evaluation: lower T may help on levels where the policy is confident; higher T may help on novel layouts
- The level encoding needed by `PolicyPrior` and `ValueEvaluator` must be precomputed when the planner is initialized on a new level (call `set_level()` once per level, not per node)
- Batch evaluation requires accumulating all leaf nodes before the forward pass; implement a "pending evaluation" queue that the search tree fills and flushes at the end of each halving round

**Relevant files:** `nclone/value_function/neural_value_pbrs.py`, `nclone/offline_rl/networks/value_networks.py`, new `nclone/planning/gumbel_mcts.py`

---

### T5.5 — Implement the MPC planning loop

**What:** Implement `nclone/planning/mpc_planner.py`, the top-level planner that calls Gumbel MCTS at each decision point, executes the best action in the real environment, and replans from the new state.

**Why:** MCTS alone is not an execution policy — it produces the best action from the current state given a simulation budget. MPC wraps MCTS into a closed-loop controller: plan → execute first action → receive new state → replan. This rolling-horizon approach makes the planner robust to execution noise and allows it to incorporate feedback from the actual game state after each action.

**Acceptance criteria:**
- `MPCPlanner.plan_and_execute(env, level_path, max_steps=500)` runs the full MPC loop:
  1. `env.reset()` with the given level
  2. At each decision step: run Gumbel MCTS with `n_simulations=50`, `k=6`, `duration=4`
  3. Execute the best action via `env.step(action, duration)`
  4. Repeat until win, death, or `max_steps` exceeded
  5. Return `(solved, steps, trajectory)` where `trajectory` is the sequence of (state, action, reward) tuples
- ReZero subtree reuse: after executing action `a*`, keep the subtree rooted at the child node corresponding to `a*` as the root for the next search; discard other branches
- `n_simulations`, `k`, `duration`, `planning_horizon` are all configurable parameters
- The planner logs per-decision-step statistics: simulations used, best action, best action prior probability, Q-value of best action vs. Q-value of policy-greedy action (to measure search improvement)

**Key implementation notes:**
- ReZero (2024): after executing action `a*`, the child `SearchNode` already has `visit_count > 0` from the just-completed search; use it directly as the new root. Update `state_snapshot` to match the actual post-execution state (may differ from the simulated state if contact interrupt fired)
- Decision steps vs. primitive steps: one "decision step" executes the chosen action for `duration` frames; the MPC replans every decision step, not every primitive frame
- The planner should not have any knowledge of the level format — it operates purely through the `env.step()` and `env.reset()` interfaces

**Relevant files:** new `nclone/planning/mpc_planner.py`, `nclone/planning/gumbel_mcts.py`, `nclone/gym_environment/npp_environment.py`

---

### T5.6 — Tune MCTS hyperparameters and validate on benchmark levels

**What:** Run the MCTS planner on the 10-level held-out benchmark from Epic 6 and systematically vary `n_simulations`, `k`, and `duration` to find a configuration that lifts solve rate by ≥15 percentage points over the policy-only baseline.

**Why:** MCTS performance is highly sensitive to hyperparameters. Too few simulations and the search is dominated by noise. Too many and the planning time exceeds practical budgets. The duration per step determines how many macro-actions fit in the planning horizon.

**Acceptance criteria:**
- Grid search over: `n_simulations ∈ {20, 50, 100, 200}`, `k ∈ {3, 6}`, `duration ∈ {2, 4, 8}`
- Results table recorded in `docs/generalizable_agent_v2/05-test-time-mcts-planning/tuning-results.md`: each config's solve rate on the 10-level benchmark and mean planning time per decision step
- Best configuration achieves: solve rate ≥ (policy-only + 15pp), planning time ≤ 500ms per decision step
- ReZero subtree reuse is confirmed to reduce simulation count by ≥30% on a reference level (compare simulation call counts with and without ReZero across 10 runs)
- Findings documented: which hyperparameter has the most impact, diminishing returns threshold, recommended production config

**Key implementation notes:**
- Run each configuration 5 times per level (MCTS has Gumbel sampling noise) and report mean and std of solve rate
- "Policy-only baseline" means running the IQL policy greedily (argmax on logits) with no tree search, using the same `duration` setting
- For the 10-minute planning budget scenario (Epic 6's "extended" evaluation): use `n_simulations=200` and report the solve rate separately from the fast (<500ms) configuration

**Relevant files:** new `nclone/planning/mpc_planner.py`, `docs/generalizable_agent_v2/05-test-time-mcts-planning/tuning-results.md`
