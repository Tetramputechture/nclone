# Epic 3: Environment and Reward Integration

## Goal

Wire the exported IQL V-network into `NppEnvironment` as a PBRS potential, replacing the current level-specific A\* HWM shaping. After this epic, any level within the target entity scope can be loaded and trained on with no per-level configuration — the reward signal is universally derived from the learned V(s).

## Background

The existing reward pipeline uses `CachedPathDistanceCalculator` (A\* over the tile graph) as the PBRS potential. This is level-specific: for each new level, the graph must be built, the A\* distances computed, and the potential cached. For single-level training this is fine; for multi-level training across 200+ levels it is a bottleneck and a maintenance burden.

`NeuralValuePBRS` already exists and is already integrated into the `TwoPhaseValuePBRS` factory. The missing piece is the configuration path: `EnvironmentConfig` does not yet have a `neural_pbrs_checkpoint` field, so `BaseNppEnvironment._try_load_value_function_pbrs()` has no way to know to use it.

The reward machine (3-state FSM: pre-switch, post-switch, terminal) correctly ensures Φ(terminal) = 0 and provides the phase boundary for two-phase PBRS. This must be verified end-to-end with the neural PBRS before training begins.

## Success Criteria

- `NppEnvironment` loads the neural PBRS from a checkpoint path in `EnvironmentConfig` with no other per-level changes
- `env.reset()` and `env.step()` complete without error on 10 random levels from the filtered corpus
- Reward values are in a reasonable range: between `−2.0` and `+6.0` per step (sparse `+1` events plus PBRS shaping)
- Phase transition: at the frame when the switch is collected, the PBRS potential source switches from `Φ_switch` to `Φ_exit`
- At episode end (win or death), Φ(terminal) = 0 — verify no non-zero terminal potential leaks into the final PBRS delta
- Shaping weight α anneals from 1.0 to 0.1 over training; the annealing is driven by a step counter, not tied to wall time

## Key Files

| File | Role |
|------|------|
| `nclone/gym_environment/config.py` | `EnvironmentConfig` — add `neural_pbrs_checkpoint: Optional[str]` |
| `nclone/gym_environment/base_environment.py` | `_try_load_value_function_pbrs()` — check for neural checkpoint |
| `nclone/gym_environment/reward_calculation/pbrs_potentials.py` | `PBRSCalculator` — add `alpha_shaping` annealing parameter |
| `nclone/gym_environment/reward_calculation/reward_machine.py` | `RewardMachine` — 3-state FSM, `get_total_potential()` |
| `nclone/value_function/two_phase_pbrs.py` | `create_neural_pbrs()` factory |
| `nclone/value_function/neural_value_pbrs.py` | `NeuralValuePBRS` — set_level, get_potential |

---

## Tickets

### T3.1 — Add `neural_pbrs_checkpoint` to `EnvironmentConfig`

**What:** Add an optional `neural_pbrs_checkpoint: Optional[str] = None` field to `EnvironmentConfig` in `nclone/gym_environment/config.py`.

**Why:** This is the configuration entry point that allows callers (training scripts, evaluation scripts) to specify which V-network to use for PBRS without modifying environment internals. Without this field, there is no clean way to opt into neural PBRS.

**Acceptance criteria:**
- `EnvironmentConfig` has `neural_pbrs_checkpoint: Optional[str] = None`
- Setting `neural_pbrs_checkpoint = "/path/to/v_network.pt"` is accepted without error
- The field is documented with a docstring comment explaining its purpose and expected format
- Existing default behavior (no neural checkpoint) is unchanged: `None` falls through to the existing A\* HWM shaping
- `EnvironmentConfig.for_training()` and `for_evaluation()` factory methods pass through the field correctly

**Key implementation notes:**
- Do not add any logic to `config.py` itself — just the field declaration; the loading logic goes in `base_environment.py` (T3.2)
- `EnvironmentConfig` is a dataclass; the new field must have a default value (`None`) to remain backward-compatible with all existing call sites that do not pass this argument

**Relevant files:** `nclone/gym_environment/config.py`

---

### T3.2 — Wire `NeuralValuePBRS` into the environment PBRS loading path

**What:** Modify `BaseNppEnvironment._try_load_value_function_pbrs()` to check `self.config.neural_pbrs_checkpoint` and, if set, create a `TwoPhaseValuePBRS` backed by the neural network instead of the graph-based potential.

**Why:** This is the critical integration point connecting the IQL V-network to the running environment. Once wired, every level loaded in a training environment automatically gets universal PBRS shaping.

**Acceptance criteria:**
- When `config.neural_pbrs_checkpoint` is set:
  - `_try_load_value_function_pbrs()` calls `create_neural_pbrs(checkpoint_path, tile_grid, switch_pos, exit_pos, mine_positions, device)`
  - The resulting `TwoPhaseValuePBRS` is stored as `self._pbrs_calculator.pbrs_potential` (or equivalent depending on the current wiring)
  - The method succeeds silently when the level is loaded; no errors for valid levels
- When `config.neural_pbrs_checkpoint` is `None`: existing behavior is unchanged
- `set_level()` is called on `NeuralValuePBRS` every time a new level is loaded via `env.reset()` (not just once at init)
- Mine positions are passed from the loaded level's entity list, not hardcoded

**Key implementation notes:**
- `_try_load_value_function_pbrs()` currently checks for a precomputed `.npz` file; the neural checkpoint should be checked first (higher priority), with `.npz` and then HWM A\* as fallbacks
- `create_neural_pbrs()` from `two_phase_pbrs.py` takes `(checkpoint_path, tile_grid, switch_pos, exit_pos, mine_positions, device="cpu")`; `tile_grid` is `np.ndarray (42, 23)`, positions are pixel coordinates
- The device should default to `cpu` for training to avoid CUDA memory contention with the policy network; consider a `neural_pbrs_device` config field if GPU inference is later desired
- After `set_level()`, `NeuralValuePBRS` caches the 128D CNN output; subsequent `get_potential()` calls are cheap (~0.01ms)

**Relevant files:** `nclone/gym_environment/base_environment.py`, `nclone/value_function/two_phase_pbrs.py`, `nclone/gym_environment/config.py`

---

### T3.3 — Verify reward machine terminal potential handling

**What:** Write a unit test that runs a complete episode (win and death) through `NppEnvironment` with neural PBRS enabled and verifies that the reward machine correctly zeros the PBRS potential at the terminal step.

**Why:** The PBRS guarantee `R_shaped = R_env + γΦ(s') − Φ(s)` only preserves optimal policy structure when Φ(terminal) = 0. If the terminal state has a non-zero potential, the PBRS delta at the final step is corrupted, causing the agent to over- or under-value terminal transitions.

**Acceptance criteria:**
- On a win step: `PBRS_delta = γ * 0 - Φ(s_before_win)`. The reward includes `R_win = +1` (sparse) plus the shaping delta; the combined value is finite and in range
- On a death step: `PBRS_delta = γ * 0 - Φ(s_before_death)`. Combined reward includes `R_death = -0.5` (or configured value) plus the shaping delta
- `RewardMachine.get_total_potential()` returns `0.0` when `is_terminal=True`
- PBRS clipping: `|γΦ(s') - Φ(s)| ≤ 2.0` — if the delta exceeds this, it should be clipped (verify the clipping is in place in `PBRSCalculator`)

**Test location:** `nclone/gym_environment/reward_calculation/tests/test_pbrs_terminal.py` (new file)

**Key implementation notes:**
- To force a death: step the environment until the ninja reaches a toggled mine
- To force a win: construct a simple level with the exit door directly adjacent to spawn and no mines, collect the switch, then walk to the exit
- `PBRSCalculator` currently has reward clipping parameters; check that `|γΦ(s') - Φ(s)| ≤ 2.0` is enforced; add it if missing with a configurable `pbrs_clip_max: float = 2.0` in `RewardConfig`

**Relevant files:** `nclone/gym_environment/reward_calculation/reward_machine.py`, `nclone/gym_environment/reward_calculation/pbrs_potentials.py`

---

### T3.4 — Add shaping weight annealing (α: 1.0 → 0.1)

**What:** Add an `alpha_shaping` parameter and linear annealing schedule to `PBRSCalculator` so the PBRS shaping weight decreases from 1.0 to 0.1 over the first 60% of training steps.

**Why:** IQL's V-function introduces a small bias under function approximation — the learned potential is an approximation, not the true value function. Annealing α reduces the agent's dependence on this biased signal over time, letting the true sparse rewards dominate in the later training phase where the policy is already near-optimal on easy levels.

**Acceptance criteria:**
- `PBRSCalculator` accepts `alpha_shaping: float = 1.0` and `total_training_steps: int = 0` parameters
- A `update_step(step: int)` method (or equivalent) computes the current α using linear interpolation: α = max(0.1, 1.0 - 0.9 * min(1.0, step / (0.6 * total_training_steps)))
- The shaped reward uses the current α: `R_shaped = R_env + alpha_shaping * PBRS_delta`
- `alpha_shaping` is logged to TensorBoard every 1K steps (as a diagnostic metric)
- When `total_training_steps = 0`, α remains fixed at 1.0 (backward compatible)

**Key implementation notes:**
- `PBRSCalculator` is currently wired to receive the step count from the training loop via the callback system; follow the same pattern rather than polling wall time
- The existing `alpha` field in `PBRSCalculator` may refer to the IQL blend ratio (not the PBRS shaping weight); be careful not to confuse these; the new parameter should be named `alpha_shaping` to be unambiguous
- The 60% threshold (start of flat phase at α=0.1) should be a configurable parameter, not a hardcoded constant

**Relevant files:** `nclone/gym_environment/reward_calculation/pbrs_potentials.py`, `nclone/gym_environment/reward_calculation/reward_config.py`

---

### T3.5 — Integration test: NppEnvironment with neural PBRS across 10 random levels

**What:** Write and run an integration test that loads 10 random levels from the filtered corpus, runs each for 500 steps under a random policy, and verifies that the full reward pipeline works correctly end-to-end.

**Why:** Unit tests for individual components do not catch integration bugs (e.g., mine position extraction from entity list not matching mine features in the observation, level cache not being invalidated between resets, or device mismatches between policy and PBRS network).

**Acceptance criteria:**
- All 10 levels complete 500 steps without error or NaN reward
- `env.reset()` on a new level calls `NeuralValuePBRS.set_level()` with the correct tile grid, switch position, exit position, and mine positions
- Rewards are finite and in `[-2.0, +6.0]` throughout
- `env.reset()` called twice on the same level produces identical initial observations
- Step timing: `env.step()` including PBRS calculation completes in <5ms per step on CPU (the neural PBRS per-step cost should be negligible)
- The test runs with no GPU dependency (`device="cpu"`)

**Test location:** `nclone/gym_environment/tests/test_neural_pbrs_integration.py` (new file)

**Key implementation notes:**
- Select 10 levels from different level ID ranges to cover diverse tile layouts and mine counts
- Use `EnvironmentConfig` with `neural_pbrs_checkpoint` set; do not patch or mock the network — use the real exported `v_network.pt`
- Verify that on a new `env.reset()` (new level), the cached level repr in `NeuralValuePBRS` is updated and does not carry over from the previous level
- Time each `env.step()` call with `time.perf_counter()` to confirm the performance budget is met

**Relevant files:** `nclone/gym_environment/base_environment.py`, `nclone/gym_environment/config.py`, `nclone/value_function/neural_value_pbrs.py`
